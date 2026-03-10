
import os.path as osp
from collections import OrderedDict
import math

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from scipy.fft import fft, ifft
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from PIL import Image
import torchvision.transforms as transforms

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from trainers.dictionary import class_descriptions

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        #print("in text model")
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        #print("in text2 model")
        x = self.ln_final(x).type(self.dtype)
        #print("in text3 model")

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        #print("tokenized_prompts.argmax(dim=-1):", tokenized_prompts.argmax(dim=-1))
        #print("self.text_projection:", self.text_projection)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        #print("in text4 model")
        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COCOOP.N_CTX
        ctx_init = cfg.TRAINER.COCOOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))
        
        if cfg.TRAINER.COCOOP.PREC == "fp16":
            self.meta_net.half()

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            text_features = clip_model.encode_text(tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        #print("Shape of text_features:", text_features.shape)
        self.text_features = text_features
        description_texts = [class_descriptions[name] for name in classnames]
        tokenized_descriptions = [clip.tokenize(desc) for desc in description_texts]

        # Convert each description into embeddings and compute the mean
        description_embeddings = []
        for tokens in tokenized_descriptions:
           with torch.no_grad():
               description_embedding = clip_model.encode_text(tokens)
               description_embeddings.append(description_embedding)

        # Step 2: Take the mean across all embeddings for each class's description
        mean_description_embeddings = torch.stack([torch.mean(desc_embeds, dim=0) for desc_embeds in description_embeddings])

        prompts1 = ["Satellite photo of a" + " " + name + "." for name in classnames]

        tokenized_prompts1 = torch.cat([clip.tokenize(p) for p in prompts1])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding1 = clip_model.token_embedding(tokenized_prompts1).type(dtype)
            text_features1 = clip_model.encode_text(tokenized_prompts1)
            text_features1 = text_features1 / text_features1.norm(dim=-1, keepdim=True)
        #print("Shape of text_features:", text_features.shape)
        self.text_features1 = text_features1

        prompts2 = ["Aerial photo of a" + " " + name + "." for name in classnames]

        tokenized_prompts2 = torch.cat([clip.tokenize(p) for p in prompts2])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding2 = clip_model.token_embedding(tokenized_prompts2).type(dtype)
            text_features2 = clip_model.encode_text(tokenized_prompts2)
            text_features2 = text_features2 / text_features2.norm(dim=-1, keepdim=True)
        #print("Shape of text_features:", text_features.shape)
        self.text_features2 = text_features2

        prompts3 = ["Overhead photo of a" + " " + name + "." for name in classnames]

        tokenized_prompts3 = torch.cat([clip.tokenize(p) for p in prompts3])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding3 = clip_model.token_embedding(tokenized_prompts3).type(dtype)
            text_features3 = clip_model.encode_text(tokenized_prompts3)
            text_features3 = text_features3 / text_features3.norm(dim=-1, keepdim=True)
        #print("Shape of text_features:", text_features.shape)
        self.text_features3 = text_features3
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        tokenized_prefix = clip.tokenize(prompt_prefix)
        tokenized_dot = clip.tokenize(["."])
        with torch.no_grad():
            text_features_prefix = clip_model.encode_text(tokenized_prefix)
            text_features_dot = clip_model.encode_text(tokenized_dot)
        #print("Shape of text_features_prefix:", text_features_prefix.shape)
        #print("Shape of mean_description_embeddings:", mean_description_embeddings.shape)
        #print("Shape of text_features_dot:", text_features_dot.shape)
        # Step 5: Concatenate prefix, mean description embeddings, and dot embeddings
        # Ensure concatenation on the correct dimension (dim=1 for feature dimension)
        '''
        text_features_combined = torch.cat([text_features_prefix, mean_description_embeddings, text_features_dot], dim=1)
        text_features_combined = text_features_combined / text_features_combined.norm(dim=-1, keepdim=True)

        # Store in the class attribute
        self.text_features_original = text_features_combined
        '''
        # Expand `text_features_prefix` and `text_features_dot` to match the batch size of `mean_description_embeddings`
        text_features_prefix_expanded = text_features_prefix.expand(mean_description_embeddings.size(0), -1)  # shape [16, 512]
        text_features_dot_expanded = text_features_dot.expand(mean_description_embeddings.size(0), -1)        # shape [16, 512]

        # Add the three features along dimension 0
        text_features_combined = (text_features_prefix_expanded + mean_description_embeddings + text_features_dot_expanded)/3

        # Normalize and store the result
        text_features_combined = text_features_combined / text_features_combined.norm(dim=-1, keepdim=True)
        self.text_features_original = text_features_combined

        #print("Shape of text_features_combined:", text_features_combined.shape)

        self.text_features_original = text_features_combined
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
    
    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,     # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, im_features):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx                     # (n_ctx, ctx_dim)
        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)           # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)             # (1, n_ctx, ctx_dim)
        ctx_shifted = torch.add(ctx,bias)           # (batch, n_ctx, ctx_dim)
        
        # Use instance-conditioned context tokens for all classes
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)
        
        return prompts, ctx_shifted

class Projector(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_fc = None  # Fully connected layers will be initialized dynamically
        self.second_fc = None
        self.attention = None  # Self-attention layer will be initialized dynamically

    def forward(self, x):
        input_dim = x.shape[1]
        middle_start = input_dim // 4
        middle_end = 3 * input_dim // 4
        #selected_features = x[:, middle_start:middle_end]  # Extract middle half of features
        selected_features = x
        # Initialize layers dynamically based on input dimensions
        if self.first_fc is None or self.second_fc is None or self.attention is None:
            reduced_dim = selected_features.shape[1]  # Half of input_dim
            
            self.first_fc = nn.Sequential(
                nn.Linear(reduced_dim, input_dim),
                nn.GELU(),
                nn.Linear(input_dim, int(input_dim*2)),
                nn.GELU(),
                nn.Linear(int(input_dim*2), input_dim),
                nn.ReLU()
            ).to(x.device)
            
            self.second_fc = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.ReLU()
            ).to(x.device)
            
            # Learnable self-attention module
            self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=4).to(x.device)
        with autocast():
           # Process selected features
           projected_features = self.first_fc(selected_features)*x.squeeze(0)

           # Apply self-attention to the input
           x = x.unsqueeze(0)  # Add batch dimension for compatibility with attention
           attn_output, _ = self.attention(x, x, x)
           attn_output = attn_output.squeeze(0)  # Remove batch dimension
           query = projected_features.unsqueeze(0)  # Add batch dimension
           key = selected_features.unsqueeze(0) # Use original input as key (and value)
           value = selected_features.unsqueeze(0)

            # Apply cross-attention
           attn_output1, _ = self.attention(query, key, value)
           attn_output1 = attn_output.squeeze(0)
           attn_output_final = self.second_fc(attn_output+attn_output1)
           # Combine self-attention output with projected features
           combined_features1 = 0.3*projected_features*attn_output
           combined_features = torch.where( x.squeeze(0)!= 0, combined_features1 / x.squeeze(0), torch.zeros_like(combined_features1))
           #combined_features1 = 0.5*self.second_fc(combined_features)*x.squeeze(0)+attn_output

           # Refine features through the second fully connected layer with residual connection
           #refined_features = self.second_fc(combined_features + x.squeeze(0))  # Residual connection
           refined_features = combined_features+x.squeeze(0)
           return refined_features
'''
class Projector(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_fc = None  # Fully connected layers will be initialized dynamically
        self.second_fc = None
        self.attention = None  # Self-attention layer will be initialized dynamically

    def forward(self, x):
        input_dim = x.shape[1]
        middle_start = input_dim // 4
        middle_end = 3 * input_dim // 4
        #selected_features = x[:, middle_start:middle_end]  # Extract middle half of features
        selected_features = x
        # Initialize layers dynamically based on input dimensions
        if self.first_fc is None or self.second_fc is None or self.attention is None:
            reduced_dim = selected_features.shape[1]  # Half of input_dim
            
            self.first_fc = nn.Sequential(
                nn.Linear(reduced_dim, input_dim, bias=False),
                nn.GELU(),
                nn.Linear(input_dim, int(input_dim*2), bias=False),
                nn.GELU(),
                nn.Linear(int(input_dim*2), input_dim, bias=False),
                nn.ReLU()
            ).to(x.device)
            
            self.second_fc = nn.Sequential(
                nn.Linear(input_dim, input_dim, bias=False),
                nn.ReLU()
            ).to(x.device)
            
            # Learnable self-attention module
            self.cross_attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=8).to(x.device)
        with autocast():
           # Process selected features
           projected_features = self.first_fc(selected_features)*x.squeeze(0) 

           # Apply self-attention to the input
           # Prepare query, key, and value for cross-attention
           query = projected_features.unsqueeze(0)  # Add batch dimension
           key = selected_features.unsqueeze(0) # Use original input as key (and value)
           value = selected_features.unsqueeze(0)

            # Apply cross-attention
           attn_output, _ = self.cross_attention(query, key, value)
           attn_output = attn_output.squeeze(0) 
           #combined_features1 = 0.5*self.second_fc(combined_features)*x.squeeze(0)+attn_output

           # Refine features through the second fully connected layer with residual connection
           refined_features1 = self.second_fc(attn_output + x.squeeze(0))*x.squeeze(0)   # Residual connection
           query1 = refined_features1.unsqueeze(0)
           attn_output1, _ = self.cross_attention(query1, key, value)
           attn_output1 = attn_output1.squeeze(0) 
           refined_features = 0.05*attn_output1+x.squeeze(0)
           return refined_features
'''

class AdaptiveFourierFiltering(torch.nn.Module):
    def __init__(self, energy_threshold=0.75):
        """
        Adaptive Fourier-based filtering module for feature extraction.

        Parameters:
        - energy_threshold (float): Percentage of spectral energy to retain (default 95%).
        """
        super().__init__()
        self.energy_threshold = energy_threshold  # Store as a parameter

    def forward(self, x):
        """
        Forward pass: Applies FFT, retains adaptive low frequencies, and applies IFFT.

        Parameters:
        - x (torch.Tensor): Input tensor of shape (batch_size, feature_dim)

        Returns:
        - torch.Tensor: Processed tensor with adaptive low-frequency retention.
        """

        # Apply FFT (PyTorch uses complex tensors)
        fft_x = torch.fft.fft(x, dim=-1)  

        # Compute energy (magnitude squared) and cumulative energy across frequencies
        spectral_energy = fft_x.abs() ** 2  
        cumulative_energy = torch.cumsum(spectral_energy, dim=-1) / torch.sum(spectral_energy, dim=-1, keepdim=True)

        # Find the number of components to retain for each batch item
        num_components = (cumulative_energy < self.energy_threshold).sum(dim=-1) + 1  # Adaptive cutoff

        # Create a mask to retain only the necessary low frequencies
        mask = torch.zeros_like(fft_x, dtype=torch.bool)
        for i in range(x.shape[0]):  # Iterate over batch
            mask[i, :num_components[i]] = True  # Retain only top low-frequencies

        # Apply the mask to filter FFT components
        fft_filtered = torch.where(mask, fft_x, torch.zeros_like(fft_x))

        # Apply IFFT
        ifft_x = torch.fft.ifft(fft_filtered, dim=-1).real  

        return ifft_x
'''
import torch
import torch.fft

class KalmanAdaptiveFourier(torch.nn.Module):
    def __init__(self, init_cutoff=75, process_var=1e-3, measure_var=1e-2):
        """
        Kalman Filter based Adaptive Fourier Filtering.

        Parameters:
        - init_cutoff: Initial estimate for low-frequency cutoff.
        - process_var: Process variance (how much the cutoff changes over time).
        - measure_var: Measurement variance (uncertainty in the estimated cutoff).
        """
        super().__init__()
        self.cutoff_est = torch.tensor(init_cutoff, dtype=torch.float32)  # Initial cutoff
        self.process_var = process_var
        self.measure_var = measure_var
        self.kalman_gain = 1.0  # Kalman gain (will be updated)

    def forward(self, x):
        """
        Forward pass: Applies FFT, retains adaptive low frequencies using Kalman filtering, and applies IFFT.

        Parameters:
        - x (torch.Tensor): Input tensor of shape (batch_size, feature_dim)

        Returns:
        - torch.Tensor: Processed tensor with dynamically estimated frequency retention.
        """
        batch_size, feature_dim = x.shape

        # Step 1: Apply FFT
        fft_x = torch.fft.fft(x, dim=-1)

        # Step 2: Compute spectral energy
        spectral_energy = fft_x.abs() ** 2  
        cumulative_energy = torch.cumsum(spectral_energy, dim=-1) / torch.sum(spectral_energy, dim=-1, keepdim=True)

        # Step 3: Measure the number of required components based on 95% energy threshold
        measured_cutoff = (cumulative_energy < 0.95).sum(dim=-1).float().mean()  # Mean cutoff across batch

        # Step 4: Kalman Filter Update
        # Prediction step: Predict next cutoff estimate
        self.cutoff_est += self.process_var  
        
        # Compute Kalman gain
        self.kalman_gain = self.cutoff_est / (self.cutoff_est + self.measure_var)  

        # Update step: Blend prediction with measurement
        self.cutoff_est = (1 - self.kalman_gain) * self.cutoff_est + self.kalman_gain * measured_cutoff  

        # Convert to integer for indexing
        final_cutoff = int(self.cutoff_est.item())  

        # Step 5: Apply the estimated cutoff
        mask = torch.zeros_like(fft_x, dtype=torch.bool)
        mask[:, :final_cutoff] = True  # Retain only adaptive low frequencies
        fft_filtered = torch.where(mask, fft_x, torch.zeros_like(fft_x))

        # Step 6: Apply IFFT
        ifft_x = torch.fft.ifft(fft_filtered, dim=-1).real  

        return ifft_x
'''
import torch
import torch.nn as nn

class KalmanAdaptiveFourier(nn.Module):
    def __init__(self, init_cutoff=75, process_var=1e-3, measure_var=1e-2, temperature=1.0, energy_thresh=0.95):
        """
        Differentiable Kalman-Adaptive Fourier Filter Block.
        
        Parameters:
        - init_cutoff: Initial estimate for the low-frequency cutoff.
        - process_var: Process variance (Q) - uncertainty in the system model.
        - measure_var: Measurement variance (R) - uncertainty in the energy threshold measurement.
        - temperature: Controls the steepness of the soft sigmoid mask.
        - energy_thresh: The spectral energy threshold to dynamically target (e.g., 95%).
        """
        super().__init__()
        
        # 1. Kalman States (Using buffers so they move to GPU and save in state_dict)
        self.register_buffer('cutoff_est', torch.tensor(init_cutoff, dtype=torch.float32))
        self.register_buffer('error_cov', torch.tensor(1.0, dtype=torch.float32)) # Initial uncertainty (P)
        
        # 2. Learnable Variances (Allows the network to optimize the Kalman trust dynamically)
        self.process_var = nn.Parameter(torch.tensor(process_var, dtype=torch.float32))
        self.measure_var = nn.Parameter(torch.tensor(measure_var, dtype=torch.float32))
        
        # 3. Hyperparameters
        self.temperature = temperature
        self.energy_thresh = energy_thresh

    def forward(self, x):
        """
        Forward pass: Applies FFT, uses Kalman tracking for adaptive bandwidth, 
        applies a differentiable soft mask, and returns IFFT.
        """
        orig_dtype = x.dtype
        
        # 2. Force to FP32 for stable FFT math
        x_f32 = x.float()
        # Step 1: Apply FFT
        fft_x = torch.fft.fft(x_f32, dim=-1)
        
        # Step 2: Compute spectral energy and measure required components
        spectral_energy = fft_x.abs() ** 2  
        cumulative_energy = torch.cumsum(spectral_energy, dim=-1) / (torch.sum(spectral_energy, dim=-1, keepdim=True) + 1e-8)
        
        # Measurement (z_t): How many components are needed for the energy threshold?
        # Note: This measurement calculation does not require gradients.
        measured_cutoff = (cumulative_energy < self.energy_thresh).sum(dim=-1).float().mean()

        # Step 3: Kalman Filter Update (Only update state during training)
        if self.training:
            # Predict step
            # Use .abs() to ensure variances remain positive during optimization
            predicted_cov = self.error_cov + self.process_var.abs()
            
            # Update step
            kalman_gain = predicted_cov / (predicted_cov + self.measure_var.abs() + 1e-8)
            
            # Update state estimates (using moving average logic)
            self.cutoff_est = self.cutoff_est + kalman_gain * (measured_cutoff - self.cutoff_est)
            self.error_cov = (1 - kalman_gain) * predicted_cov

        # Step 4: Differentiable Soft Masking
        # Create an array of frequency indices [0, 1, 2, ..., feature_dim-1]
        freq_indices = torch.arange(fft_x.shape[-1], device=x.device).float()
        
        # Create the soft mask using Sigmoid
        # Frequencies below cutoff_est get values near 1.0; above get values near 0.0
        soft_mask = torch.sigmoid((self.cutoff_est - freq_indices) / self.temperature)
        
        # Apply the mask
        fft_filtered = fft_x * soft_mask

        # Step 5: Apply IFFT to return to the spatial/feature domain
        ifft_x = torch.fft.ifft(fft_filtered, dim=-1).real  

        return ifft_x.to(orig_dtype)

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.ori_embed = self.prompt_learner.text_features_original
        self.ori_embedding = self.prompt_learner.text_features
        self.ori_embedding1 = self.prompt_learner.text_features1
        self.ori_embedding2 = self.prompt_learner.text_features2
        self.ori_embedding3 = self.prompt_learner.text_features3
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.image_features_store = []
        self.projector = Projector()
        self.adaptive = KalmanAdaptiveFourier()

    def process_all_vectors_torch(self, matrix, num_components):
        """
        Applies FFT, retains low frequencies, and performs IFFT with zero preservation
        for all vectors in a PyTorch tensor.
        """
        def retain_low_frequencies(fft_vec, num_components):
            """
            Retains the top high-frequency components of the FFT vector.
            """
            #sorted_indices = np.argsort(np.abs(fft_vec))[-num_components:]
            sorted_indices = np.argsort(np.abs(fft_vec))[-num_components:][::-1]
            filtered_fft = np.zeros_like(fft_vec, dtype=complex)  # Ensure complex type
            filtered_fft[sorted_indices] = fft_vec[sorted_indices]
            return filtered_fft

        def ifft_preserve_zeros(original_vector, fft_filtered):
            """
            Performs IFFT and preserves zeros at the positions of the original vector.
            """
            ifft_result = np.real(ifft(fft_filtered))  # Perform IFFT
            # Replace values where the original vector was zero
            #ifft_result[original_vector == 0] = 0
            return ifft_result
        """
        Applies FFT, retains low frequencies, and performs IFFT with zero preservation
        for all vectors in a PyTorch tensor.
        """
        # Move the tensor to CPU before converting to NumPy
        matrix_np = matrix.detach().cpu().numpy()  # Convert to NumPy for FFT operations
        processed_vectors = []
        for vector in matrix_np:
           fft_vector = fft(vector)  # Apply FFT
           fft_filtered = retain_low_frequencies(fft_vector, num_components)  # Retain low frequencies
           ifft_vector = ifft_preserve_zeros(vector, fft_filtered)  # Apply IFFT with zero preservation
           processed_vectors.append(ifft_vector)
        processed_matrix_np = np.array(processed_vectors)
        # Convert back to PyTorch tensor and move it back to the original device
        return torch.tensor(processed_matrix_np, dtype=matrix.dtype, device=matrix.device)
   

    def forward(self, image, label, aug_images):
        #print("in model")
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        image_features = self.image_encoder(image.type(self.dtype))
        #image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features = self.projector(image_features)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        #print(image_features.min())
        #print(image_features.max())
        #image_features = self.process_all_vectors_torch(image_features, 350)
        image_features = self.adaptive(image_features) 
        # Check if aug_images is None, and handle accordingly
        if aug_images is not None:
           image_features1 = self.image_encoder(aug_images.type(self.dtype))
        else:
        # Use the original input if aug_images is None
           image_features1 = self.image_encoder(image.type(self.dtype))
        #image_features1, data = self.image_encoder(aug_images.type(self.dtype))
        image_features1 = image_features1 / image_features1.norm(dim=-1, keepdim=True)        
        #print(image_features.min())
        #print(image_features.max())
        #torch.set_printoptions(threshold=float('inf'), linewidth=200)
        #self.image_features_store.append(image_features.cpu().detach())
        #print(image_features.shape)
        #print(image_features)
        #print("in2 model")
        text_features_old4 = self.ori_embed
        text_features_old = self.ori_embedding
        text_features_old1 = self.ori_embedding1
        text_features_old2 = self.ori_embedding2
        text_features_old3 = self.ori_embedding3
        prompts, ctx_shifted = self.prompt_learner(image_features)
        _, ctx_shifted1 = self.prompt_learner(image_features1)
        x = F.softmax(ctx_shifted, dim=-1)  # Normalize each row (batch size dimension)
        y = F.softmax(ctx_shifted1, dim=-1)

        # Compute the KL Divergence between x and y
        kl_div = F.kl_div(y.log(), x, reduction='batchmean')        
        #print("in3 model")
        
        logits = []
        for pts_i, imf_i in zip(prompts, image_features):
            #print("in4 model")
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            #print("in5 model")
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = logit_scale * imf_i @ text_features.t()
            logits.append(l_i)
        logits = torch.stack(logits)

        #print("in6 model")
        cos = torch.nn.CosineSimilarity(dim=1,eps=1e-07)
        text_features_old = text_features_old / text_features_old.norm(dim=-1, keepdim=True)
        text_features_old1 = text_features_old1 / text_features_old1.norm(dim=-1, keepdim=True)
        text_features_old2 = text_features_old2 / text_features_old2.norm(dim=-1, keepdim=True)
        text_features_old3 = text_features_old3 / text_features_old3.norm(dim=-1, keepdim=True)
        text_features_old4 = text_features_old4 / text_features_old4.norm(dim=-1, keepdim=True)
        device = text_features.device  # Ensure this uses the same device as text_features
        score0 = cos(text_features.to(device), text_features_old.to(device))
        score1 = cos(text_features.to(device), text_features_old1.to(device))
        score2 = cos(text_features.to(device), text_features_old2.to(device))
        score3 = cos(text_features.to(device), text_features_old3.to(device))
        score4 = cos(text_features.to(device), text_features_old4.to(device)) 

        #score = cos(text_features,text_features_old)
        score0 = 1.0-torch.mean(score0)
        score1 = 1.0-torch.mean(score1)
        score2 = 1.0-torch.mean(score2)
        score3 = 1.0-torch.mean(score3)
        score4 = 1.0-torch.mean(score4)
        score = ((score0+score1+score2+score3))+(0.25*score4) 
        if self.prompt_learner.training:
            return F.cross_entropy(logits, label)+0.8*score+1.0*kl_div
        
        return logits, ctx_shifted, label


@TRAINER_REGISTRY.register()
class CoCoOp(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.COCOOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COCOOP.PREC == "fp32" or cfg.TRAINER.COCOOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"
        
        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                param.requires_grad_(False)
        
        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COCOOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        #image, label = self.parse_batch_train(batch)
        image, label, impath_list = self.parse_batch_train(batch)
        model = self.model
        optim = self.optim
        scaler = self.scaler
        #print(self.cfg.TRAINER.COCOOP.PREC)
        #print("in")
        aug_images = [Image.open(path) for path in impath_list]
        transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                         std=[0.26862954, 0.26130258, 0.27577711])
        ])
        aug_images = [transform(img).to(self.device) for img in aug_images]
        #aug_images = torch.tensor(aug_images)
        aug_images = torch.stack(aug_images).to(self.device)
        prec = self.cfg.TRAINER.COCOOP.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label, aug_images)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            #print("in2")
            loss = model(image, label, aug_images)
            #print("in3")
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        impath_list = batch["impath"]
        impath_list = [path.replace("/images/", "/aug_images/") for path in impath_list]
        #aug_images = [torch.load(path).to(self.device) for path in impath_list]
        #impath_list = impath_list.replace("/images/", "/aug_images/")
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label, impath_list
    '''
    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = ((self.epoch + 1) %
                                self.cfg.TRAIN.CHECKPOINT_FREQ == 0 if
                                self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False)

        if do_test:
            curr_result = self.test()
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(self.epoch,
                                self.output_dir,
                                model_name="model-best.pth.tar")

            self.set_model_mode("train")

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)
    '''
    def after_epoch(self):
       last_epoch = (self.epoch + 1) == self.max_epoch  # Check if this is the last epoch
       do_test = not self.cfg.TEST.NO_TEST
       meet_test_freq = (self.epoch + 1) in [100]  # Test at epochs 10 and 20

       if do_test and meet_test_freq:  # Perform testing at specified epochs
          curr_result = self.test()
          is_best = curr_result > self.best_result
          if is_best:
             self.best_result = curr_result
             self.save_model(self.epoch,
                            self.output_dir,
                            model_name="model-best.pth.tar")

          self.set_model_mode("train")

       meet_checkpoint_freq = (
          (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0 
          if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 
          else False
       )

       if meet_checkpoint_freq or last_epoch:
           self.save_model(self.epoch, self.output_dir) 
    
    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
    @torch.no_grad()
    def model_inference(self, input, label, aug_images=None):
        if aug_images is not None:
          # Inference logic with augmented images
          pass
        else:
          # Inference logic without augmented images
          pass
        return self.model(input, label, aug_images)
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            logits, ctx_shifted, label = self.model_inference(input, label, aug_images=None)
            self.evaluator.process(logits, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]
