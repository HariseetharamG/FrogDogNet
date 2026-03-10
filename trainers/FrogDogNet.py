import os.path as osp
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from scipy.fft import fft, ifft
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    """Loads the CLIP model to CPU based on the configuration."""
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
    """Encodes tokenized prompts using the CLIP transformer."""
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # take features from the eot embedding
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class PromptLearner(nn.Module):
    """Generates learnable, instance-conditioned prompts via Meta-Net."""
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

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
    
    def construct_prompts(self, ctx, prefix, suffix, label=None):
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return prompts

    def forward(self, im_features):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx                     
        bias = self.meta_net(im_features)  
        bias = bias.unsqueeze(1)           
        ctx = ctx.unsqueeze(0)             
        ctx_shifted = torch.add(ctx, bias) 
        
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)
        
        return prompts, ctx_shifted


class Projector(nn.Module):
    """Refines visual features using dynamically instantiated fully connected and attention layers."""
    def __init__(self):
        super().__init__()
        self.first_fc = None
        self.second_fc = None
        self.attention = None

    def forward(self, x):
        input_dim = x.shape[1]
        selected_features = x
        
        # Initialize layers dynamically, matching the device AND dtype of the input 'x'
        if self.first_fc is None or self.second_fc is None or self.attention is None:
            reduced_dim = selected_features.shape[1]
            
            self.first_fc = nn.Sequential(
                nn.Linear(reduced_dim, input_dim),
                nn.GELU(),
                nn.Linear(input_dim, int(input_dim * 2)),
                nn.GELU(),
                nn.Linear(int(input_dim * 2), input_dim),
                nn.ReLU()
            ).to(device=x.device, dtype=x.dtype)
            
            self.second_fc = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.ReLU()
            ).to(device=x.device, dtype=x.dtype)
            
            self.attention = nn.MultiheadAttention(
                embed_dim=input_dim, num_heads=4
            ).to(device=x.device, dtype=x.dtype)

        with autocast():
           projected_features = self.first_fc(selected_features) * x.squeeze(0)

           x_unsqueezed = x.unsqueeze(0) 
           attn_output, _ = self.attention(x_unsqueezed, x_unsqueezed, x_unsqueezed)
           attn_output = attn_output.squeeze(0) 
           
           query = projected_features.unsqueeze(0) 
           key = selected_features.unsqueeze(0) 
           value = selected_features.unsqueeze(0)

           attn_output1, _ = self.attention(query, key, value)
           attn_output1 = attn_output1.squeeze(0)
           
           attn_output_final = self.second_fc(attn_output + attn_output1)
           
           combined_features1 = 0.3 * projected_features * attn_output_final
           combined_features = torch.where(
               x.squeeze(0) != 0, 
               combined_features1 / x.squeeze(0), 
               torch.zeros_like(combined_features1)
           )

           refined_features = combined_features + x.squeeze(0)
           return refined_features


class CustomCLIP(nn.Module):
    """Main model combining CLIP with FrogDogNet modifications."""
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.image_features_store = []
        self.projector = Projector()

    def process_all_vectors_torch(self, matrix, num_components):
        """
        Applies FFT, retains low frequencies, and performs IFFT with zero preservation
        for all vectors in a PyTorch tensor.
        """
        def retain_low_frequencies(fft_vec, num_components):
            sorted_indices = np.argsort(np.abs(fft_vec))[-num_components:][::-1]
            filtered_fft = np.zeros_like(fft_vec, dtype=complex)
            filtered_fft[sorted_indices] = fft_vec[sorted_indices]
            return filtered_fft

        def ifft_preserve_zeros(original_vector, fft_filtered):
            ifft_result = np.real(ifft(fft_filtered))
            ifft_result[original_vector == 0] = 0
            return ifft_result

        matrix_np = matrix.detach().cpu().numpy()
        processed_vectors = []
        for vector in matrix_np:
           fft_vector = fft(vector)
           fft_filtered = retain_low_frequencies(fft_vector, num_components)
           ifft_vector = ifft_preserve_zeros(vector, fft_filtered)
           processed_vectors.append(ifft_vector)
           
        processed_matrix_np = np.array(processed_vectors)
        return torch.tensor(processed_matrix_np, dtype=matrix.dtype, device=matrix.device)

    def forward(self, image, label):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = self.projector(image_features)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Apply static FFT noise filtering
        image_features = self.process_all_vectors_torch(image_features, 350) 

        prompts, ctx_shifted = self.prompt_learner(image_features)
        
        logits = []
        for pts_i, imf_i in zip(prompts, image_features):
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = logit_scale * imf_i @ text_features.t()
            logits.append(l_i)
        logits = torch.stack(logits)
        
        if self.prompt_learner.training:
            return F.cross_entropy(logits, label)
        
        return logits, ctx_shifted, label


@TRAINER_REGISTRY.register()
class FrogDogNet(TrainerX):
    """FrogDogNet Training Protocol."""
    def check_cfg(self, cfg):
        assert cfg.TRAINER.COCOOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COCOOP.PREC == "fp32" or cfg.TRAINER.COCOOP.PREC == "amp":
            clip_model.float()

        print("Building custom CLIP (FrogDogNet)")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"
        
        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                param.requires_grad_(False)
        
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COCOOP.PREC == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        model = self.model
        optim = self.optim
        scaler = self.scaler
        prec = self.cfg.TRAINER.COCOOP.PREC
        
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label)
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
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label
    
    def after_epoch(self):
       last_epoch = (self.epoch + 1) == self.max_epoch 
       do_test = not self.cfg.TEST.NO_TEST
       meet_test_freq = (self.epoch + 1) in [100]  # Configure test frequency here

       if do_test and meet_test_freq: 
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

            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)

    @torch.no_grad()
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
            logits, ctx_shifted, label = self.model_inference(input, label)
            self.evaluator.process(logits, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]