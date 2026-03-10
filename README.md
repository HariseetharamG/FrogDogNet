# FrogDogNet: Fourier frequency Retained visual prompt Output Guidance for Domain Generalization of CLIP

Official repository of FrogDogNet, which introduces a static Fourier cutoff mechanism to filter high-frequency sensor noise for unknown class and domain generalization in Remote Sensing by adapting pre-trained vision-language models (VLM) like [CLIP](https://arxiv.org/abs/2103.00020).

## **CVPRw 2025 (EarthVision)**

[![paper](https://img.shields.io/badge/Conference-Paper-blue)](https://openaccess.thecvf.com/content/CVPR2025W/EarthVision/papers/Gunduboina_FrogDogNet_Fourier_frequency_Retained_visual_prompt_Output_Guidance_for_Domain_CVPRW_2025_paper.pdf)
[![supplement](https://img.shields.io/badge/Supplementary-Material-F9D371)](https://openaccess.thecvf.com/content/CVPR2025W/EarthVision/supplemental/Gunduboina_FrogDogNet_Fourier_frequency_CVPRW_2025_supplemental.pdf) 
[![arXiv](https://img.shields.io/badge/arXiv-Paper-brightgreen)](https://arxiv.org/pdf/2504.16433)

## Abstract

![teaser](images/teaser.png)

In recent years, the success of large-scale vision-language models (VLMs) such as CLIP has led to their increased usage in various computer vision tasks. However, the potential of VLMs for generalization tasks in remote sensing (RS) is often bottlenecked by severe domain shifts caused by high-frequency spatial clutter and sensor noise. To address this, we propose FrogDogNet (Fourier frequency Retained visual prompt Output Guidance). 

Unlike standard spatial-domain adapters that inadvertently absorb non-transferable background biases, FrogDogNet introduces a static Fourier cutoff mechanism. By explicitly filtering high-frequency noise in the Fourier domain before visual features interact with the Meta-Net, we prevent the prompt tokens from overfitting to domain-specific spatial clutter. To validate FrogDogNet, we curated four available RS benchmarks and utilized experimental protocols for base-to-new, cross-dataset, and domain generalization tasks, demonstrating superior out-of-distribution performance over existing baseline methods.

## Architecture

![architecture](images/architecture.png)

FrogDogNet is composed of a text encoder, an image encoder, and a static Fourier frequency filter designed for visual feature refinement. The image encoder extracts visual features, which are then transformed into the Fourier domain. A heuristic frequency thresholding module strips away high-frequency sensor noise. These cleaned, domain-invariant features are then passed to the Meta-Net to generate robust conditional bias tokens for the text prompts.

## Datasets
- For Base-to-New Class and Cross-Dataset Generalization:
  - [PatternNet](https://sites.google.com/view/zhouwx/dataset)
  - [RSICD](https://github.com/201528014227051/RSICD_optimal)
  - [RESISC45](https://www.tensorflow.org/datasets/catalog/resisc45)
  - [MLRSNet](https://data.mendeley.com/datasets/7j9bv9vwsx/3)

## Version 2 Datasets (Version-2):
- For Domain Generalization:
  - [PatternNetv2](https://drive.google.com/file/d/1K-GZ2KjQ3hn17JJBrxnmXsTxAFeg2XUT/view?usp=sharing)
  - [RSICDv2](https://drive.google.com/file/d/1uhlTHQCHkE0KD04YGBAKsxPgG14eQez_/view?usp=sharing)
  - [RESISC45v2](https://drive.google.com/file/d/1Zfsko5swyQqu5HiuRwZe5jIGoUKfBgxq/view?usp=sharing)
  - [MLRSNetv2](https://drive.google.com/file/d/1OJrAwU1i9hYe7kEsHIIq_TodJDiwnnAz/view?usp=sharing)

## How to install

### Create your environment:

```bash
$conda create -n frogdognet python=3.8$ conda activate frogdognet
$conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=10.2 -c pytorch$ pip install -r requirements.txt
```
## Code Instructions
