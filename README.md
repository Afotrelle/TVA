# Transferable-Video-Attack
# From Pretrain to Pain: Adversarial Vulnerability of Video Foundation Models Without Task Knowledge

This repository contains the official code for our paper **"From Pretrain to Pain: Adversarial Vulnerability of Video Foundation Models Without Task Knowledge"**. We investigate the adversarial vulnerability of large-scale video foundation models (VFMs) under a task-agnostic threat model. Our approach does not rely on downstream task labels or architectures, and directly perturbs the input video to manipulate intermediate representations of pretrained models.

## Overview

The codebase provides a framework to evaluate adversarial attacks on video foundation models across three representative downstream benchmarks:

- **TAD (Temporal Action Detection):** Evaluates the temporal robustness of VFMs under attacks using pretrained VideoMAE-base backbones as visual encoders. 
  
- **MVBench:** A diverse suite of video understanding tasks covering action reasoning, object dynamics, and temporal ordering. Attacks are trained on frozen VFMs (LanguageBind, QFormer, SigLIP), with predictions made via LLaVA-NeXT, Video-LLaVA, and Video-chat2.
  
- **SEEDBench:** A multimodal benchmark involving vision-language tasks such as video question answering and captioning. 

## Structure

The repository is modular, with code organized for:

- **Attack implementation:** Task-agnostic loss functions and optimization routines that operate without requiring task-specific ground truth.
- **Model wrappers:** Interface code for loading and extracting intermediate features from state-of-the-art VFMs, including VideoMAE for TAD.
- **Dataset support:** Preprocessing, dataloaders, and evaluation scripts tailored for TAD.
- **Experiment scripts:** Configurable pipelines for perturbation training and evaluation. Under the `tad/configs` dictionary, `thumos_videomae_b` is used for training and the others used for evaluation.

**Train:**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 t_attack.py ./configs/thumos_videomae_b_16.py --na train --max_sample 100 --eps 8 --stepsize_adv 4 --flow 1 --bicos 1
```

**Test:**
```bash
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 test_npy.py ./configs/thumos_b2b.py --checkpoint "/file/from/opentad/adatad_thumos_actionformer_videomae_b_768x1_160_adapter_epoch_51_c3872325.pth" --na test --file_path "/perturbation/path/adv_l1_8.0_4_bicon1_flow1_174_thumos/" --max_sample 100
```

## Key Features

- Plug-and-play loss formulations that generalize across tasks.
- Compatible with popular attack method (I-FGSM, MI-FGSM, DI-FGSM,...).
- Emphasis on feature-level manipulation.

## Preparation

Please make sure you have followed the official setup instructions provided in the [Video-chat-2 GitHub respository](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2), [SEEDBench GitHub respository](https://github.com/AILab-CVC/SEED-Bench), [Video-LLaVA GitHub repository](https://github.com/PKU-YuanGroup/Video-LLaVA), [LLaVA-NeXT GitHub repository](https://github.com/LLaVA-VL/LLaVA-NeXT), [OpenTAD GitHub repository](https://github.com/sming256/OpenTAD), including data preparation, environment setup and downloading necessary model checkpoints.

## Acknowledgements

We build upon several public resources including MVBench, SEEDBench, OpenTAD and open-source implementations of video foundation models. We thank the authors of these projects for their contributions to the community.

