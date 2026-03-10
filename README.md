# SIMSPINE: A Biomechanics-Aware Simulation Framework for 3D Spine Motion Annotation and Benchmarking

<div align="center">

[![Home](https://img.shields.io/badge/Project-Homepage-pink.svg)](https://saifkhichi.com/research/simspine/)
[![arXiv](https://img.shields.io/badge/arXiv-2602.20792-B31B1B.svg)](https://arxiv.org/abs/2602.20792)

![SIMSPINE Teaser](teaser.gif)
</div>

---

> __Abstract__: _Modeling spinal motion is fundamental to understanding human biomechanics, yet remains underexplored in computer vision due to the spine's complex multi-joint kinematics and the lack of large-scale 3D annotations. We present a biomechanics-aware keypoint simulation framework that augments existing human pose datasets with anatomically consistent 3D spinal keypoints derived from musculoskeletal modeling. Using this framework, we create the first open dataset, named SIMSPINE, which provides sparse vertebra-level 3D spinal annotations for natural full-body motions in indoor multi-camera capture without external restraints. With 2.14 million frames, this enables data-driven learning of vertebral kinematics from subtle posture variations and bridges the gap between musculoskeletal simulation and computer vision. In addition, we release pretrained baselines covering fine-tuned 2D detectors, monocular 3D pose lifting models, and multi-view reconstruction pipelines, establishing a unified benchmark for biomechanically valid spine motion estimation. Specifically, our 2D spine baselines improve the state-of-the-art from 0.63 to 0.80 AUC in controlled environments, and from 0.91 to 0.93 AP for in-the-wild spine tracking. Together, the simulation framework and SIMSPINE dataset advance research in vision-based biomechanics, motion analysis, and digital human modeling by enabling reproducible, anatomically grounded 3D spine estimation under natural conditions._

## Overview

Official repository for the CVPR 2026 workshop paper "SIMSPINE: A Biomechanics-Aware Simulation Framework for 3D Spine Motion Annotation and Benchmarking" by Muhammad Saif Ullah Khan and Didier Stricker.

More details to be added soon. For now, please refer to the [arXiv preprint](https://arxiv.org/abs/2602.20792) for the full paper and supplementary materials.


## Roadmap

- [ ] Release SIMSPINE dataset on [HuggingFace](https://huggingface.co/datasets/dfki-av/simspine)
- [ ] Release ONNX models for inference via the [SpinePose library](https://github.com/dfki-av/spinepose)
  - [ ] 2D Models
    - [ ] SpinePose-SIMSPINE (small, medium, large)
    - [ ] HRNet-SIMSPINE (w32)
    - [ ] RTMPose-SIMSPINE (medium)
    - [ ] ViTPose-SIMSPINE (base)
  - [ ] 2D-to-3D Lifting Model
- [ ] Release Pytorch checkpoints and evaluation config files
- [ ] Release evaluation code
  - [ ] 2D Pose Estimation
  - [ ] 2D-to-3D Lifting
  - [ ] Multiview 3D Triangulation 
- [ ] Release training code
