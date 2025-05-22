# Discriminative Skeleton-Based Action Recognition via Co-Learning with Motion Diffusion Model

This repository is the official implementation of "Discriminative Skeleton-Based Action Recognition via Co-Learning with Motion Diffusion Model" (AVSS, 2025).

## About
Skeleton-based action recognition is vital for numerous real-world applications, yet it continues to face challenges due to limited and imbalanced 3D motion data, complex movement dynamics, and sensitivity to viewpoint variations. To tackle these issues, we introduce a novel co-learning framework that unifies generative and discriminative modeling by combining a 3D Motion Diffusion Model (MDM) with its inverse counterpart, I-MDM, for more robust recognition. Through the incorporation of high-quality synthetic motions guided by discriminative features from I-MDM, our method achieves state-of-the-art top-1 accuracy on HumanAct12 (94.63%) and NTU-13 (99.25%), while delivering over three times faster inference (5.32 ms) than prior methods. These results highlight the potential of diffusion-based generative augmentation guided by discriminative feedback, paving a new path for efficient and accurate skeleton-based action recognition.


## Overview
![abstract_figure](https://github.com/user-attachments/assets/7634cd7c-233c-4ec1-9dac-2497cf6194e1)


## Datasets
