# 4D TTT

## 📖 Overview
This is the implementation for the Test Time Training for 4D Medical ImageInterpolation

## Code Fundation
Our code is partially built on [UVI-Net](https://github.com/jungeun122333/UVI-Net)

## Dataset
We use the same data set that are shown in [UVI-Net](https://github.com/jungeun122333/UVI-Net), including [ACDC](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb) and [4D Lung](https://www.cancerimagingarchive.net/collection/4d-lung/).
After downloading all the datasets, you need to place them in the `dataset` folder (or create one).
└── dataset
    ├── ACDC
    │   └── database
    │       ├── training
    │       │   ├── patient001
    │       │   │   ├── patient001_4d.nii.gz
    │       │   │   ├── patient001_frame01.nii.gz
    │       │   │   ├── patient001_frame01_gt.nii.gz
    │       │   │   ├── patient001_frame12.nii.gz
    │       │   │   ├── patient001_frame12_gt.nii.gz
    │       │   │   ├── MANDATORY_CITATION.md
    │       │   │   └── Info.cfg
    │       │   ├── patient002
    │       │   │       :
    │       │   └── patient100
    │       ├── testing
    │       │   ├── patient101
    │       │   │       :
    │       │   └── patient150
    │       └── MANDATORY_CITATION.md
    └── 4D-Lung
        ├── 100_HM10395
        │   ├── 09-15-1997-NA-p4-69351
        │   │             :
        │   └── 07-02-2003-NA-p4-14571
        ├── 101_HM10395
        │     :
        └── 119_HM10395

## 🛠️ Requirements
Use  `-pip install -r requirements.txt` to install all the required libraries.

## Usage
# Training
To pre train the model, run the following codes:
For the Rotation Predictor: 
 Run `python trainrp.py --dataset cardiac` for Cardiac dataset
 Run `python trainrp.py --dataset lung` for 4D lung dataset

For the 3D MAE:
You need to adjust the Patch embedding setting for the datasets for they don't share the same sizes.
For Cardiac dataset:
Go to models/mae3d/patch_embed.py, set code in line10 to img_size = (img_size, img_size, 8)  # cardiac
Then Run `python train.py --dataset cardiac`

For Lung dataset:
Go to models/mae3d/patch_embed.py, set code in line10 to img_size = (img_size, img_size, 32)  # lung
Then Run `python train.py --dataset cardiac`

# Evaluation
After pre-training the models, you can evaluate the codes using:
 NOTE that you can choose 1 of the 3 TTT modes: naive/online/mini_batch. Just simply type after --ttt_mode, e.g. --ttt_mode naive
For the Rotation Predictor: 
 Run `python evaluationrp.py --dataset cardiac --ttt_mode naive/online/mini_batch ` for Cardiac dataset
 Run `python evaluationrp.py --dataset lung --ttt_mode naive/online/mini_batch` for 4D lung dataset

For the 3D MAE:
 You need to adjust the Patch embedding setting for the datasets for they don't share the same sizes.
For Cardiac dataset:
Go to models/mae3d/patch_embed.py, set code in line10 to img_size = (img_size, img_size, 8)  # cardiac
 Then Run `python evaluation.py --dataset cardiac --ttt_mode naive/online/mini_batch ` for Cardiac dataset

For Lung dataset:
Go to models/mae3d/patch_embed.py, set code in line10 to img_size = (img_size, img_size, 32)  # lung
 Then Run `python evaluation.py --dataset lung --ttt_mode naive/online/mini_batch` for 4D lung dataset
