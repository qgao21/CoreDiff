# CoreDiff: Contextual Error-Modulated Generalized Diffusion Model for Low-Dose CT Denoising and Generalization
This is the official implementation of the paper "CoreDiff: Contextual Error-Modulated Generalized Diffusion Model for Low-Dose CT Denoising and Generalization". The pre-print version can be found in [arxiv](https://arxiv.org/abs/2304.01814); the published version can be found in [TMI](https://ieeexplore.ieee.org/document/10268250).

## Updates
- Oct, 2024: Uploaded the pre-trained model on the original Mayo 2016 'DICOM' format data (25% dose): [ema_model-150000](https://drive.google.com/drive/folders/1rGb34H_6ktP79vMYYJOLSoCE3579TDZ5?usp=drive_link).
- Dec, 2023: Updated the code for preprocessing the original Mayo 2016 "DICOM" format data (`data_preporcess/prep_mayo2016.py`) and its corresponding training demo (`train_mayo2016.sh`).
- Oct, 2023: initial commit.


## Data Preparation
- The AAPM-Mayo dataset can be found from: [Mayo 2016](https://ctcicblog.mayo.edu/2016-low-dose-ct-grand-challenge/). 
- The "Low Dose CT Image and Projection Data" can be found from [Mayo 2020](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=52758026#527580262a84e4aa87794b6583c78dccf041269f).
- The Piglet Dataset can be found from: [SAGAN](https://github.com/xinario/SAGAN).
- The Phantom Dataset can be found from: [XNAT](https://xnat.bmia.nl/app/template/XDATScreen_report_xnat_projectData.vm/search_element/xnat:projectData/search_field/xnat:projectData.ID/search_value/stwstrategyps4).


## Training & Inference
Please check `train.sh` for training script (or `test.sh` for inference script) once the data is well prepared. Specify the setting in the script, and simply run it in the terminal.

For one-shot learning framework，please check `train_osl_framework_training.sh` for training script (or `test_osl_framework.sh` for inference script)

## Training loss and evaluation metrics. 
These curves are calculated based on our simulated 5% dose data.
![Image text](https://github.com/qgao21/CoreDiff/blob/main/figs/loss_and_metrics.png)

## Requirements
```
- Linux Platform
- python==3.8.13
- cuda==10.2
- torch==1.10.1
- torchvision=0.11.2
- numpy=1.23.1
- scipy==1.10.1
- h5py=3.7.0
- pydicom=2.3.1
- natsort=8.2.0
- scikit-image=0.21.0
- einops=0.4.1
- tqdm=4.64.1
- wandb=0.13.3
```

## Acknowledge
- Our codebase builds heavily on [DU-GAN](https://github.com/Hzzone/DU-GAN) and [Cold Diffusion](https://github.com/arpitbansal297/Cold-Diffusion-Models). Thanks for open-sourcing!
- Low-dose CT data simulation refers to [LD-CT-simulation](https://github.com/smuzd/LD-CT-simulation). Thanks for open-sourcing!



## Citation
If you find our work and code helpful, please kindly cite the corresponding paper:
```
@article{gao2023corediff,
  title={CoreDiff: Contextual Error-Modulated Generalized Diffusion Model for Low-Dose CT Denoising and Generalization},
  author={Gao, Qi and Li, Zilong and Zhang, Junping and Zhang, Yi and Shan, Hongming},
  journal={IEEE Transactions on Medical Imaging},
  volume={43},
  number={2},
  pages={745--759},
  year={2024}
}
```
