# RobustVideoMatting


## Introduction

- Implemented the paper Robust High Resolution Video Matting with temporal Guidance. Uses neural network models that process frames as independent images. 
- Video Matting is used for separating the video into two or more layers, usually foreground and background. Enables us to extract the foreground, and replace it with a different background.

## Results Demo
- RVM uses a recurrent neural network to process videos with temporal memory. It can perform a real time matting on any video without any additional inputs.
- To see the model’s performance click [here](https://drive.google.com/drive/folders/1pUNSMO37Y5ozjZbXPH7IU9Qk9wZj9tV3) for Demo.


## Installation
Follow the mentioned steps to install and run the demo on your system:
**Step 1**: Clone this github repository on your system.
**Step 2**: Install pytorch dependencies: pip install -r requirements_inference.txt

## Inference 
**Step 1**: Download the pretrained model 
- [rvm_mobilenetv3.pth](https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth) for mobilenet model (Recommended for most use cases)
- [rvm_renet50.pth](https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50.pth) for Resnet model
**Step 2**: Load the model
```
import torch
from model import MattingNetwork

model = MattingNetwork('mobilenetv3').eval().cuda()  # or "resnet50"
model.load_state_dict(torch.load('rvm_mobilenetv3.pth'))
```
**Step 5**: Convert videos
```
from inference import convert_video
convert_video(
    model,                           # The model, can be on any device (cpu or cuda).
    input_source='input.mp4',        # A video file or an image sequence directory.
    output_type='video',             # Choose "video" or "png_sequence"
    output_composition='com.mp4',    # File path if video; directory path if png sequence.
    output_alpha="pha.mp4",          # [Optional] Output the raw alpha prediction.
    output_foreground="fgr.mp4",     # [Optional] Output the raw foreground prediction.
    output_video_mbps=4,             # Output video mbps. Not needed for png sequence.
    downsample_ratio=None,           # A hyperparameter to adjust or use None for auto.
    seq_chunk=12,                    # Process n frames at once for better parallelism.
)
```
## Training and Evaluation:

**Model**
- Decoder: decoder.py Includes a recurrent network in which the Bottleneck block at 1/16 feature scale. 
  - The ConvGRU layer is operated on only half of the channels by splitting. This significantly reduces parameters and computation.
  - Upsampling block operates at 1/8, 1/4, and 1/2 scale.  It contains layers that increase the spatial resolution of images and generate a high-resolution image.
  - Output block uses regular convolutions to refine the results. It employs 2 repeated convolutions, Batch Normalization, and ReLU stacks which generates the final hidden features. These features are projected to outputs, including 1-channel alpha prediction, 3-channel foreground 

- Deep_guided_filter.py: Consists of a feedforward convolutional neural network and is used for high resolution upsampling. This module is optional.
- lraspp.py: Used for semantic segmentation as a part of encoder module
- mobilenetv3.py : Includes mobilenet model which serves as the encoder for this architecture. The encoder module operates on individual frames and extracts features at 1/2 , 1/4 , 1/8 , and 1/16 scales for the recurrent decoder.
- Resnet.py: Loads resnet model architecture from torchvision models.
- Model.py: Connects all the above files and forms the sequence for whole model architecture.

## Training

- Train_config.py: Contains all the paths for train/valid/train dataset. Can be updated according to the path on the user's system for dataset locations.
- Train.py: After updating the paths in train_config.py, run python train.py with all the arguments mentioned like model-variant, dataset, resolution, learning rate, checkpoint-dir, epoch. 
- Train_loss.py: Defines all the losses(matting, segmentation, laplacian etc).

## Evaluation

- Evaluation_hr: Run evaluation_hr.py to get the evaluation metric for high resolution samples. 
- Evaluation_lr: Run evaluation_lr.py to get the evaluation metric for low resolution samples. 

## Dataset 

- Augmentation.py: Does motion augmentation (fine translation, blurring, scaling sharpness) and temporal augmentation (clip reversal, frame skipping, speed change) on foreground and background.
- Coco.py: Performs augmentation (color jittering, shearing, translation etc) on COCO dataset. 
- Imagmatte.py: Adds Noise, jittering, grayscale,  sharpness, blurring augmentation on imagematte dataset.
- spd.py: Extracts RGB channels of images from image directory and luminance of images from segmentation directory for  Supervisely person dataset.
- Videomatte.py: Performs motion augmentation( foreground, background affine, noise, sharpness, blur,  flip etc.) on Videomatte training dataset. 
- Youtubevis.py: Performs resizing, horizontal flipping, color jittering etc on YoutubeVis dataset.

## Inference
- Inference.py, Inference_utils.py, Inference_speed_test.py: Used to generate the results.  
- Documentation: Includes all the documentation for training(training.md) and inference(inference.md) purposes. 

## Project Members
- Anish Phule - asp5607@psu.edu
- Ramona Devi - rfd5319@psu.edu
- Saquib Khan - msk5752@psu.edu
- Vishnu Sharma - vxh5104@psu.edu


Contributors: Implemented code from this [repo](https://github.com/PeterL1n/RobustVideoMatting). 
