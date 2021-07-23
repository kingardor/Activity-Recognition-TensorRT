# Activity Recognition TensorRT

Perform video classification using 3D ResNets trained on Kinetics-700 and Moments in Time dataset - accelerated with TensorRT 8.0

[![ActivityGIF](resources/act.gif)](https://youtu.be/snlZtGCyJMg)

P.S Click on the gif to watch the full-length video!

## Index

- [Activity Recognition TensorRT](#activity-recognition-tensorrt)
- [Index](#index)
- [TensorRT Installation](#tensorrt-installation)
- [Python Dependencies](#python-dependencies)
- [Clone the repository](#clone-the-repository)
- [Download Pretrained Models](#download-pretrained-models)
- [Running the code](#running-the-code)
- [Citations](#citations)

## TensorRT 8 Installation

Assuming you have CUDA already installed, go ahead and download TensorRT 8 from [here](https://developer.nvidia.com/tensorrt-getting-started).

Follow instructions of installing the system binaries and python package for tensorrt [here](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-tar).

## Python dependencies

Install the necessary python dependencies by running the following command -

```sh
pip3 install -r requirements.txt
```

## Clone the repository

This is a straightforward step, however, if you are new to git recommend glancing threw the steps.

First, install git

```sh
sudo apt install git
```

Next, clone the repository

```sh
# Using HTTPS
https://github.com/kn1ghtf1re/Activity-Recognition-TensorRT.git
# Using SSH
git@github.com:kn1ghtf1re/Activity-Recognition-TensorRT.git
```

## Download Pretrained Models

Download models from [google-drive](https://drive.google.com/drive/folders/1PumnUl_-eVvk0tFpn463vPxpqchGEuWB?usp=sharing) and place them in the current directory.

## Running the code

The code supports a number of command line arguments. Use help to see all supported arguments

```sh
➜ python3 action_recognition_tensorrt.py --help
usage: action_recognition_tensorrt.py [-h] [--stream STREAM] [--model MODEL] [--fp16] [--frameskip FRAMESKIP] [--save_output SAVE_OUTPUT]

Action Recognition using TensorRT 8

optional arguments:
  -h, --help            show this help message and exit
  --stream STREAM       Path to use video stream
  --model MODEL         Path to model to use
  --fp16                To enable fp16 precision
  --frameskip FRAMESKIP
                        Number of frames to skip
  --save_output SAVE_OUTPUT
                        Save output as video
```

Run the script this way:

```sh
# Video
python3 action_recognition_tensorrt.py --stream /path/to/video --model resnet-18-kinetics-moments.onnx --fp16 --frameskip 2

# Webcam
python3 action_recognition_tensorrt.py --stream webcam --model resnet-18-kinetics-moments.onnx --fp16 --frameskip 2
```

## Citations

```text
@article{hara3dcnns,
  author={Kensho Hara and Hirokatsu Kataoka and Yutaka Satoh},
  title={Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?},
  journal={arXiv preprint},
  volume={arXiv:1711.09577},
  year={2017},
}
```
