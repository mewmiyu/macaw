# Macaw (Mobile Augmented Campus Assistant for Wayfinding)
<p align="center">
<img src="https://github.com/mewmiyu/macaw/assets/60306066/f0041e53-3405-4ce8-bd7f-866726efeb29" alt="animated" />
</p>
MACAW (Mobile Augmented Campus Assistant for Wayfinding) is a mobile application that recognizes university buildings 
and applies natural feature tracking to track the movement of the buildings inside the frame. 
The user sees the tracked building highlighted with a bounding box and the name of the building.
The application is designed to help students and visitors of the TU Darmstadt to find their way around the campus. Currently only German language is supported for the overlays with the information.
</p>
<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#introduction">Introduction</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ol>
        <li>
          <a href="#prerequisites">Prerequisites</a>
        </li>
        <li>
          <a href="#installation">Installation</a>
        </li>
      </ol>
    </li>
    <li>
      <a href="#usage">Usage</a>
      <ol>
        <li>
          <a href="#configuration">Configuration</a>
        </li>
        <li>
          <a href="#visualizing">Visualizing</a>
        </li>
      </ol>
    </li>
    <li>
      <a href="#contacts">Contacts</a>
    </li>
  </ol>
</details>

<!-- Introduction -->
## Introduction

MACAW was created as a part of the course Augmented Vision at the Technical University of Darmstadt
given by Dr.-Ing. Pavel Rojtberg. The goal of the project was to create an application that uses augmented reality.


<!-- Getting Started -->
## Getting Started

### Prerequisites
Make sure you have a running Python 3.10 environment. We recommend using [Anaconda](https://www.anaconda.com/products/individual) for managing your Python environment. 

### Installation

Make sure you are in a conda environment, but if you would like to create a new one, you can do so by:

    conda create -n macaw python=3.10
    conda activate macaw

In order to run our script you need to install the following packages:

    conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
    
    conda install matplotlib pycocotools imutils wandb -c conda-forge

    pip install opencv-python
    
    pip install huggingface_hub 

<!-- USAGE -->
## Usage
The following sections describe how to use the application in detail.

### Configuration

In order to use MACAW you need to provide a config file, describing what you want to do. An example file with all possible configurations can be found in [configs/base.yaml](configs/base.yaml). We also provided more specific config files, based on the method:

- [configs/run-macaw.yaml](configs/run-macaw.yaml) runs the application for a given video file. The weights for the detector are also automatically downloaded if "Download" is set to True.
- [configs/train.yaml](configs/train.yaml) configures macaw to train the detection model on the specified dataset. You can define the backbone architecture of the FasterRCNN, Hyperparameters for the training, as well as the usage of weights and biases.
- [configs/eval.yaml](configs/eval.yaml) configures macaw to see the results of the detection model. The config file therefore contains the data as well as the model-checkpoint. If "Download" is set to true, macaw tries to download the model-weights.
- [configs/label.yaml](configs/label.yaml) configures macaw to label a dataset. The annotation json-file stores all annotations made by the user. If the mode is set to "review" the already made annotations are displayed.

### Visualizing

To run MACAW you need to open a console/terminal in the working directory and run the following command:

    python src/macaw.py --config configs/run-macaw.yaml

After that a window will open that either shows the video or the webcam stream. 

<!-- CONTACTS -->
## Contacts
* Johannes Beck - [JohBeck](https://github.com/JohBeck)
* Darya Nikitina - [mewmiyu](https://github.com/mewmiyu)
* Dennis Hoebelt - [Hoebelt](https://github.com/Hoebelt)
* Danail Iordanov - [DanailIordanov](https://github.com/DanailIordanov)

<!-- SOURCES -->
## Sources
* [Icon](https://www.canva.com/ai-image-generator/)
* [Information](https://www.tu-darmstadt.de/universitaet/campus/stadtmitte_3/index.de.jsp)
