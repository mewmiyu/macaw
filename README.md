# macaw

MACAW is a project we did for our Augmented Vision course at TU Darmstadt. It is an application that recognizes university buildings and applies natural feature tracking to track the movement of the buildings inside the frame. We also show a bounding box, capturing the tracked features of the building front, along with information about the building. 

## Installation

Make sure you are in a conda environment, but if you would like to create a new one, you can do so by:

    conda create -n macaw python=3.10
    conda activate macaw

In order to run our script you need to install the following packages:

    conda install numpy matplotlib opencv pycocotools imutils wandb gst-plugins-base gst-plugins-good gstreamer -c conda-forge

    conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
    
    pip install huggingface_hub 

## Configuration

In order to use macaw you need to provide a config file, describing what you want to do. An example file with all possible configurations can be found in [configs/base.yaml](configs/base.yaml). We also provided more specific config files, based on the method:

- [configs/run-macaw.yaml](configs/run-macaw.yaml) runs the application for a given video file. The weights for the detector are also automatically downloaded if "Download" is set to True.
- [configs/train.yaml](configs/train.yaml) configures macaw to train the detection model on the specified dataset. You can define the backbone architecture of the FasterRCNN, Hyperparameters for the training, as well as the usage of weights and biases.
- [configs/eval.yaml](configs/eval.yaml) configures macaw to see the results of the detection model. The config file therefore contains the data as well as the model-checkpoint. If "Download" is set to true, macaw tries to download the model-weights.
- [configs/label.yaml](configs/label.yaml) configures macaw to label a dataset. The annotation json-file stores all annotations made by the user. If the mode is set to "review" the already made annotations are displayed.

## Run macaw

To run macaw you need to open a console/terminal in the working directory and run the following command

    python src/macaw.py --config configs/run-macaw.yaml
