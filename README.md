# Automated Training GUI for Brain Segmentation

A user-friendly interface designed to automate and streamline the process of training deep learning algorithms for brain segmentation using SynthSeg and nnUNet.

## Overview

We made a GUI

**Created by:** @Emoney and @Kenevan-Carter

## Features

- **Automated Workflow**: Streamlines the entire training process from data preparation and preprocessing to inference
- **Custom Presets**: Save and reuse configurations for different training sessions
- **Modular Execution**: Toggle individual steps on/off based on your needs

## Requirements

- SynthSeg repository
- dcan-nn-unet repository
- Properly structured data directories
- Training and test datasets

## Directory Structure

Your data should be organized as shown below prior to running our program:

```
Project_Root/
├── nnUNet_raw_data_base/
│   ├── nnUNet_raw_data/
│   │   ├── Task000/
│   │   │   ├── imagesTr/
│   │   │   ├── imagesTs/
│   │   │   ├── labelsTr/
│   │   │   └── labelsTs/
│   │   ├── Task001/
│   │   ├── Task002/
│   │   └── ...
│   ├── nnUNet_preprocessed/ (created automatically)
│   └── nnUNet_cropped_data/ (created automatically)
```

## Configuration Parameters

### Required Paths

| Parameter | Description |
|-----------|-------------|
| **Dcan-nn-unet Path** | Path to your dcan-nn-unet repository |
| **SynthSeg Path** | Path to your SynthSeg repository |
| **Task Path** | Path to the folder containing train and test data for your specific task |
| **Raw Data Base Path** | Path to the folder containing raw, preprocessed, and cropped data folders |
| **Results Path** | Path where inferred segmentations and plots will be saved |
| **Trained Models Path** | Path where trained models will be stored |

### Training Arguments

| Parameter | Options | Description |
|-----------|---------|-------------|
| **Modality** | `t1`, `t2`, `t1t2` | Dataset modality type |
| **Task Number** | Integer | Unique identifier for your task (must match task path) |
| **Distribution** | `uniform`, `normal` | Data distribution type |
| **Number of SynthSeg Images** | Integer | Number of synthetic images for SynthSeg to generate per age group |

## Training Steps

The GUI provides 8 configurable training steps:

### 1. Resize Images
- **Purpose**: Initial setup step, formats your data to uniformly to be used by SynthSeg and nNUnet
- **Output**: Uniformly sized dataset

### 2. Mins/Maxes
- **Purpose**: Creates priors for SynthSeg image generation
- **Output**: Prior files stored in `GUI_repo/min_maxes/` subfolder

### 3. SynthSeg Image Creation
- **Purpose**: Generates synthetic training images and segmentations using SynthSeg
- **Output**: Files will be stored in your task directory. If you want to take a look at these before they are merged with the rest of the data, do not run the following steps

### 4. Copying Over SynthSeg Images
- **Purpose**: Moves synthetic data to training folders
- **Output**: Synthetic data is put into existing training folders

### 5. Create JSON File
- **Purpose**: Generates metadata required by nnUNet
- **Output**: JSON file will be put in your task folder

### 6. Plan and Preprocess
- **Purpose**: Sets up your dataset and extracts from it the necessary info that nNUnet will need in the model training step
- **Output**: Preprocessed data and extracted training parameters

### 7. Training the Model
- **Purpose**: Executes nnUNet model training
- **Output**: Trained model saved to your trained models path

### 8. Running Inference
- **Purpose**: Generates predictions on test data and plots of the model's performance compared to ground truth
- **Output**: 
  - `###_infer/`: Segmentation predictions
  - `###_results/`: Comparison plots

## Usage
(As of now, this has you must have access to the faird group on MSI)

1. **Launch the GUI**: Run pyqt_test to open the main UI window
2. **Configure Paths**: Fill in all required directory paths
3. **Set Parameters**: Specify modality, task number, distribution, and image count
4. **Select Steps**: Choose which training steps to execute (default: all selected)
5. **Execute**: Press Run

For questions or issues, please contact the development team: @Emoney and @Kenevan-Carter