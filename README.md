# Automated Training GUI for Brain Segmentation

A user-friendly interface designed to automate and streamline the process of training deep learning algorithms for brain segmentation using SynthSeg and nnUNet.

## Documentation Overview

This documentation explains how to run model training using both Version 1 and Version 2 of the nnUNet workflow. Sections labeled with V1 or V2 apply only to that version. If a section does not specify a version, the instructions apply to both workflows.

## Features

- **Automated Workflow**: Streamlines the entire training process from data preparation and preprocessing to inference
- **Custom Presets**: Save and reuse configurations for different training sessions
- **Modular Execution**: Toggle individual steps on/off based on your needs

## Requirements

- SynthSeg repository
- dcan-nnUNet repository
- Properly structured data directories; see Directory Structure V1 and V2
- Training and test datasets
- Access to the faird group on MSI
  
## Directory Structure V1

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

## Directory Structure V2

Your data should be organized as shown below prior to running our program:

```
Project_Root/
├── nnUNet_raw/
│   ├── Dataset[Task Number]_[DatasetName]                 e.g Dataset644_AnomalousInfant
│   │   ├── imagesTr/
│   │   ├── imagesTs/
│   │   ├── labelsTr/
│   │   └── labelsTs/
│   ├── Dataset[Task Number]_[DatasetName]/
│   ├── Dataset[Task Number]_[DatasetName]/
│   └── ...
├── nnUNet_preprocessed/ (created automatically)  ?
└── nnUNet_results/ (created automatically)  ?
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

### Training Arguments V1

| Parameter | Options | Description |
|-----------|---------|-------------|
| **Modality** | `t1`, `t2`, `t1t2` | Dataset modality type |
| **Task Number** | Integer | Unique identifier for your task (must match task path) |
| **Distribution** | `uniform`, `normal` | Data distribution type |
| **Number of SynthSeg Images** | Integer | Number of synthetic images for SynthSeg to generate per age group |

### Training Arguments V2

| Parameter | Options | Description |
|-----------|---------|-------------|
| **Modality** | `t1`, `t2`, `t1t2` | Dataset modality type |
| **Task Number** | Integer | Unique identifier for your task (must match task path) |
| **Distribution** | `uniform`, `normal` | Data distribution type |
| **Number of SynthSeg Images** | Integer | Number of synthetic images for SynthSeg to generate per age group |
| **Dataset Name** | `e.g AnomalousInfant` | Name of training dataset |
| **Model Type**| `lifespan`, `infant` | Specify age parameter of Dataset | 

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
(As of now, to run this, you must have access to the faird group on MSI)
1. **Launch Environment**: Run ```source /projects/standard/faird/shared/code/external/envs/miniconda3/load_miniconda3.sh``` and ```conda activate SynthSeg-fixed-perms``` to gain access to the environment variables needed to run this program.
2. **Launch the GUI**: Run:```python trainer_gui.py```to open the main UI window
3. **Select version**: Either  V1 or V2
4. **Configure Paths**: Fill in all required directory paths
5. **Set Parameters**: Specify modality, task number, distribution, image count etc..
6. **Select Steps**: Choose which training steps to execute (default: all selected)
7. **Execute**: Press Run

## Canceling Process
To stop a running process, press the cancel button in the GUI if available. If the process does not stop cleanly, terminate it from the terminal where the GUI was launched.

For questions or issues, please contact the development team: @Emoney and @Kenevan-Carter
