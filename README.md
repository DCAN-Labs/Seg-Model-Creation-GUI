# Automated Training GUI for Brain Segmentation

A user-friendly interface designed to automate and streamline the process of training deep learning algorithms for brain segmentation using SynthSeg and nnUNet.

## Overview

This GUI addresses the challenges of manually executing commands in the terminal for brain segmentation model training. By providing an intuitive interface, it eliminates the tedious and error-prone process of copying and pasting documentation while significantly reducing workflow downtime.

**Created by:** @Emoney and @Kenevan-Carter

## Features

- **Automated Workflow**: Streamlines the entire training process from data preparation to inference
- **Custom Presets**: Save and reuse parameter configurations for different training sessions
- **Modular Execution**: Toggle individual steps on/off based on your needs
- **Error Reduction**: Eliminates manual command-line operations and potential copy-paste errors
- **Time Efficient**: Focus on analytical aspects rather than technical implementation details

## Requirements

- SynthSeg repository
- dcan-nn-unet repository
- Properly structured data directories
- Training and test datasets

## Directory Structure

Your data should be organized as follows:

```
Project_Root/
├── nnUNet_raw_data_base/
│   ├── nnUNet_raw_data/
│   │   ├── Task001_YourTask/
│   │   │   ├── imagesTr/
│   │   │   ├── imagesTs/
│   │   │   ├── labelsTr/
│   │   │   └── labelsTs/
│   │   ├── Task002_AnotherTask/
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
| **Number of SynthSeg Images** | Integer | Synthetic images to generate per age group |

## Training Steps

The GUI provides 8 configurable training steps:

### 1. Resize Images
- **Purpose**: Initial data formatting for SynthSeg and nnUNet compatibility
- **Output**: Uniformly formatted dataset

### 2. Mins/Maxes
- **Purpose**: Creates priors for SynthSeg image generation
- **Output**: Prior files stored in `GUI_repo/min_maxes/` subfolder

### 3. SynthSeg Image Creation
- **Purpose**: Generate synthetic training images
- **Output**: Synthetic images/labels in separate task subfolder

### 4. Copying Over SynthSeg Images
- **Purpose**: Move synthetic data to training folders
- **Output**: Synthetic data integrated into training dataset

### 5. Create JSON File
- **Purpose**: Generate metadata required by nnUNet
- **Output**: JSON configuration file in task folder

### 6. Plan and Preprocess
- **Purpose**: nnUNet preprocessing and dataset preparation
- **Output**: Preprocessed data and extracted training parameters

### 7. Training the Model
- **Purpose**: Execute nnUNet model training
- **Output**: Trained model saved to specified trained models path

### 8. Running Inference
- **Purpose**: Generate predictions on test data
- **Output**: 
  - `###_infer/`: Segmentation predictions
  - `###_results/`: Comparison plots with ground truth

## Usage

1. **Launch the GUI**: Open the main UI window
2. **Configure Paths**: Fill in all required directory paths
3. **Set Parameters**: Specify modality, task number, distribution, and image count
4. **Select Steps**: Choose which training steps to execute (default: all selected)
5. **Create/Load Preset**: Save current configuration or load existing preset
6. **Execute**: Run the selected training pipeline

## Tips

- **Full Pipeline**: Keep all checkboxes selected for complete start-to-finish training
- **Partial Execution**: Uncheck steps you want to skip for debugging or resuming workflows
- **Presets**: Use custom presets to quickly switch between different experimental configurations
- **Directory Permissions**: Ensure you have proper access permissions to all specified directories

## Output Files

- **Models**: Trained models saved to your specified trained models directory
- **Segmentations**: Inference results in `###_infer` folders
- **Visualizations**: Comparison plots in `###_results` folders
- **Metadata**: JSON configuration files in task directories
- **Preprocessing**: Automated creation of preprocessed and cropped data folders

## Troubleshooting

- Verify all paths are correctly specified and accessible
- Ensure task numbers match between configuration and directory structure
- Check that required repositories (SynthSeg, dcan-nn-unet) are properly installed
- Confirm sufficient disk space for synthetic image generation and model training

---

For questions or issues, please contact the development team: @Emoney and @Kenevan-Carter