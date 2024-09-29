#Introduction to the Automated Training GUI and How to Use It.

The primary objective of creating a GUI was to develop a user-friendly
interface to automate and streamline the process of training a deep
learning algorithm for the purpose of brain segementation. The traditional
approach prior to the app's creation of manually executing commands in the
terminal, involving extensive copying and pasting of documentation, proved
to be very tedius and time-consuming for users. This repetitive process not
only made way for a ridiculous amount of potential errors, but also added a
significant amount of unnecessary downtime in the workflow. The GUI
addresses these challanges by providing an intuitive interface that
simplifies and accelerates the training process which allows users to focus
more on the analytical aspects of the process instead of the intricacies of
command-line operations.

In this NoteBook,the different functions of the GUI created by @Emoney and
@Kenevan-Carter for the purpose of automating the training process will be
detailed.
                                        ---Functionality---
Parameters:

Paths - On launch of the main Ui window, the user will see a list of path arguments that must be filled out that are specific to where their own directories are located. Every user will not be able to access the same files due to permission errors so it is important that the user has their own paths to a SynthSeg directory, Dcan-nn-unet, a Task Folder where all of the data is stored, and their raw data base.
    
    Dcan-nn-unet Path - Path to your dcan-nn-unet repo
    SynthSeg Path - Path to your SynthSeg repo
    Task Path - Path to the folder containing the train and test data for your specific task
    Raw Data Base Path - Path to the folder that will contain the raw, preprocessed and cropped data folders for your tasks. This should be a couple directories above your task folder.
    
    Folder structure should look something like this:
    
    Some general name/s
    +-- nnUNet_raw_data_base
        +-- nnUNet_raw_data
        |   +-- Task...
        |   |   +-- imagesTr
        |   |   +-- imagesTs
        |   |   +-- labelsTr
        |   |   +-- labelsTs
        |   +-- Task...
        |   +-- Task...
        +-- nnUNet_preprocessed
        |    ...
        +-- nnUNet_cropped_data
             ...
            
     Your prepocessed and cropped folders as well as the task folders within them will be created automatically by the program the first time around.
     
     Results path - Path to the folder where you want your inferred segmentations and plots to go
     Trained models path - Path to the folder where you want to keep the models created by the train step
    
Arguments - Users will also be asked to input certain parameters to guide the model training process.

    Modality - The user will also be asked to specify the modality their dataset is comprised of. Options: t1, t2, or t1t2
    Distribution - Options: uniform, or normal
    Task Number - The number id given to your specific task. This should match up with your task path.
    Number of SynthSeg Generated Images - The number of synthetic images you want SynthSeg to create per age group.
    
Presets: 
    The user can create custom presets, which streamline the setup process for new training sessions. This feature not only saves time but also reduces the likelihood of errors, ensuring a smoother and more efficient start.
Check Boxes:
    Listed on the right side of the screen when launching the main UI window are 8 different check boxes . These are the necessary steps for the model training process. If you want to run the model training from start to finish, keep all the boxes selected, otherwise, you can pick and choose the steps you want to run.
    
    Resize Images - Initial setup step, formats your data to uniformly to be used by SynthSeg and nNUnet.
    Mins/Maxes - Creates priors for SynthSeg image generation. These files will be stored within the GUI repo in a subfolder called min_maxes.
    SynthSeg Image Creation - Synthetic image creation step, SynthSeg will produce the number of synthetic images/labels you specified. These will be put into a seperate folder within your task folder if you want to do anything with them before copying them over.
    Copying Over SynthSeg Images - Moves your synthetic data into your corresponding train folders. 
    Create JSON File - Creates the metadata that nNUnet needs to train. This JSON file will be put in your task folder.
    Plan and Preprocess - nNUnet preprocess step, sets up your dataset and extracts from it the necessary info that nNUnet will need in the model training step.
    Training the Model - nnUNet model training step, trains a model using your train data. The model will output to your trained models path.
    Running Inference - nnUNet inference step, the trained model will create inferred segmentations for your test data and create plots comparing it to the ground truth labels. These will go in your results folder with ###_infer containing the segmentations and ###_results containing the plots.




 

                                         ---How to Launch the GUI---
***Requirements***

*Run this on a persistent desktop within MSI*

*Must be within the SynthSeg-fixed-perms environment*

source /home/faird/shared/code/external/envs/miniconda3/load_miniconda3.sh

conda activate SynthSeg-fixed-perms
---To launch the GUI, cd into the Seg-Model-Creation-GUI and use the command:
python pyqt_test.py

Or run 
python *path to pyqt_test.py*, 
