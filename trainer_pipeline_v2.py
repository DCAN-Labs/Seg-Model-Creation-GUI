import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path

# region ### SLURM SCRIPTS ###
# Note: create_min_maxes.sh and SynthSeg_image_generation.sh are unchanged from v1
SCRIPTS = [
    "SynthSeg_image_generation_v2.sh",
    "NnUnet_plan_and_preprocess_v2_agate.sh",
    "NnUnetTrain_v2_agate.sh",
    "infer_v2_agate.sh",
    "create_min_maxes_v2.sh"
]
# endregion

# region ### UTILITY FUNCTIONS ###
 
def wait_for_file(path: Path, timeout=10000, interval=5):
    '''
    Wait until a specified file path appears on disk
    Args:
        path: File path to watch for
        timeout: Max number of polling attempts
        interval: Seconds between each check
    Out: True if file appeared within timeout, False otherwise.
    '''
    for _ in range(timeout):
        if path.exists():
            return True
        time.sleep(interval)
    return False
 
def write_log(filepath, job_id):
    '''
    Writes the given job id to the specified log file path. Used for keeping track of active SLURM jobs.
    Args:
        filepath: path to the log file to write to
        job_id: the SLURM job id to write to the log file
    Out: None
    '''
    
    # Write job id to job log file
    with open(filepath, "a") as f:
        f.write(f"{job_id}\n")
 
def is_job_running(job_id):
    '''
    Checks if a job with the given job id is currently running in SLURM.
    Args:
        job_id: the SLURM job id to check
    Out: True if the job is running, False otherwise
    '''
    
    result = subprocess.run(['squeue', '--job', str(job_id)], capture_output=True, text=True) # Check squeue output
    return str(job_id) in result.stdout
 
def wait_for_job_to_finish(job_id, fold, check_interval=60):
    '''
    Waits for a SLURM job with the given job id to finish (specifically used for training and inference status updates in this program).
    Args:
        job_id: the SLURM job id to wait for
        fold: the fold number associated with the job (1-4 for training folds, -1 for inference)
        check_interval: how many seconds to wait between checks
    Out: None
    '''
    print_counter = 0
    while is_job_running(job_id):
        if fold >= 0 and print_counter % 1140 == 0:
            print(f"Waiting for fold {fold} to complete training...")
        elif fold == -1 and print_counter % 60 == 0:
            print("Waiting for inference to complete...")
        print_counter += 1
        time.sleep(check_interval)
 
def monitor_log_file(file_path, process):
    '''
    Monitors the output of a log file (meant for printing output of SLURM scripts to terminal)
    Args:
        file_path: path to the log file to monitor
        process: the subprocess object representing the running SLURM job
    Out: None
    '''
    with open(file_path, 'r') as f: 
        f.seek(0, os.SEEK_END)
        while process.poll() is None:
            line = f.readline()
            if line:
                print(line, end='')
            else:
                time.sleep(1)
 
def submit_job(command, log_path, wait_file=""):
    '''
    Submits a SLURM job given a bunch of parameters
    Args:
        command: list of command line arguments to submit the job, e.g. ["sbatch", "script.sh", "arg1", "arg2"]
        log_path: path to the log file where the job id will be written
        wait_file: special use case for certain steps that want to wait for a specific output file to be made before proceeding (e.g. min_maxes and synthseg steps)
    Out: the job id of the submitted job
    '''
    
    process = subprocess.Popen(command, stdout=subprocess.PIPE) # Start the job and capture the output
    job_id = process.stdout.readline().strip().split()[-1].decode("utf-8") # Extract the job id from the sbatch output
    write_log(log_path, job_id)
 
    file = None # Special case bug fixes
    if wait_file == "min_maxes":
        file = log_path.parent / f"Create_min_maxes-{job_id}.err"
    elif wait_file == "synthseg":
        file = log_path.parent / f"SynthSeg_image_generation-{job_id}.err"
 
    if file:
        if not wait_for_file(file):
            print(f"Timeout waiting for {file}. Canceling job {job_id}.")
            subprocess.run(["scancel", job_id])
            exit(1)
        monitor_log_file(file, process)
    process.wait()
    return job_id
 
def check_complete(err_path, fold):
    '''
    Used for training step. Checks if a training job has actually completed or if it was stopped due to hitting the SLURM time limit (in which case it needs to be re-submitted with the continue flag)
    Args:
        err_path: path to the error file
        fold: the fold number associated with the job
    Out: True if the job is complete, False otherwise
    '''

    # Does this by checking the error file for the "DUE TO TIME LIMIT" message that SLURM outputs when a job is stopped due to hitting the time limit 
    if err_path.exists():
        with open(err_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if "DUE TO TIME LIMIT" in line:
                    print(f"Fold {fold} training stopped due to time limit.")
                    return False
    print(f"Fold {fold} Training Complete.")
    return True
 
def move_matching_files(src: Path, dst: Path, pattern: str):
    '''
    Moves files from src to dst if they contain the specified pattern in their filename (Used for moving SynthSeg generated files that were misplaced in the wrong folders)
    Args:
        src: the source directory to look for files in
        dst: the destination directory to move matching files to
        pattern: the string pattern to look for in filenames to determine if they should be moved
    Out: None
    '''
    for file in os.listdir(src):
        if pattern in file:
            shutil.move(Path(src) / file, Path(dst) / file)
 
def set_up_slurm_scripts(task_logs: Path, all_slurm: Path):
    '''
    Sets up SLURM scripts for the training pipeline
    Args:
        task_logs: the directory where task logs will be stored (and where the SLURM scripts will be copied to)
        all_slurm: the directory containing all SLURM scripts
    Out: None
    '''
    
    # Copies SLURM scripts to the correct task log folder
    task_logs.mkdir(parents=True, exist_ok=True)
    for script in SCRIPTS:
        dest = task_logs / script
        shutil.copyfile(all_slurm / script, dest)
    (task_logs / "active_jobs.txt").write_text("") # Create an empty log file to store active job ids
 
def get_dataset_folder(task_number, dataset_name):
    # Returns the v2 dataset folder name, e.g. Dataset645_AnomalousInfant
    return f"Dataset{task_number}_{dataset_name}"
 
def get_nnunet_raw(raw_data_base_path):
    # In v2, nnUNet_raw lives directly under the base path
    return str(Path(raw_data_base_path) / "nnUNet_raw")
 
def get_nnunet_preprocessed(raw_data_base_path):
    # In v2, nnUNet_preprocessed lives directly under the base path
    return str(Path(raw_data_base_path) / "nnUNet_preprocessed")
 
def get_training_log_path(logs_path, task_number, fold, job_id):
    # Returns the v2 training log path (the .out file for specific fold and job id)
    return logs_path / f"Train_{fold}_{task_number}_nnUNetv2-{job_id}.out"
 
def get_training_error_path(logs_path, task_number, fold, job_id):
    # Returns the v2 training error path (the .err file for specific fold and job id)
    return logs_path / f"Train_{fold}_{task_number}_nnUNetv2-{job_id}.err"
 
def get_fold_dir(trained_models_path, task_number, dataset_name, fold):
    # Returns path to the v2 fold directory inside nnUNet_results (used as a backup check for fold 0 setup completion in case the training log files aren't being written for some reason)

    trained_models_path = Path(trained_models_path)
    dataset_folder = get_dataset_folder(task_number, dataset_name)
    return (
        # This is where nnUnet v2 automatically saves fold checkpoints and training logs. Basically just serves as a backup of the .out training log files that we expect to be written to the logs folder (there have been bugs where logs weren't being written here)
        trained_models_path / dataset_folder / "nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres" / f"fold_{fold}"
    )
 
def get_latest_training_log(fold_dir):
    # Returns the most recently modified training_log file in the dir, used to distinguish the specific training log to check for fold 0 setup completion in case there are multiple training logs in the fold directory for some reason (e.g. from multiple failed training attempts)
    if not fold_dir.exists():
        return None
 
    logs = [p for p in fold_dir.iterdir() if p.is_file() and p.name.startswith("training_log")]
    if not logs:
        return None
 
    return max(logs, key=lambda p: p.stat().st_mtime)
 
def file_has_epoch0(out_file):
    '''
    Checks if the given training log file contains the "epoch: 0" message that indicates fold 0 has completed its initial setup. Fold 0 setup for nnUNet has to finish before starting training on other folds
    Args:
        out_file: path to the training log file to check
    Out: True if the file contains the "epoch: 0" message, False otherwise
    '''
    
    if out_file is None or not out_file.exists():
        return False
 
    with out_file.open() as f:
        for line in f:
            if "epoch: 0" in line or "epoch:  0" in line:
                return True
    return False
 
def is_training_ready(out_file, trained_models_path, task_number, dataset_name):
    '''
    Checks if fold 0 has completed its initial setup and training on fold 0 can begin, checks training logs folder and backup fold directory in nnUNet_results incase necessary
    Args:
        out_file: path to the fold 0 training log file
        trained_models_path: base path to the nnUNet_results directory where fold directories and backup training logs are stored
        task_number: the task number of the dataset
        dataset_name: the name of the dataset
    Out: True if fold 0 setup is complete and training can begin, False otherwise
    '''
    
    # Helper function to confirm fold 0 initial setup is done before launching other folds
    if file_has_epoch0(out_file):
        print("Preparation complete. Ready to continue training on the rest of the folds.")
        return True
 
    # Fallback: check the backup training log inside the results folder
    fold_dir = get_fold_dir(trained_models_path, task_number, dataset_name, 0)
    latest_log = get_latest_training_log(fold_dir)
    if latest_log is not None and file_has_epoch0(latest_log):
        print("Preparation complete. Ready to continue training on the rest of the folds.")
        return True
 
    return False
 
def wait_fold_0_setup(out_file, err_file, trained_models_path, task_number, dataset_name):
    '''
    Waits for fold 0 to complete its initial setup before allowing the training pipeline to continue (fold 0 has to complete setup before launching training on other folds or else there will be errors)
    Args:
        out_file: path to the fold 0 .out log file
        err_file: path to the fold 0 .err error file
        trained_models_path: base path to the nnUNet_results directory where fold directories and backup training logs are stored (used as a backup check for fold 0 setup completion in case the training log files aren't being written for some reason)
        task_number: the task number of the dataset
        dataset_name: the name of the dataset
    Out: None
    '''
    print_counter = 0
    while not is_training_ready(out_file, trained_models_path, task_number, dataset_name):
        if err_file.exists():
            with err_file.open() as f:
                if any("Error" in line for line in f): # Check for any error messages in the fold 0 error log and exit if any are found to avoid waiting indefinitely for fold 0 setup to complete
                    print("Error detected in training log.")
                    exit(1)
        if print_counter % 30 == 0:
            print("Setup in progress...")
        print_counter += 1
        time.sleep(60)
 
def get_job_id_from_squeue(job_name):
    '''
    Gets the job id of a currently running job with the specified name by parsing the output of squeue
    Args:
        job_name: the name of the job to look for in squeue (e.g. "12345_0_Train_nnUNetv2")
    Out: the job id associated with the job name if found, None otherwise
    '''
    result = subprocess.run(['squeue', '--name', job_name, '--format', '%.18i'], capture_output=True, text=True)
    lines = result.stdout.strip().splitlines()
    if len(lines) > 1:
        return lines[1].strip()
    return None
 
# endregion

# region ### TRAINING FUNCTIONS ###
 
### Resize Images ###
def resize_images(args):
    '''
    Resizes images to the correct dimensions for nnUNet v2 training using the resize_images.py script from the dcan repo
    Args:
        args: the command line arguments passed to the program
    Out: None
    '''
    
    print("--- Now Resizing Images ---")
    task_path = Path(args.task_path)
    resize_script = str(Path(args.dcan_path) / "dcan" / "img_preproc" / "resize_images.py")
    data_dirs = ["imagesTr", "imagesTs", "labelsTr", "labelsTs"]
 
    for dir_name in data_dirs:
        curr_dir = task_path / dir_name
        old_dir = task_path / f"Old_{dir_name}"
 
        # Rename current dir to Old_xxx
        curr_dir.rename(old_dir)
        curr_dir.mkdir(exist_ok=True)
 
        print(f"Resizing {dir_name}...")
        subprocess.run(["python", resize_script, str(old_dir), str(curr_dir), f"--model={args.model_type}"])
 
        # Remove the Old_ directory once resize is complete
        shutil.rmtree(old_dir)
 
    print("--- Images Resized ---")
    
### Min Maxes ###
def min_max(args, logs_path, log_file_path, script_dir):
    '''
    Creates the min max files needed for SynthSeg augmented image creation by submitting a SLURM job to run the create_min_maxes_v2.sh script
    Args:
        args: the command line arguments passed to the program
        logs_path: the path to the logs directory where the SLURM script is located and where the job out and err files will be written
        log_file_path: the path to the log file where the active job ids are stored
        script_dir: the path to the directory where this script lives (used for constructing the output path for the min maxes)
    Out: None
        '''
    print("--- Now Creating Min Maxes ---")
    os.chdir(logs_path)
    time.sleep(3)
    output_path = Path(script_dir) / "min_maxes" / f"mins_maxes_{get_dataset_folder(args.task_number, args.dataset_name)}.npy"
    submit_job(["sbatch", "-W", str(logs_path / "create_min_maxes_v2.sh"), args.synth_path, args.task_path, str(output_path)], log_file_path, "min_maxes")
    print("--- Min Maxes Created ---")
    
### SynthSeg Image Creation ###
def SynthSeg_img(args, logs_path, log_file_path, script_dir):
    '''
    Creates augmented synthetic images using SynthSeg by submitting a SLURM job to run the SynthSeg_image_generation_v2.sh script
    Args:
        args: the command line arguments passed to the program
        logs_path: the path to the logs directory where the SLURM script is located and where the job out and err files will be written
        log_file_path: the path to the log file where the active job ids are stored
        script_dir: the path to the directory where this script lives (used for finding the min maxes output from the previous step)
    Out: None
    '''
    print("--- Now Creating Synthetic Images ---")
    os.chdir(logs_path)
    time.sleep(3)
    output_path = Path(script_dir) / "min_maxes" / f"mins_maxes_{get_dataset_folder(args.task_number, args.dataset_name)}.npy"
    submit_job([
        "sbatch", "-W",
        str(logs_path / "SynthSeg_image_generation_v2.sh"),
        args.synth_path, args.task_path, str(output_path),
        args.synth_img_amt,
        f"--modalities={args.modality}",
        f"--distribution={args.distribution}",
        args.task_number
    ], log_file_path, "synthseg")
    print("--- SynthSeg Images Generated ---")
    
### Moving Over SynthSeg Images ###
def copy_SynthSeg(args):
    '''
    Copies the SynthSeg generated images from the output folder to the correct imagesTr and labelsTr folders for nnUNet training, also moves any files that were misplaced in the wrong folders by the SynthSeg script (bug fixes)
    Args:
        args: the command line arguments passed to the program
    Out: None
    '''
    
    print("--- Now Moving Over SynthSeg Generated Images ---")
    util_dir = Path(args.dcan_path) / "dcan" / "util"
    task_path = Path(args.task_path)
 
    # Initial copy over
    subprocess.run(["python", str(util_dir / "copy_over_augmented_image_files.py"),
        str(task_path / "SynthSeg_generated" / "images"),
        str(task_path / "imagesTr"),
        str(task_path / "labelsTr")])
    subprocess.run(["python", str(util_dir / "copy_over_augmented_image_files.py"),
        str(task_path / "SynthSeg_generated" / "labels"),
        str(task_path / "imagesTr"),
        str(task_path / "labelsTr")])
 
    # Move any files that were misplaced in the wrong folders by the SynthSeg script (bug fixes)
    move_matching_files(task_path / "imagesTr", task_path / "labelsTr", "_SynthSeg_generated_0000.nii.gz")
    move_matching_files(task_path / "imagesTr", task_path / "labelsTr", "_SynthSeg_generated_0001.nii.gz")
 
    # Remove the SynthSeg_generated folder once all files have been moved
    if (task_path / "SynthSeg_generated").exists():
        shutil.rmtree(task_path / "SynthSeg_generated")
    print("--- Images Moved ---")
    
### Creating Dataset JSON ###
def create_json(args):
    '''
    Creates the dataset JSON file needed for nnUNet training using the dataset conversion script from the dcan repo
    Args:
        args: the command line arguments passed to the program
    Out: None
    '''
    
    print("--- Now Creating Dataset JSON ---")
    dataset_folder = get_dataset_folder(args.task_number, args.dataset_name)
    conversion_script = Path(args.dcan_path) / "dcan" / "dataset_conversion" / f"{dataset_folder}.py"
 
    if not conversion_script.exists():
        print(f"ERROR: Dataset conversion script not found: {conversion_script}")
        exit(1)
 
    # Don't think this is necassary but just in case, export the nnUNet paths here as well for the conversion script to use
    env = os.environ.copy()
    env.update({
        "nnUNet_raw":          get_nnunet_raw(args.raw_data_base_path),
        "nnUNet_preprocessed": get_nnunet_preprocessed(args.raw_data_base_path),
        "nnUNet_results":      args.trained_models_path,
    })
    
    subprocess.run(["python", str(conversion_script)], env=env)
    print("--- Dataset JSON Created ---")

### Plan and Preprocess ###
def p_and_p(args, logs_path, log_file_path, script_dir):
    '''
    Runs the preliminary plan and preprocess step of nnUNet v2 by submitting a SLURM job to run the NnUnet_plan_and_preprocess_v2_agate.sh script
    Args:
        args: the command line arguments passed to the program
        logs_path: the path to the logs directory where the SLURM script is located and where the job out and err files will be written
        log_file_path: the path to the log file where the active job ids are stored
        script_dir: the path to the directory where this script lives (Not actually used in this function but included for consistency with other functions)
    Out: None
    '''
    
    print("--- Now Running Plan and Preprocess ---")
    os.chdir(logs_path)
    time.sleep(3)
    submit_job([
        "sbatch", "-W",
        str(logs_path / "NnUnet_plan_and_preprocess_v2_agate.sh"),
        "faird",
        args.dcan_path,
        args.task_number,
        get_nnunet_raw(args.raw_data_base_path),
        get_nnunet_preprocessed(args.raw_data_base_path),
        args.trained_models_path
    ], log_file_path)
    print("--- Finished Plan and Preprocessing ---")
    
### Training Model ###
def model_training(args, logs_path, log_file_path, script_dir):
    '''
    Runs 5 folds of nnUNet v2 training by submitting SLURM jobs to run the NnUnetTrain_v2_agate.sh script, uses your Tr data folders
    Args:
        args: the command line arguments passed to the program
        logs_path: the path to the logs directory where the SLURM script is located and where the job out and err files will be written
        log_file_path: the path to the log file where the active job ids are stored
        script_dir: the path to the directory where this script lives (Not actually used in this function but included for consistency with other functions)
    Out: None
    '''
    print("--- Now Running NnUNet v2 Training ---")
    os.chdir(logs_path)
    job_ids = [None, None, None, None, None]
    complete = [False, False, False, False, False]
 
    nnunet_raw = get_nnunet_raw(args.raw_data_base_path)
    nnunet_preprocessed = get_nnunet_preprocessed(args.raw_data_base_path)
 
    # Defines the command to submit a training job for a specific fold, with an optional continue flag for re-submitting folds that hit the time limit
    def _train_cmd(fold, continue_flag=""):
        cmd = [
            "sbatch", "-W",
            str(logs_path / "NnUnetTrain_v2_agate.sh"),
            str(fold),                 # $1 fold
            "faird",                   # $2 account
            args.task_number,          # $3 dataset task number
            args.dcan_path,            # $4 dcan_path
            nnunet_raw,                # $5 nnUNet_raw
            nnunet_preprocessed,       # $6 nnUNet_preprocessed
            args.trained_models_path   # $7 nnUNet_results
        ]
        if continue_flag:
            cmd.append(continue_flag)  # $8 --c (optional)
        return cmd
 
    # Start fold 0 and wait for initial setup to complete before launching remaining folds
    time.sleep(3)
    submit_job(_train_cmd(0), log_file_path)
    job_ids[0] = get_job_id_from_squeue(f"{args.task_number}_0_Train_nnUNetv2")
    wait_fold_0_setup(
        get_training_log_path(logs_path, args.task_number, 0, job_ids[0]),
        get_training_error_path(logs_path, args.task_number, 0, job_ids[0]),
        args.trained_models_path,
        args.task_number,
        args.dataset_name
    )
    print("Begin training Fold 0")
 
    # Launch folds 1-4 after the initial setup, they will be automatically stopped if they hit the time limit and can be re-submitted with the continue flag
    for i in range(1, 5):
        print(f"Begin training Fold {i}")
        time.sleep(3)
        submit_job(_train_cmd(i), log_file_path)
        job_ids[i] = get_job_id_from_squeue(f"{args.task_number}_{i}_Train_nnUNetv2")
 
    # Re-submit any folds that hit the SLURM time limit with the --c (continue) flag
    while not all(complete):
        for i in range(5):
            wait_for_job_to_finish(job_ids[i], i)
            err_file = get_training_error_path(logs_path, args.task_number, i, job_ids[i])
            if check_complete(err_file, i):
                complete[i] = True
            else:
                time.sleep(3)
                submit_job(_train_cmd(i, "--c"), log_file_path)
                job_ids[i] = get_job_id_from_squeue(f"{args.task_number}_{i}_Train_nnUNetv2")
 
    print("--- Training Complete ---")
 
### Create Inferred Segmentations and Plots ###
def inference(args, logs_path, log_file_path, script_dir):
    '''
    Runs inference on the set aside test (Ts) data by submitting a SLURM job to run the infer_v2_agate.sh script, then creates dice plots of the results by running the evaluate_results.py script from the dcan repo
    Args:
        args: the command line arguments passed to the program
        logs_path: the path to the logs directory where the SLURM script is located and where the job out and err files will be written
        log_file_path: the path to the log file where the active job ids are stored
        script_dir: the path to the directory where this script lives (Not actually used in this function but included for consistency with other functions)
    Out: None
    '''
    
    print("--- Starting Inference ---")
    dataset_folder = get_dataset_folder(args.task_number, args.dataset_name)
    inferred_dir = Path(args.results_path) / f"{dataset_folder}_infer"
    inferred_dir.mkdir(parents=True, exist_ok=True)
 
    os.chdir(logs_path)
    time.sleep(3)
    submit_job([
        "sbatch", "-W",
        str(logs_path / "infer_v2_agate.sh"),
        "faird",
        args.task_number,
        dataset_folder,
        args.dcan_path,
        get_nnunet_raw(args.raw_data_base_path),
        get_nnunet_preprocessed(args.raw_data_base_path),
        args.trained_models_path,
        str(inferred_dir)
    ], log_file_path)
    job_id = get_job_id_from_squeue(f"{args.task_number}_infer_v2")
    wait_for_job_to_finish(job_id, -1)
    print("--- Inference Complete ---")
 
    # Create dice plots (still uses the SynthSeg conda env as before)
    print("--- Creating Plots ---")
    results_dir = Path(args.results_path) / f"{dataset_folder}_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    paper_dir = Path(args.synth_path) / "SynthSeg" / "dcan" / "paper"
    os.chdir(paper_dir)
    subprocess.run([
        "python", "evaluate_results.py",
        str(Path(args.task_path) / "labelsTs"),
        str(inferred_dir),
        str(results_dir)
    ])
    print("--- Plots Created ---")
 
# endregion

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="nnUNet v2 training pipeline with SynthSeg augmentation")
 
    # --- Paths (same as v1) ---
    parser = argparse.ArgumentParser()
    parser.add_argument('dcan_path')
    parser.add_argument('task_path')
    parser.add_argument('synth_path')
    parser.add_argument('raw_data_base_path')
    parser.add_argument('results_path')
    parser.add_argument('trained_models_path')
    parser.add_argument('modality')
    parser.add_argument('task_number')
    parser.add_argument('distribution')
    parser.add_argument('synth_img_amt')
    
    # New V2 arguments
    parser.add_argument('dataset_name') 
    parser.add_argument('model_type') # for resize images (infant or lifespan)
    
    parser.add_argument('list')
    
    args = parser.parse_args()
 
    # Export necessary paths
    os.environ.update({
        "PYTHONPATH": f"{args.synth_path}:{Path(args.synth_path) / 'SynthSeg'}:{args.dcan_path}:{Path(args.dcan_path) / 'dcan'}",
        "nnUNet_raw": str(Path(args.raw_data_base_path) / "nnUNet_raw"),
        "nnUNet_preprocessed": str(Path(args.raw_data_base_path) / "nnUNet_preprocessed"),
        "nnUNet_results": args.trained_models_path
    })
 
    # Some setup stuff - create logs folder, copy over SLURM scripts, set up log file path
    script_dir = Path(__file__).resolve().parent
    logs_path = script_dir / "logs" / get_dataset_folder(args.task_number, args.dataset_name)
    log_file_path = logs_path / "active_jobs.txt"
 
    set_up_slurm_scripts(logs_path, script_dir / "scripts" / "slurm_scripts_v2")
 
    # List of all the steps in the pipeline in the order they should be run
    run_list = [
        resize_images,
        min_max,
        SynthSeg_img,
        copy_SynthSeg,
        create_json,
        p_and_p,
        model_training,
        inference
    ]
 
    # Decode which steps to run from the GUI's encoded list
    flags = [args.list[i * 3 + 1] == '1' for i in range(len(run_list))]
 
    for step, should_run in zip(run_list, flags):
        if should_run:
            if step in [min_max, SynthSeg_img, p_and_p, model_training, inference]:
                step(args, logs_path, log_file_path, script_dir)
            else:
                step(args)
 
    print("PROGRAM COMPLETE!")
