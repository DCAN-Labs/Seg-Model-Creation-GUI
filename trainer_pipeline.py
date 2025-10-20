import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path

# region ### SLURM SCRIPTS ###
SCRIPTS = [
    "SynthSeg_image_generation.sh",
    "NnUnet_plan_and_preprocess_agate.sh",
    "NnUnetTrain_agate.sh",
    "infer_agate.sh",
    "create_min_maxes.sh"
]
# endregion

# region ### UTILITY FUNCTIONS ###

def wait_for_file(path: Path, timeout=10000, interval=5):
    # Wait for a specified file to be made
    for _ in range(timeout):
        if path.exists():
            return True
        time.sleep(interval)
    return False

def write_log(filepath, job_id):
    # Write job id to job log file
    with open(filepath, "a") as f:
        f.write(f"{job_id}\n")

def is_job_running(job_id):
    # Checks to see if a specific job is running
    result = subprocess.run(['squeue', '--job', str(job_id)], capture_output=True, text=True)
    return str(job_id) in result.stdout

def wait_for_job_to_finish(job_id, fold, check_interval=60):
    # Waits for a specific job to finish (used for training and inference)
    print_counter = 0
    while is_job_running(job_id):
        if fold >= 0 and print_counter % 1140 == 0: # Case where this is being called for the train step
            print(f"Waiting for fold {fold} to complete training...")
        elif fold == -1 and print_counter % 60 == 0: # Case where this is being called for the inference step
            print("Waiting for inference to complete...")
        print_counter += 1
        time.sleep(check_interval)

def monitor_log_file(file_path, process):
    # Monitors the output of a log file. (Meant for printing output of SLURM scripts to terminal)
    with open(file_path, 'r') as f:
        f.seek(0, os.SEEK_END)
        while process.poll() is None:
            line = f.readline()
            if line:
                print(line, end='')
            else:
                time.sleep(1)

def submit_job(command, log_path, wait_file=""):
    # Submits a SLURM job given a bunch of parameters
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    job_id = process.stdout.readline().strip().split()[-1].decode("utf-8") # Gets job id and adds it to the active jobs log file
    write_log(log_path, job_id)

    file = None
    if wait_file == "min_maxes": # Waits for min max output file (.err for some reason) so that it can be monitored and printed to the terminal
        file = log_path.parent / f"Create_min_maxes-{job_id}.err"
    elif wait_file == "synthseg": # Waits for synthseg output file (.err for some reason) so that it can be monitored and printed to the terminal
        file = log_path.parent / f"SynthSeg_image_generation-{job_id}.err"

    if file:
        if not wait_for_file(file): 
            print(f"Timeout waiting for {file}. Canceling job {job_id}.")
            subprocess.run(["scancel", job_id])
            exit(1)
        monitor_log_file(file, process) # Monitor output file once it's created
    process.wait()
    return job_id

def check_complete(err_path, fold):
    # Checks to see if training jobs are actually finished, or if they need to be run again
    if err_path.exists(): # Searches through error file. If the job finished due to a time limit, trainig is not complete
        with open(err_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if "DUE TO TIME LIMIT" in line:
                    print(f"Fold {fold} training stopped due to time limit.")
                    return False
    print(f"Fold {fold} Training Complete.")
    return True

def move_matching_files(src: Path, dst: Path, pattern: str):
    # Used in synthseg step to move misplaced files
    for file in os.listdir(src):
        if pattern in file:
            shutil.move(Path(src) / file, Path(dst) / file)

def set_up_slurm_scripts(task_logs: Path, all_slurm: Path):
    # Run before any step starts, it just copies over slurm scripts to a task folder within logs
    task_logs.mkdir(parents=True, exist_ok=True)
    for script in SCRIPTS:
        dest = task_logs / script
        shutil.copyfile(all_slurm / script, dest)
    (task_logs / "active_jobs.txt").write_text("")

def get_training_log_path(logs_path, task_number, fold, job_id):
    # Returns the training output path
    return logs_path / f"Train_{fold}_{task_number}_nnUNet-{job_id}.out"

def get_training_error_path(logs_path, task_number, fold, job_id):
    # Returns the training error path
    return logs_path / f"Train_{fold}_{task_number}_nnUNet-{job_id}.err"

def is_training_ready(out_file):
    # Helper function to read fold 0 output to make sure initial setup is done
    if not out_file.exists():
        return False
    with out_file.open() as f:
        for line in f:
            if "epoch: 0" in line or "epoch:  0" in line: # If epochs have started, training is ready to continue for other folds
                print("Preparation complete. Ready to continue training on the rest of the folds.")
                return True
    return False

def wait_fold_0_setup(out_file, err_file):
    # Waits for fold 0 to finish setup before other folds start running
    print_counter = 0
    while not is_training_ready(out_file): # Continuously reads output file to detect if its ready to continue
        if err_file.exists(): # If theres an error in the preparation, exit
            with err_file.open() as f:
                if any("Error" in line for line in f):
                    print("Error detected in training log.")
                    exit(1)
        if print_counter % 30 == 0:
            print("Setup in progress...")
        print_counter += 1
        time.sleep(60)

def get_job_id_from_squeue(job_name):
    # Gets job id from a job name input
    result = subprocess.run(['squeue', '--name', job_name, '--format', '%.18i'], capture_output=True, text=True)
    lines = result.stdout.strip().splitlines()
    if len(lines) > 1:
        return lines[1].strip()
    return None
# endregion

#region ### TRAINING FUNCTIONS ###

### Resize Images
def resize_images(args):
    print("--- Now Resizing Images ---")
    subprocess.run(["python", str(Path(args.dcan_path) / "dcan" / "img_processing" / "resize_images_test.py"), args.task_path])
    print("--- Images Resized ---")

### Min Maxes ###
def min_max(args, logs_path, log_file_path, script_dir):
    print("--- Now Creating Min Maxes ---")
    os.chdir(logs_path)
    time.sleep(3)
    output_path = Path(script_dir) / "min_maxes" / f"mins_maxes_task_{args.task_number}.npy"
    submit_job(["sbatch", "-W", str(logs_path / "create_min_maxes.sh"), args.synth_path, args.task_path, str(output_path)], log_file_path, "min_maxes")
    print("--- Min Maxes Created ---")

### SynthSeg Image Creation ###
def SynthSeg_img(args, logs_path, log_file_path, script_dir):
    print("--- Now Creating Synthetic Images ---")
    os.chdir(logs_path)
    time.sleep(3)
    output_path = Path(script_dir) / "min_maxes" / f"mins_maxes_task_{args.task_number}.npy"
    submit_job([
        "sbatch", "-W",
        str(logs_path / "SynthSeg_image_generation.sh"),
        args.synth_path, args.task_path, str(output_path),
        args.synth_img_amt,
        f"--modalities={args.modality}",
        f"--distribution={args.distribution}",
        args.task_number
    ], log_file_path, "synthseg")
    print("--- SynthSeg Images Generated ---")

### Moving Over SynthSeg Images ###
def copy_SynthSeg(args):
    # Copies over synthseg generated images from SynthSeg_generated to raw data folder
    print("--- Now Moving Over SynthSeg Generated Images ---")
    util_dir = Path(args.dcan_path) / "dcan" / "util"
    subprocess.run(["python", str(util_dir / "copy_over_augmented_image_files.py"), str(Path(args.task_path) / "SynthSeg_generated" / "images"), str(Path(args.task_path) / "imagesTr"), str(Path(args.task_path) / "labelsTr")])
    subprocess.run(["python", str(util_dir / "copy_over_augmented_image_files.py"), str(Path(args.task_path) / "SynthSeg_generated" / "labels"), str(Path(args.task_path) / "imagesTr"), str(Path(args.task_path) / "labelsTr")])

    task_path = Path(args.task_path)
    # Some files don't get put in the right folder and need to be moved
    move_matching_files(task_path / "imagesTr", task_path / "labelsTr", "_SynthSeg_generated_0000.nii.gz")
    move_matching_files(task_path / "imagesTr", task_path / "labelsTr", "_SynthSeg_generated_0001.nii.gz")

    if (task_path / "SynthSeg_generated").exists():
        shutil.rmtree(task_path / "SynthSeg_generated")
    print("--- Images Moved ---")

### Creating Dataset Json ###
def create_json(args):
    print("--- Now Creating Dataset json ---")
    task_path = Path(args.task_path)
    # Json gets created
    subprocess.run([
        "python",
        str(Path(args.dcan_path) / "dcan" / "dataset_conversion" / "create_json_file.py"),
        f"Task{args.task_number}",
        str(Path(args.dcan_path) / "look_up_tables" / "Freesurfer_LUT_DCAN.txt"),
        f"--modalities={args.modality}"
    ])
    # Some errors in json need to be fixed
    subprocess.run([
        "python",
        str(Path(args.dcan_path) / "dcan" / "dataset_conversion" / "fix_json_file.py"),
        str(task_path / 'dataset.json'), str(task_path / 'dataset2.json'),
        str(Path(args.dcan_path) / "look_up_tables" / "Freesurfer_LUT_DCAN.txt")
    ])
    (task_path / 'dataset.json').unlink()
    (task_path / 'dataset2.json').rename(task_path / 'dataset.json')
    print("--- Dataset json Created ---")

### Plan and Preprocess ###
def p_and_p(args, logs_path, log_file_path, script_dir):
    print("--- Now Running Plan and Preprocess ---")
    os.chdir(logs_path)
    time.sleep(3)
    submit_job(["sbatch", "-W", "NnUnet_plan_and_preprocess_agate.sh", args.raw_data_base_path, args.task_number, args.trained_models_path], log_file_path)
    print("--- Finished Plan and Preprocessing ---")

### Training Model ###
def model_training(args, logs_path, log_file_path, script_dir):
    print("--- Now Running NnUNet Training ---")
    os.chdir(logs_path)
    job_ids = [None, None, None, None, None]
    complete = [False, False, False, False, False]
    
    # Start fold 0 training and wait until it finishes the setup to run next folds
    time.sleep(3)
    submit_job(["sbatch", "-W", "NnUnetTrain_agate.sh", "0", "faird", args.task_number, args.raw_data_base_path, args.trained_models_path], log_file_path)
    job_ids[0] = get_job_id_from_squeue(f"{args.task_number}_0_Train_nnUNet")
    wait_fold_0_setup(
        get_training_log_path(logs_path, args.task_number, 0, job_ids[0]),
        get_training_error_path(logs_path, args.task_number, 0, job_ids[0])
    )
    print("Begin training Fold 0")

    # Once setup is ready, start training the next folds
    for i in range(1, 5):
        print(f"Begin training Fold {i}")
        time.sleep(3)
        submit_job(["sbatch", "-W", "NnUnetTrain_agate.sh", str(i), "faird", args.task_number, args.raw_data_base_path, args.trained_models_path], log_file_path)
        job_ids[i] = get_job_id_from_squeue(f"{args.task_number}_{i}_Train_nnUNet")

    # If folds finish training due to SLURM time limit, continue training with -c argument
    while not all(complete):
        for i in range(5):
            wait_for_job_to_finish(job_ids[i], i)
            err_file = get_training_error_path(logs_path, args.task_number, i, job_ids[i])
            if check_complete(err_file, i):
                complete[i] = True
            else:
                time.sleep(3)
                submit_job(["sbatch", "-W", "NnUnetTrain_agate.sh", str(i), "faird", args.task_number, args.raw_data_base_path, args.trained_models_path, "-c"], log_file_path)
                job_ids[i] = get_job_id_from_squeue(f"{args.task_number}_{i}_Train_nnUNet")
    print("--- Training Complete ---")

### Create Inferred Segmentations and Plots ###
def inference(args, logs_path, log_file_path, script_dir):
    # Created inferred segmentations
    print("--- Starting Inference ---")
    inferred_dir = Path(args.results_path) / f"{args.task_number}_infer"
    inferred_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(logs_path)
    time.sleep(3)
    submit_job(["sbatch", "-W", "infer_agate.sh", "faird", args.task_number, args.raw_data_base_path, args.trained_models_path], log_file_path)
    job_id = get_job_id_from_squeue(f"{args.task_number}_infer")
    wait_for_job_to_finish(job_id, -1)
    print("--- Inference Complete ---")

    # Create dice plots
    print("--- Creating Plots ---")
    results_dir = Path(args.results_path) / f"{args.task_number}_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    paper_dir = Path(args.synth_path) / "SynthSeg" / "dcan" / "paper"
    os.chdir(paper_dir)
    subprocess.run(["python", "evaluate_results.py",
                    str(Path(args.task_path) / "labelsTs"),
                    str(inferred_dir),
                    str(results_dir)])
    print("--- Plots Created ---")
# endregion

if __name__ == '__main__':
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
    parser.add_argument('list')
    args = parser.parse_args()

    # Export necessary paths
    os.environ.update({
        "PYTHONPATH": f"{args.synth_path}:{Path(args.synth_path) / 'SynthSeg'}:{args.dcan_path}:{Path(args.dcan_path) / 'dcan'}",
        "nnUNet_raw_data_base": args.raw_data_base_path,
        "nnUNet_preprocessed": str(Path(args.raw_data_base_path) / "nnUNet_preprocessed"),
        "RESULTS_FOLDER": args.trained_models_path
    })

    # Some setup stuff
    script_dir = Path(__file__).resolve().parent
    logs_path = script_dir / "logs" / f"Task{args.task_number}"
    log_file_path = logs_path / "active_jobs.txt"

    set_up_slurm_scripts(logs_path, script_dir / "scripts" / "slurm_scripts")

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
    
    # Figures out what functions user wants to run from selection in GUI and runs only those ones
    flags = [args.list[i * 3 + 1] == '1' for i in range(len(run_list))]

    for step, should_run in zip(run_list, flags):
        if should_run:
            if step in [min_max, SynthSeg_img, p_and_p, model_training, inference]: # These functions need extra arguments
                step(args, logs_path, log_file_path, script_dir)
            else:
                step(args)

    print("PROGRAM COMPLETE!")