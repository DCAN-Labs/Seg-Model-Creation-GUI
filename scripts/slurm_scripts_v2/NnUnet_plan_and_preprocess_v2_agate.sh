#!/bin/sh

### nnUNetv2 Plan and Preprocess
### Args: $1=account, $2=dcan_path, $3=dataset_id (numeric), $4=nnUNet_raw, $5=nnUNet_preprocessed, $6=nnUNet_results
### Sample invocation: ./NnUnet_plan_and_preprocess_v2_agate.sh faird /path/to/dcan-nnunet-v2 645 /raw/ /preprocessed/ /results/
 
#SBATCH --job-name=plan_and_preprocess_v2
#SBATCH --time=24:00:00
#SBATCH --mem=90g
#SBATCH -p a100-4
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=6
#SBATCH -A $1
 
#SBATCH -e Plan_and_preprocess_v2-%j.err
#SBATCH -o Plan_and_preprocess_v2-%j.out
 
module load gcc cuda/11.2
module load python3/3.12.4_anaconda2024.06-1_libmamba
 
cd $2
source $2/.venv/bin/activate
 
export nnUNet_raw="$4"
export nnUNet_preprocessed="$5"
export nnUNet_results="$6"
 
nnUNetv2_plan_and_preprocess -d $3 --verify_dataset_integrity
