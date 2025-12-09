#!/bin/sh

### Argument to this script is the fold number (between 0 and 4 
### inclusive) and -A argument 
### Sample invocation: sbatch script.sh <fold_number> <argument>

#SBATCH --job-name=plan_and_preprocess # job name
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)

#SBATCH --mem=90g                 # memory per cpu-core (what is the default?)
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
#SBATCH -p a100-4     
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=6               # total number of tasks across all nodes

#SBATCH -e Train_plan_and_preprocess-%j.err
#SBATCH -o Train_plan_and_preprocess-%j.out

module load gcc cuda/11.2
source /projects/standard/faird/shared/code/external/envs/miniconda3/load_miniconda3.sh
conda activate /projects/standard/faird/shared/code/external/envs/miniconda3/mini3/envs/pytorch_1.11.0


export nnUNet_raw_data_base="$1"
export nnUNet_preprocessed="$1/nnUNet_preprocessed"
export RESULTS_FOLDER="$3"

nnUNet_plan_and_preprocess -t $2 --verify_dataset_integrity
