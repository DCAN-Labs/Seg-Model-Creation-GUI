#!/bin/bash
sbatch <<EOT
#!/bin/sh

#SBATCH --job-name=$2_infer 
#SBATCH --mem=64g       
#SBATCH --time=8:00:00          # (HH:MM:SS)

#SBATCH -p a100-4     
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH -e infer_$2-%j.err
#SBATCH -o infer_$2-%j.out

#SBATCH -A $1

## build script here
module load gcc cuda/11.2
source /projects/standard/faird/shared/code/external/envs/miniconda3/load_miniconda3.sh
conda activate /projects/standard/faird/shared/code/external/envs/miniconda3/mini3/envs/pytorch_1.11.0


export nnUNet_raw_data_base="$3"
export nnUNet_preprocessed="$3/nnUNet_preprocessed/"
export RESULTS_FOLDER="$4"



nnUNet_predict -i $3/nnUNet_raw_data/Task$2/imagesTs -o /projects/standard/faird/shared/data/nnUNet_lundq163/$2_infer/ -t $2 -tr nnUNetTrainerV2_noMirroring -m 3d_fullres --disable_tta
EOT
