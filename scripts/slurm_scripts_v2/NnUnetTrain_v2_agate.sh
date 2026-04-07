#!/bin/bash
sbatch <<EOT
#!/bin/sh
 
### nnUNetv2 Training
### Args: $1=fold, $2=account, $3=dataset_id (numeric), $4=dcan_path,
###       $5=nnUNet_raw, $6=nnUNet_preprocessed, $7=nnUNet_results, [$8=--c (continue flag)]
### Sample invocation: ./NnUnetTrain_v2_agate.sh 0 faird 645 /path/to/dcan-nnunet-v2 /raw/ /preprocessed/ /results/
### Continue invocation: ./NnUnetTrain_v2_agate.sh 0 faird 645 /path/to/dcan-nnunet-v2 /raw/ /preprocessed/ /results/ --c
 
#SBATCH --job-name=${3}_${1}_Train_nnUNetv2
#SBATCH --mem=90g
#SBATCH --time=24:00:00
#SBATCH -p msigpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=6
#SBATCH -A $2
 
#SBATCH -e Train_${1}_${3}_nnUNetv2-%j.err
#SBATCH -o Train_${1}_${3}_nnUNetv2-%j.out
 
module load gcc cuda/11.2
module load python3/3.12.4_anaconda2024.06-1_libmamba
 
cd $4
source $4/.venv/bin/activate
 
export nnUNet_raw="$5"
export nnUNet_preprocessed="$6"
export nnUNet_results="$7"
 
# nnUNetv2_train arg order: <dataset_id> <config> <fold> -tr <trainer> [--c]
nnUNetv2_train $3 3d_fullres $1 -tr nnUNetTrainerNoMirroring $8
EOT