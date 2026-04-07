#!/bin/bash
sbatch <<EOT
#!/bin/sh
 
### nnUNetv2 Inference
### Args: $1=account, $2=dataset_id (numeric), $3=dataset_folder (Dataset###_NAME),
###       $4=dcan_path, $5=nnUNet_raw, $6=nnUNet_preprocessed, $7=nnUNet_results, $8=output_path
### Sample invocation: ./infer_v2_agate.sh faird 645 Dataset645_AnomalousInfant /path/to/dcan-nnunet-v2 /raw/ /preprocessed/ /results/ /output/
 
#SBATCH --job-name=${2}_infer_v2
#SBATCH --mem=64g
#SBATCH --time=8:00:00
#SBATCH -p a100-4
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH -A $1
 
#SBATCH -e infer_v2_${2}-%j.err
#SBATCH -o infer_v2_${2}-%j.out
 
module load gcc cuda/11.2
module load python3/3.12.4_anaconda2024.06-1_libmamba
 
cd $4
source $4/.venv/bin/activate
 
export nnUNet_raw="$5"
export nnUNet_preprocessed="$6"
export nnUNet_results="$7"
 
# nnUNetv2_predict flags: -d (dataset id), -c (config), -tr (trainer)
nnUNetv2_predict -i $5/$3/imagesTs -o $8 -d $2 -c 3d_fullres -tr nnUNetTrainerNoMirroring
EOT