#!/bin/sh

### Argument to this script is: 
### Sample invocation:

#SBATCH --job-name=SynthSeg_image_generation
#SBATCH --time=48:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mem-per-cpu=8GB
#SBATCH --cpus-per-task=4
#SBATCH -A faird
#SBATCH --tmp=20gb
#SBATCH -p msismall

#SBATCH -e SynthSeg_image_generation-%j.err
#SBATCH -o SynthSeg_image_generation-%j.out

## build script here

source /home/faird/shared/code/external/envs/miniconda3/load_miniconda3.sh
conda activate SynthSeg-fixed-perms

export PYTHONPATH=${PYTHONPATH}:$1
export PYTHONPATH=${PYTHONPATH}:$1/SynthSeg/

cd $1

python ./SynthSeg/dcan/image_generation_for_all_ages.py $2 $2/SynthSeg_generated/ $3 $4 $5 $6
