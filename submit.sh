#!/bin/bash
#SBATCH -p gpu
#SBATCH --mem=32GB
#SBATCH --gres=gpu
#SBATCH --mail-type=begin,end
#SBATCH --mail-user=ams90@kent.ac.uk
#SBATCH --array=1

source /home/bwc/ams90/miniconda3/etc/profile.d/conda.sh
conda activate torchenv

# Define your commands
COMMANDS=("python Bigram.py"

num_jobs=${#COMMANDS[@]}


# Set up an index for array job
INDEX=$((SLURM_ARRAY_TASK_ID - 1))

# Execute the command
${COMMANDS[$INDEX]}
