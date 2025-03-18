#!/bin/bash

#################################################
## TEMPLATE VERSION 1.01                       ##
#################################################
## ALL SBATCH COMMANDS WILL START WITH #SBATCH ##
## DO NOT REMOVE THE # SYMBOL                  ## 
#################################################

#SBATCH --nodes=1                   # Use 1 node
#SBATCH --cpus-per-task=10          # Increase to 10 CPUs for faster processing
#SBATCH --mem=48GB                  # Increase memory to 48GB
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --constraint=a100           # Target A100 GPUs specifically
#SBATCH --time=02-00:00:00          # Maximum run time of 2 days
##SBATCH --mail-type=BEGIN,END,FAIL  # Email notifications for job start, end, and failure
#SBATCH --output=%u.%x              # Log file location

################################################################
## EDIT %u.%j.out AFTER THIS LINE IF YOU ARE OKAY WITH DEFAULT SETTINGS ##
################################################################

#SBATCH --partition=researchshort                 # Partition assigned
#SBATCH --account=sunjunresearch   # Account assigned (use myinfo command to check)
#SBATCH --qos=research-1-qos         # QOS assigned (use myinfo command to check)
#SBATCH --job-name=diagnostic    # Default job name (will be overridden)
#SBATCH --mail-user=myatmin.nay.2022@phdcs.smu.edu.sg  # Email notifications

#################################################
##            END OF SBATCH COMMANDS           ##
#################################################

# Set job name based on model and dataset
export SLURM_JOB_NAME="${MODEL_NAME}_${DATASET_NAME}"

# Purge the environment, load the modules we require.
# Refer to https://violet.smu.edu.sg/origami/module/ for more information
module purge
module load Python/3.10.16-GCCcore-13.3.0 
module load CUDA/12.6.0

# Activate the environment
source ~/myenv/bin/activate

python diagnostic.py