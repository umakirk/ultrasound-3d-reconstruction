#!/bin/bash

##SBATCH -J sim_1
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -p mit_normal_gpu
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=umakirk@mit.edu
#SBATCH --gres=gpu:1
#SBATCH --mem=10G

module load miniforge/24.3.0-0
source activate torch

export nnUNet_raw="/home/umakirk/nnUNet_raw"
export nnUNet_preprocessed="/home/umakirk/nnUNet_preprocessed"
export nnUNet_results="/home/umakirk/nnUNet_results"

#nnUNetv2_plan_and_preprocess -d 878 --verify_dataset_integrity
nnUNetv2_train 100 2d 0
