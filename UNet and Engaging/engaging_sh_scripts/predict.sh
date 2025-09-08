#!/bin/bash

#SBATCH -J nnunet_predict
#SBATCH -o %j_predict.out
#SBATCH -e %j_predict.err
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

nnUNetv2_predict \
  -i /home/umakirk/inference_data/poster_artifact_added_pass5 \
  -o /home/umakirk/inference_predictions/poster_artifact_added_pass5 \
  -d Dataset100_KidneyUS \
  -c 2d \
  -f 0
