import subprocess


def write_predict_script(name: str, output_path="engaging_sh_scripts/predict.sh"):
    script_content = f"""#!/bin/bash

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

nnUNetv2_predict \\
  -i /home/umakirk/inference_data/{name} \\
  -o /home/umakirk/inference_predictions/{name} \\
  -d Dataset100_KidneyUS \\
  -c 2d \\
  -f 0
"""
    with open(output_path, "w", newline='\n') as f:
        f.write(script_content)
