#!/bin/bash  
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --job-name=BPS-FH_training
#SBATCH --mem=8G
#SBATCH --gres=gpu:K80:1

#SBATCH --mail-type=ALL
#SBATCH --mail-user=gianluca.micchi@univ-lille.fr

module load tensorflow/1.12.0/anaconda3/2018.12
python main.py
