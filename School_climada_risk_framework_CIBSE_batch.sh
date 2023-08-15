#!/bin/bash -l
#SBATCH --mem=32GB
#SBATCH --ntasks=8
#SBATCH --output=cibse_risk.txt
#SBATCH --error=cibse_risk.err
#SBATCH --time=00-06:00:00
# SBATCH --qos=normal
#SBATCH --export=NONE
conda activate climada_env_2022

python School_climada_risk_framework_CIBSE.py $1 $2 $3
