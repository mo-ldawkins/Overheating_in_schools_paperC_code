#!/bin/bash -l
#SBATCH --mem=32GB
#SBATCH --ntasks=8
#SBATCH --output=logs/clim_end2.txt
#SBATCH --error=logs/clim_end2.err
#SBATCH --time=00-06:00:00
# SBATCH --qos=normal
#SBATCH --export=NONE
conda activate climada_env_2022

python /net/home/h01/ldawkins/Documents/UKCR/SchoolApp/FinalCode/Schools_climada_risk_framework.py