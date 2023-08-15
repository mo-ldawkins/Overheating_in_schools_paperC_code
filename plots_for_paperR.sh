#!/bin/bash -l
#SBATCH --qos=normal
#SBATCH --mem=64G
#SBATCH --ntasks=8
#SBATCH --time=00-06:00:00
#SBATCH --export=NONE
#SBATCH --mail-user=laura.dawkins@metoffice.gov.uk
#SBATCH --mail-type=ALL

R CMD BATCH --silent --no-restore /home/h01/ldawkins/Documents/UKCR/SchoolApp/FinalCode/plots_for_paperR.R /home/h01/ldawkins/Documents/UKCR/SchoolApp/FinalCode/log_paperplotsR.out

