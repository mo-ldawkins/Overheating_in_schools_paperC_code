#!/bin/bash -l
#SBATCH --qos=long
#SBATCH --mem=96G
#SBATCH --ntasks=8
#SBATCH --time=00-72:00:00
#SBATCH --export=NONE
#SBATCH --mail-user=laura.dawkins@metoffice.gov.uk
#SBATCH --mail-type=ALL
module unload R
module load R/3.6.1

# Rscript --silent --no-restore /home/h01/ldawkins/Documents/UKCR/SchoolApp/FinalCode/FitGAM.R 1 35
# Rscript --silent --no-restore /home/h01/ldawkins/Documents/UKCR/SchoolApp/FinalCode/FitGAM.R 2 35
# Rscript --silent --no-restore /home/h01/ldawkins/Documents/UKCR/SchoolApp/FinalCode/FitGAM.R 3 35
# Rscript --silent --no-restore /home/h01/ldawkins/Documents/UKCR/SchoolApp/FinalCode/FitGAM.R 1 26
Rscript --silent --no-restore /home/h01/ldawkins/Documents/UKCR/SchoolApp/FinalCode/FitGAM.R 2 26
# Rscript --silent --no-restore /home/h01/ldawkins/Documents/UKCR/SchoolApp/FinalCode/FitGAM.R 3 26

