#!/bin/bash -l
#SBATCH --qos=long
#SBATCH --mem=64G
#SBATCH --ntasks=8
#SBATCH --time=00-72:00:00
#SBATCH --export=NONE
#SBATCH --mail-user=laura.dawkins@metoffice.gov.uk
#SBATCH --mail-type=ALL

R CMD BATCH --silent --no-restore /home/h01/ldawkins/Documents/UKCR/SchoolApp/FinalCode/CIBSE_hazard_excel_representation.R /home/h01/ldawkins/Documents/UKCR/SchoolApp/FinalCode/log_CIBSE_hazard_excel_representation.out
