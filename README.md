# Code for analysis in paper C: Quantifying overheating risk in UK schools: A spatially coherent climate risk assessment

Code is saved in /home/h01/ldawkins/Documents/UKCR/SchoolApp/FinalCode/ and on github at 

See paper /home/h01/ldawkins/Documents/UKCR/SchoolApp/Paper/paperC.pdf

1. school_temp_analysis_to_excel.ipynb (code written by Kate Brown in SIRA) - this code fits a linear model to the relationship between outdoor and indoor temperature to estimate the vulnerability step function for each school archetype. This code can output validation plots, the linear model parameters and the vulnerability function excel spreadsheet that is read into the CLIMADA risk assessment. The indoor temperature metric (mean/max and air/operative), and indoor threshold must be specified. For the paper, the output information is produced for indoor daily maximum operative temperature for indoor thresholds of 26C and 35C, saved in the home directory /home/h01/ldawkins/Documents/UKCR/SchoolApp/FinalCode/.

To run this using conda environment ~/.conda/envs/climada_env_2022 (need to set up locally)
module load climada_env_2022
jupyter notebook school_temp_analysis_to_excel.ipynb

2. Schools_climada_risk_framework.py (code originally written by Kate Brown and modifed by Laura Dawkins) - this code applies the CLIMADA risk assessment to assess overheating risk in ~20,000 schools in England and Wales. The hazard, exposure and vulnerability information used is as per the paper. Outputs the Expected Annual Impact (EAI) metric for each warming level and ensemble member, saved in data/users/ldawkins/UKCR/DataForSchoolApp/Impact/. Currently require you to manually alter the overheating threshold to be considered. This script also contains the code to apply the CLIMADA calculation to the 'observed' HadUK-Grid temperature data (shown in Fig 7 d and h).

I either run in Pycharm (might crash due to memory), or on SPICE using the associated .sh file (need to change info in this .sh file if running). This also uses the conda environment ~/.conda/envs/climada_env_2022

3. CIBSE_hazard_excel_representation.R - R code to create the CIBSE style 'region location' hazard data - used to estimate comparison risk as shown in Figure 1 of the paper. Outputs this data to /data/users/ldawkins/UKCR/DataForSchoolApp/CIBSE

4. School_climada_risk_framework_CIBSE.py (and associated _batch and _calling scripts) - as in 2., but applied to the CIBSE style 'region location' hazard data.  This loops over warming level (current, 2C, 4C), overheating threshold (26C and 35C) and UKCP18 ensemble member to speed things up (code in 2. could be adapted to do this too).

5. FitGAM.R (and 6x FitGAM.sh which run the code on SPICE for each of the global warming level (current, 2C, 4C) and overheating threshold (26C and 35C) combinations. Inputs risk maps from the 12 UKCP18 ensemble members calculated in Schools_climada_risk_framework.py, (i.e. applied to risk from spatially coherent hazard data, not region location version) and outputs the fitted GAM information as a .RData file. If run locally you might need to install some of the R libraries to your local R library.

5. plots_for_paperR.R (and associated .sh file) - makes plots for the paper that are based on R code. This also includes code to simulate from the GAM. Outputs plots and GAM samples saved in /data/users/ldawkins/UKCR/DataForSchoolApp/Impact/*/GAM_SAMPLES_* 

6. plotsforschoolpaper.py - makes plots for the paper that are based on python code.
