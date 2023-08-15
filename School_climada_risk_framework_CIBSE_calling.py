import os

def run_all():
    """ Launches batch jobs to preprocess raw rcm data

        """
    thresholds = ['26', '35']
    warming_levels = ['current', '2deg', '4deg']

    for i in range(0, len(warming_levels)):
        warming_level = warming_levels[i]

        if warming_level == '4deg':
            members = ['01', '04', '05', '06', '07', '09', '11', '12', '13']
        else:
            members = ['01', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '15']

        for member in members:
            for threshold in thresholds:
                #print(warming_level)
                #print(member)
                #print(threshold)
                cmd = 'sbatch School_climada_risk_framework_CIBSE_batch.sh ' + str(threshold) + ' ' + str(
                    warming_level) + ' ' + str(member)
                print(cmd)
                os.system(cmd)
    return

