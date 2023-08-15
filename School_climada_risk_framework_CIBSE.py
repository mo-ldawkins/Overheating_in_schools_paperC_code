#!/usr/bin/env python
# coding: utf-8

# In[1]:


def define_hazard(file_name, nc1, variable, haz_type):
    """
    Define the hazard data and read it in from netcdf files to create a Heat
    stress  instance of a hazard object

    Inputs
    ------
        file_name: (string) name of netcdf file to be read in
        nc1:


    Returns
    -------
         hazard: (Hazard Object)
    """
    nyears = round(nc1.dimensions['time'].size)/ (195 + 30) # scale to represent 195 days of school per year, but we have zeroed out August (30 days), so each year is 225 days

    # Variables that help defined the Heat stress instance
    hazard_args = {'intensity_var': variable,
                   'event_id': np.arange(1, len(nc1.variables['time'])+1),
                   'frequency': np.full(len(nc1.variables['time']), 1/nyears),
                   'haz_type': haz_type,
                   'description': 'Hazard data',
                   'replace_value': np.nan,
                   'fraction_value': 1.0}

    # read in hazard data from netcdf and use the previously defined variables
    # to help define the data
    hazard1 = Hazard.from_netcdf(file_name, **hazard_args)

    hazard1.check()  #This needs to come before the plots

    # Plots

    #hazard.centroids.plot() #blue centroid plot
    #plt.show(block=True)
    #hazard1.plot_intensity(event=1, smooth=False) #firtst event
    #hazard1.plot_intensity(event=-1) # largest single event
    #hazard1.plot_intensity(event=0)  # max tas at each point
    #hazard1.plot_rp_intensity([5,10]) #return period plots

    # Each blue dot is a centroid
    #hazard1.plot_intensity(centr=3213) #Time series for centroid 3213
    return(hazard1)





# In[2]:


def create_exposures():
    """
    Create the exposure data - and store as a exposure attribute. This contains
    geolocalised values of anything exposed to the hazard - in this case it
    has been set to 1 which is the a working school day (associated with the
    building and not the number of pupils in the building). The impf_HS links
    the building to the impact function to use.


    Inputs
    ------



    Returns
    -------
         data: (pandas dataframe)   with lat, long, value and impf_HS
    """


    # An exposure dataset needs latitude, longitude and a value
    # create some dummy data
    # Rough the SE is from 1.5W to 1.5E

    longitude = np.random.uniform(low=-1.5, high=1.5, size=50)
    latitude = np.random.uniform(low=50, high=53, size=50)

    value = np.ones(50) #(loc=2000, scale = 20, size=50)
    impf_HS = np.random.randint(low=1, high=4, size=50)

    print(max(value), min(value))

    data = pd.DataFrame({'latitude': latitude, 'longitude': longitude, 'value': value,
                              'impf_HS': impf_HS })
    return data


# In[3]:


def create_exposure_dict():

    #Create a dictionary that will be used to match up to the regression equations

    school_type = ['Primary', 'Secondary']
    region_type = ["Z"+str(i).zfill(2) for i in range(1,14)]
    construction_type = ['Pre 1919', 'Interwar', '1945-66', '1967-76', 'Post 1976']

    combinations = list(product(school_type, region_type, construction_type))

    # Create the keys of the dictionary in the same order that the regression equations
    # were derived (Impact functions)

    key_HS = [i_perm[1] + '_' + i_perm[2] + '_' + i_perm[0] for i_perm in combinations]
    value_HS = np.arange(1, len(key_HS)+1)

    HS_dict1 = dict(zip(key_HS, value_HS))

    return HS_dict1



# In[4]:


def read_exposures(moved = True):
    construction_dict = {'Pre 1919': 1, "Interwar": 2,
                         '1945-66': 3, "1967-76": 4,
                         'Post 1976': 5}

    HS_dict = create_exposure_dict()

    all_schools = pd.read_csv("/data/users/ldawkins/UKCR/DataForSchoolApp/schools_lat_lon.csv")

    # drop missing lines
    all_schools = all_schools.dropna(how="all")

    if(moved == True):
        df_schools = all_schools[["school_type", "climateregion_ref", "era_type",
                                  "climateregion_name", "geo_lat_new", "geo_long_new"]]
    else:
        df_schools = all_schools[["school_type", "climateregion_ref", "era_type",
                                  "climateregion_name", "geo_lat", "geo_long"]]

    # Remove unknow construction type  observations
    df_schools = df_schools[df_schools["era_type"] != "Unknown"]

    # Replace string building era with digits to represent different types
    # df_schools = df_schools.replace({"era_type": construction_dict})

    #Assume that region 16 is region 13 in duncan's dataset reset the region 16 to be 13
    df_schools.loc[df_schools.loc[:, 'climateregion_ref']==16, 'climateregion_ref'] = 13

    # Remove schools where we had no files to create the impact functions
    # secondary schools  "Post 1976" for z05, z08, z10, z13
    # secondary schools  "Pre 1919" for Z13
    # secondary schools  "1967-76" for z13

    print(" Before Dropped missing post 1976 schools", df_schools.shape)
    index_04 = df_schools[ (df_schools['school_type'] == 'Secondary') &
                           (df_schools['era_type'] == 'Post 1976') &
                           (df_schools['climateregion_ref'].isin([5, 8, 10, 13])) ].index

    df_schools.drop(index_04, inplace = True)

    print("Dropped missing post 1976 schools", df_schools.shape)
    index_z13 = df_schools[(df_schools['school_type'] == 'Secondary') &
                           (df_schools['era_type'].isin(['Pre 1919', '1967-76'])) &
                           (df_schools['climateregion_ref'] == 13) ].index
    df_schools.drop(index_z13, inplace = True)
    print("Dropped missing Welsh schools", df_schools.shape)

    #####################
    # Subset data to try  and find location of bug problem

    #Drop secondary schools
    #index_secondary = df_schools[df_schools['school_type'] == 'Secondary'].index
    #df_schools.drop(index_secondary, inplace = True)

    print("*** Secondary schools dropped", df_schools.shape)

    # Drop multiple regions
    #index_regions = df_schools[df_schools['climateregion_ref'].isin([1,3,4,5,6,
    #                                                                 7,8,9,
    #                                                                 10,11,12,13])].index
    #df_schools.drop(index_regions, inplace = True)

    print("*** All regions but wales dropped", df_schools.shape)

    ###################
    df_schools=df_schools.dropna(subset=['climateregion_ref','school_type',
                                         'era_type'])
    print(df_schools.shape)



    # Find the maximum number of climate regions and architecture types
    #max_regions = len(np.unique(df_schools["climateregion_ref"]))
    #max_architype = len(np.unique(df_schools["era_type"]))

    # create a unique identifier for each different school_type, climate region and
    # architecture type
    #df_schools["impf_HS"] = ((df_schools["climateregion_ref"] - 1.0)) * 5 + \
    #    pd.to_numeric(df_schools["era_type"]) + \
    #   (df_schools["school_type"] == "Secondary") * max_regions * max_architype


    ref_HS = ['Z' + str(int(x)).zfill(2) + '_' +  str(y) + '_' + z for x, y, z in zip(df_schools["climateregion_ref"], df_schools["era_type"], df_schools["school_type"])]



    df_schools["impf_HS"] = [HS_dict[i] for i in ref_HS]

    # set the value of the asset to be 1. This represents 1 school day lost.
    df_schools["value"] = 1

    # subset the data we want and set appropriate variable names
    if(moved == True):
        data = df_schools[['geo_lat_new','geo_long_new', 'value', 'impf_HS']]
    else:
        data = df_schools[['geo_lat', 'geo_long', 'value', 'impf_HS']]

    data.columns = ['latitude', 'longitude', 'value', 'impf_HS']

    data = data[['latitude', 'longitude', 'value', 'impf_HS']]

    return data


# In[5]:


def exposure_instance(data):
    """
    Create an exposure instance of the Exposures class for Heat stress and
    produce some plots

    Inputs
    ------
       data: (Pandas data frame)  with lat, long, value and impf_HS


    Returns
    -------
         expl_temp: (Object)  Heat stress exposure instance for an
         exposure class
    """
    expl_temp = Exposures(data)

    # set geometry attribute (shapely Points) from GeoDataFrame from
    # latitude and longitude
    expl_temp.set_geometry_points()

    # always apply the check() method in the end. It puts in metadata that has
    # not been assigned, and causes an error if missing mandatory data
    expl_temp.check()


    #Set the value unit
    expl_temp.value_unit = 'Number of schools'

    # print('\n' + '\x1b[1;03;30;30m'  + 'exp looks like:' + '\x1b[0m')
    # print(expl_temp)

    # plot exposures

    ## If two points are near to each other the totals are summed together
    #axs = expl_temp.plot_hexbin()
    return(expl_temp)


# In[6]:


# set the entity exposure object
def set_entity(data):
    """
    Put the exposures into the Entity class

    Inputs
    ------
       data: (Object)  Heat stress exposure instance for an
         exposure class

    Returns
    -------
         ent: (Object)  an Entity class with a Heat stress instance of
                        exposures
    """
    ent = Entity()
    ent.exposures = data
    return ent


# In[7]:


# Define impact functions

def define_impact_fn(file_name, ent1):
    """
    Adds impact functions to Entity class

    Inputs
    ------
       file_name: (Object)  Heat stress exposure instance for an
         exposure class
       ent1: (Object)  an Entity class with a Heat stress instance of
         exposures

    Returns
    -------
         ent1:  (Object) Entity class with Heat stress instance of exposures and
                impact funtcion instances for Heat stress.
         imp_set1: (object) with impact functions
    """


    imp_set1 = ImpactFuncSet.from_excel(file_name)

    imp_set1.check()
    # plot the 'impact funtions'

    #imp_set1.plot()

    # adjust the plots
    #plt.subplots_adjust(right=1, top=1, hspace=0.4, wspace=0.4)

    #plt.legend(loc="upper left")

    #plt.show()

    ent1.impact_funcs = imp_set1

    return ent1, imp_set1





# # Calculate the impact (risk)
#

# In[8]:


def calc_impact(ent1, hazard1):
    """
    Create an impact class

    Inputs
    ------
       ent1: (Object)  an Entity class with a Heat stress instance of
         exposures and impact function instance for Heat stress

    Returns
    -------
         imp1: (Object) an Object that contains the results of impact
         calculations
    """

    imp1 = Impact()

    imp1.calc(ent1.exposures, ent1.impact_funcs, hazard1, save_mat='True')

    return(imp1)



# In[9]:


# Tweak 1_main_climada.py using insights from other tutorials + to be more like our use case
#
#   Laura's code for her fake data
#

# Make sure we are pointing to the right place

import warnings
from netCDF4 import Dataset
import sys
import numpy as np
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
home_dir = '/net/home/h01/ldawkins/' # change to location of the Sup Material folder

sys.path.append(home_dir + 'climada_netcdf/climada_python/')
warnings.filterwarnings('ignore')

from itertools import product

#import climada functions
from climada.hazard import Hazard
from climada.entity import Exposures
from climada.entity import ImpactFunc, ImpactFuncSet
from climada.entity import Entity
from climada.engine import Impact

from climada.entity import ImpactFuncSet

import sys
#data_dir = home_dir + 'Data/'
#data_source = 'Mean_temp_bias_corrected'

data_dir = '/data/users/ldawkins/UKCR/DataForSchoolApp/'
data_source = 'UKCP_BC/'

# In[10]:


def read_hazard(warming_level="current", ens_mem='01'):

    if warming_level == 'current':
        netcdf_file_path = glob.glob(
                data_dir + data_source + '/*' + ens_mem + '_*tas*1998*')
    else:
        netcdf_file_path = glob.glob(
            data_dir + data_source + '/*'+ ens_mem + '_*tas_*'+warming_level+'*')

    print(netcdf_file_path)

    # load in the hazard data (mean_temperature)

    netcdf_file = Dataset(netcdf_file_path[0])
    return netcdf_file_path[0], netcdf_file


# In[11]:


def check_years_ge_15(nc_data):

    nyears = round(nc_data.dimensions['time'].size/360)
    return (nyears >= 15)



def main():

    threshold = sys.argv[1]
    warming_level = sys.argv[2]
    member = sys.argv[3]

    #thresholds = ['35']#['26', '35']
    #warming_levels = ['2deg', '4deg']#['current', '2deg', '4deg']

    #for threshold in thresholds:
    print(threshold)
    if threshold == '26':
        lim = 120
    else:
        lim = 30
    print(lim)
    #for warming_level in warming_levels:
    print(warming_level)
    print(member)
    hazard = Hazard('HS')
    hazard.read_excel('/data/users/ldawkins/UKCR/DataForSchoolApp/CIBSE/HazardCIBSE_rep_' + warming_level + '_member' + member + '.xlsx')
    hazard.tag.haz_type = 'HS'
    expl_temp = read_exposures(moved=False)
    expl_inst = exposure_instance(expl_temp)
    ent = set_entity(expl_inst)
    filename = "/home/h01/ldawkins/Documents/UKCR/SchoolApp/FinalCode/Operative"+threshold+"_temperature_relationships.xlsx"/entity_sch_281122.xlsx"
    ent, imp_set = define_impact_fn(filename, ent)
    imp = calc_impact(ent, hazard)

    imp.plot_basemap_eai_exposure(s=6, pop_name=False, vmin=0, vmax=lim)
    plt.savefig(
        '/data/users/ldawkins/UKCR/SchoolApp/SchClose_' + warming_level + '_CIBSE_rep_' + threshold + '_member' + member + '.png',
        dpi=500)
    plt.show(block=True)
    imp.write_csv("/data/users/ldawkins/UKCR/DataForSchoolApp/Impact/" + \
                                       warming_level + "/CIBSE_output_thresh"+threshold+"_member" + member + "_opttemp.csv")



if __name__ == "__main__":
    main()
