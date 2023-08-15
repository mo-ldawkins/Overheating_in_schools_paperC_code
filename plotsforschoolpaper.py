"""
Code to generate python plots for the paper C
/home/h01/ldawkins/Documents/UKCR/SchoolApp/Paper/paperC.pdf
"""
plot_out_dir = '/data/users/ldawkins/UKCR/OutputPlots/Plotsforpaper/SchoolPaper/'

import matplotlib.pyplot as plt
import warnings
from netCDF4 import Dataset
from scipy import sparse
import sys
import numpy as np
import glob
from matplotlib import colors
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

#from Schools_climada_risk_framework import read_exposures, read_hazard, define_hazard, exposure_instance, set_entity, define_impact_fn, calc_impact
import Schools_climada_risk_framework as schfns

# Climada part

sys.path.append('/net/home/h01/ldawkins/climada_netcdf/climada_python/')
warnings.filterwarnings('ignore')

#from pandas import DataFrame
from climada.hazard import Hazard
from climada.entity import Exposures
from climada.entity import ImpactFunc, ImpactFuncSet
from climada.entity import Entity
from climada.engine import Impact

############################################################################################
# Figure 1
############################################################################################

# current ens mean
#threshold = 26
warming_level='current'
ens_mems = ['01', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '15']
ens_mem = ens_mems[0]
threshold = '26'
ds = pd.read_csv('/data/users/ldawkins/UKCR/DataForSchoolApp/Impact/'+warming_level+'/CIBSE_output_thresh'+threshold+'_member' + ens_mem + '_opttemp.csv', error_bad_lines=False)
df = ds[['exp_lon','exp_lat','eai_exp']]
df['member'] = ens_mem
df['schoolno'] = df.index + 1
for ens_mem in ens_mems[1:len(ens_mems)]:
    ds = pd.read_csv(
        '/data/users/ldawkins/UKCR/DataForSchoolApp/Impact/' + warming_level + '/CIBSE_output_thresh'+threshold+'_member' + ens_mem + '_opttemp.csv',
        error_bad_lines=False)
    df2 = ds[['exp_lon', 'exp_lat', 'eai_exp']]
    df2['member'] = ens_mem
    df2['schoolno'] = df2.index + 1
    df = pd.concat([df, df2])
# find ensemble mean EAI
ens_mean_current_26_cibse = df.groupby(['schoolno']).mean()
ens_mean_current_26_cibse = ens_mean_current_26_cibse.reset_index(level=0)
ens_mean_current_26_cibse = ens_mean_current_26_cibse.reset_index(level=0)
spaceagg_current_26_cibse = df[['eai_exp','member']].groupby(['member']).sum().mean()

# threshold = 35
ens_mem = ens_mems[0]
threshold = '35'
ds = pd.read_csv('/data/users/ldawkins/UKCR/DataForSchoolApp/Impact/'+warming_level+'/CIBSE_output_thresh'+threshold+'_member' + ens_mem + '_opttemp.csv', error_bad_lines=False)
df = ds[['exp_lon','exp_lat','eai_exp']]
df['member'] = ens_mem
df['schoolno'] = df.index + 1
for ens_mem in ens_mems[1:len(ens_mems)]:
    ds = pd.read_csv(
        '/data/users/ldawkins/UKCR/DataForSchoolApp/Impact/' + warming_level + '/CIBSE_output_thresh'+threshold+'_member' + ens_mem + '_opttemp.csv',
        error_bad_lines=False)
    df2 = ds[['exp_lon', 'exp_lat', 'eai_exp']]
    df2['member'] = ens_mem
    df2['schoolno'] = df2.index + 1
    df = pd.concat([df, df2])
# find ensemble mean EAI
ens_mean_current_35_cibse = df.groupby(['schoolno']).mean()
ens_mean_current_35_cibse = ens_mean_current_35_cibse.reset_index(level=0)
ens_mean_current_35_cibse = ens_mean_current_35_cibse.reset_index(level=0)
spaceagg_current_35_cibse = df[['eai_exp','member']].groupby(['member']).sum().mean()

# 2 deg ens mean
#threshold = 26
warming_level='2deg'
ens_mems = ['01', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '15']
ens_mem = ens_mems[0]
threshold = '26'
ds = pd.read_csv('/data/users/ldawkins/UKCR/DataForSchoolApp/Impact/'+warming_level+'/CIBSE_output_thresh'+threshold+'_member' + ens_mem + '_opttemp.csv', error_bad_lines=False)
df = ds[['exp_lon','exp_lat','eai_exp']]
df['member'] = ens_mem
df['schoolno'] = df.index + 1
for ens_mem in ens_mems[1:len(ens_mems)]:
    ds = pd.read_csv(
        '/data/users/ldawkins/UKCR/DataForSchoolApp/Impact/' + warming_level + '/CIBSE_output_thresh'+threshold+'_member' + ens_mem + '_opttemp.csv',
        error_bad_lines=False)
    df2 = ds[['exp_lon', 'exp_lat', 'eai_exp']]
    df2['member'] = ens_mem
    df2['schoolno'] = df2.index + 1
    df = pd.concat([df, df2])
# find ensemble mean EAI
ens_mean_2deg_26_cibse = df.groupby(['schoolno']).mean()
ens_mean_2deg_26_cibse = ens_mean_2deg_26_cibse.reset_index(level=0)
ens_mean_2deg_26_cibse = ens_mean_2deg_26_cibse.reset_index(level=0)
spaceagg_2deg_26_cibse = df[['eai_exp','member']].groupby(['member']).sum().mean()

# threshold = 35
ens_mem = ens_mems[0]
threshold = '35'
ds = pd.read_csv('/data/users/ldawkins/UKCR/DataForSchoolApp/Impact/'+warming_level+'/CIBSE_output_thresh'+threshold+'_member' + ens_mem + '_opttemp.csv', error_bad_lines=False)
df = ds[['exp_lon','exp_lat','eai_exp']]
df['member'] = ens_mem
df['schoolno'] = df.index + 1
for ens_mem in ens_mems[1:len(ens_mems)]:
    ds = pd.read_csv(
        '/data/users/ldawkins/UKCR/DataForSchoolApp/Impact/' + warming_level + '/CIBSE_output_thresh'+threshold+'_member' + ens_mem + '_opttemp.csv',
        error_bad_lines=False)
    df2 = ds[['exp_lon', 'exp_lat', 'eai_exp']]
    df2['member'] = ens_mem
    df2['schoolno'] = df2.index + 1
    df = pd.concat([df, df2])
# find ensemble mean EAI
ens_mean_2deg_35_cibse = df.groupby(['schoolno']).mean()
ens_mean_2deg_35_cibse = ens_mean_2deg_35_cibse.reset_index(level=0)
ens_mean_2deg_35_cibse = ens_mean_2deg_35_cibse.reset_index(level=0)
spaceagg_2deg_35_cibse = df[['eai_exp','member']].groupby(['member']).sum().mean()


# 4 deg ens mean
#threshold = 26
warming_level='4deg'
ens_mems = ['01', '04', '05', '06', '07', '09', '11', '12', '13']
ens_mem = ens_mems[0]
threshold = '26'
ds = pd.read_csv('/data/users/ldawkins/UKCR/DataForSchoolApp/Impact/'+warming_level+'/CIBSE_output_thresh'+threshold+'_member' + ens_mem + '_opttemp.csv', error_bad_lines=False)
df = ds[['exp_lon','exp_lat','eai_exp']]
df['member'] = ens_mem
df['schoolno'] = df.index + 1
for ens_mem in ens_mems[1:len(ens_mems)]:
    ds = pd.read_csv(
        '/data/users/ldawkins/UKCR/DataForSchoolApp/Impact/' + warming_level + '/CIBSE_output_thresh'+threshold+'_member' + ens_mem + '_opttemp.csv',
        error_bad_lines=False)
    df2 = ds[['exp_lon', 'exp_lat', 'eai_exp']]
    df2['member'] = ens_mem
    df2['schoolno'] = df2.index + 1
    df = pd.concat([df, df2])
# find ensemble mean EAI
ens_mean_4deg_26_cibse = df.groupby(['schoolno']).mean()
ens_mean_4deg_26_cibse = ens_mean_4deg_26_cibse.reset_index(level=0)
ens_mean_4deg_26_cibse = ens_mean_4deg_26_cibse.reset_index(level=0)
spaceagg_4deg_26_cibse = df[['eai_exp','member']].groupby(['member']).sum().mean()

# threshold = 35
ens_mem = ens_mems[0]
threshold = '35'
ds = pd.read_csv('/data/users/ldawkins/UKCR/DataForSchoolApp/Impact/'+warming_level+'/CIBSE_output_thresh'+threshold+'_member' + ens_mem + '_opttemp.csv', error_bad_lines=False)
df = ds[['exp_lon','exp_lat','eai_exp']]
df['member'] = ens_mem
df['schoolno'] = df.index + 1
for ens_mem in ens_mems[1:len(ens_mems)]:
    ds = pd.read_csv(
        '/data/users/ldawkins/UKCR/DataForSchoolApp/Impact/' + warming_level + '/CIBSE_output_thresh'+threshold+'_member' + ens_mem + '_opttemp.csv',
        error_bad_lines=False)
    df2 = ds[['exp_lon', 'exp_lat', 'eai_exp']]
    df2['member'] = ens_mem
    df2['schoolno'] = df2.index + 1
    df = pd.concat([df, df2])
# find ensemble mean EAI
ens_mean_4deg_35_cibse = df.groupby(['schoolno']).mean()
ens_mean_4deg_35_cibse = ens_mean_4deg_35_cibse.reset_index(level=0)
ens_mean_4deg_35_cibse = ens_mean_4deg_35_cibse.reset_index(level=0)
spaceagg_4deg_35_cibse = df[['eai_exp','member']].groupby(['member']).sum().mean()


fig = plt.figure(figsize=(12,18))
gs = GridSpec(nrows=3, ncols=2)
ax1 = fig.add_subplot(gs[0, 0],projection=ccrs.PlateCarree(),aspect='auto')
ax2 = fig.add_subplot(gs[1, 0],projection=ccrs.PlateCarree(),aspect='auto')
ax3 = fig.add_subplot(gs[2, 0],projection=ccrs.PlateCarree(),aspect='auto')
ax4 = fig.add_subplot(gs[0, 1],projection=ccrs.PlateCarree(),aspect='auto')
ax5 = fig.add_subplot(gs[1, 1],projection=ccrs.PlateCarree(),aspect='auto')
ax6 = fig.add_subplot(gs[2, 1],projection=ccrs.PlateCarree(),aspect='auto')

ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
cp = ax1.scatter(ens_mean_current_26_cibse['exp_lon'],ens_mean_current_26_cibse['exp_lat'],c=ens_mean_current_26_cibse['eai_exp'],s=2,cmap='gist_heat_r',vmin=0,vmax=120)
cbar = plt.colorbar(cp,ax=ax1,shrink=1)
cbar.set_label('Value (No. of days school overheats)')
ax1.title.set_text('(a) Overheating threshold = 26C')
ax1.coastlines()
ax1.set_xlim(-6.1,2)
ax1.set_ylim(49.5,56.1)

ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')
cp = ax2.scatter(ens_mean_2deg_26_cibse['exp_lon'],ens_mean_2deg_26_cibse['exp_lat'],c=ens_mean_2deg_26_cibse['eai_exp'],s=2,cmap='gist_heat_r',vmin=0,vmax=120)
cbar = plt.colorbar(cp,ax=ax2,shrink=1)
cbar.set_label('Value (No. of days school overheats)')
ax2.title.set_text('(b)')
ax2.coastlines()
ax2.set_xlim(-6.1,2)
ax2.set_ylim(49.5,56.1)

ax3.set_xlabel('Longitude')
ax3.set_ylabel('Latitude')
cp = ax3.scatter(ens_mean_4deg_26_cibse['exp_lon'],ens_mean_4deg_26_cibse['exp_lat'],c=ens_mean_4deg_26_cibse['eai_exp'],s=2,cmap='gist_heat_r',vmin=0,vmax=120)
cbar = plt.colorbar(cp,ax=ax3,shrink=1)
cbar.set_label('Value (No. of days school overheats)')
ax3.title.set_text('(c)')
ax3.coastlines()
ax3.set_xlim(-6.1,2)
ax3.set_ylim(49.5,56.1)


ax4.set_xlabel('Longitude')
ax4.set_ylabel('Latitude')
cp = ax4.scatter(ens_mean_current_35_cibse['exp_lon'],ens_mean_current_35_cibse['exp_lat'],c=ens_mean_current_35_cibse['eai_exp'],s=2,cmap='gist_heat_r',vmin=0,vmax=25)
cbar = plt.colorbar(cp,ax=ax4,shrink=1)
cbar.set_label('Value (No. of days school overheats)')
ax4.title.set_text('(d) Overheating threshold = 35C')
ax4.coastlines()
ax4.set_xlim(-6.1,2)
ax4.set_ylim(49.5,56.1)

ax5.set_xlabel('Longitude')
ax5.set_ylabel('Latitude')
cp = ax5.scatter(ens_mean_2deg_35_cibse['exp_lon'],ens_mean_2deg_35_cibse['exp_lat'],c=ens_mean_2deg_35_cibse['eai_exp'],s=2,cmap='gist_heat_r',vmin=0,vmax=25)
cbar = plt.colorbar(cp,ax=ax5,shrink=1)
cbar.set_label('Value (No. of days school overheats)')
ax5.title.set_text('(e)')
ax5.coastlines()
ax5.set_xlim(-6.1,2)
ax5.set_ylim(49.5,56.1)

ax6.set_xlabel('Longitude')
ax6.set_ylabel('Latitude')
cp = ax6.scatter(ens_mean_4deg_35_cibse['exp_lon'],ens_mean_4deg_35_cibse['exp_lat'],c=ens_mean_4deg_35_cibse['eai_exp'],s=2,cmap='gist_heat_r',vmin=0,vmax=25)
cbar = plt.colorbar(cp,ax=ax6,shrink=1)
cbar.set_label('Value (No. of days school overheats)')
ax6.title.set_text('(f)')
ax6.coastlines()
ax6.set_xlim(-6.1,2)
ax6.set_ylim(49.5,56.1)

lons = [-0.6069083,  0.3271254, -1.8211129, -4.2678039, -2.5075592, -1.8082776, -2.5329118, -2.9209004, -1.9944422,
        -1.3484200, -0.9091481,  0.5724432, -2.7919652]
lats = [51.58657, 51.14009, 50.89316, 50.64821, 51.48465, 52.51090, 53.48230, 54.50481, 55.26194, 54.34327, 53.37068,
        52.32775, 52.72512]

ax6.scatter(lons,lats,marker='x',s=20,c='black')


plt.tight_layout()
plt.savefig(
      plot_out_dir + 'Figure1.png', dpi=500)
plt.show(block=True)

############################################################################################
# Figure 5
############################################################################################

ens_mem = '01'
warming_level='current'

data_dir = '/data/users/ldawkins/UKCR/DataForSchoolApp/'
data_source = 'UKCP_BC/'
# Check that I can set these variables to what I like?
variable = "tas"  # check that this is something that I set
haz_type = "HS"
filename = "/net/home/h01/ldawkins/Documents/UKCR/SchoolApp/FinalCode/Operative35_temperature_relationships_max.xlsx"
expl_temp = schfns.read_exposures(moved=True)
nc_path, nc = schfns.read_hazard(warming_level=warming_level, ens_mem=ens_mem)
hazard = schfns.define_hazard(nc_path,nc,variable,haz_type)
expl_temp = schfns.read_exposures(moved=True)
expl_inst = schfns.exposure_instance(expl_temp)
ent = schfns.set_entity(expl_inst)
ent, imp_set = schfns.define_impact_fn(filename, ent)
imp = schfns.calc_impact(ent, hazard)
# move back to actual lonlats
expl_temp = schfns.read_exposures(moved=False)
expl_temp = Exposures(expl_temp)
expl_temp.set_geometry_points()
lon_true = expl_temp.gdf.longitude.to_numpy()
lat_true = expl_temp.gdf.latitude.to_numpy()
lonlat_true = np.transpose((lat_true, lon_true))
#hazard.change_centroids(lonlat_true)
imp.coord_exp = lonlat_true
expl_inst.gdf.longitude = lon_true
expl_inst.gdf.latitude = lat_true
imp.unit = 'Number of schools overheated'

# info for 26 deg thresh
filename2 = "/net/home/h01/ldawkins/Documents/UKCR/SchoolApp/FinalCode/Operative26_temperature_relationships_max.xlsx"
ent2 = schfns.set_entity(expl_inst)
ent2, imp_set2 = schfns.define_impact_fn(filename2, ent2)
imp2 = schfns.calc_impact(ent2, hazard)
imp2.coord_exp = lonlat_true
imp2.unit = 'Number of schools overheated'

fig = plt.figure(figsize=(12,12))
gs = GridSpec(nrows=3, ncols=3)
ax1 = fig.add_subplot(gs[0, 0],projection=ccrs.PlateCarree(),aspect='auto')
ax2 = fig.add_subplot(gs[0, 1],projection=ccrs.PlateCarree(),aspect='auto')
ax3 = fig.add_subplot(gs[0, 2],projection=ccrs.PlateCarree(),aspect='auto')
ax4 = fig.add_subplot(gs[1, 0],aspect='auto')
ax5 = fig.add_subplot(gs[1, 1],projection=ccrs.PlateCarree(),aspect='auto')
ax6 = fig.add_subplot(gs[1, 2],aspect='auto')
ax7 = fig.add_subplot(gs[2, 0],aspect='auto')
ax8 = fig.add_subplot(gs[2, 1],projection=ccrs.PlateCarree(),aspect='auto')
ax9 = fig.add_subplot(gs[2, 2],aspect='auto')

# hazard map in extreme event
hazard.plot_intensity(axis=ax1, event=1644, vmin=0, vmax=30)
ax1.set_xlim(-6.1,2)
ax1.set_ylim(49.5,56.1)
ax1.title.set_text('(a)')

# All schools
expl_inst.value_unit = 'Number of schools'
expl_inst.plot_scatter(pop_name=False, axis=ax2, s=0.1,cmap='Reds',vmin=0,vmax=1)#, norm=norm,s=11)
plt.legend('', frameon=False)
ax2.set_xlim(-6.1,2)
ax2.set_ylim(49.5,56.1)
ax2.title.set_text('(b)')

# Archetypes in Thames Valley
ax3.set_xlabel('Longitude')
ax3.set_ylabel('Latitude')
NUM_COLORS = 10
cm = plt.get_cmap('gist_rainbow')
ax3.set_prop_cycle(color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])

labels = ['Primary,\npre-1919','Secondary,\npre-1919','Primary,\n1919-44','Secondary,\n1919-44','Primary,\n1945-66',
          'Secondary,\n1945-66','Primary,\n1967-76','Secondary,\n1967-76','Primary,\npost-1976','Secondary,\npost-1976']
archs = [1,66,2,67,3,68,4,69,5,70]
mtype = ['.','^','.','^','.','^','.','^','.','^']
for i in range(0,10):
    ind = np.where(expl_inst.gdf['impf_HS'] == archs[i])[0]
    cp = ax3.scatter(np.array(expl_inst.gdf['longitude'])[ind],np.array(expl_inst.gdf['latitude'])[ind],s=2,label=labels[i],marker=mtype[i])#,cmap='magma_r')
#ax3.legend(fontsize='x-small')
ax3.title.set_text('(c)')
ax3.coastlines()
ax3.set_xlim(-1.8,1.8)
ax3.set_ylim(50.5,52.5)
ax3.legend(loc= 'lower right', fontsize='x-small',facecolor='white', framealpha=1, markerscale=3)

# vuln functions for Thames valley
ax4.set_xlabel('Outdoor Daily \n Mean Temperature (degrees C)')
ax4.set_ylabel('Impact')
ax4.set_prop_cycle(color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
linetypes=['solid','dashed','solid','dashed','solid','dashed','solid','dashed','solid','dashed']
for i in range(0,10):
    func = imp_set.get_func('HS',archs[i])
    x = func.intensity
    y = func.mdd
    ax4.plot(x,y,label=labels[i],linestyle=linetypes[i])
ax4.title.set_text('(d)')
ax4.legend(loc= 'lower right', fontsize='x-small')
ax4.set_xlim(21,24)

impact_at_events_exp = imp._build_exp_event(1644)
impact_at_events_exp.plot_scatter(axis=ax5,pop_name=False,s=0.5,cmap='Greys',vmin=0,vmax=1)#, norm=norm,s=16)
ax5.title.set_text('(e)')
ax5.set_xlim(-6.1,2)
ax5.set_ylim(49.5,56.1)

freq_curve = imp.calc_freq_curve() # impact exceedence frequency curve
freq_curve.plot(axis=ax6)
ax6.title.set_text('(f)')
ax6.set_ylim(0,20000)

#same for 26 deg threshold

# vuln functions for Thames valley
ax7.set_xlabel('Outdoor Daily \n Mean Temperature (degrees C)')
ax7.set_ylabel('Impact')
ax7.set_prop_cycle(color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
linetypes=['solid','dashed','solid','dashed','solid','dashed','solid','dashed','solid','dashed']
for i in range(0,10):
    func = imp_set2.get_func('HS',archs[i])
    x = func.intensity
    y = func.mdd
    ax7.plot(x,y,label=labels[i],linestyle=linetypes[i])
ax7.title.set_text('(g)')
ax7.legend(loc= 'lower right', fontsize='x-small')
ax7.set_xlim(12,16)

impact_at_events_exp2 = imp2._build_exp_event(1644)
impact_at_events_exp2.plot_scatter(axis=ax8,pop_name=False,s=0.5,cmap='Greys',vmin=0,vmax=1)#, norm=norm,s=16)
ax8.title.set_text('(h)')
ax8.set_xlim(-6.1,2)
ax8.set_ylim(49.5,56.1)

freq_curve = imp2.calc_freq_curve() # impact exceedence frequency curve
freq_curve.plot(axis=ax9)
ax9.title.set_text('(i)')
ax9.set_ylim(0,20000)

plt.tight_layout()
plt.savefig(
      plot_out_dir + 'Figure3.png', dpi=500)
plt.show(block=True)



############################################################################################
# Figure 6
############################################################################################


#Threshold = 26oC
# calc ensemble mean based on all members for plot (b)
warming_level='current'
ens_mems = ['01', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '15']
ens_mem = ens_mems[0]
threshold = '26'
ds = pd.read_csv('/data/users/ldawkins/UKCR/DataForSchoolApp/Impact/'+warming_level+'/ens_'+ens_mem+'_output__thresh'+threshold+'_opttemp.csv', error_bad_lines=False)
df = ds[['exp_lon','exp_lat','eai_exp']]
df['member'] = ens_mem
df['schoolno'] = df.index + 1
for ens_mem in ens_mems[1:len(ens_mems)]:
    ds = pd.read_csv(
        '/data/users/ldawkins/UKCR/DataForSchoolApp/Impact/' + warming_level + '/ens_' + ens_mem + '_output__thresh' + threshold + '_opttemp.csv',
        error_bad_lines=False)
    df2 = ds[['exp_lon', 'exp_lat', 'eai_exp']]
    df2['member'] = ens_mem
    df2['schoolno'] = df2.index + 1
    df = pd.concat([df, df2])
# find ensemble mean EAI
ens_mean = df.groupby(['schoolno']).mean()
ens_mean = ens_mean.reset_index(level=0)
ens_mean = ens_mean.reset_index(level=0)

# Find spatially aggregated EAI for each ensemble member
spaceagg = df[['eai_exp','member']].groupby(['member']).sum()

# Load in for observed
ds = pd.read_csv('/data/users/ldawkins/UKCR/DataForSchoolApp/Obs/Obs_output_'+threshold+'.csv', error_bad_lines=False)
obs_eai = ds[['exp_lon','exp_lat','eai_exp']]
obs_spaceagg = obs_eai[['eai_exp']].sum()
# calculate bias
bias_eai = pd.merge(obs_eai, ens_mean, how = 'left', left_on = ['exp_lon','exp_lat'],
                    right_on = ['exp_lon','exp_lat'])
bias_eai['bias'] = (bias_eai['eai_exp_y'] - bias_eai['eai_exp_x'])#/bias_eai['eai_exp_x']


#Threshold = 35oC
# calc ensemble mean based on all members for plot (b)
warming_level='current'
ens_mems = ['01', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '15']
ens_mem = ens_mems[0]
threshold = '35'
ds = pd.read_csv('/data/users/ldawkins/UKCR/DataForSchoolApp/Impact/'+warming_level+'/ens_'+ens_mem+'_output__thresh'+threshold+'_opttemp.csv', error_bad_lines=False)
df = ds[['exp_lon','exp_lat','eai_exp']]
df['member'] = ens_mem
df['schoolno'] = df.index + 1
for ens_mem in ens_mems[1:len(ens_mems)]:
    ds = pd.read_csv(
        '/data/users/ldawkins/UKCR/DataForSchoolApp/Impact/' + warming_level + '/ens_' + ens_mem + '_output__thresh' + threshold + '_opttemp.csv',
        error_bad_lines=False)
    df2 = ds[['exp_lon', 'exp_lat', 'eai_exp']]
    df2['member'] = ens_mem
    df2['schoolno'] = df2.index + 1
    df = pd.concat([df, df2])
# find ensemble mean EAI
ens_mean1 = df.groupby(['schoolno']).mean()
ens_mean1 = ens_mean1.reset_index(level=0)
ens_mean1 = ens_mean1.reset_index(level=0)

# Find spatially aggregated EAI for each ensemble member
spaceagg1 = df[['eai_exp','member']].groupby(['member']).sum()

# Load in for observed
ds = pd.read_csv('/data/users/ldawkins/UKCR/DataForSchoolApp/Obs/Obs_output_'+threshold+'.csv', error_bad_lines=False)
obs_eai1 = ds[['exp_lon','exp_lat','eai_exp']]
obs_spaceagg1 = obs_eai1[['eai_exp']].sum()
# calculate bias
bias_eai1 = pd.merge(obs_eai1, ens_mean1, how = 'left', left_on = ['exp_lon','exp_lat'],
                    right_on = ['exp_lon','exp_lat'])
bias_eai1['bias'] = (bias_eai1['eai_exp_y'] - bias_eai1['eai_exp_x'])#/bias_eai['eai_exp_x']


plt.figure(figsize=(13,18))
ax1 = plt.subplot(4,2,1,projection=ccrs.PlateCarree())
ax2 = plt.subplot(4,2,3,projection=ccrs.PlateCarree())
ax3 = plt.subplot(4,2,5)
ax4 = plt.subplot(4,2,7,projection=ccrs.PlateCarree())

ax5 = plt.subplot(4,2,2,projection=ccrs.PlateCarree())
ax6 = plt.subplot(4,2,4,projection=ccrs.PlateCarree())
ax7 = plt.subplot(4,2,6)
ax8 = plt.subplot(4,2,8,projection=ccrs.PlateCarree())

ds = pd.read_csv('/data/users/ldawkins/UKCR/DataForSchoolApp/Impact/'+warming_level+'/ens_01_output__thresh26_opttemp.csv', error_bad_lines=False)
df = ds[['exp_lon','exp_lat','eai_exp']]

ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
cp = ax1.scatter(df['exp_lon'],df['exp_lat'],c=df['eai_exp'],s=2,cmap='gist_heat_r',vmin=0,vmax=100)
cbar = plt.colorbar(cp,ax=ax1,shrink=1)
cbar.set_label('Value (No. of days school overheats)')
ax1.title.set_text('(a) Overheating threshold = 26C')
ax1.coastlines()
ax1.set_xlim(-6.1,2)
ax1.set_ylim(49.5,56.1)

#ens_mean_order = pd.merge(df, ens_mean, how = 'left', left_on = ['exp_lon','exp_lat'],
#                    right_on = ['exp_lon','exp_lat'])
ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')
#cp = ax2.scatter(ens_mean_order['exp_lon'],ens_mean_order['exp_lat'],c=ens_mean_order['eai_exp_y'],s=2,cmap='gist_heat_r',vmin=0,vmax=100)
cp = ax2.scatter(ens_mean['exp_lon'],ens_mean['exp_lat'],c=ens_mean['eai_exp'],s=2,cmap='gist_heat_r',vmin=0,vmax=100)
cbar = plt.colorbar(cp,ax=ax2,shrink=1)
cbar.set_label('Value (No. of days school overheats)')
ax2.title.set_text('(b)')
ax2.coastlines()
ax2.set_xlim(-6.1,2)
ax2.set_ylim(49.5,56.1)

ax3.set_xlabel('Spatially aggregated risk \n (Total no. of days schools overheat)')
ax3.set_ylabel('Frequency')
ax3.set_xlim(1.16e6,1.2e6)
ax3.hist(spaceagg['eai_exp'],density=False,color='cornflowerblue')
ax3.vlines(obs_spaceagg[0],ymin=0,ymax=4,linestyles='solid',colors='purple',linewidth=2)
ax3.vlines(spaceagg.mean()[0],ymin=0,ymax=4,linestyles='solid',colors='blue')
#sns.kdeplot(data=spaceagg['annual_impact'],ax=ax3)
ax3.title.set_text('(c)')
legend_elements = [Patch(facecolor='cornflowerblue', edgecolor='cornflowerblue',label='UKCP18 ensemble'),
                    Line2D([0], [0], color='blue', linestyle='solid', label='Ensemble mean'),
                   Line2D([0], [0], color='purple', linestyle='solid', label='Observed',linewidth=2)]
ax3.legend(handles=legend_elements,loc='upper right')

ax4.set_xlabel('Longitude')
ax4.set_ylabel('Latitude')
cp = ax4.scatter(bias_eai['exp_lon'],bias_eai['exp_lat'],c=bias_eai['bias'],vmin=-5,vmax=5,s=0.5,cmap='bwr')
cbar = plt.colorbar(cp,ax=ax4,shrink=1)
cbar.set_label('Bias (UKCP18 ensemble mean risk - Observed risk)')
ax4.title.set_text('(d)')
ax4.coastlines()


ds = pd.read_csv('/data/users/ldawkins/UKCR/DataForSchoolApp/Impact/'+warming_level+'/ens_01_output__thresh35_opttemp.csv', error_bad_lines=False)
df = ds[['exp_lon','exp_lat','eai_exp']]

ax5.set_xlabel('Longitude')
ax5.set_ylabel('Latitude')
cp = ax5.scatter(df['exp_lon'],df['exp_lat'],c=df['eai_exp'],s=2,cmap='gist_heat_r',vmin=0,vmax=6)
cbar = plt.colorbar(cp,ax=ax5,shrink=1)
cbar.set_label('Value (No. of days school overheats)')
ax5.title.set_text('(e) Overheating threshold = 35C')
ax5.coastlines()
ax5.set_xlim(-6.1,2)
ax5.set_ylim(49.5,56.1)

#ens_mean_order = pd.merge(df, ens_mean1, how = 'left', left_on = ['exp_lon','exp_lat'],
#                    right_on = ['exp_lon','exp_lat'])
ax6.set_xlabel('Longitude')
ax6.set_ylabel('Latitude')
#cp = ax6.scatter(ens_mean_order['exp_lon'],ens_mean_order['exp_lat'],c=ens_mean_order['eai_exp_y'],s=2,cmap='gist_heat_r',vmin=0,vmax=6)
cp = ax6.scatter(ens_mean1['exp_lon'],ens_mean1['exp_lat'],c=ens_mean1['eai_exp'],s=2,cmap='gist_heat_r',vmin=0,vmax=6)
cbar = plt.colorbar(cp,ax=ax6,shrink=1)
cbar.set_label('Value (No. of days school overheats)')
ax6.title.set_text('(f)')
ax6.coastlines()
ax6.set_xlim(-6.1,2)
ax6.set_ylim(49.5,56.1)

ax7.set_xlabel('Spatially aggregated risk \n (Total no. of days schools overheat)')
ax7.set_ylabel('Frequency')
ax7.set_xlim(2e4,2.9e4)
ax7.hist(spaceagg1['eai_exp'],density=False,color='cornflowerblue')
ax7.vlines(obs_spaceagg1[0],ymin=0,ymax=4,linestyles='solid',colors='purple',linewidth=2)
ax7.vlines(spaceagg1.mean()[0],ymin=0,ymax=4,linestyles='solid',colors='blue')
#sns.kdeplot(data=spaceagg['annual_impact'],ax=ax3)
ax7.title.set_text('(g)')
legend_elements = [Patch(facecolor='cornflowerblue', edgecolor='cornflowerblue',label='UKCP18 ensemble'),
                    Line2D([0], [0], color='blue', linestyle='solid', label='Ensemble mean'),
                   Line2D([0], [0], color='purple', linestyle='solid', label='Observed',linewidth=2)]
ax7.legend(handles=legend_elements,loc='upper right')

ax8.set_xlabel('Longitude')
ax8.set_ylabel('Latitude')
cp = ax8.scatter(bias_eai1['exp_lon'],bias_eai1['exp_lat'],c=bias_eai1['bias'],vmin=-1.4,vmax=1.4,s=0.5,cmap='bwr')
cbar = plt.colorbar(cp,ax=ax8,shrink=1)
cbar.set_label('Bias (UKCP18 ensemble mean risk - Observed risk)')
ax8.title.set_text('(h)')
ax8.coastlines()

plt.tight_layout()
plt.savefig(
      plot_out_dir + 'Figure4.png', dpi=500)
plt.show(block=True)

############################################################################################
# Figure 7
############################################################################################


# current ens mean  - calculated above
ens_mean_current_26 = ens_mean
ens_mean_current_35 = ens_mean1

spaceagg_current_26 = spaceagg
spaceagg_current_35 = spaceagg1

# 2 deg ens mean
warming_level='2deg'
ens_mems = ['01', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '15']
ens_mem = ens_mems[0]
threshold = '26'
ds = pd.read_csv('/data/users/ldawkins/UKCR/DataForSchoolApp/Impact/'+warming_level+'/ens_'+ens_mem+'_output__thresh'+threshold+'_opttemp.csv', error_bad_lines=False)
df = ds[['exp_lon','exp_lat','eai_exp']]
df['member'] = ens_mem
df['schoolno'] = df.index + 1
for ens_mem in ens_mems[1:len(ens_mems)]:
    ds = pd.read_csv(
        '/data/users/ldawkins/UKCR/DataForSchoolApp/Impact/' + warming_level + '/ens_' + ens_mem + '_output__thresh' + threshold + '_opttemp.csv',
        error_bad_lines=False)
    df2 = ds[['exp_lon', 'exp_lat', 'eai_exp']]
    df2['member'] = ens_mem
    df2['schoolno'] = df2.index + 1
    df = pd.concat([df, df2])
# find ensemble mean EAI
ens_mean_2deg_26 = df.groupby(['schoolno']).mean()
ens_mean_2deg_26 = ens_mean_2deg_26.reset_index(level=0)
ens_mean_2deg_26 = ens_mean_2deg_26.reset_index(level=0)

spaceagg_2deg_26 = df[['eai_exp','member']].groupby(['member']).sum()

ens_mem = ens_mems[0]
threshold = '35'
ds = pd.read_csv('/data/users/ldawkins/UKCR/DataForSchoolApp/Impact/'+warming_level+'/ens_'+ens_mem+'_output__thresh'+threshold+'_opttemp.csv', error_bad_lines=False)
df = ds[['exp_lon','exp_lat','eai_exp']]
df['member'] = ens_mem
df['schoolno'] = df.index + 1
for ens_mem in ens_mems[1:len(ens_mems)]:
    ds = pd.read_csv(
        '/data/users/ldawkins/UKCR/DataForSchoolApp/Impact/' + warming_level + '/ens_' + ens_mem + '_output__thresh' + threshold + '_opttemp.csv',
        error_bad_lines=False)
    df2 = ds[['exp_lon', 'exp_lat', 'eai_exp']]
    df2['member'] = ens_mem
    df2['schoolno'] = df2.index + 1
    df = pd.concat([df, df2])
# find ensemble mean EAI
ens_mean_2deg_35 = df.groupby(['schoolno']).mean()
ens_mean_2deg_35 = ens_mean_2deg_35.reset_index(level=0)
ens_mean_2deg_35 = ens_mean_2deg_35.reset_index(level=0)

# Find spatially aggregated EAI for each ensemble member
spaceagg_2deg_35 = df[['eai_exp','member']].groupby(['member']).sum()


# 4 deg ens mean
warming_level='4deg'
ens_mems = ['01', '04', '05', '06', '07', '09', '11', '12', '13']
ens_mem = ens_mems[0]
threshold = '26'
ds = pd.read_csv('/data/users/ldawkins/UKCR/DataForSchoolApp/Impact/'+warming_level+'/ens_'+ens_mem+'_output__thresh'+threshold+'_opttemp.csv', error_bad_lines=False)
df = ds[['exp_lon','exp_lat','eai_exp']]
df['member'] = ens_mem
df['schoolno'] = df.index + 1
for ens_mem in ens_mems[1:len(ens_mems)]:
    ds = pd.read_csv(
        '/data/users/ldawkins/UKCR/DataForSchoolApp/Impact/' + warming_level + '/ens_' + ens_mem + '_output__thresh' + threshold + '_opttemp.csv',
        error_bad_lines=False)
    df2 = ds[['exp_lon', 'exp_lat', 'eai_exp']]
    df2['member'] = ens_mem
    df2['schoolno'] = df2.index + 1
    df = pd.concat([df, df2])
# find ensemble mean EAI
ens_mean_4deg_26 = df.groupby(['schoolno']).mean()
ens_mean_4deg_26 = ens_mean_4deg_26.reset_index(level=0)
ens_mean_4deg_26 = ens_mean_4deg_26.reset_index(level=0)

spaceagg_4deg_26 = df[['eai_exp','member']].groupby(['member']).sum()

ens_mem = ens_mems[0]
threshold = '35'
ds = pd.read_csv('/data/users/ldawkins/UKCR/DataForSchoolApp/Impact/'+warming_level+'/ens_'+ens_mem+'_output__thresh'+threshold+'_opttemp.csv', error_bad_lines=False)
df = ds[['exp_lon','exp_lat','eai_exp']]
df['member'] = ens_mem
df['schoolno'] = df.index + 1
for ens_mem in ens_mems[1:len(ens_mems)]:
    ds = pd.read_csv(
        '/data/users/ldawkins/UKCR/DataForSchoolApp/Impact/' + warming_level + '/ens_' + ens_mem + '_output__thresh' + threshold + '_opttemp.csv',
        error_bad_lines=False)
    df2 = ds[['exp_lon', 'exp_lat', 'eai_exp']]
    df2['member'] = ens_mem
    df2['schoolno'] = df2.index + 1
    df = pd.concat([df, df2])
# find ensemble mean EAI
ens_mean_4deg_35 = df.groupby(['schoolno']).mean()
ens_mean_4deg_35 = ens_mean_4deg_35.reset_index(level=0)
ens_mean_4deg_35 = ens_mean_4deg_35.reset_index(level=0)

# Find spatially aggregated EAI for each ensemble member
spaceagg_4deg_35 = df[['eai_exp','member']].groupby(['member']).sum()


# histogram + CIBSE spatial agg

fig = plt.figure(figsize=(10,18))
gs = GridSpec(nrows=4, ncols=2)
ax1 = fig.add_subplot(gs[0, 0],projection=ccrs.PlateCarree(),aspect='auto')
ax2 = fig.add_subplot(gs[1, 0],projection=ccrs.PlateCarree(),aspect='auto')
ax3 = fig.add_subplot(gs[2, 0],projection=ccrs.PlateCarree(),aspect='auto')
ax4 = fig.add_subplot(gs[3, 0])
ax5 = fig.add_subplot(gs[0, 1],projection=ccrs.PlateCarree(),aspect='auto')
ax6 = fig.add_subplot(gs[1, 1],projection=ccrs.PlateCarree(),aspect='auto')
ax7 = fig.add_subplot(gs[2, 1],projection=ccrs.PlateCarree(),aspect='auto')
ax8 = fig.add_subplot(gs[3, 1])

ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
cp = ax1.scatter(ens_mean_current_26['exp_lon'],ens_mean_current_26['exp_lat'],c=ens_mean_current_26['eai_exp'],s=2,cmap='gist_heat_r',vmin=0,vmax=120)
cbar = plt.colorbar(cp,ax=ax1,shrink=1)
cbar.set_label('Value (No. of days school overheats)')
ax1.title.set_text('(a) Overheating threshold = 26C')
ax1.coastlines()
ax1.set_xlim(-6.1,2)
ax1.set_ylim(49.5,56.1)

ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')
cp = ax2.scatter(ens_mean_2deg_26['exp_lon'],ens_mean_2deg_26['exp_lat'],c=ens_mean_2deg_26['eai_exp'],s=2,cmap='gist_heat_r',vmin=0,vmax=120)
cbar = plt.colorbar(cp,ax=ax2,shrink=1)
cbar.set_label('Value (No. of days school overheats)')
ax2.title.set_text('(b)')
ax2.coastlines()
ax2.set_xlim(-6.1,2)
ax2.set_ylim(49.5,56.1)

ax3.set_xlabel('Longitude')
ax3.set_ylabel('Latitude')
cp = ax3.scatter(ens_mean_4deg_26['exp_lon'],ens_mean_4deg_26['exp_lat'],c=ens_mean_4deg_26['eai_exp'],s=2,cmap='gist_heat_r',vmin=0,vmax=120)
cbar = plt.colorbar(cp,ax=ax3,shrink=1)
cbar.set_label('Value (No. of days school overheats)')
ax3.title.set_text('(c)')
ax3.coastlines()
ax3.set_xlim(-6.1,2)
ax3.set_ylim(49.5,56.1)

binwidth = 3e4
binsuse = np.arange(1e6, 1.9e6  + binwidth, binwidth)
ax4.set_xlabel('Spatially aggregated risk \n (Total no. of days schools overheat)')
ax4.set_ylabel('Frequency')
ax4.set_xlim(1e6,2e6)
ax4.hist(spaceagg_current_26['eai_exp'],density=False,color='cornflowerblue',label='Current climate: UKCP18 ensemble',bins=binsuse)
ax4.hist(spaceagg_2deg_26['eai_exp'],density=False,color='palegreen',label='2oC warming level: UKCP18 ensemble',bins=binsuse)
ax4.hist(spaceagg_4deg_26['eai_exp'],density=False,color='darksalmon',label='4oC warming level: UKCP18 ensemble',bins=binsuse)
ax4.title.set_text('(d)')
ax4.vlines(spaceagg_current_26['eai_exp'].mean(),ymin=0,ymax=10,linestyles='solid',colors='blue',linewidth=2,label='Current climate: UKCP18 mean')
ax4.vlines(spaceagg_2deg_26['eai_exp'].mean(),ymin=0,ymax=10,linestyles='solid',colors='green',linewidth=2,label='2oC warming level: UKCP18 mean')
ax4.vlines(spaceagg_4deg_26['eai_exp'].mean(),ymin=0,ymax=10,linestyles='solid',colors='red',linewidth=2,label='4oC warming level: UKCP18 mean')
ax4.vlines(spaceagg_current_26_cibse,ymin=0,ymax=10,linestyles='dashed',colors='blue',linewidth=2,label='Current climate: Region location')
ax4.vlines(spaceagg_2deg_26_cibse,ymin=0,ymax=10,linestyles='dashed',colors='green',linewidth=2,label='2oC warming level: Region location')
ax4.vlines(spaceagg_4deg_26_cibse,ymin=0,ymax=10,linestyles='dashed',colors='red',linewidth=2,label='4oC warming level: Region location')
ax4.vlines(obs_spaceagg[0],ymin=0,ymax=10,linestyles='dotted',colors='purple',linewidth=2,label='Current climate: Observed')
legend1 = ax4.legend(loc='upper right',fontsize='x-small',framealpha=1)
legend1.get_frame().set_facecolor('white')

ax5.set_xlabel('Longitude')
ax5.set_ylabel('Latitude')
cp = ax5.scatter(ens_mean_current_35['exp_lon'],ens_mean_current_35['exp_lat'],c=ens_mean_current_35['eai_exp'],s=2,cmap='gist_heat_r',vmin=0,vmax=25)
cbar = plt.colorbar(cp,ax=ax5,shrink=1)
cbar.set_label('Value (No. of days school overheats)')
ax5.title.set_text('(e) Overheating threshold = 35C')
ax5.coastlines()
ax5.set_xlim(-6.1,2)
ax5.set_ylim(49.5,56.1)

ax6.set_xlabel('Longitude')
ax6.set_ylabel('Latitude')
cp = ax6.scatter(ens_mean_2deg_35['exp_lon'],ens_mean_2deg_35['exp_lat'],c=ens_mean_2deg_35['eai_exp'],s=2,cmap='gist_heat_r',vmin=0,vmax=25)
cbar = plt.colorbar(cp,ax=ax6,shrink=1)
cbar.set_label('Value (No. of days school overheats)')
ax6.title.set_text('(f)')
ax6.coastlines()
ax6.set_xlim(-6.1,2)
ax6.set_ylim(49.5,56.1)

ax7.set_xlabel('Longitude')
ax7.set_ylabel('Latitude')
cp = ax7.scatter(ens_mean_4deg_35['exp_lon'],ens_mean_4deg_35['exp_lat'],c=ens_mean_4deg_35['eai_exp'],s=2,cmap='gist_heat_r',vmin=0,vmax=25)
cbar = plt.colorbar(cp,ax=ax7,shrink=1)
cbar.set_label('Value (No. of days school overheats)')
ax7.title.set_text('(g)')
ax7.coastlines()
ax7.set_xlim(-6.1,2)
ax7.set_ylim(49.5,56.1)

binwidth = 5e3
binsuse = np.arange(1e4, 3e5 + binwidth, binwidth)
binwidth2 = 1e4
binsuse2 = np.arange(1e4, 3e5 + binwidth2, binwidth2)
ax8.set_xlabel('Spatially aggregated risk \n (Total no. of days schools overheat)')
ax8.set_ylabel('Frequency')
ax8.set_xlim(1e3,3.2e5)
ax8.hist(spaceagg_current_35['eai_exp'],density=False,color='cornflowerblue',label='Current climate: UKCP18 ensemble',bins=binsuse)
ax8.hist(spaceagg_2deg_35['eai_exp'],density=False,color='palegreen',label='2oC warming level: UKCP18 ensemble',bins=binsuse2)
ax8.hist(spaceagg_4deg_35['eai_exp'],density=False,color='darksalmon',label='4oC warming level: UKCP18 ensemble',bins=binsuse2)
ax8.title.set_text('(h)')
ax8.vlines(spaceagg_current_35['eai_exp'].mean(),ymin=0,ymax=10,linestyles='solid',colors='blue',linewidth=2,label='Current climate: UKCP18 mean')
ax8.vlines(spaceagg_2deg_35['eai_exp'].mean(),ymin=0,ymax=10,linestyles='solid',colors='green',linewidth=2,label='2oC warming level: UKCP18 mean')
ax8.vlines(spaceagg_4deg_35['eai_exp'].mean(),ymin=0,ymax=10,linestyles='solid',colors='red',linewidth=2,label='4oC warming level: UKCP18 mean')
ax8.vlines(spaceagg_current_35_cibse,ymin=0,ymax=10,linestyles='dashed',colors='blue',linewidth=2,label='Current climate: Region location')
ax8.vlines(spaceagg_2deg_35_cibse,ymin=0,ymax=10,linestyles='dashed',colors='green',linewidth=2,label='2oC warming level: Region location')
ax8.vlines(spaceagg_4deg_35_cibse,ymin=0,ymax=10,linestyles='dashed',colors='red',linewidth=2,label='4oC warming level: Region location')
ax8.vlines(obs_spaceagg1[0],ymin=0,ymax=10,linestyles='dotted',colors='purple',linewidth=2,label='Current climate: Observed')
legend2 = ax8.legend(loc='upper right',fontsize='x-small',framealpha=1)
legend2.get_frame().set_facecolor('white')

plt.tight_layout()
plt.savefig(
      plot_out_dir + 'Figure5.png', dpi=500)
plt.show(block=True)



############################################################################################
# Figure 8
############################################################################################


diff_eai_26 = pd.merge(ens_mean_2deg_26_cibse, ens_mean_2deg_26, how = 'left', left_on = ['exp_lon','exp_lat'],
                    right_on = ['exp_lon','exp_lat'])
diff_eai_26['diff'] = (diff_eai_26['eai_exp_x'] - diff_eai_26['eai_exp_y'])#/diff_eai_26['eai_exp_y']

diff_eai_35 = pd.merge(ens_mean_2deg_35_cibse, ens_mean_2deg_35, how = 'left', left_on = ['exp_lon','exp_lat'],
                    right_on = ['exp_lon','exp_lat'])
diff_eai_35['diff'] = (diff_eai_35['eai_exp_x'] - diff_eai_35['eai_exp_y'])#/diff_eai_35['eai_exp_y']


fig = plt.figure(figsize=(12,6))
gs = GridSpec(nrows=1, ncols=2)
ax1 = fig.add_subplot(gs[0, 0],projection=ccrs.PlateCarree(),aspect='auto')
ax2 = fig.add_subplot(gs[0, 1],projection=ccrs.PlateCarree(),aspect='auto')

ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
cp = ax1.scatter(diff_eai_26['exp_lon'],diff_eai_26['exp_lat'],c=diff_eai_26['diff'],s=2,cmap='bwr',vmin=-46,vmax=46)
cbar = plt.colorbar(cp,ax=ax1,shrink=1)
cbar.set_label('Value (Difference in no. of days school overheats)')
ax1.title.set_text('(a) Overheating threshold = 26C')
ax1.coastlines()
ax1.set_xlim(-6.1,2)
ax1.set_ylim(49.5,56.1)

ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')
cp = ax2.scatter(diff_eai_35['exp_lon'],diff_eai_35['exp_lat'],c=diff_eai_35['diff'],s=2,cmap='bwr',vmin=-5.5,vmax=5.5)
cbar = plt.colorbar(cp,ax=ax2,shrink=1)
cbar.set_label('Value (Difference in no. of days school overheats)')
ax2.title.set_text('(b) Overheating threshold = 35C')
ax2.coastlines()
ax2.set_xlim(-6.1,2)
ax2.set_ylim(49.5,56.1)


plt.tight_layout()
plt.savefig(
      plot_out_dir + 'Figure6.png', dpi=500)
plt.show(block=True)


# Figure 8


############################################################################################
# Figure 9
############################################################################################


# NEED TO RUN plots_for_paperR.R first as this generates GAM samples

# GAM samples

def risk_func(x):
    risk = 10**x - 1
    return(risk)

def load_info(warming_level, threshold):
    ds = pd.read_csv(
        '/data/users/ldawkins/UKCR/DataForSchoolApp/Impact/' + warming_level + '/GAM_SAMPLES_output__thresh' + threshold + '_opttemp.csv',
        error_bad_lines=False)
    ds = ds.drop(['Unnamed: 0'], axis=1)
    ds = ds.apply(risk_func, axis=1)

    EAI_lower = np.quantile(ds,0.025,axis=0)
    EAI_upper = np.quantile(ds,0.975,axis=0)
    EAI_mean = np.nanmean(ds,axis=0)
    EAI_lower_diff = EAI_lower - EAI_mean
    EAI_upper_diff = EAI_upper - EAI_mean

    spaceagg_sim = ds.apply(np.sum,axis=1)

    return(EAI_mean,EAI_lower_diff,EAI_upper_diff,spaceagg_sim)

# threshold = 26
#current
EAI_mean_c_26, EAI_lower_diff_c_26, EAI_upper_diff_c_26, spaceagg_sim_c_26 = load_info('current', '26')
# 2 deg
EAI_mean_2deg_26, EAI_lower_diff_2deg_26, EAI_upper_diff_2deg_26, spaceagg_sim_2deg_26 = load_info('2deg', '26')
# 4deg
EAI_mean_4deg_26, EAI_lower_diff_4deg_26, EAI_upper_diff_4deg_26, spaceagg_sim_4deg_26 = load_info('4deg', '26')

# threshold = 35
#current
EAI_mean_c_35, EAI_lower_diff_c_35, EAI_upper_diff_c_35, spaceagg_sim_c_35 = load_info('current', '35')
# 2 deg
EAI_mean_2deg_35, EAI_lower_diff_2deg_35, EAI_upper_diff_2deg_35, spaceagg_sim_2deg_35 = load_info('2deg', '35')
# 4deg
EAI_mean_4deg_35, EAI_lower_diff_4deg_35, EAI_upper_diff_4deg_35, spaceagg_sim_4deg_35 = load_info('4deg', '35')

lon = ens_mean_2deg_35['exp_lon']
lat = ens_mean_2deg_35['exp_lat']

fig = plt.figure(figsize=(15,24))
gs = GridSpec(nrows=6, ncols=3)
ax1 = fig.add_subplot(gs[0, 0],projection=ccrs.PlateCarree())
ax2 = fig.add_subplot(gs[0, 1],projection=ccrs.PlateCarree())
ax3 = fig.add_subplot(gs[0, 2],projection=ccrs.PlateCarree())
ax4 = fig.add_subplot(gs[1, 0],projection=ccrs.PlateCarree())
ax5 = fig.add_subplot(gs[1, 1],projection=ccrs.PlateCarree())
ax6 = fig.add_subplot(gs[1, 2],projection=ccrs.PlateCarree())

ax7 = fig.add_subplot(gs[3, 0],projection=ccrs.PlateCarree())
ax8 = fig.add_subplot(gs[3, 1],projection=ccrs.PlateCarree())
ax9 = fig.add_subplot(gs[3, 2],projection=ccrs.PlateCarree())
ax10 = fig.add_subplot(gs[4, 0],projection=ccrs.PlateCarree())
ax11 = fig.add_subplot(gs[4, 1],projection=ccrs.PlateCarree())
ax12 = fig.add_subplot(gs[4, 2],projection=ccrs.PlateCarree())

ax13 = fig.add_subplot(gs[2, 0:3])
ax14 = fig.add_subplot(gs[5, 0:3])

ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
cp = ax1.scatter(lon,lat,c=EAI_lower_diff_2deg_26,s=1,vmin=-80,vmax=80,cmap='bwr')
cbar = plt.colorbar(cp,ax=ax1,shrink=0.8)
cbar.set_label('No. of days school overheats \n (lower bound of 95% CI - mean)')
ax1.title.set_text('(a) Overheating threshold = 26C')
ax1.coastlines()
ax1.set_xlim(-6.1,2)
ax1.set_ylim(49.5,56.1)

ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')
cp = ax2.scatter(lon,lat,c=EAI_mean_2deg_26,s=1,cmap='gist_heat_r',vmin=0,vmax=120)
cbar = plt.colorbar(cp,ax=ax2,shrink=0.8)
cbar.set_label('No. of days school overheats (mean)')
ax2.title.set_text('(b)')
ax2.coastlines()
ax2.set_xlim(-6.1,2)
ax2.set_ylim(49.5,56.1)

ax3.set_xlabel('Longitude')
ax3.set_ylabel('Latitude')
cp = ax3.scatter(lon,lat,c=EAI_upper_diff_2deg_26,s=1,vmin=-80,vmax=80,cmap='bwr')
cbar = plt.colorbar(cp,ax=ax3,shrink=0.8)
cbar.set_label('No. of days school overheats \n (upper bound of 95% CI - mean)')
ax3.title.set_text('(c)')
ax3.coastlines()
ax3.set_xlim(-6.1,2)
ax3.set_ylim(49.5,56.1)

ax4.set_xlabel('Longitude')
ax4.set_ylabel('Latitude')
cp = ax4.scatter(lon,lat,c=EAI_lower_diff_4deg_26,s=1,vmin=-80,vmax=80,cmap='bwr')
cbar = plt.colorbar(cp,ax=ax4,shrink=0.8)
cbar.set_label('No. of days school overheats \n (lower bound of 95% CI - mean)')
ax4.title.set_text('(d)')
ax4.coastlines()
ax4.set_xlim(-6.1,2)
ax4.set_ylim(49.5,56.1)

ax5.set_xlabel('Longitude')
ax5.set_ylabel('Latitude')
cp = ax5.scatter(lon,lat,c=EAI_mean_4deg_26,s=1,cmap='gist_heat_r',vmin=0,vmax=120)
cbar = plt.colorbar(cp,ax=ax5,shrink=0.8)
cbar.set_label('No. of days school overheats (mean)')
ax5.title.set_text('(e)')
ax5.coastlines()
ax5.set_xlim(-6.1,2)
ax5.set_ylim(49.5,56.1)

ax6.set_xlabel('Longitude')
ax6.set_ylabel('Latitude')
cp = ax6.scatter(lon,lat,c=EAI_upper_diff_4deg_26,s=1,vmin=-80,vmax=80,cmap='bwr')
cbar = plt.colorbar(cp,ax=ax6,shrink=0.8)
cbar.set_label('No. of days school overheats \n (upper bound of 95% CI - mean)')
ax6.title.set_text('(f)')
ax6.coastlines()
ax6.set_xlim(-6.1,2)
ax6.set_ylim(49.5,56.1)

ax7.set_xlabel('Longitude')
ax7.set_ylabel('Latitude')
cp = ax7.scatter(lon,lat,c=EAI_lower_diff_2deg_35,s=1,vmin=-20,vmax=20,cmap='bwr')
cbar = plt.colorbar(cp,ax=ax7,shrink=0.8)
cbar.set_label('No. of days school overheats \n (lower bound of 95% CI - mean)')
ax7.title.set_text('(h) Overheating threshold = 35C')
ax7.coastlines()
ax7.set_xlim(-6.1,2)
ax7.set_ylim(49.5,56.1)

ax8.set_xlabel('Longitude')
ax8.set_ylabel('Latitude')
cp = ax8.scatter(lon,lat,c=EAI_mean_2deg_35,s=1,cmap='gist_heat_r',vmin=0,vmax=25)
cbar = plt.colorbar(cp,ax=ax8,shrink=0.8)
cbar.set_label('No. of days school overheats (mean)')
ax8.title.set_text('(i)')
ax8.coastlines()
ax8.set_xlim(-6.1,2)
ax8.set_ylim(49.5,56.1)

ax9.set_xlabel('Longitude')
ax9.set_ylabel('Latitude')
cp = ax9.scatter(lon,lat,c=EAI_upper_diff_2deg_35,s=1,vmin=-20,vmax=20,cmap='bwr')
cbar = plt.colorbar(cp,ax=ax9,shrink=0.8)
cbar.set_label('No. of days school overheats \n (upper bound of 95% CI - mean)')
ax9.title.set_text('(j)')
ax9.coastlines()
ax9.set_xlim(-6.1,2)
ax9.set_ylim(49.5,56.1)

ax10.set_xlabel('Longitude')
ax10.set_ylabel('Latitude')
cp = ax10.scatter(lon,lat,c=EAI_lower_diff_4deg_35,s=1,vmin=-20,vmax=20,cmap='bwr')
cbar = plt.colorbar(cp,ax=ax10,shrink=0.8)
cbar.set_label('No. of days school overheats \n (lower bound of 95% CI - mean)')
ax10.title.set_text('(k)')
ax10.coastlines()
ax10.set_xlim(-6.1,2)
ax10.set_ylim(49.5,56.1)

ax11.set_xlabel('Longitude')
ax11.set_ylabel('Latitude')
cp = ax11.scatter(lon,lat,c=EAI_mean_4deg_35,s=1,cmap='gist_heat_r',vmin=0,vmax=25)
cbar = plt.colorbar(cp,ax=ax11,shrink=0.8)
cbar.set_label('No. of days school overheats (mean)')
ax11.title.set_text('(l)')
ax11.coastlines()
ax11.set_xlim(-6.1,2)
ax11.set_ylim(49.5,56.1)

ax12.set_xlabel('Longitude')
ax12.set_ylabel('Latitude')
cp = ax12.scatter(lon,lat,c=EAI_upper_diff_4deg_35,s=1,vmin=-20,vmax=20,cmap='bwr')
cbar = plt.colorbar(cp,ax=ax12,shrink=0.8)
cbar.set_label('No. of days school overheats \n (upper bound of 95% CI - mean)')
ax12.title.set_text('(m)')
ax12.coastlines()
ax12.set_xlim(-6.1,2)
ax12.set_ylim(49.5,56.1)

binwidth1 = 5e3
binsuse1 = np.arange(1e6, 1.9e6  + binwidth1, binwidth1)
binwidth2 = 3e4
binsuse2 = np.arange(1e6, 1.9e6  + binwidth2, binwidth2)
ax13.set_xlabel('Spatially aggregated risk \n (Total no. of days schools overheat)')
ax13.set_ylabel('Frequency (GAM samples)')
ax13.set_xlim(1e6,2e6)
ax13.hist(spaceagg_sim_c_26,density=False,color='darkblue',label='Current climate (recent past): GAM',alpha=0.5,bins=binsuse1)
ax13.hist(spaceagg_sim_2deg_26,density=False,color='green',label='2oC warming level: GAM',alpha=0.5,bins=binsuse2)
ax13.hist(spaceagg_sim_4deg_26,density=False,color='red',label='4oC warming level: GAM',alpha=0.5,bins=binsuse2)
ax15 = ax13.twinx()
ax15.hist(spaceagg_current_26['eai_exp'],density=False,color='cornflowerblue',label='Current climate: UKCP18 ensemble',alpha=0.6,bins=binsuse1)
ax15.hist(spaceagg_2deg_26['eai_exp'],density=False,color='palegreen',label='2oC warming level: UKCP18 ensemble',alpha=0.6,bins=binsuse2)
ax15.hist(spaceagg_4deg_26['eai_exp'],density=False,color='darksalmon',label='4oC warming level: UKCP18 ensemble',alpha=0.6,bins=binsuse2)
ax15.title.set_text('(g)')
ax15.set_ylabel('Frequency (UKCP18 ensemble)')
ax13.legend(loc='upper right')
ax15.legend(loc='upper center')
ax13.set_ylim(0,300)
ax15.set_ylim(0,8)

binwidth3 = 3e3
binsuse3 = np.arange(1e4, 3e5 + binwidth3, binwidth3)
binwidth4 = 5e3
binsuse4 = np.arange(1e4, 3e5 + binwidth4, binwidth4)
ax14.set_xlabel('Spatially aggregated risk \n (Total no. of days schools overheat)')
ax14.set_ylabel('Frequency (GAM samples)')
ax14.set_xlim(1e3,3.2e5)
ax14.hist(spaceagg_sim_c_35,density=False,color='darkblue',label='Current climate (recent past): GAM',alpha=0.5,bins=binsuse3)
ax14.hist(spaceagg_sim_2deg_35,density=False,color='green',label='2oC warming level: GAM',alpha=0.5,bins=binsuse4)
ax14.hist(spaceagg_sim_4deg_35,density=False,color='red',label='4oC warming level: GAM',alpha=0.5,bins=binsuse4)
ax16 = ax14.twinx()
ax16.hist(spaceagg_current_35['eai_exp'],density=False,color='cornflowerblue',label='Current climate: UKCP18 ensemble',alpha=0.6,bins=binsuse3)
ax16.hist(spaceagg_2deg_35['eai_exp'],density=False,color='palegreen',label='2oC warming level: UKCP18 ensemble',alpha=0.6,bins=binsuse4)
ax16.hist(spaceagg_4deg_35['eai_exp'],density=False,color='darksalmon',label='4oC warming level: UKCP18 ensemble',alpha=0.6,bins=binsuse4)
ax16.title.set_text('(n)')
ax16.set_ylabel('Frequency (UKCP18 ensemble)')
ax14.legend(loc='upper right')
ax16.legend(loc='upper center')
ax14.set_ylim(0,420)
ax16.set_ylim(0,8)

plt.tight_layout()
plt.savefig(
       plot_out_dir + 'Figure8.png', dpi=500)
plt.show(block=True)



############################################################################################
# Figure 10
############################################################################################

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.lines as mlines
import cartopy.crs as ccrs
from cartopy.io.img_tiles import OSM

test = np.empty(EAI_mean_2deg_35.shape[0])
use = EAI_mean_2deg_35
test[use < np.quantile(use,0.8)] = 1
test[(use < np.quantile(use,0.9)) & (use> np.quantile(use,0.8))] = 2
test[use > np.quantile(use,0.9)] = 3

#test2 = (EAI_mean_4deg_35 + EAI_upper_diff_4deg_35)
#sum =  (EAI_mean_4deg_35 + EAI_upper_diff_4deg_35)
#test2[sum < np.quantile(sum,0.9)] = 1
#test2[(sum < np.quantile(sum,0.95)) & (sum > np.quantile(sum,0.9))] = 2
#test2[sum > np.quantile(sum,0.95)] = 3

markers = ['o','^','d']
labs = ['low','medium','high']
cols = [(0.3, 1, 0.3), (1,0.6,0.1), (1, 0, 0)]
df1 = pd.DataFrame()
df1['lon'] = lon
df1['lat'] = lat
df1['risk'] = test.astype(np.int)
df1['marker'] = df1["risk"].apply(lambda x: markers[x - 1])
df1['col'] = df1["risk"].apply(lambda x: cols[x - 1])
df1['lab'] = df1["risk"].apply(lambda x: labs[x - 1])
df1 = df1.sort_values('risk')
#df2 = pd.DataFrame()
#df2['lon'] = lon
#df2['lat'] = lat
#df2['risk'] = test2.astype(np.int)
#df2['marker'] = df2["risk"].apply(lambda x: markers[x - 1])
#df2['col'] = df2["risk"].apply(lambda x: cols[x - 1])
#df2['lab'] = df2["risk"].apply(lambda x: labs[x - 1])
#df2 = df2.sort_values('risk')

imagery = OSM()
fig = plt.figure(figsize=(16,10))
gs = GridSpec(nrows=2, ncols=2)
ax1 = fig.add_subplot(gs[0, 0],projection=ccrs.PlateCarree())
ax2 = fig.add_subplot(gs[0, 1],projection=ccrs.PlateCarree())
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1],projection=ccrs.PlateCarree())
#ax5 = fig.add_subplot(gs[0, 2],projection=ccrs.PlateCarree())
#ax6 = fig.add_subplot(gs[1, 2],projection=ccrs.PlateCarree())

colors = [(0.3, 1, 0.3), (1,0.6,0.1), (1, 0, 0)]
n_bins = 3 # Discretizes the interpolation into bins
cmap_name = 'my_list'
cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
low =  mlines.Line2D([], [], color=colors[0], marker='o', linestyle='None', markersize=10, label='Low')
med =  mlines.Line2D([], [], color=colors[1], marker='^', linestyle='None', markersize=10, label='Medium')
high =  mlines.Line2D([], [], color=colors[2], marker='d', linestyle='None', markersize=10, label='High')

ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
for marker, d in df1.groupby('marker',sort=False):
    ax1.scatter(d['lon'],d['lat'],c=d['col'],s=1, marker=marker)
cbar.set_label('School risk category')
ax1.legend(handles=[low,med,high], loc=1)
ax1.title.set_text('(a)')
ax1.coastlines()
ax1.set_xlim(-6.1,2)
ax1.set_ylim(49.5,56.1)

#ax3.set_xlabel('Longitude')
#ax3.set_ylabel('Latitude')
#for marker, d in df2.groupby('marker',sort=False):
#    ax3.scatter(d['lon'],d['lat'],c=d['col'],s=1, marker=marker)
#cbar.set_label('School risk category')
#ax3.title.set_text('(d)')
#ax3.coastlines()
#ax3.set_xlim(-6.1,2)
#ax3.set_ylim(49.5,56.1)

ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')
for i in range(0,10):
    ind = np.where(expl_inst.gdf['impf_HS'] == archs[i])[0]
    df1ind = df1.loc[ind,:]
    df1ind = df1ind.sort_values('risk')
    for marker, d in df1ind.groupby('marker',sort=False):
        ax2.scatter(d['lon'],d['lat'],c=d['col'],s=3, marker=marker)
ax2.title.set_text('(b)')
ax2.coastlines()

#ax4.set_xlabel('Longitude')
#ax4.set_ylabel('Latitude')
#for i in range(0,10):
#    ind = np.where(expl_inst.gdf['impf_HS'] == archs[i])[0]
#    df2ind = df2.loc[ind,:]
#    df2ind = df2ind.sort_values('risk')
#    for marker, d in df2ind.groupby('marker',sort=False):
#        ax4.scatter(d['lon'],d['lat'],c=d['col'],s=3, marker=marker)
#ax4.title.set_text('(e)')
#ax4.coastlines()

test = np.empty((10,3))
for i in range(0,10):
    ind = np.where(expl_inst.gdf['impf_HS'] == archs[i])[0]
    df1ind = df1.loc[ind,:]
    lenall = df1ind.shape[0]
    test[i,0] = (len(np.where(df1ind['lab'] == 'low')[0])/lenall) * 100
    test[i,1] = (len(np.where(df1ind['lab'] == 'medium')[0])/lenall) * 100
    test[i, 2] = (len(np.where(df1ind['lab'] == 'high')[0]) / lenall) * 100

atypes = ('Primary,\npre-1919','Secondary,\npre-1919','Primary,\n1919-44','Secondary,\n1919-44','Primary,\n1945-66',
          'Secondary,\n1945-66','Primary,\n1967-76','Secondary,\n1967-76','Primary,\npost-1976','Secondary,\npost-1976')

Low = test[:,0]
Medium = test[:,1]
High = test[:,2]

width=0.5
bottom = np.arange(10)

ax3.set_xlabel('Archetype')
ax3.set_ylabel('Proportion of archetype at each risk level')
p1 = ax3.bar(bottom, Low, width = width, color = colors[0])
p2 = ax3.bar(bottom, Medium, width = width, bottom = Low, color = colors[1])
p3 = ax3.bar(bottom, High, width = width, bottom = Low + Medium, color = colors[2])

legend_elements = [Patch(facecolor=colors[0], edgecolor=colors[0], label='Low'),
                    Patch(facecolor=colors[1], edgecolor=colors[1], label='Medium'),
                    Patch(facecolor=colors[2], edgecolor=colors[2], label='High')]
ax3.legend(handles=legend_elements, loc="upper right")
ax3.title.set_text('(c)')
ax3.set_xticks(bottom,atypes)


ax4.set_extent((-0.08, 0.08, 51.45,51.55))
# Add the imagery to the map.
ax4.add_image(imagery, 14)
for marker, d in df1.groupby('marker',sort=False):
    ax4.scatter(d['lon'],d['lat'],c=d['col'],s=20, marker=marker, edgecolors='black')
ax4.title.set_text('(d)')


# ax6.set_extent((-0.08, 0.08, 51.45,51.55))
# # Add the imagery to the map.
# ax6.add_image(imagery, 14)
# for marker, d in df2.groupby('marker',sort=False):
#     ax6.scatter(d['lon'],d['lat'],c=d['col'],s=20, marker=marker, edgecolors='black')
# ax6.title.set_text('(f)')


plt.tight_layout()
plt.savefig(
       plot_out_dir + 'Figure9.png', dpi=500)
plt.show(block=True)





# Figure 9 - example of prioritising schools - using CIBSE style data

use = np.array(ens_mean_2deg_35_cibse.eai_exp)
test = np.empty(use.shape[0])
test[use < np.quantile(use,0.8)] = 1
test[(use < np.quantile(use,0.9)) & (use> np.quantile(use,0.8))] = 2
test[use > np.quantile(use,0.9)] = 3

markers = ['o','^','d']
labs = ['low','medium','high']
cols = [(0.3, 1, 0.3), (1,0.6,0.1), (1, 0, 0)]
df1 = pd.DataFrame()
df1['lon'] = lon
df1['lat'] = lat
df1['risk'] = test.astype(np.int)
df1['marker'] = df1["risk"].apply(lambda x: markers[x - 1])
df1['col'] = df1["risk"].apply(lambda x: cols[x - 1])
df1['lab'] = df1["risk"].apply(lambda x: labs[x - 1])
df1 = df1.sort_values('risk')

imagery = OSM()
fig = plt.figure(figsize=(16,10))
gs = GridSpec(nrows=2, ncols=2)
ax1 = fig.add_subplot(gs[0, 0],projection=ccrs.PlateCarree())
ax2 = fig.add_subplot(gs[0, 1],projection=ccrs.PlateCarree())
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1],projection=ccrs.PlateCarree())

colors = [(0.3, 1, 0.3), (1,0.6,0.1), (1, 0, 0)]
n_bins = 3 # Discretizes the interpolation into bins
cmap_name = 'my_list'
cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
low =  mlines.Line2D([], [], color=colors[0], marker='o', linestyle='None', markersize=10, label='Low')
med =  mlines.Line2D([], [], color=colors[1], marker='^', linestyle='None', markersize=10, label='Medium')
high =  mlines.Line2D([], [], color=colors[2], marker='d', linestyle='None', markersize=10, label='High')

ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
for marker, d in df1.groupby('marker',sort=False):
    ax1.scatter(d['lon'],d['lat'],c=d['col'],s=1, marker=marker)
cbar.set_label('School risk category')
ax1.legend(handles=[low,med,high], loc=1)
ax1.title.set_text('(a)')
ax1.coastlines()
ax1.set_xlim(-6.1,2)
ax1.set_ylim(49.5,56.1)

ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')
for i in range(0,10):
    ind = np.where(expl_inst.gdf['impf_HS'] == archs[i])[0]
    df1ind = df1.loc[ind,:]
    df1ind = df1ind.sort_values('risk')
    for marker, d in df1ind.groupby('marker',sort=False):
        ax2.scatter(d['lon'],d['lat'],c=d['col'],s=3, marker=marker)
ax2.title.set_text('(b)')
ax2.coastlines()

test = np.empty((10,3))
for i in range(0,10):
    ind = np.where(expl_inst.gdf['impf_HS'] == archs[i])[0]
    df1ind = df1.loc[ind,:]
    lenall = df1ind.shape[0]
    test[i,0] = (len(np.where(df1ind['lab'] == 'low')[0])/lenall) * 100
    test[i,1] = (len(np.where(df1ind['lab'] == 'medium')[0])/lenall) * 100
    test[i, 2] = (len(np.where(df1ind['lab'] == 'high')[0]) / lenall) * 100

atypes = ('Primary,\npre-1919','Secondary,\npre-1919','Primary,\n1919-44','Secondary,\n1919-44','Primary,\n1945-66',
          'Secondary,\n1945-66','Primary,\n1967-76','Secondary,\n1967-76','Primary,\npost-1976','Secondary,\npost-1976')

Low = test[:,0]
Medium = test[:,1]
High = test[:,2]

width=0.5
bottom = np.arange(10)

ax3.set_xlabel('Archetype')
ax3.set_ylabel('Proportion of archetype at each risk level')
p1 = ax3.bar(bottom, Low, width = width, color = colors[0])
p2 = ax3.bar(bottom, Medium, width = width, bottom = Low, color = colors[1])
p3 = ax3.bar(bottom, High, width = width, bottom = Low + Medium, color = colors[2])

legend_elements = [Patch(facecolor=colors[0], edgecolor=colors[0], label='Low'),
                    Patch(facecolor=colors[1], edgecolor=colors[1], label='Medium'),
                    Patch(facecolor=colors[2], edgecolor=colors[2], label='High')]
ax3.legend(handles=legend_elements, loc="upper right")
ax3.title.set_text('(c)')
ax3.set_xticks(bottom,atypes)


ax4.set_extent((-0.08, 0.08, 51.45,51.55))
# Add the imagery to the map.
ax4.add_image(imagery, 14)
for marker, d in df1.groupby('marker',sort=False):
    ax4.scatter(d['lon'],d['lat'],c=d['col'],s=20, marker=marker, edgecolors='black')
ax4.title.set_text('(d)')


plt.tight_layout()
plt.savefig(
       plot_out_dir + 'Figure9_CIBSE_supmat.png', dpi=500)
plt.show(block=True)
