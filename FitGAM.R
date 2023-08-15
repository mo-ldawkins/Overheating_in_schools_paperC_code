
# This code fits a GAM to the 12 UKCP18 ensemble member representations of Risk (expected annual impact)
# The model is as follows:

#log_{10}(Risk(m,s,a)) ~ N(\mu(m,s,a), \sigma(s)^2), 
#mu(m,s,a) = f(lon(s), lat(s)) + \xi_m + \beta_a, 
#sigma(s) = f(lon(s), lat(s)), 
#xi_m ~ N(0, \lambda^2),

#wherem represents the ensemble member, s represents the spatial location, a represents the school building archetype and log_10() is the log base 10 function which is used to transform risk at each school to the log scale. This transform is performed to ensure risk (the output variable of the model) is symmetrically distributed (as required by the specified GAM model). The GAM then models this log transformed risk using a Normal distribution with a mean that varies in space (by longitude and latitude), by ensemble member, and by school building archetype, mu(m,s,a), and standard deviation that varies in space (by longitude and latitude), sigma(s). The mean is modelled as a combination of a smooth function of space, f(lon(s), lat(s)), (i.e., a spatial regression), a spatially constant ensemble member random effect, xi_m, and a school building archetype fixed effect, beta_a. The ensemble member random effect random effect is in turn modelled using a Normal distribution with zero mean and standard deviation lambda.


library(openxlsx)
library(fields)
library(maps)
library(dplyr)
library(mgcv)
library(RNetCDF)
library(lubridate)
library(dplyr)
library(reshape2)
library(ggplot2)
library(gridExtra)
library(MASS)
library("RColorBrewer")
library(fields)
library(ggplot2)

library(RNetCDF)
library(dplyr)
library(tidyr)
library(reshape2)
library(ggplot2)
library(gridExtra)
library(RColorBrewer)
library(ggpubr)



exposures = read.csv('/data/users/ldawkins/UKCR/DataForSchoolApp/exp_info.csv')


era = c('Pre 1919', "Interwar",'1945-66', "1967-76",'Post 1976')
region = c('Thames Valley', 'South Eastern', 'Southern', 'South Western', 'Severn Valley', 'Midland', 'West Pennines', 'North Western', 'Borders', 'North Eastern', 'East Pennines', 'East Anglia', 'Wales')  #seq(1,13,1)
type = c('Primary','Secondary')


eras <- rep(era,13*2)
regions <- rep(rep(region,each=5),2)
types <- rep(type,each=5*13)

archetype_all <- data.frame(atype=seq(1,130,1),era = as.factor(eras),region=as.factor(regions),type=as.factor(types))


# Reading in arguments from the shell script - which warming level and which overheating threshold
args <- commandArgs(trailingOnly = TRUE)
wl <- args[1]
threshold <- args[2]

print(wl)
print(threshold)

# make data frame for GAM

warming_levels= c('current','2deg','4deg')

warming_level = warming_levels[as.numeric(wl)]

print(warming_level)

if(warming_level == '4deg'){
  ens_mems = c('01', '04', '05', '06', '07', '09', '11', '12', '13')
} else {
  ens_mems = c('01', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '15')
}

i = 1
ens_mem = ens_mems[i]
risk_data = read.csv(paste0('/data/users/ldawkins/UKCR/DataForSchoolApp/Impact/',warming_level,'/ens_',ens_mem,'_output__thresh',threshold,'_opttemp.csv'))
lon <- exposures$long_true
lat <- exposures$lat_true
EAI <- risk_data$eai_exp
archetype <- exposures$impf_HS
school_no <- seq(1,length(lon),1)

df = data.frame(lon=as.vector(lon),lat=as.vector(lat),imp=as.vector(EAI),atype=archetype,sch_no=school_no,member=as.factor(ens_mem))
for(i in 2:length(ens_mems)){
  ens_mem = ens_mems[i]
  risk_data = read.csv(paste0('/data/users/ldawkins/UKCR/DataForSchoolApp/Impact/',warming_level,'/ens_',ens_mem,'_output__thresh',threshold,'_opttemp.csv'))
  EAI <- risk_data$eai_exp
  df_new = data.frame(lon=as.vector(lon),lat=as.vector(lat),imp=as.vector(EAI),atype=archetype,sch_no=school_no,member=as.factor(ens_mem))
  df = rbind(df,df_new)
}
df$atype = as.factor(df$atype)
df <- merge(df,archetype_all,by='atype',sort=F)
# 
# Plots to explore the relationships in the data 
# 
#df_test = df[df$member == '05',]
# 
#ggplot(df_test) + geom_point(aes(x=lon,y=imp,col=era)) + facet_wrap(~region)
#ggplot(df_test) + geom_point(aes(x=lat,y=imp,col=era)) + facet_wrap(~region)
# 
#ggplot(df_test) + geom_point(aes(x=lon,y=imp,col=region)) + facet_wrap(~era)
#ggplot(df_test) + geom_point(aes(x=lat,y=imp,col=region)) + facet_wrap(~era)
# 
#ggplot(df_test) + geom_point(aes(x=lon,y=imp,col=type)) + facet_wrap(~era)
#ggplot(df_test) + geom_point(aes(x=lat,y=imp,col=type)) + facet_wrap(~era)
# 

# ggplot(df) + geom_point(aes(x=lon,y=lat,col=imp),cex=1) + facet_wrap(~member)
# ggplot(df) + geom_point(aes(x=lat,y=imp,col=member))
# 
#ggplot(df) + geom_histogram(aes(x=log(imp+1))) + facet_wrap(~member)
#ggplot(df) + geom_histogram(aes(x=sqrt(imp+0.1))) + facet_wrap(~member)
# 

df$imp = log(df$imp+1,base=10)
# Plots to explore the relationships in the data 
# ggplot(df) + geom_histogram(aes(x=imp)) + facet_wrap(~member)
# ggplot(df) + geom_point(aes(x=lon,y=imp,col=region)) + facet_wrap(~member)
# ggplot(df) + geom_point(aes(x=lat,y=imp,col=region)) + facet_wrap(~member)

mod_mean = imp ~ ti(lon, lat, k=16,bs="tp") + ti(member,bs='re') + atype
mod_sd = ~ s(lon,k=8) + s(lat,k=8) 
mod = list(mod_mean,mod_sd)

fit = gam(mod, data=df, family='gaulss', method='REML')
save(fit,file = paste0('/data/users/ldawkins/UKCR/fit_schools_',warming_level,'_gaulss_thresh',threshold,'_4.RData'))

