# Make CIBSE equivalent data 

region = c('Thames Valley', 'South Eastern', 'Southern', 'South Western', 'Severn Valley', 'Midland', 'West Pennines', 'North Western', 'Borders', 'North Eastern', 'East Pennines', 'East Anglia', 'Wales')  #seq(1,13,1)


warming_level = 'current'
exposures = read.csv('/data/users/ldawkins/test.csv')

era = c('_00_', '_01_','_02_', '_03_','_04_')
region = paste0('Z',  sprintf("%02d",1:13))
type = c('P','S')


archetypes = array(dim=c(5,13,2))
for(i in 1:5){
  for(j in 1:13){
    for(k in 1:2){
      archetypes[i,j,k] = region[j]
    }
  }
}
archetypes = data.frame(atypeno = seq(1,130,1), labels=as.vector(archetypes))

lons = lats = array(dim=13)

at = 1
ind = which(archetypes$labels == paste0('Z',  sprintf("%02d",at)))
lon = exposures$longitude[which(exposures$impf_HS %in% ind)]
lat = exposures$latitude[which(exposures$impf_HS %in% ind)]
plot(lon,lat,col=at,xlim=c(-6,2),ylim=c(50,56))
x = min(lon) + 0.5*(abs(max(lon) - min(lon)))
y =  min(lat) + 0.5*(abs(max(lat) - min(lat)))
points(x,y,cex=2,pch='x')
lons[at] = x
lats[at] = y

for(at in 2:13){
  ind = which(archetypes$labels == paste0('Z',  sprintf("%02d",at)))
  lon = exposures$longitude[which(exposures$impf_HS %in% ind)]
  lat = exposures$latitude[which(exposures$impf_HS %in% ind)]
  points(lon,lat,col=at)
  x = min(lon) + 0.5*(abs(max(lon) - min(lon)))
  y =  min(lat) + 0.5*(abs(max(lat) - min(lat)))
  points(x,y,cex=2,pch='x')
  lons[at] = x
  lats[at] = y
}

df = data.frame(long = lons, lat = lats)

# find indices in UKCP data 
library(RNetCDF)
library(fields)

nc = open.nc('/data/users/ldawkins/UKCR/DataForSchoolApp/UKCP_BC/Timeseries_01_tas_1998_2017_BC.nc')
lon_ukcp = var.get.nc(nc,'longitude')
lat_ukcp = var.get.nc(nc,'latitude')
close.nc(nc)

lons_use = lats_use = array(dim=13)
ids_all = array(dim=c(13,2))

for(j in 1:13){
  dists = array(dim=dim(lon_ukcp))
  
  for(i in 1:dim(lon_ukcp)[1]){
    for(k in 1:dim(lon_ukcp)[2]){
      dists[i,k] = rdist.earth(matrix(c(lons[j],lats[j]),ncol=2),matrix(c(lon_ukcp[i,k],lat_ukcp[i,k]),ncol=2))
    }
  }
  
  ids = which(dists==min(dists),arr.ind = T)
  lons_use[j] = lon_ukcp[ids[1],ids[2]]
  lats_use[j] = lat_ukcp[ids[1],ids[2]]

  ids_all[j,] <- ids
}


at = 1
ind = which(archetypes$labels == paste0('Z',  sprintf("%02d",at)))
lon = exposures$longitude[which(exposures$impf_HS %in% ind)]
print(length(lon))
llon = length(lon)
lat = exposures$latitude[which(exposures$impf_HS %in% ind)]
plot(lon,lat,col=at,xlim=c(-6,2),ylim=c(50,56))
points(lons_use[at],lats_use[at],cex=2,pch='x')
for(at in 2:13){
  ind = which(archetypes$labels == paste0('Z',  sprintf("%02d",at)))
  lon = exposures$longitude[which(exposures$impf_HS %in% ind)]
  print(length(lon))
  llon = llon + length(lon)
  lat = exposures$latitude[which(exposures$impf_HS %in% ind)]
  points(lon,lat,col=at)
  points(lons_use[at],lats_use[at],cex=2,pch='x')
}


##### make excel
exposures = read.csv('/data/users/ldawkins/UKCR/DataForSchoolApp/exp_info.csv')

#for(warming_level in c('current','2deg','4deg')){
warming_level = 'current'
  if(warming_level == '4deg'){
    ens_mems = c('01', '04', '05', '06', '07', '09', '11', '12', '13')
  } else {
    ens_mems = c('01', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '15')
  }
  for(mem in ens_mems){
  
  # sheet 1: lon and lats

  S1 = data.frame('centroid_id'=seq(1,dim(exposures)[1],1)-1, 'longitude'=exposures$long_true, 'latitude'=exposures$lat_true)
  
  # sheet 2: daily mean temp from weather files 
  
  centroid_id=seq(1,dim(exposures)[1],1)-1
  impf = exposures$impf_HS
  
  if(warming_level=='4deg'){
    ndays = 330*17
  } else {
    ndays = 330*20
  }
  
  hazard_dat = array(dim=c(length(centroid_id),ndays)) 

  era = c('_00_', '_01_','_02_', '_03_','_04_')
  region = paste0('Z',  sprintf("%02d",1:13))
  type = c('P','S')


  archetypes2 = array(dim=c(5,13,2))
  for(i in 1:5){
    for(j in 1:13){
      for(k in 1:2){
        archetypes2[i,j,k] = paste(era[i], region[j], type[k])
      }
    }
  }
  archetypes2 = data.frame(atypeno = seq(1,130,1), labels=as.vector(archetypes2))
  
  
  if(warming_level == 'current'){
    nc = open.nc(paste0('/data/users/ldawkins/UKCR/DataForSchoolApp/UKCP_BC/Timeseries_',mem,'_tas_1998_2017_BC.nc'))
  } else if(warming_level == '2deg') {
    nc = open.nc(list.files(path = '/data/users/ldawkins/UKCR/DataForSchoolApp/UKCP_BC/',pattern=glob2rx(paste0('*_',mem,'_*2deg*')),full.names=T))
  } else{
    nc = open.nc(list.files(path = '/data/users/ldawkins/UKCR/DataForSchoolApp/UKCP_BC/',pattern=glob2rx(paste0('*_',mem,'_*4deg*')),full.names=T))
  }
  
  tas = var.get.nc(nc,'tas')
  month = var.get.nc(nc,'month_number') 
  
  #hazard_dat = array(dim=dim(tas))
  
  
  for(at in 1:13){
    labs = as.character(archetypes2$labels[which(archetypes2$atypeno == at)])
    labs_era = strsplit(labs,' ')[[1]][1]
    labs_region = strsplit(labs,' ')[[1]][2]
    labs_type = strsplit(labs,' ')[[1]][3]

    if(warming_level == 'current'){
      simtype = '2020High'
    } else if(warming_level == '2deg') {
      simtype = '2050Medium'
    } else if(warming_level == '4deg') {
      simtype = '2080Medium'
    }

    if(labs_region == 'Z03' & warming_level=='2deg'){
      simtype = '2050High'
    }
    # 
    #file = read.csv(paste0('/data/users/hadkb/118091-MetOffice/',labs_region,'_TRY_',simtype,'50_/',labs_type,'_',labs_region,labs_era,'Nat/eplusout.csv'))
    #southout = na.omit(file$SOUTH.Zone.Outdoor.Air.Drybulb.Temperature..C..Daily.)
    #southout[213:(213+30)] <- NA
    #southout = na.omit(southout)
    
    
    southout = tas[ids_all[at,1],ids_all[at,2],]
    southout[month == 8] <- NA
    southout = na.omit(southout)
    
    #ind = which(impf == at)
    ind = which(archetypes$labels == paste0('Z',  sprintf("%02d",at)))
    inds = which(impf %in% ind)
    for(i in 1:length(inds))
      hazard_dat[inds[i], ] = southout
  }
  
  S2= data.frame(centroid_id,hazard_dat)
  
  names(S2) = c('centroid_id/event_id',sprintf('%s',seq(1:ndays)))
  
  
  # sheet 3
  
  S3 = data.frame('event_id'=seq(1:ndays),'event_name'=paste0('event',sprintf('%03d',seq(1:ndays))),'frequency'=rep(1/(ndays/195),ndays),'orig_event_flag'=1,'event_date'=rep(as.numeric(substr(simtype, 1, 4)),ndays))
  
  require(openxlsx)
  list_of_datasets <- list("centroids" = S1, "hazard_intensity" = S2, "hazard_frequency" = S3)
  write.xlsx(list_of_datasets, file = paste0('/data/users/ldawkins/UKCR/DataForSchoolApp/CIBSE/HazardCIBSE_rep_',warming_level,'_member',mem,'.xlsx'))
}#}



