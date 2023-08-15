# Plots for school paper in R


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


# Exposure maps in Fig 3 

library(maps)

nickColors <- function(n, h = c(120,400), l = c(.40,.70), s = c(.8,1), alpha = 1){
  require(colorspace)
  require(scales)
  return (alpha(hex(HLS(seq(h[1],h[2],length.out = n), seq(l[1],l[2],length.out = n), seq(s[1],s[2],length.out=n))), alpha))
}


library(openxlsx)
vuln_functions <- read.xlsx('/home/h01/ldawkins/Documents/UKCR/SchoolApp/FinalCode/Operative35_temperature_relationships_max.xlsx')


arch = vuln_functions[,1]
uni_arch = unique(arch)


era = c('Pre 1919', "Interwar",'1945-66', "1967-76",'Post 1976')
region = c('Thames Valley', 'South Eastern', 'Southern', 'South Western', 'Severn Valley', 'Midland', 'West Pennines', 'North Western', 'Borders', 'North Eastern', 'East Pennines', 'East Anglia', 'Wales')  #seq(1,13,1)
type = c('Primary','Secondary')

archetypes = array(dim=c(5,13,2))
for(i in 1:5){
  for(j in 1:13){
    for(k in 1:2){
      archetypes[i,j,k] = paste(era[i], region[j], type[k])
    }
  }
}
archetypes = as.vector(archetypes)

exposures = read.csv('/data/users/ldawkins/UKCR/DataForSchoolApp/exp_info.csv')

par(mfrow=c(1,2))

i = 1
ind = which(exposures$impf_HS >= (i*5 -4) & exposures$impf_HS <= (i*5))
sub = exposures[ind,]
colsuse = rep(nickColors(5),13)
plot(sub$long_true,sub$lat_true, pch=19,col=colsuse[sub$impf_HS],xlim=c(-6,2),ylim=c(50,56),cex=0.4,xlab='Longitude',ylab='Latitude')
map('world',add=T)
legend('topright',legend=c('Primary: Pre 1919', 'Primary: Interwar', 'Primary: 1945-66', 'Primary: 1967-76', 'Primary: Post 1976'),col=colsuse[1:5],pch=19,cex=1)
for(i in 2:13){
  ind = which(exposures$impf_HS >= (i*5 -4) & exposures$impf_HS <= (i*5))
  sub = exposures[ind,]
  points(sub$long_true,sub$lat_true, pch=19,col=colsuse[sub$impf_HS],cex=0.4)
}

i = 1
ind = which(exposures$impf_HS >= 65+(i*5 -4) & exposures$impf_HS <= 65+(i*5))
sub = exposures[ind,]
colsuse = rep(nickColors(5),26)
plot(sub$long_true,sub$lat_true, pch=19,col=colsuse[sub$impf_HS],xlim=c(-6,2),ylim=c(50,56),cex=0.4,xlab='Longitude',ylab='Latitude')
map('world',add=T)
legend('topright',legend=c('Secondary: Pre 1919', 'Secondary: Interwar', 'Secondary: 1945-66', 'Secondary: 1967-76', 'Secondary: Post 1976'),col=colsuse[1:5],pch=19,cex=1)
for(i in 2:13){
  ind = which(exposures$impf_HS >= 65+(i*5 -4) & exposures$impf_HS <= 65+(i*5))
  sub = exposures[ind,]
  points(sub$long_true,sub$lat_true, pch=19,col=colsuse[sub$impf_HS],cex=0.4)
}



#Figure 4 - vulnerability function construction

# plot data for North West pre-1980 + Thames Valley 1967-76


era = c('_00_', '_01_','_02_', '_03_','_04_')
region = paste0('Z',  sprintf("%02d",1:13))
type = c('P','S')


archetypes = array(dim=c(5,13,2))
for(i in 1:5){
  for(j in 1:13){
    for(k in 1:2){
      archetypes[i,j,k] = paste(era[i], region[j], type[k])
    }
  }
}
archetypes = data.frame(atypeno = seq(1,130,1), labels=as.vector(archetypes))

at = int = slop = array(dim=2)
at[1] = which(archetypes$labels == '_00_ Z05 S')
at[2] = which(archetypes$labels == '_03_ Z01 P')

df_plot <- list()

for(i in 1:2){
  df_all <- data.frame(day=NA,outdoor=NA,indoor=NA,pred=NA)
  for(warming_level in c('current','2deg','4deg')) {
    

    labs = as.character(archetypes$labels[which(archetypes$atypeno == at[i])])
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
    file = read.csv(paste0('/data/users/hadkb/118091-MetOffice/',labs_region,'_TRY_',simtype,'50_/',labs_type,'_',labs_region,labs_era,'Nat/eplusout.csv'))

    print(paste0('/data/users/hadkb/118091-MetOffice/',labs_region,'_TRY_',simtype,'50_/',labs_type,'_',labs_region,labs_era,'Nat/eplusout.csv'))
    #[/data/users/hadkb/118091-MetOffice/Z01_TRY_2020High50_/P_Z01_04_Nat/eplusout.csv, /data/users/hadkb/118091-MetOffice/Z01_TRY_2050Medium50_/P_Z01_04_Nat/eplusout.csv, /data/users/hadkb/118091-MetOffice/Z01_TRY_2080Medium50_/P_Z01_04_Nat/eplusout.csv]

    df <- data.frame(day = seq(1,365,1),outdoor = na.omit(file$SOUTH.Zone.Outdoor.Air.Drybulb.Temperature..C..Daily.))

    file$day = rep(1:365, each=24)

    daily_max_in_op_time = array(dim=365)
    for(d in 1:365){
      ind = which(file$day == d)
      indh = which(file$SOUTH.Zone.People.Occupant.Count....Hourly.[ind]>0)
      #indh <- 10:16 #8:17
      daily_max_in_op_time[d] = max(file$SOUTH.Zone.Operative.Temperature..C..Hourly.[ind[indh]])
    }

    df$indoor = daily_max_in_op_time

    df <- df[-which(df$indoor == -Inf),]

    params = read.xlsx('/net/home/h01/ldawkins/Documents/UKCR/SchoolApp/FinalCode/Operative35_temperature_relationship_parameters_max.xlsx')
    pars = params[at[i],]
    int[i] = pars$Intercept + pars$`direction[T.SOUTH]`
    slop[i] = pars$outdoor_temp + pars$`outdoor_temp:direction[T.SOUTH]`
    df$pred = int[i] + slop[i]*df$outdoor

    #df <- df[-which(df$outdoor < 12),]

    df_all <- rbind(df_all,df)
  }
df_plot[[i]] <- df_all
}

df1 = df_plot[[1]]
df2 = df1[-which(df1$outdoor < 12),]

df3 = df_plot[[2]]
df4 = df3[-which(df3$outdoor < 12),]

params2 = read.xlsx('/net/home/h01/ldawkins/Documents/UKCR/SchoolApp/FinalCode/Operative26_temperature_relationship_parameters_max.xlsx')
thresh26_1 = params2$outdoor.threshold[at[1]]
thresh35_1 = params$outdoor.threshold[at[1]]
thresh26_2 = params2$outdoor.threshold[at[2]]
thresh35_2 = params$outdoor.threshold[at[2]]


g1 = ggplot(df1) + geom_point(aes(x = day, y = outdoor, color = 'Outdoor daily mean \n air temperature',shape='Outdoor daily mean \n air temperature')) +
  geom_point(aes(x = day, y = indoor, color = 'Indoor daily maximum \n operative temperature',shape='Indoor daily maximum \n operative temperature')) +
  geom_hline(aes(yintercept=26,linetype='Indoor temperature \n threshold = 26C')) +
  geom_hline(aes(yintercept=35,linetype='Indoor temperature \n threshold = 35C')) +
  #geom_line(aes(x = day, y = outdoor, color = 'Outdoor daily mean \n air temperature')) + 
  #geom_line(aes(x = day, y = indoor, color = 'Indoor daily maximum \n operative temperature')) +
  scale_linetype_manual("Thresholds",values=c('Indoor temperature \n threshold = 26C'=2,'Indoor temperature \n threshold = 35C'=3)) +
  theme_bw() + ylim(-2,45) +
  labs(y = 'Temperature (degrees C)', x = 'Day of the year', colour = 'Variables', shape = 'Variables') + ggtitle('(a) North-west, Pre-1918, Secondary')


g2 = ggplot(df3) + geom_point(aes(x = day, y = outdoor, color = 'Outdoor daily mean \n air temperature',shape = 'Outdoor daily mean \n air temperature')) +
  geom_point(aes(x = day, y = indoor, color = 'Indoor daily maximum \n operative temperature', shape = 'Indoor daily maximum \n operative temperature')) +
  geom_hline(aes(yintercept=26,linetype='Indoor temperature \n threshold = 26C')) +
  geom_hline(aes(yintercept=35,linetype='Indoor temperature \n threshold = 35C')) +
  #geom_line(aes(x = day, y = outdoor, color = 'Outdoor daily mean \n air temperature')) + 
  #geom_line(aes(x = day, y = indoor, color = 'Indoor daily maximum \n operative temperature')) +
  scale_linetype_manual("Thresholds",values=c('Indoor temperature \n threshold = 26C'=2,'Indoor temperature \n threshold = 35C'=3)) +
  theme_bw() + ylim(-2,45) +
  labs(y = 'Temperature (degrees C)', x = 'Day of the year', colour = 'Variables', shape = 'Variables') + ggtitle('(e) Thames Valley, 1967-76, Primary')


g3 = ggplot() + geom_point(aes(x = sort(df2$indoor), y =  sort(df2$pred)), pch=19) +
  geom_abline(aes(intercept = 0, slope = 1,linetype='Line of y=x'))  +
  theme_bw() + xlim(20,40) + ylim(20,40) +
  labs(y = 'Predicted indoor daily maximum \n operive temperature quantiles (degrees C)', x = 'True indoor daily maximum \n operative temperature quantiles (degrees C)', linetype='Lines')  +
  ggtitle('(b)')

g4 = ggplot() + geom_point(aes(x = sort(df4$indoor), y =  sort(df4$pred)), pch=19) +
  geom_abline(aes(intercept = 0, slope = 1,linetype='Line of y=x'))  +
  theme_bw() + xlim(20,40) + ylim(20,40) +
  labs(y = 'Predicted indoor daily maximum \n operive temperature quantiles (degrees C)', x = 'True indoor daily maximum \n operative temperature quantiles (degrees C)', linetype='Lines')  +
  ggtitle('(f)')

g5 = ggplot(df2) + geom_point(aes(x = outdoor, y = indoor), pch=19, col='grey') +
  geom_abline(aes(intercept = int[1], slope = slop[1], color = 'Linear regression model'),lwd=2) +
  geom_hline(aes(yintercept=26,linetype='Indoor temperature \n threshold = 26C')) +
  geom_hline(aes(yintercept=35,linetype='Indoor temperature \n threshold = 35C')) +
  scale_linetype_manual("Thresholds",values=c('Indoor temperature \n threshold = 26C'=2,'Indoor temperature \n threshold = 35C'=3)) +
  theme_bw() + xlim(10,40) + ylim(15,45) +
  labs(y = 'Indoor daily maximum \n operative temperature (degrees C)', x = 'Outdoor daily mean \n air temperature (degrees C)')  +
  ggtitle('(c)')

g6 = ggplot(df4) + geom_point(aes(x = outdoor, y = indoor), pch=19, col='grey') +
  geom_abline(aes(intercept = int[2], slope = slop[2], color = 'Linear regression model'),lwd=2) +
  geom_hline(aes(yintercept=26,linetype='Indoor temperature \n threshold = 26C')) +
  geom_hline(aes(yintercept=35,linetype='Indoor temperature \n threshold = 35C')) +
  scale_linetype_manual("Thresholds",values=c('Indoor temperature \n threshold = 26C'=2,'Indoor temperature \n threshold = 35C'=3)) +
  theme_bw() + xlim(10,40) + ylim(15,45) +
  labs(y = 'Indoor daily maximum \n operative temperature (degrees C)', x = 'Outdoor daily mean \n air temperature (degrees C)')  +
  ggtitle('(g)')


g7 = ggplot() + geom_line(aes(x=c(10,thresh26_1,thresh26_1,40),y=c(0,0,1,1), color = 'Impact function for indoor \n temperature \n threshold = 26C'),lwd=1) +
  geom_line(aes(x=c(10,thresh35_1,thresh35_1,40),y=c(0,0,1,1), color = 'Impact function for indoor \n temperature \n threshold = 35C'),lwd=2) +
  theme_bw() + xlim(10,40) +
  labs(y = 'Impact', x = 'Outdoor daily mean \n air temperature (degrees C)',colour='Impact functions')  + ggtitle('(d)')

g8 = ggplot() + geom_line(aes(x=c(10,thresh26_2,thresh26_2,40),y=c(0,0,1,1), color = 'Impact function for indoor \n temperature \n threshold = 26C'),lwd=1) +
  geom_line(aes(x=c(10,thresh35_2,thresh35_2,40),y=c(0,0,1,1), color = 'Impact function for indoor \n temperature \n threshold = 35C'),lwd=2) +
  theme_bw() + xlim(10,40) +
  labs(y = 'Impact', x = 'Outdoor daily mean \n air temperature (degrees C)',colour='Impact functions')  + ggtitle('(h)')



g = grid.arrange(g1,g2,g5,g6,g3,g4,g7,g8,ncol=2)

ggsave(g,filename = paste0('/data/users/ldawkins/UKCR/OutputPlots/Plotsforpaper/SchoolPaper/Figure2.png'),width=15,height=15)






#===========================
# QQplots - in Sup Material


era = c('Pre 1919', "Interwar",'1945-66', "1967-76",'Post 1976')
region = c('Thames Valley', 'South Eastern', 'Southern', 'South Western', 'Severn Valley', 'Midland', 'West Pennines', 'North Western', 'Borders', 'North Eastern', 'East Pennines', 'East Anglia', 'Wales')  #seq(1,13,1)
type = c('Primary','Secondary')


eras <- rep(era,13*2)
regions <- rep(rep(region,each=5),2)
types <- rep(type,each=5*13)

archetype_all <- data.frame(atype=seq(1,130,1),era = as.factor(eras),region=as.factor(regions),type=as.factor(types))


warming_levels= c('current','2deg','4deg')

for(threshold in c('26','35')){
  for(warming_level in warming_levels){
    
    #warming_level = warming_levels[as.numeric(wl)]
    print(threshold)
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
    
    df$imp = log(df$imp+1,base=10)
    
    mod_mean = imp ~ ti(lon, lat, k=16,bs="tp") + ti(member,bs='re') + atype 
    mod_sd = ~ s(lon,k=8) + s(lat,k=8)
    mod = list(mod_mean,mod_sd)
    

    load(paste0('/data/users/ldawkins/UKCR/fit_schools_',warming_level,'_gaulss_thresh',threshold,'_4.RData'))
    
    # initial check of output
    # summary(fit)
    # newdata = df[,c('lon','lat','member','atype')]
    # pred = predict(fit,newdata,type = 'response')
    # #plot(df$imp,pred[,1])
    # #abline(0,1,col='blue')
    # 
    # plot(10^df$imp -1, 10^pred[,1] -1)
    # abline(0,1,col='blue')
    
    # Simulate posterior predictive distribution
    
    ensmean = df %>% group_by(sch_no) %>% summarise_at(vars(imp),list(ensmean=mean))
    df = merge(df,ensmean,by='sch_no',sort=F)
    
    # explore output
    model = fit
    n.sims <- 1000
    betas <- rmvn(n.sims,coef(model),model$Vp)
    
    newd <- df[df$member=='04',]
    newd = newd[order(newd$sch_no),]
    
    num.knots = c(123 + 15^2 + length(ens_mems), 1 + 7 + 7) #c(123 + 7 + 7 + 15^2 + length(ens_mems), 1 + 7 + 7 + 15^2)
    # 1000 replicate param. vectors
    betas <- rmvn(n.sims,coef(model),model$Vc) ## Vc version = Vp corrected covariance matrix for smoothing parameter uncertainty
    
    # Make predictions
    X <- predict(model,newdata = newd,type = "lpmatrix", exclude = "ti(member)") ## model matrix
    
    vcomp = gam.vcomp(model)
    rownames = row.names(vcomp)
    ind = which(rownames == 'ti(member)')
    RE_sd = vcomp[ind, 1]
    
    msamp = rnorm(n.sims,0,RE_sd)
    msamp_mat = matrix(msamp, nrow=dim(newd)[1], ncol=n.sims, byrow=TRUE)
    
    #ssamp = rnorm(n.sims,0,0.361750767)
    #ssamp_mat = matrix(ssamp, nrow=dim(newd)[1], ncol=n.sims, byrow=TRUE)
    
    Mean <- X[, 1:num.knots[1]] %*% t(betas[, 1:num.knots[1]]) + msamp_mat
    LPsd <- X[,(num.knots[1]+1):(num.knots[1]+num.knots[2])] %*% t(betas[,(num.knots[1]+1):(num.knots[1]+num.knots[2])])  #+ ssamp_mat
    Sd <- exp(LPsd) + 0.01
    
    # simulate from predictive distribution (noting that the conditional is Log-Normal)
    preds <- matrix(ncol = nrow(newd), nrow = n.sims)
    for (simnum in 1:n.sims) {
      for (rownum in 1:nrow(newd)) {
        preds[simnum,rownum] <- rnorm(1, mean = Mean[rownum,simnum], sd = Sd[rownum,simnum]) 
      }
    }
    preds[preds>quantile(preds,0.9999,na.rm = T)]<-quantile(preds,0.9999,na.rm = T)
    
    preds_df = data.frame(preds)
    colnames(preds_df) = paste0('school',1:19158)
    write.csv(preds_df, paste0('/data/users/ldawkins/UKCR/DataForSchoolApp/Impact/',warming_level,'/GAM_SAMPLES_ens_',ens_mem,'_output__thresh',threshold,'_opttemp.csv'))
    
    
    preds_risk = 10^preds - 1 
    preds_risk[preds_risk<0]<-0
    predmean = apply(preds_risk,2,mean,na.rm=T)
    predLB = apply(preds_risk,2,function(x){quantile(x,0.025,na.rm=T)})
    predUB = apply(preds_risk,2,function(x){quantile(x,0.975,na.rm=T)})
    
    #predmedian = apply(preds,2,median,na.rm=T)
    
    # plot(ensmean$ensmean, log(predmean+1,base=10))
    # abline(0,1)
    # 
    # plot(10^ensmean$ensmean - 1, predmean)
    # abline(0,1)
    
    df$imp_risk = 10^df$imp - 1
    
    
    # # QQ plot ***
    
    data = cbind(newd[,c('lon','lat','ensmean')],t(preds_risk))
    data$ensmean = 10^data$ensmean - 1
    
    EAI_quant = apply(data[,c(4:1003)],2,function(x) quantile(x,seq(0,1,length=500),na.rm=T)) %>%
      apply(1,quantile,c(0.025,0.5,0.975))%>%melt(varnames=c('quantile','index'))%>%
      spread(quantile,value)%>%mutate(true=quantile(data$ensmean,seq(0,1,length=500)))%>%
      mutate(Ensemble= factor('ensemble mean'))
    
    for(j in 1:length(ens_mems)){
      EAI_quant_new = apply(data[,c(4:1003)],2,function(x) quantile(x,seq(0,1,length=500),na.rm=T)) %>%
        apply(1,quantile,c(0.025,0.5,0.975))%>%melt(varnames=c('quantile','index'))%>%
        spread(quantile,value)%>%mutate(true=quantile(df$imp_risk[df$member == ens_mems[j]],seq(0,1,length=500)))%>%
        mutate(Ensemble= factor(paste0('member ',ens_mems[j])))
      EAI_quant = rbind(EAI_quant_new,EAI_quant)
    }
    
    data$predmean = predmean
    data$diff = data$predmean - data$ensmean
    
    
    nens = length(ens_mems)
    g1 = ggplot(data=EAI_quant,aes(x=true,y=`50%`,group=Ensemble)) +
      geom_ribbon(aes(ymin=`2.5%`,ymax=`97.5%`,fill=Ensemble),alpha=c(rep(0.1,nens*500),rep(0.3,500))) +
      geom_point(aes(x=true,y=`50%`,col=Ensemble),size=c(rep(0.5,nens*500),rep(2,500))) + ggtitle('(a)') + xlab('Ensemble member (true) risk') +
      geom_abline(intercept=0,slope=1,col="#22211d")+
      ylab('GAM sampled (modelled) risk (mean and 95% CI)')+ xlim(0,max(EAI_quant$`97.5%`)) + ylim(0,max(EAI_quant$`97.5%`))+
      scale_fill_viridis_d() + scale_color_viridis_d()
    
    g2 = ggplot(data) + geom_point(aes(x=ensmean,y=predmean)) + ggtitle('(b)') + #xlim(0,4) + ylim(0,4)+
      geom_abline(intercept=0,slope=1,col="#22211d")+
      xlab('Ensemble mean risk') + ylab('GAM sample mean risk') #+ geom_point(aes(x=ensmean,y=predUB),col='red') + geom_point(aes(x=ensmean,y=predLB),col='blue') 
    
    UK <- map_data(map = "world", region = "UK")
    g3 = ggplot(data) +
      geom_point(aes(x=lon,y=lat,color=diff),size=0.6) + xlab('Longitdue') + ylab('Latitude') + labs(color='Bias in GAM \n mean risk') +
      scale_colour_gradient2(low='blue',mid='white',high='red') +
      geom_polygon(data = UK, aes(x = long, y = lat, group = group),fill = NA, color = 'black') +
      coord_map() + ggtitle('(c)') + geom_point(aes(x=0.1720095,y=51.45396),pch=1,col='black',size=0.6)
    
    if(threshold == '26'){
      g1 = g1 + ylim(0,110)
      g2 = g2 + ylim(0,110)
    }
    
    g=grid.arrange(g1,g2,g3,ncol=3)
    
    ggsave(g,filename = paste0('/data/users/ldawkins/UKCR/OutputPlots/Plotsforpaper/SchoolPaper/QQplot_',warming_level,'_',threshold,'.png'),width=15,height=5)
  
    #Histograms
    # sort data for grid cell histograms
    data = cbind(newd[,c('lon','lat','ensmean')],t(preds))
    nens = length(ens_mems)
    data_use = data
    #find grid cell near London
    ind = which(data_use$lat>51 & data_use$lat<52 & data_use$lon>0.1 & data_use$lon<0.2 & newd$era == 'Post 1976')[1]
    plotdatag1=data.frame(x=as.numeric(data_use[ind,c(4:1003)]))
    addlinesdata = df[df$lon==data_use$lon[ind] & df$lat==data_use$lat[ind],]
    addlinesdata <- rbind(c(NA,NA,NA,NA,addlinesdata$ensmean[1],NA,NA,NA,NA,NA),addlinesdata)
    addlinesdata$ensemble = c('ensemble mean',rep('individual members',nens))
    
    #sort data for agg histograms
    EAI_agg_true = df[,c('imp','member')] %>% group_by(member) %>% summarise(AEAI=sum(imp))
    EAI_agg_true = rbind(c(NA,sum(data$ensmean)),EAI_agg_true)
    EAI_agg_true$ensemble = c('ensemble mean',rep('individual members',nens))
    EAI_agg_sim = apply(data[,c(4:1003)],2,sum,na.rm=T)
    plotdatag3=data.frame(x=as.numeric(EAI_agg_sim))
    
    
    g1=ggplot()+geom_histogram(data=plotdatag1,aes(x=x),bins=15,col='black',fill='magenta') +
      geom_vline(data=addlinesdata, aes(xintercept = imp, col=ensemble),lty=c(1,rep(2,nens)),lwd=c(2,rep(0.5,nens)),col='blue') +
      #scale_color_manual(values=c('black','blue'))+
      geom_vline(aes(xintercept=c(quantile(plotdatag1$x,0.025,na.rm=T),mean(plotdatag1$x),quantile(plotdatag1$x,0.975,na.rm=T))),col='grey',lty=c(2,1,2),lwd=2)+
      xlab('Risk (Number of days) \n grid cell (lon: 0.1720, lat: 51.454)') + ylab('Frequency') + ggtitle('(a)')  +
      theme(legend.position="none")
    
    g2=ggplot()+geom_histogram(data=plotdatag3,aes(x=x),bins=15,col='black') +
      geom_vline(data=EAI_agg_true, aes(xintercept = AEAI, col=ensemble),lty=c(1,rep(2,nens)),lwd=c(2,rep(0.5,nens)),col='blue') +
      #scale_color_manual(values=c('black','blue'))+
      geom_vline(aes(xintercept=c(quantile(plotdatag3$x,0.025,na.rm=T),mean(plotdatag3$x),quantile(plotdatag3$x,0.975,na.rm=T))),col='grey',lty=c(2,1,2),lwd=2)+
      xlab('Spatially aggregated risk \n (Number of days)') + ylab('Frequency') + ggtitle('(b)')  +
      theme(legend.position="none")
    
    g = grid.arrange(g1,g2,ncol=2)
    ggsave(g,filename = paste0('/data/users/ldawkins/UKCR/OutputPlots/Plotsforpaper/SchoolPaper/Histogram_',warming_level,'_',threshold,'.png'),width=10,height=5)
    
  }
}











