# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 16:01:00 2020

@author: ATeklesadik
"""
from mainpipeline import Activetyphoon
import sys
import os
import pandas as pd
import feedparser
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from datetime import datetime
from datetime import timedelta
import smtplib
from smtplib import SMTP_SSL as SMTP
import geopandas as gpd
import fiona
from ftplib import FTP
import shutil
from os.path import relpath
import re
import zipfile
import os.path
from os.path import relpath, abspath
from os import listdir
from os.path import isfile, join
from pybufrkit.decoder import Decoder
from pybufrkit.renderer import FlatTextRenderer
from sys import platform
from io import StringIO
import numpy as np
from bs4 import BeautifulSoup
import subprocess
from geopandas.tools import sjoin
import geopandas as gpd
import xarray as xr
from pathlib import Path
from liquer import *
from liquer.context import Context

#decoder = Decoder() # TODO: Does not seem to be used
##path='C:/Users/ATeklesadik/OneDrive - Rode Kruis/Documents/documents/Typhoon-Impact-based-forecasting-model/'
##Path='home/fbf/'

@first_command
def home_path():
    return str(Path(__file__).parent.absolute())  # os.path.split(abspath(__file__))[0]

#%%
sys.path.insert(0, os.path.join(home_path(),'lib'))
os.chdir(path)

from settings import fTP_LOGIN, fTP_PASSWORD, uCL_USERNAME, uCL_PASSWORD
from secrets import *
 
from climada.hazard import Centroids, TropCyclone,TCTracks
from climada.hazard.tc_tracks import estimate_roci,estimate_rmw
from climada.hazard.tc_tracks_forecast import TCForecast
from utility_fun import track_data_clean,Rainfall_data,Check_for_active_typhoon,Sendemail,ucl_data

#%% check for active typhoons
@first_command
def active_typhoons(remote_dir=None, typhoon=None, context=None):
    if context is None:
        context=Context()

    if typhoon in (None,""):
        print('---------------------check for active typhoons---------------------------------')
        print(str(datetime.now()))
        context.info("check for active typhoons")
        Activetyphoon=Check_for_active_typhoon.check_active_typhoon()
    else:
        context.info(f"selected typhoon: {typhoon}")
        Activetyphoon=[typhoon]
    #forecast_available_fornew_typhoon= False#'False'

    if Activetyphoon==[]:
        if remote_dir in (None,""):
            remote_dir='20210421120000' #for downloading test data otherwise set it to None
        Activetyphoon=['SURIGAE']  #name of typhoon for test
        context.info(f"No active typhoon, using {Activetyphoon} at {remote_dir}")
    else:
        #remote_dir=None #for downloading real time data
        if remote_dir == "":
            remote_dir=None #'20210518120000' #for downloading test data  Activetyphoon=['SURIGAE']
  
    print("currently active typhoon list= %s"%Activetyphoon)
    context.info(f"currenly active typhoon list is {Activetyphoon}")
    return dict(remote_dir=remote_dir, active_typhoons=Activetyphoon)


#%% Download Rainfaall

@command
def forecast_setup(config):
    path=config["path"]
    Alternative_data_point=(datetime.strptime(datetime.now().strftime("%Y%m%d%H"), "%Y%m%d%H")-timedelta(hours=24)).strftime("%Y%m%d")
        
    Input_folder=os.path.join(path,'forecast/Input/%s/Input/'%(datetime.now().strftime("%Y%m%d%H")))
    Output_folder=os.path.join(path,'forecast/Output/%s/Output/'%(datetime.now().strftime("%Y%m%d%H")))

    if not os.path.exists(Input_folder):
        os.makedirs(Input_folder)
    if not os.path.exists(Output_folder):
        os.makedirs(Output_folder)   
    return dict(
        alternative_data_point = Alternative_data_point,
        input_folder = Input_folder,
        output_folder = Output_folder,
        **config
    )

@first_command
def basic_config():
    return dict(
        path = home_path(),
        uCL_USERNAME=uCL_USERNAME,
        uCL_PASSWORD=uCL_PASSWORD,
        admin_shp_path = str((Path(home_path()) / "data-raw" / "phl_admin3_simpl2.shp") .absolute())
    )

@command
def download_rainfall(config, skip_nomads_download=False, context=None):
    if context is None:
        context=Context()
#NOAA rainfall
    Input_folder = config["input_folder"]
    path = config["path"]
    Alternative_data_point = config["alternative_data_point"]
    uCL_USERNAME = config["uCL_USERNAME"]
    uCL_PASSWORD = config["uCL_PASSWORD"]

    if not skip_nomads_download:
        context.info("download rainfall nomads")
        Rainfall_data.download_rainfall_nomads(Input_folder,path,Alternative_data_point)
    try:
        ucl_data.create_ucl_metadata(path,uCL_USERNAME,uCL_PASSWORD)
        ucl_data.process_ucl_data(path,Input_folder,uCL_USERNAME,uCL_PASSWORD)

    except:
        pass

    return config

_CENTROID = None
@first_command(volatile=True)
def grid_points_centroid():
    global _CENTROID
    ##Create grid points to calculate Winfield
    if _CENTROID is None:
        cent = Centroids()
        cent.set_raster_from_pnt_bounds((118,6,127,19), res=0.05)
        cent.check()
        _CENTROID = cent
    return _CENTROID

@command
def plot_centroid(cent):
    from matplotlib.pyplot import gcf
    ########################################## uncomment the following line to see intermidiate results
    cent.plot()
    ####
    return gcf()

@command
def to_cent_df(cent):
    df = pd.DataFrame(data=cent.coord)
    df["centroid_id"] = "id"+(df.index).astype(str)  
    centroid_idx=df["centroid_id"].values
    ncents = cent.size
    df=df.rename(columns={0: "lat", 1: "lon"})
    return df

@first_command
def cent_df(cent):
    return to_cent_df(grid_points_centroid())

@command
def get_admin(config):
    admin_shp_path = config["admin_shp_path"]
    print (f"Admin shp path: {admin_shp_path}")
    #admin=gpd.read_file("C:/Users/ATeklesadik/OneDrive - Rode Kruis/Documents/documents/Typhoon-Impact-based-forecasting-model/data-raw/phl_admin3_simpl2.shp")
    admin=gpd.read_file(admin_shp_path)
    return admin

@command
def get_admin_with_cent(config):
    admin = get admin(config)
    df=cent_df()

    df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))
    #df.to_crs({'init': 'epsg:4326'})
    df.crs = {'init': 'epsg:4326'}
    df_admin = sjoin(df, admin, how="left")
    df_admin=df_admin.dropna()
    return df_admin

#%% Download ECMWF data 

@command(volatile=True)
def ecmwf_fcast(config):
    remote_dir = config["remote_dir"]
    Activetyphoon = config["active_typhoons"]  
    bufr_files = TCForecast.fetch_bufr_ftp(remote_dir=remote_dir)
    fcast = TCForecast()
    fcast.fetch_ecmwf(files=bufr_files)

    #%% filter for active typhoons (or example typhoon) 

    # filter tracks with name of current typhoons and drop tracks with only one timestep
    fcast.data = [tr for tr in fcast.data if tr.name in Activetyphoon]
    fcast.data = [tr for tr in fcast.data if tr.time.size>1]
    return fcast


#typhhon_df.to_csv('//TyphoonModel/Typhoon-Impact-based-forecasting-model/forecast/intensity.csv')


#%%#Download rainfall (old pipeline)

def prepare_calculation(typhoons):
    fname=open(os.path.join(path,'forecast/Input/',"typhoon_info_for_model.csv"),'w')
    fname.write('source,filename,event,time'+'\n')            
    line_='Rainfall,'+'%sRainfall' % Input_folder +',' +typhoons+','+ datetime.now().strftime("%Y%m%d%H")  #StormName #
    fname.write(line_+'\n')
    line_='Output_folder,'+'%s' % Output_folder +',' +typhoons+',' + datetime.now().strftime("%Y%m%d%H")  #StormName #
    #line_='Rainfall,'+'%sRainfall/' % Input_folder +','+ typhoons + ',' + datetime.now().strftime("%Y%m%d%H") #StormName #
    fname.write(line_+'\n')

    #typhoons='SURIGAE'  # to run it manually for any typhoon 
                # select windspeed for HRS model
                
    fcast.data1=[tr for tr in fcast.data if tr.name==typhoons]
    tr_HRS=[tr for tr in fcast.data1 if (tr.is_ensemble=='False') & (tr.ensemble_number==1)]
    HRS_SPEED=(tr_HRS[0].max_sustained_wind.values/0.84).tolist()  ############# 0.84 is conversion factor for ECMWF 10MIN TO 1MIN AVERAGE
    
    dfff=tr_HRS[0].to_dataframe()
    dfff[['VMAX','LAT','LON']]=dfff[['max_sustained_wind','lon','lat']]
    dfff['YYYYMMDDHH']=dfff.index.values
    dfff[['YYYYMMDDHH','VMAX','LAT','LON']].to_csv(os.path.join(Input_folder,'ecmwf_hrs_track.csv'), index=False)
    
    
    line_='ecmwf,'+'%secmwf_hrs_track.csv' % Input_folder+ ',' +typhoons+','+ datetime.now().strftime("%Y%m%d%H")   #StormName #
    #line_='Rainfall,'+'%sRainfall/' % Input_folder +','+ typhoons + ',' + datetime.now().strftime("%Y%m%d%H") #StormName #
    fname.write(line_+'\n')
    
    # Adjust track time step
    
    data_forced=[tr.where(tr.time <= max(tr_HRS[0].time.values),drop=True) for tr in fcast.data1]
        
    data_forced = [track_data_clean.track_data_force_HRS(tr,HRS_SPEED) for tr in data_forced] # forced with HRS windspeed
    
    
    
    fcast.data = [track_data_clean.track_data_clean(tr) for tr in data_forced] # taking speed of ENS
    # interpolate to 3h steps from the original 6h
    fcast.equal_timestep(3)

    
    # calculate windfields for each ensamble
    threshold=0            #(threshold to filter dataframe /reduce data )
    df = pd.DataFrame(data=cent.coord)
    df["centroid_id"] = "id"+(df.index).astype(str)  
    centroid_idx=df["centroid_id"].values
    ncents = cent.size
    df=df.rename(columns={0: "lat", 1: "lon"})
    
    #calculate wind field for each ensamble members 
    list_intensity=[]
    distan_track=[]
    for tr in fcast.data:
        print(tr.name)
        track = TCTracks() 
        typhoon = TropCyclone()
        track.data=[tr]
        typhoon.set_from_tracks(track, cent, store_windfields=True)
        windfield=typhoon.windfields
        nsteps = windfield[0].shape[0]
        centroid_id = np.tile(centroid_idx, nsteps)
        intensity_3d = windfield[0].toarray().reshape(nsteps, ncents, 2)
        intensity = np.linalg.norm(intensity_3d, axis=-1).ravel()
        
        timesteps = np.repeat(tr.time.values, ncents)
        timesteps = timesteps.reshape((nsteps, ncents)).ravel()
        inten_tr = pd.DataFrame({
                'centroid_id': centroid_id,
                'value': intensity,
                'timestamp': timesteps,})
        inten_tr = inten_tr[inten_tr.value > threshold]
        inten_tr['storm_id'] = tr.sid
        inten_tr['ens_id'] =tr.sid+'_'+str(tr.ensemble_number)
        inten_tr['name'] = tr.name
        list_intensity.append(inten_tr)
        distan_track1=[]
        for index, row in df.iterrows():
            dist=np.min(np.sqrt(np.square(tr.lat.values-row['lat'])+np.square(tr.lon.values-row['lon'])))
            distan_track1.append(dist*111)
            print(f'centroid_id:   {len(centroid_idx)}  {centroid_idx}')
            print(f'distan_track1: {len(distan_track1)}   {distan_track1}')
            dist_tr = pd.DataFrame({'centroid_id': centroid_idx,'value': distan_track1})
            dist_tr['storm_id'] = tr.sid
            dist_tr['name'] = tr.name
            dist_tr['ens_id'] =tr.sid+'_'+str(tr.ensemble_number)
            distan_track.append(dist_tr)                
    df_intensity = pd.concat(list_intensity)
    df_intensity=pd.merge(df_intensity, df_admin, how='outer', on='centroid_id')
    df_intensity=df_intensity.dropna()
    
    df_intensity_=df_intensity.groupby(['adm3_pcode','ens_id'],as_index=False).agg({"value":['count', 'max']}) 
    # rename columns
    df_intensity_.columns = [x for x in ['adm3_pcode','storm_id','value_count','v_max']] 
    distan_track1= pd.concat(distan_track)
    distan_track1=pd.merge(distan_track1, df_admin, how='outer', on='centroid_id')
    distan_track1=distan_track1.dropna()
    
    distan_track1=distan_track1.groupby(['adm3_pcode','name','ens_id'],as_index=False).agg({'value':'min'}) 
    distan_track1.columns = [x for x in ['adm3_pcode','name','storm_id','dis_track_min']]#join_left_df_.columns.ravel()] 
    typhhon_df = pd.merge(df_intensity_, distan_track1,  how='left', on=['adm3_pcode','storm_id']) 

    typhhon_df.to_csv(os.path.join(Input_folder,'windfield.csv'), index=False)

    line_='windfield,'+'%swindfield.csv' % Input_folder+ ',' +typhoons+','+ datetime.now().strftime("%Y%m%d%H")   #StormName #
    #line_='Rainfall,'+'%sRainfall/' % Input_folder +','+ typhoons + ',' + datetime.now().strftime("%Y%m%d%H") #StormName #
    fname.write(line_+'\n')
    fname.close()

def automation_sript(path):
    print('---------------------AUTOMATION SCRIPT STARTED---------------------------------')
    print(str(datetime.now()))
#    Activetyphoon=Check_for_active_typhoon.check_active_typhoon()
    ##############################################################################
    ### download metadata from UCL
    ##############################################################################
    if not Activetyphoon==[]:
        #delete_old_files()      
        for typhoons in Activetyphoon:
            fname=open(os.path.join(path,'forecast/Input/',"typhoon_info_for_model.csv"),'w')
            fname.write('source,filename,event,time'+'\n')            
            line_='Rainfall,'+'%sRainfall' % Input_folder +',' +typhoons+','+ datetime.now().strftime("%Y%m%d%H")  #StormName #
            fname.write(line_+'\n')
            line_='Output_folder,'+'%s' % Output_folder +',' +typhoons+',' + datetime.now().strftime("%Y%m%d%H")  #StormName #
            #line_='Rainfall,'+'%sRainfall/' % Input_folder +','+ typhoons + ',' + datetime.now().strftime("%Y%m%d%H") #StormName #
            fname.write(line_+'\n')

            #typhoons='SURIGAE'  # to run it manually for any typhoon 
                        # select windspeed for HRS model
                        
            fcast.data1=[tr for tr in fcast.data if tr.name==typhoons]
            tr_HRS=[tr for tr in fcast.data1 if (tr.is_ensemble=='False') & (tr.ensemble_number==1)]
            HRS_SPEED=(tr_HRS[0].max_sustained_wind.values/0.84).tolist()  ############# 0.84 is conversion factor for ECMWF 10MIN TO 1MIN AVERAGE
            
            dfff=tr_HRS[0].to_dataframe()
            dfff[['VMAX','LAT','LON']]=dfff[['max_sustained_wind','lon','lat']]
            dfff['YYYYMMDDHH']=dfff.index.values
            dfff[['YYYYMMDDHH','VMAX','LAT','LON']].to_csv(os.path.join(Input_folder,'ecmwf_hrs_track.csv'), index=False)
            
            
            line_='ecmwf,'+'%secmwf_hrs_track.csv' % Input_folder+ ',' +typhoons+','+ datetime.now().strftime("%Y%m%d%H")   #StormName #
            #line_='Rainfall,'+'%sRainfall/' % Input_folder +','+ typhoons + ',' + datetime.now().strftime("%Y%m%d%H") #StormName #
            fname.write(line_+'\n')
            
            # Adjust track time step
            
            data_forced=[tr.where(tr.time <= max(tr_HRS[0].time.values),drop=True) for tr in fcast.data1]
             
            data_forced = [track_data_clean.track_data_force_HRS(tr,HRS_SPEED) for tr in data_forced] # forced with HRS windspeed
            
            
          
            fcast.data = [track_data_clean.track_data_clean(tr) for tr in data_forced] # taking speed of ENS
            # interpolate to 3h steps from the original 6h
            fcast.equal_timestep(3)
 
            
            # calculate windfields for each ensamble
            threshold=0            #(threshold to filter dataframe /reduce data )
            df = pd.DataFrame(data=cent.coord)
            df["centroid_id"] = "id"+(df.index).astype(str)  
            centroid_idx=df["centroid_id"].values
            ncents = cent.size
            df=df.rename(columns={0: "lat", 1: "lon"})
            
            #calculate wind field for each ensamble members 
            list_intensity=[]
            distan_track=[]
            for tr in fcast.data:
                print(tr.name)
                track = TCTracks() 
                typhoon = TropCyclone()
                track.data=[tr]
                typhoon.set_from_tracks(track, cent, store_windfields=True)
                windfield=typhoon.windfields
                nsteps = windfield[0].shape[0]
                centroid_id = np.tile(centroid_idx, nsteps)
                intensity_3d = windfield[0].toarray().reshape(nsteps, ncents, 2)
                intensity = np.linalg.norm(intensity_3d, axis=-1).ravel()
                
                timesteps = np.repeat(tr.time.values, ncents)
                timesteps = timesteps.reshape((nsteps, ncents)).ravel()
                inten_tr = pd.DataFrame({
                        'centroid_id': centroid_id,
                        'value': intensity,
                        'timestamp': timesteps,})
                inten_tr = inten_tr[inten_tr.value > threshold]
                inten_tr['storm_id'] = tr.sid
                inten_tr['ens_id'] =tr.sid+'_'+str(tr.ensemble_number)
                inten_tr['name'] = tr.name
                list_intensity.append(inten_tr)
                distan_track1=[]
                for index, row in df.iterrows():
                    dist=np.min(np.sqrt(np.square(tr.lat.values-row['lat'])+np.square(tr.lon.values-row['lon'])))
                    distan_track1.append(dist*111)
                    print(f'centroid_id:   {len(centroid_idx)}  {centroid_idx}')
                    print(f'distan_track1: {len(distan_track1)}   {distan_track1}')
                    dist_tr = pd.DataFrame({'centroid_id': centroid_idx,'value': distan_track1})
                    dist_tr['storm_id'] = tr.sid
                    dist_tr['name'] = tr.name
                    dist_tr['ens_id'] =tr.sid+'_'+str(tr.ensemble_number)
                    distan_track.append(dist_tr)                
            df_intensity = pd.concat(list_intensity)
            df_intensity=pd.merge(df_intensity, df_admin, how='outer', on='centroid_id')
            df_intensity=df_intensity.dropna()
            
            df_intensity_=df_intensity.groupby(['adm3_pcode','ens_id'],as_index=False).agg({"value":['count', 'max']}) 
            # rename columns
            df_intensity_.columns = [x for x in ['adm3_pcode','storm_id','value_count','v_max']] 
            distan_track1= pd.concat(distan_track)
            distan_track1=pd.merge(distan_track1, df_admin, how='outer', on='centroid_id')
            distan_track1=distan_track1.dropna()
            
            distan_track1=distan_track1.groupby(['adm3_pcode','name','ens_id'],as_index=False).agg({'value':'min'}) 
            distan_track1.columns = [x for x in ['adm3_pcode','name','storm_id','dis_track_min']]#join_left_df_.columns.ravel()] 
            typhhon_df = pd.merge(df_intensity_, distan_track1,  how='left', on=['adm3_pcode','storm_id']) 
        
            typhhon_df.to_csv(os.path.join(Input_folder,'windfield.csv'), index=False)
        
            line_='windfield,'+'%swindfield.csv' % Input_folder+ ',' +typhoons+','+ datetime.now().strftime("%Y%m%d%H")   #StormName #
            #line_='Rainfall,'+'%sRainfall/' % Input_folder +','+ typhoons + ',' + datetime.now().strftime("%Y%m%d%H") #StormName #
            fname.write(line_+'\n')
            fname.close()

            

            #############################################################
            #### Run IBF model 
            #############################################################
            os.chdir(path)
            
            if platform == "linux" or platform == "linux2": #check if running on linux or windows os
                # linux
                try:
                    p = subprocess.check_call(["Rscript", "run_model.R", str(rainfall_error)])
                except subprocess.CalledProcessError as e:
                    raise ValueError(str(e))
            elif platform == "win32": #if OS is windows edit the path for Rscript
                try:
                    p = subprocess.check_call(["C:/Program Files/R/R-3.6.3/bin/Rscript", "run_model.R", str(rainfall_error)])
                except subprocess.CalledProcessError as e:
                    raise ValueError(str(e))
                
   
            #############################################################
            ### send email in case of landfall-typhoon
            #############################################################
    
            landfall_typhones=[]
            try:
                fname2=open("forecast/%s_file_names.csv" % typhoons,'r')
                for lines in fname2.readlines():
                    print(lines)
                    if (lines.split(' ')[1].split('_')[0]) !='"Nolandfall':
                        if lines.split(' ')[1] not in landfall_typhones:
                            landfall_typhones.append(lines.split(' ')[1])
                fname2.close()
            except:
                pass
            
            if not landfall_typhones==[]:
                image_filename=landfall_typhones[0]
                data_filename=landfall_typhones[1]
                html = """\
                <html>
                <body>
                <h1>IBF model run result </h1>
                <p>Please find below a map and data with updated model run</p>
                <img src="cid:Impact_Data">
                </body>
                </html>
                """
                Sendemail.sendemail(from_addr  = EMAIL_FROM,
                        to_addr_list = EMAIL_LIST,
                        cc_addr_list = CC_LIST,
                        message = message(
                            subject='Updated impact map for a new Typhoon in PAR',
                            html=html,
                            textfile=data_filename,
                            image=image_filename),
                        login  = EMAIL_LOGIN,
                        password= EMAIL_PASSWORD,
                        smtpserver=SMTP_SERVER)


    print('---------------------AUTOMATION SCRIPT FINISHED---------------------------------')
    print(str(datetime.now()))
    
 


 
#%% 

automation_sript(path)