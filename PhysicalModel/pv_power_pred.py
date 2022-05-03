#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 09:48:26 2022

@author: tim
"""

from pvodataset import PVODataset, UDFClass, QCfunc
# built-in python modules
import datetime
import inspect
import os
import glob
import tarfile
import shutil
import re

# scientific python add-ons
import numpy as np
import pandas as pd
import netCDF4 as nc

# plotting stuff
# first line makes the plots appear in the notebook
import matplotlib.pyplot as plt
import matplotlib as mpl

# finally, we import the pvlib library
import pvlib
from pvlib import solarposition, irradiance, atmosphere, pvsystem, inverter, temperature, location, clearsky
from pvlib.forecast import GFS, Location
from pvlib.modelchain import ModelChain
import download_gfs


# 1. Extract position data for PV farm that should be modelled and time horizon
path = "/Users/tim/Documents/Vorlseungen/Semester_1/Einführung_Umweltmodellierung/Praktikum/PV_Forecast/Datasets/PVODdatasets_v1/"
pvod = PVODataset(path=path,timezone="UTC+8")

s_id = 8
metadata = pvod.read_metadata()
ori_data = pvod.read_ori_data(station_id=s_id)
station_info = pvod.station_info(station_id=s_id)
# select data within date range 
time_zone = "Asia/Shanghai"
t1 = '2019-3-01 00:00'
t2 = '2019-3-31 00:00'
time_data = pvod.select_daterange(station_id=s_id, start_date=t1, end_date=t2)
# convert time data to UTC
time_data.index = time_data.date_time
time_data.index = time_data.index.tz_convert("UTC")
# read all relevant metadata
lon = metadata["Longitude"][s_id]
lat = metadata["Latitude"][s_id]
surface_tilt = int(metadata["Array_Tilt"][s_id][-4:-1])
surface_azimuth = 180
modules_per_string = int(metadata["Layout"][s_id].split("\n")[0][-2:])
strings_per_inverter = int(metadata["Layout"][s_id].split("\n")[1][-1:])

# 2. Load gfs archive data for the position and the time horizon

start_date =  time_data.index[0] - pd.Timedelta(1, "d") # add buffer date to account for whole time period
end_date = time_data.index[-1] + pd.Timedelta(1, "d") 
model = 'ds084.1'
params = 'TMP/DSWRF/T CDC/V GRD/U GRD/GUST/' #'M CDC/H CDC/L CDCT CDC'
levels = 'ISBL:1000;EATM:0;BCY:0;HCY:0;MCY:0;LCY:0;CCY:0;SFC:0'
products = "3-hour Forecast/6-hour Forecast/9-hour Forecast/3-hour Average (initial+0 to initial+3)/6-hour Average (initial+0 to initial+6)/3-hour Average (initial+6 to initial+9)"
path_gfs = "/Users/tim/Documents/Vorlseungen/Semester_1/Einführung_Umweltmodellierung/Praktikum/PV_Forecast/get_gfs_ucar"
path_res = "/Users/tim/Documents/Vorlseungen/Semester_1/Einführung_Umweltmodellierung/Praktikum/PV_Forecast/Datasets/GHI_gfs/test"
coord = [round(lat+1),round(lat-1),round(lon-1),round(lon+1)] # NLat, SLat, WLon, ELon
    
download_gfs.load_file(start_date, end_date, params, levels, products, path_gfs, model, coord)

# get gfs output data for the exact desired position
startdate = str(start_date)[0:10].replace("-", "") + "0000"
enddate = str(end_date)[0:10].replace("-", "") + "0000"
# open tar file
tar_name = [targ_dir for targ_dir in os.listdir(path_gfs) if startdate[0:8] in targ_dir]
res_tar = path_gfs + "/" + tar_name[0]
if res_tar.endswith("tar.gz"):
    tar = tarfile.open(res_tar, "r:gz")
    tar.extractall()
    tar.close(f"/Users/tim/Documents/Vorlseungen/Semester_1/Einführung_Umweltmodellierung/Praktikum/PV_Forecast/get_gfs_ucar/{startdate}-{enddate}_gfs_data")
elif res_tar.endswith("tar"):
    tar = tarfile.open(res_tar, "r:")
    tar.extractall(f"/Users/tim/Documents/Vorlseungen/Semester_1/Einführung_Umweltmodellierung/Praktikum/PV_Forecast/get_gfs_ucar/{startdate}-{enddate}_gfs_data")
    tar.close()
original = res_tar
target = path_gfs + "/tar_archive/" + tar_name[0]
shutil.move(original, target)
# get variable data
dir_str = path_gfs + "/" + f"{startdate}-{enddate}_gfs_data"+"/gfs*"
file_names = []
for file in glob.glob(dir_str):
    file_names.append(file)

var_list = ["T_CDC_L224_Avg_1", "T_CDC_L244", "T_CDC_L10_Avg_1",
            "TMP_L1", "GUST_L1", "T_CDC_L234_Avg_1", "T_CDC_L214_Avg_1",
            "DSWRF_L1_Avg_1", "T_CDC_L211_Avg_1", "U_GRD_L100","V_GRD_L100"]

gfs_model_names = [
 'Medium_cloud_cover_middle_cloud_Mixed_intervals_Average',
 'Total_cloud_cover_convective_cloud',
 'Total_cloud_cover_entire_atmosphere_Mixed_intervals_Average',
 'Temperature_surface',
 'Wind_speed_gust_surface',
 'High_cloud_cover_high_cloud_Mixed_intervals_Average',
 'Low_cloud_cover_low_cloud_Mixed_intervals_Average',
 'Downward_Short-Wave_Radiation_Flux_surface_Mixed_intervals_Average',
 'Total_cloud_cover_boundary_layer_cloud_Mixed_intervals_Average',
 'u-component_of_wind_isobaric',
 'v-component_of_wind_isobaric']

df_cols = ["datetime", "date", "model_run", "forecast_hour",
           "lon", "lat"] + gfs_model_names
res_df = pd.DataFrame(columns=df_cols)
error = 0
for names in file_names:
        inter_res = []
        ds = nc.Dataset(names)
        lons = list(ds["lon"][:])
        lats = list(ds["lat"][:])
        
        lon_ind = (np.abs(lons - lon)).argmin()
        lat_ind = (np.abs(lats - lat)).argmin()
        
        # convert date into format that is used by pv-lib
        date = re.search("0p25.(.*).f", names).group(1)[:-2]
        run = re.search("0p25.(.*).f", names).group(1)[-2:]
        forecast = int(re.search("0p25(.*).grib2", names).group(1)[-3:])
        date_str = date+run+":00"
        time_str = str(pd.to_datetime(date_str, format="%Y%m%d%H:%M"))
        unix_time = pd.Timestamp(time_str).timestamp()
        unix_time += forecast*3600
        inter_res.append(pd.to_datetime(unix_time, unit="s"))
        # tm_stemp = pd.Timestamp(time_str, tz = time_zone)
        # tm_stemp = tm_stemp.tz_convert("UTC")
        
        inter_res.append(re.search("0p25.(.*).f", names).group(1)[:-2])
        inter_res.append(re.search("0p25.(.*).f", names).group(1)[-2:])
        inter_res.append(re.search("0p25(.*).grib2", names).group(1)[-3:])
        inter_res.append(lon)
        inter_res.append(lat)
        
        for var in var_list:
            try:
                curr_var = ds[var][:]
                curr_val = curr_var[0,lat_ind,lon_ind]
                inter_res.append(curr_val)
            except:
                inter_res.append(np.NAN)
                print(f"variable {var} in file {names} not found. Replaced by nan")
                error += 1
        
        res_df.loc[-1] = inter_res
        res_df = res_df.reset_index(drop=True)

dir_str = path_gfs + "/" + f"{startdate}-{enddate}_gfs_data"
shutil.rmtree(dir_str)
res_path = path_res + "/" + startdate + "-" + enddate + ".csv"
res_df.to_csv(res_path, index=False)
print("Total errors:", error)
# return res_path

# 3. Calculate irradiation properties using campbell-norman model (interpolate to PV time steps)
dir_str = "/Users/tim/Documents/Vorlseungen/Semester_1/Einführung_Umweltmodellierung/Praktikum/PV_Forecast/Datasets/GHI_gfs/test/" + str(start_date)[:10].replace("-","") + "*"
res_path = glob.glob(dir_str)[0]

gfs_archive = pd.read_csv(res_path)
gfs_archive["datetime"] = pd.DatetimeIndex(gfs_archive["datetime"], tz = 'UTC') #.tz_convert(time_zone)
gfs_archive = gfs_archive.set_index(gfs_archive["datetime"])
gfs_archive = gfs_archive.drop(["datetime"], axis=1)
gfs_archive = gfs_archive.sort_index()

f_hour = 3
st_data = gfs_archive[gfs_archive["forecast_hour"] == f_hour]
data = st_data[st_data.columns[7:]]
data = data.sort_index()
data = data.apply(pd.to_numeric)
model = GFS()
model.start = start_date
model.end = end_date
model.location = Location(lat, lon, time_zone)

data = model.rename(data)
# convert air temperature
data['temp_air'] = model.kelvin_to_celsius(data['temp_air'])
# convert wind components to wind speed
data['wind_speed'] = model.uv_to_speed(data)   
# interpolation is acceptble when only interpolated btw. past time and most recent forecast!
data_resample = data.resample("15min").interpolate()
irrad_data_camp = model.cloud_cover_to_irradiance(data_resample['total_clouds'], how='campbell_norman')

irrad_data_camp = irrad_data_camp[irrad_data_camp.index>= time_data.index[0]]
irrad_data_camp = irrad_data_camp[irrad_data_camp.index<= time_data.index[-1]]
time_data = pd.concat([time_data,irrad_data_camp], axis=1)
data_resample = data_resample[data_resample.index>= time_data.index[0]]
data_resample = data_resample[data_resample.index<= time_data.index[-1]]
time_data = pd.concat([time_data,data_resample[["temp_air", "wind_speed"]]], axis=1)
time_data.index = time_data.index.tz_convert(time_zone)

# 4. Calculate solar power output
module_para = pvsystem.retrieve_sam("CECMod")["Yingli_Energy__China__YL250P_29b"]
pos = location.Location(lat, lon, tz="Asia/Shanghai")
inverter_para = pvsystem.retrieve_sam("cecinverter")["Advanced_Energy_Industries__Solaron_500kW__3159500_XXXX___480V_"]
inverter_para["Pdco"] = 567000
inverter_para["Vdco"] = 315
inverter_para["Vdcmax"] = 1000
inverter_para["Idcmax"] = 1134
inverter_para["Mppt_low"] = 460
inverter_para["Mppt_high"] = 950

thermal_params = temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

system = pvsystem.PVSystem(surface_tilt=surface_tilt, surface_azimuth=180,
                  module_parameters=module_para,
                  inverter_parameters=inverter_para,
                  modules_per_string=modules_per_string, strings_per_inverter=strings_per_inverter,
                  temperature_model_parameters=thermal_params)

mc = ModelChain(system, pos, transposition_model="perez",
                solar_position_method="nrel_numpy",
                # orientation_strategy="south_at_latitude_tilt",
                aoi_model="physical",spectral_model="no_loss")

weather = pd.DataFrame(data=time_data[["lmd_totalirrad", "lmd_diffuseirrad", 
                     "lmd_temperature", "lmd_windspeed"]].values, 
                       columns=["ghi","dhi", "temp_air","wind_speed"], 
                       index = time_data.date_time)
mc.complete_irradiance(weather)
mc.run_model(mc.results.weather)

test=mc.results.cell_temperature
mc.results.ac.plot()

weather = pd.DataFrame(data=time_data[["ghi", "dhi", 
                     "dni", "temp_air", "wind_speed"]].values, 
                       columns=["ghi","dhi", "dni", "temp_air","wind_speed"], 
                       index = time_data.date_time)
#mc.complete_irradiance(weather)
mc.run_model(weather)


time_data["model_ac_pw_pred"] = np.array(mc.results.ac * 1e-6 * 716)
time_data[["model_ac_pw", "power"]].plot()

rmse = (1/len(time_data["model_ac_pw"]) * sum((time_data["model_ac_pw"]-time_data["power"])**2))**0.5
rmse_rel = rmse/time_data["power"].mean()

mae = 1/len(time_data["model_ac_pw"]) * sum(abs(time_data["model_ac_pw"]-time_data["power"]))
nmae = mae/time_data["power"].mean()

mae = 1/len(time_data["model_ac_pw"]) * sum(abs(time_data["model_ac_pw_pred"]-time_data["power"]))
nmae = mae/time_data["power"].mean()


time_data_slice = time_data[time_data.index < "2019-03-05"]
data_slice = data[data.index < "2019-03-05"]
# plot NWP data
cloud_vars = ['total_clouds', 'low_clouds',
              'high_clouds']
data_slice[cloud_vars].plot();
plt.ylabel('Cloud cover [%]');
plt.xlabel('Forecast Time ({})'.format(time_zone));
plt.title('Cloud forecast for lat={}, lon={}'.format(round(lat), round(lon)));
plt.legend(loc="lower right");
plt.savefig("cloud_cover.png")

time_data_slice[["nwp_globalirrad", "lmd_totalirrad", "ghi"]].plot()
plt.ylabel('GHI [W/m2]');
plt.xlabel('Forecast Time ({})'.format(time_zone));
plt.title('GHI Comparison for lat={}, lon={}'.format(round(lat), round(lon)));
plt.legend(["nwp dataset", "measured", "gfs model"], loc="lower right");
plt.savefig("ghi_pred.png")


time_data_slice[["model_ac_pw_pred","model_ac_pw", "power"]].plot()
plt.ylabel('Power MW');
plt.xlabel('Forecast Time ({})'.format(time_zone));
plt.title('Power Output Comparison for lat={}, lon={}'.format(round(lat), round(lon)));
plt.legend(["predicted GFS", "predicted msr", "measured"],loc="upper left");
plt.savefig("power_output.png")









