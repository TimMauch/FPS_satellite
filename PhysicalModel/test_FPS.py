#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 11:12:02 2022

@author: tim
"""
from pvodataset import PVODataset, UDFClass, QCfunc

# scientific python add-ons
# import numpy as np
import pandas as pd
# import netCDF4 as nc

# plotting stuff
# first line makes the plots appear in the notebook
import matplotlib.pyplot as plt
import matplotlib as mpl

# finally, we import the pvlib library
# import pvlib
from pvlib import solarposition, irradiance, atmosphere, pvsystem, inverter, temperature, location, clearsky
from pvlib.forecast import GFS, Location
from pvlib.modelchain import ModelChain


# 1. Load PVO Dataset
path = "/Users/tim/Documents/Vorlseungen/Semester_1/Einführung_Umweltmodellierung/Praktikum/PV_Forecast/Datasets/PVODdatasets_v1/"
pvod = PVODataset(path=path,timezone="UTC+8")

# 2. Define station parameters
s_id = 8
metadata = pvod.read_metadata()
ori_data = pvod.read_ori_data(station_id=s_id)
station_info = pvod.station_info(station_id=s_id)

# 3. select data within date range 
time_zone = "Asia/Shanghai"
t1 = '2018-07-01 00:00'
t2 = '2018-07-05 00:00'
time_data = pvod.select_daterange(station_id=s_id, start_date=t1, end_date=t2)
# convert time data to UTC
time_data.index = time_data.date_time
time_data.index = time_data.index.tz_convert("UTC")

# 4. read all relevant metadata
lon = metadata["Longitude"][s_id]
lat = metadata["Latitude"][s_id]
pos = location.Location(lat, lon, tz="Asia/Shanghai") ### maybe add altitude here ###
surface_tilt = int(metadata["Array_Tilt"][s_id][-4:-1])
surface_azimuth = 180
modules_per_string = int(metadata["Layout"][s_id].split("\n")[0][-2:])
strings_per_inverter = int(metadata["Layout"][s_id].split("\n")[1][-1:])


# 4. Get inverter and panel data from database
module_para = pvsystem.retrieve_sam("CECMod")["Yingli_Energy__China__YL250P_29b"]
inverter_para = pvsystem.retrieve_sam("cecinverter")["Advanced_Energy_Industries__Solaron_500kW__3159500_XXXX___480V_"]
inverter_para["Pdco"] = 567000
inverter_para["Vdco"] = 315
inverter_para["Vdcmax"] = 1000
inverter_para["Idcmax"] = 1134
inverter_para["Mppt_low"] = 460
inverter_para["Mppt_high"] = 950

###
# at this part of the code also panel and inverter training models could be implemented
###

# 5. Define thermal model
thermal_params = temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

# 6. Define PV System (Comprises all information at level of one inverter)
system = pvsystem.PVSystem(surface_tilt=surface_tilt, surface_azimuth=180,
                  module_parameters=module_para,
                  inverter_parameters=inverter_para,
                  modules_per_string=modules_per_string, strings_per_inverter=strings_per_inverter,
                  temperature_model_parameters=thermal_params)

# 7. Set up model chain for standardized PV modelling
#    This contains three components: 
#    1. PVSystem/ 2. Location/ 3. Model attributes
#    Model chain works for one inverter and therefore needs to be multiplied with number of total inverters
mc = ModelChain(system, pos, transposition_model="perez",
                solar_position_method="nrel_numpy",
                # orientation_strategy="south_at_latitude_tilt",
                aoi_model="physical",spectral_model="no_loss")
### add some more sophisticated methods ###

# 8. Read weather data
# At this point satellite data need to be included
weather = pd.DataFrame(data=time_data[["lmd_totalirrad", "lmd_diffuseirrad", 
                     "lmd_temperature", "lmd_windspeed"]].values, 
                       columns=["ghi","dhi", "temp_air","wind_speed"], 
                       index = time_data.date_time)
mc.complete_irradiance(weather)
mc.run_model(mc.results.weather)

# 9. Plot everything
time_data["model_ac_pw_pred"] = mc.results.ac*720*1e-6
time_data[["model_ac_pw_pred", "power"]].plot()
plt.ylabel('Power MW');
plt.xlabel('Forecast Time ({})'.format(time_zone));
plt.title('Power Output Comparison for lat={}, lon={}'.format(round(lat), round(lon)));
plt.legend(["predicted", "measured"],loc="upper left");
plt.savefig("power_output.png")













