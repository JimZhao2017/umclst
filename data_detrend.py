#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import rasterio
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[2]:



folder_path = "C:/Users/zhaoj/Desktop/umclst/dataimage/jjj/"
fullname = folder_path+'Beijingdata.tif'
with rasterio.open(fullname) as src:
# Read the bands as numpy arrays
    lcz2003 = src.read(1)
    lcz2004 = src.read(2)
    lcz2005 = src.read(3)
    lcz2006 = src.read(4)
    lcz2007 = src.read(5)
    lcz2008 = src.read(6)
    lcz2009 = src.read(7)
    lcz2010 = src.read(8)
    lcz2011 = src.read(9)
    lcz2012 = src.read(10)
    lcz2013 = src.read(11)
    lcz2014 = src.read(12)
    lcz2015 = src.read(13)
    lcz2016 = src.read(14)
    lcz2017 = src.read(15)
    lcz2018 = src.read(16)
    lcz2019 = src.read(17)
    lcz2020 = src.read(18)

    lst2003 = src.read(19)
    lst2004 = src.read(20)
    lst2005 = src.read(21)
    lst2006 = src.read(22)
    lst2007 = src.read(23)
    lst2008 = src.read(24)
    lst2009 = src.read(25)
    lst2010 = src.read(26)
    lst2011 = src.read(27)
    lst2012 = src.read(28)
    lst2013 = src.read(29)
    lst2014 = src.read(30)
    lst2015 = src.read(31)
    lst2016 = src.read(32)
    lst2017 = src.read(33)
    lst2018 = src.read(34)
    lst2019 = src.read(35)
    lst2020 = src.read(36)
    
    data = src.read()
    # Convert the data to a three-dimensional numpy array
    data_3d = np.moveaxis(data, 0, -1)

    # Print the shape of the data
    print(data_3d.shape)


# In[3]:


lcz1 = lcz2020==1
lcz2 = lcz2020==2
lcz3 = (lcz2020==3)
lcz4 = lcz2020==4
lcz5 = lcz2020==5
lcz6 = lcz2020==6
lcz7 = lcz2020==7
lcz8 = lcz2020==8
lcz9 = lcz2020==9
lcz10 = lcz2020==10
forest = lcz2020==11
cropland = lcz2020==14
urban = (lcz2003<=10)&(lcz2003>0)&(lcz2003==lcz2020)


# In[6]:


lst_1  = data_3d[lcz1,19:39]
lst_1 = np.mean(lst_1, axis=(0))/100

lst_2  = data_3d[lcz2,19:39]
lst_2 = np.mean(lst_2, axis=(0))/100

lst_3  = data_3d[lcz3,19:39]
lst_3 = np.mean(lst_3, axis=(0))/100

lst_4  = data_3d[lcz4,19:39]
lst_4 = np.mean(lst_4, axis=(0))/100

lst_5  = data_3d[lcz5,19:39]
lst_5 = np.mean(lst_5, axis=(0))/100

lst_6  = data_3d[lcz6,19:39]
lst_6 = np.mean(lst_6, axis=(0))/100

lst_8  = data_3d[lcz8,19:39]
lst_8 = np.mean(lst_8, axis=(0))/100

lst_10  = data_3d[lcz10,19:39]
lst_10 = np.mean(lst_10, axis=(0))/100

lst_f = data_3d[forest,18:39]
lst_f = np.mean(lst_f, axis=(0))/100
print(lst_f.shape)

lst_c = data_3d[cropland,18:39]
lst_c = np.mean(lst_c, axis=(0))/100

lst_u = data_3d[urban,18:39]
print(lst_u)
lst_u = np.mean(lst_u, axis=(0))/100
print(lst_u)

# Create an array of x values to use for the scatter plot
x_values = np.arange(2003,2023)
print(x_values.shape)
# Plot the data using scatter()
#plt.scatter(x_values, lst_f)
#plt.scatter(x_values, lst_c)
#plt.scatter(x_values, lst_1)

# Show the plot
plt.show()

# Fit line for stable mean data
m1, b1 = np.polyfit(x_values, lst_f, 1)

# Calculate residuals
#resi = lst_1- (m1 * x_values + b1)
resi = lst_u- lst_c

year1 = 2008
year2 = 2015
mask_before = x_values < year1
mask_after = x_values > year2 
#print(resi.shape)
#print(mask_before)
#avg_before = np.mean(resi[mask_before])
#avg_after = np.mean(resi[mask_after]) 
#diff = avg_after-avg_before
#print(diff)
plt.scatter(x_values, lst_c, c='green')
plt.scatter(x_values, lst_u, c='grey')
plt.show()


# In[4]:


lst_1  = data_3d[lcz1,19:37]
lst_1 = np.mean(lst_1, axis=(0))/100

lst_2  = data_3d[lcz2,19:37]
lst_2 = np.mean(lst_2, axis=(0))/100

lst_3  = data_3d[lcz3,19:37]
lst_3 = np.mean(lst_3, axis=(0))/100

lst_4  = data_3d[lcz4,19:37]
lst_4 = np.mean(lst_4, axis=(0))/100

lst_5  = data_3d[lcz5,19:37]
lst_5 = np.mean(lst_5, axis=(0))/100

lst_6  = data_3d[lcz6,19:37]
lst_6 = np.mean(lst_6, axis=(0))/100

lst_8  = data_3d[lcz8,19:37]
lst_8 = np.mean(lst_8, axis=(0))/100

lst_10  = data_3d[lcz10,19:37]
lst_10 = np.mean(lst_10, axis=(0))/100

lst_f = data_3d[forest,18:36]
lst_f = np.mean(lst_f, axis=(0))/100
print(lst_f.shape)

lst_c = data_3d[cropland,18:36]
lst_c = np.mean(lst_c, axis=(0))/100

lst_u = data_3d[urban,18:36]
print(lst_u)
lst_u = np.mean(lst_u, axis=(0))/100
print(lst_u)

# Create an array of x values to use for the scatter plot
x_values = np.arange(2003,2021)
print(x_values.shape)
# Plot the data using scatter()
#plt.scatter(x_values, lst_f)
#plt.scatter(x_values, lst_c)
#plt.scatter(x_values, lst_1)

# Show the plot
plt.show()

# Fit line for stable mean data
m1, b1 = np.polyfit(x_values, lst_f, 1)

# Calculate residuals
#resi = lst_1- (m1 * x_values + b1)
resi = lst_u- lst_c

year1 = 2008
year2 = 2015
mask_before = x_values < year1
mask_after = x_values > year2 
#print(resi.shape)
#print(mask_before)
#avg_before = np.mean(resi[mask_before])
#avg_after = np.mean(resi[mask_after]) 
#diff = avg_after-avg_before
#print(diff)
plt.scatter(x_values, lst_c, c='green')
plt.scatter(x_values, lst_u, c='grey')
plt.show()


# In[18]:


t = x_values

lcz1 = (lcz2003==1)&(lcz2003>0)&(lcz2003==lcz2020)
lcz2 = (lcz2003==2)&(lcz2003>0)&(lcz2003==lcz2020)
lcz3 = (lcz2003==3)&(lcz2003>0)&(lcz2003==lcz2020)
lcz4 = (lcz2003==4)&(lcz2003>0)&(lcz2003==lcz2020)
lcz5 = (lcz2003==5)&(lcz2003>0)&(lcz2003==lcz2020)
lcz6 = (lcz2003==6)&(lcz2003>0)&(lcz2003==lcz2020)
lcz7 = (lcz2003==7)&(lcz2003>0)&(lcz2003==lcz2020)
lcz8 = (lcz2003==8)&(lcz2003>0)&(lcz2003==lcz2020)
lcz9 = (lcz2003==9)&(lcz2003>0)&(lcz2003==lcz2020)
lcz10 = (lcz2003==10)&(lcz2003>0)&(lcz2003==lcz2020)

lst_1  = data_3d[lcz1,19:37]
lst_1 = np.mean(lst_1, axis=(0))/100

lst_2  = data_3d[lcz2,19:37]
lst_2 = np.mean(lst_2, axis=(0))/100

lst_3  = data_3d[lcz3,19:37]
lst_3 = np.mean(lst_3, axis=(0))/100

lst_4  = data_3d[lcz4,19:37]
lst_4 = np.mean(lst_4, axis=(0))/100

lst_5  = data_3d[lcz5,19:37]
lst_5 = np.mean(lst_5, axis=(0))/100

lst_6  = data_3d[lcz6,19:37]
lst_6 = np.mean(lst_6, axis=(0))/100

lst_8  = data_3d[lcz8,19:37]
lst_8 = np.mean(lst_8, axis=(0))/100

lst_10  = data_3d[lcz10,19:37]
lst_10 = np.mean(lst_10, axis=(0))/100

print(lst_1)
plt.figure(figsize=(12, 6))
plt.plot(t, lst_1, label='lst_1',color='grey')
plt.plot(t, lst_2, label='lst_2',color='grey')
plt.plot(t, lst_3, label='lst_3',color='grey')
plt.plot(t, lst_8, label='lst_8',color='grey')
plt.plot(t, lst_10, label='lst_10',color='grey')


#plt.plot(t, detrended_y+35, label='Detrended Time Series', color='red')
#plt.plot(t, resi_urban+35, label='Detrended Time Series', color='yellow')


# In[19]:


m1, b1 = np.polyfit(x_values, lst_u, 1)


# In[9]:


import numpy as np
from scipy import signal

# Generate a sample time series with a linear trend
t = x_values
y = lst_u  # linear trend + random noise

# Detrend the time series using scipy.signal.detrend
detrended_y = signal.detrend(y)

m3, b3 = np.polyfit(x_values, lst_u, 1)

# Calculate residuals
resi_urban = lst_u - m3*x_values-b3

# Plot the original and detrended time series
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(t, y, label='Original Time Series')
plt.plot(t, detrended_y+35, label='Detrended Time Series', color='red')
plt.plot(t, resi_urban+35, label='Detrended Time Series', color='yellow')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()


# In[6]:


import numpy as np
from scipy import signal

# Generate a sample time series with a linear trend
t = np.arange(100)
y = 3 * t + 5 + np.random.randn(100)  # linear trend + random noise

# Detrend the time series using scipy.signal.detrend
detrended_y = signal.detrend(y)

# Plot the original and detrended time series
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(t, y, label='Original Time Series')
plt.plot(t, detrended_y, label='Detrended Time Series', color='red')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()


# In[184]:


#plt.scatter(x_values, lst_u-lst_c, c='green')
plt.scatter(x_values,lst_c-lst_f,c='yellow')
plt.show()


# In[158]:


import statsmodels.api as sm
import matplotlib.pyplot as plt

# Fit a linear regression model and obtain residuals
# ...

# Generate Q-Q plot
fig = sm.qqplot(lst_u-lst_f, loc=0, scale=1, line='s')

# Set plot labels and title
plt.title("Q-Q Plot")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Sample Quantiles")

# Show the plot
plt.show()


# In[7]:


import glob
import rasterio
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


# In[12]:


# Beijing UHI method
#####region needs to be modified #####
folder_path = "C:/Users/zhaoj/Desktop/umclst/dataimage/jjj/"
fullname = folder_path+'Beijingdata.tif'
with rasterio.open(fullname) as src:
# Read the bands as numpy arrays
    lcz2003 = src.read(1)
    lcz2004 = src.read(2)
    lcz2005 = src.read(3)
    lcz2006 = src.read(4)
    lcz2007 = src.read(5)
    lcz2008 = src.read(6)
    lcz2009 = src.read(7)
    lcz2010 = src.read(8)
    lcz2011 = src.read(9)
    lcz2012 = src.read(10)
    lcz2013 = src.read(11)
    lcz2014 = src.read(12)
    lcz2015 = src.read(13)
    lcz2016 = src.read(14)
    lcz2017 = src.read(15)
    lcz2018 = src.read(16)
    lcz2019 = src.read(17)
    lcz2020 = src.read(18)


lcz1 = lcz2008
lcz2 = lcz2015

year1 = 2008
year2 = 2015

new_data = {'change': [], 'diff': []}
bj_crop = pd.DataFrame(new_data)
bj_forest = pd.DataFrame(new_data)
bj_urban = pd.DataFrame(new_data)

##### calculation #####
change = lcz1*100+lcz2
#change_l = change[(lcz1!=lcz2)&(lcz2<=10)]
change_l = change[(lcz1<=10)&(lcz2<=10)]
change_list = np.unique(change_l)
print(change_list)

# Read the raster data
with rasterio.open(folder_path+'Beijingdata.tif') as src:
    data = src.read()
    data_3d = np.moveaxis(data, 0, -1)
    # Extract land cover and land surface temperature data
    lcz = data_3d[..., :18]
    lst = data_3d[..., 18:36]

# Compute the mean land surface temperature of stable land cover areas
crop = (lcz1 == lcz2) & (lcz2 == 14)
crop_data = lst[crop, :]
crop_mean_data = np.mean(crop_data, axis=0) / 100

forest = (lcz1 == lcz2) & (lcz2 == 11)
forest_data = lst[forest, :]
forest_mean_data = np.mean(forest_data, axis=0) / 100

urban = (lcz1 == lcz2) & (lcz1 <= 10)
urban_data = lst[urban, :]
urban_mean_data = np.mean(urban_data, axis=0) / 100

# Create an array of x values to use for the scatter plot
x_values = np.arange(2003, 2021)
#plt.scatter(x_values,crop_mean_data)

for x in change_list:
  
    # Compute the mean land surface temperature of change areas in 201
    change_type = change == x
    changetype_data = lst[change_type, :]
    changetype_mean_data = changetype_data/ 100

    # Calculate residuals
    resi_crop = changetype_mean_data - crop_mean_data
    resi_forest = changetype_mean_data - forest_mean_data
    resi_urban = changetype_mean_data - urban_mean_data
    # Create a mask for the data before and after 2010
    mask_before = x_values < year1
    mask_after = x_values > year2
    # Add a line for the average value of the two groups
    avg_before_crop = np.mean(resi_crop[:,mask_before],axis=1)
    avg_before_forest = np.mean(resi_forest[:,mask_before],axis=1)
    avg_after_crop = np.mean(resi_crop[:,mask_after],axis=1) 
    avg_after_forest = np.mean(resi_forest[:,mask_after],axis=1) 
    
    avg_before_urban = np.mean(resi_urban[:,mask_before],axis=1)
    avg_after_urban = np.mean(resi_urban[:,mask_after],axis=1)
    
    diff_crop = avg_after_crop-avg_before_crop
    diff_forest = avg_after_forest-avg_before_forest
    diff_urban = avg_after_urban-avg_before_urban
    
    diff_crop = pd.DataFrame(diff_crop)
    diff_crop = diff_crop.rename(columns={0:'diff'})
    diff_crop['change'] = x
    
    diff_forest = pd.DataFrame(diff_forest)
    diff_forest = diff_forest.rename(columns={0:'diff'})
    diff_forest['change'] = x
    
    diff_urban = pd.DataFrame(diff_urban)
    diff_urban = diff_urban.rename(columns={0:'diff'})
    diff_urban['change'] = x
    
    # Add a new row to the dataframe
    
    bj_crop = bj_crop.append(diff_crop, ignore_index=True)
    bj_forest = bj_forest.append(diff_forest, ignore_index=True)
    bj_urban = bj_urban.append(diff_urban, ignore_index=True)
    
bj_crop.to_csv('bj_crop_uhi.csv')
bj_forest.to_csv('bj_forest_uhi.csv')
bj_urban.to_csv('bj_urban_uhi.csv')

import pandas as pd
import matplotlib.pyplot as plt

# Concatenate dataframes and convert 'change' column to categorical
#df = pd.concat([df2008, df2009, df2010, df2011, df2012, df2013, df2014, df2015])
bj_crop['change'] = bj_crop['change'].astype('category')
bj_forest['change'] = bj_forest['change'].astype('category')
bj_urban['change'] = bj_forest['change'].astype('category')

# Create a boxplot based on the dataframe
ax = bj_forest.boxplot(column='diff', by='change')

# Set the size of the figure
fig = plt.gcf()
fig.set_size_inches(12, 6)

# Set the rotation of the x-axis label and xticks
ax.set_xlabel('change', rotation=0, fontsize=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_title("UHI Forest")
plt.ylim(-5,5)
# Show the plot
plt.show()


# In[204]:


# Beijing Regression method
#####region needs to be modified #####
folder_path = "C:/Users/zhaoj/Desktop/umclst/dataimage/jjj/"
fullname = folder_path+'Beijingdata.tif'
with rasterio.open(fullname) as src:
# Read the bands as numpy arrays
    lcz2003 = src.read(1)
    lcz2004 = src.read(2)
    lcz2005 = src.read(3)
    lcz2006 = src.read(4)
    lcz2007 = src.read(5)
    lcz2008 = src.read(6)
    lcz2009 = src.read(7)
    lcz2010 = src.read(8)
    lcz2011 = src.read(9)
    lcz2012 = src.read(10)
    lcz2013 = src.read(11)
    lcz2014 = src.read(12)
    lcz2015 = src.read(13)
    lcz2016 = src.read(14)
    lcz2017 = src.read(15)
    lcz2018 = src.read(16)
    lcz2019 = src.read(17)
    lcz2020 = src.read(18)


lcz1 = lcz2008
lcz2 = lcz2015

year1 = 2008
year2 = 2015

new_data = {'change': [], 'diff': []}
bj_crop = pd.DataFrame(new_data)
bj_forest = pd.DataFrame(new_data)
bj_urban = pd.DataFrame(new_data)

##### calculation #####
change = lcz1*100+lcz2
change_l = change[(lcz1!=lcz2)&(lcz2<=10)]
change_list = np.unique(change_l)
print(change_list)

# Read the raster data
with rasterio.open(folder_path+'Beijingdata.tif') as src:
    data = src.read()
    data_3d = np.moveaxis(data, 0, -1)
    # Extract land cover and land surface temperature data
    lcz = data_3d[..., :18]
    lst = data_3d[..., 18:36]

# Compute the mean land surface temperature of stable land cover areas
crop = (lcz1 == lcz2) & (lcz2 == 14)
crop_data = lst[crop, :]
crop_mean_data = np.mean(crop_data, axis=0) / 100

forest = (lcz1 == lcz2) & (lcz2 == 11)
forest_data = lst[forest, :]
forest_mean_data = np.mean(forest_data, axis=0) / 100

urban = (lcz1 == lcz2) & (lcz1 <= 10)
urban_data = lst[urban, :]
urban_mean_data = np.mean(urban_data, axis=0) / 100

# Create an array of x values to use for the scatter plot
x_values = np.arange(2003, 2021)
#plt.scatter(x_values,crop_mean_data)

for x in change_list:
  
    # Compute the mean land surface temperature of change areas in 201
    change_type = change == x
    changetype_data = lst[change_type, :]
    changetype_mean_data = changetype_data/ 100

    # Fit line for stable mean data
    m1, b1 = np.polyfit(x_values, crop_mean_data, 1)
    m2, b2 = np.polyfit(x_values, forest_mean_data, 1)
    m3, b3 = np.polyfit(x_values, urban_mean_data, 1)

    # Calculate residuals
    resi_crop = changetype_mean_data - m1*x_values-b1
    resi_forest = changetype_mean_data - m2*x_values-b2
    resi_urban = changetype_mean_data - m3*x_values-b3
    
    # Create a mask for the data before and after 2010
    mask_before = x_values < year1
    mask_after = x_values > year2
    # Add a line for the average value of the two groups
    avg_before_crop = np.mean(resi_crop[:,mask_before],axis=1)
    avg_before_forest = np.mean(resi_forest[:,mask_before],axis=1)
    avg_after_crop = np.mean(resi_crop[:,mask_after],axis=1) 
    avg_after_forest = np.mean(resi_forest[:,mask_after],axis=1) 
    
    avg_before_urban = np.mean(resi_urban[:,mask_before],axis=1)
    avg_after_urban = np.mean(resi_urban[:,mask_after],axis=1)
    
    diff_crop = avg_after_crop-avg_before_crop
    diff_forest = avg_after_forest-avg_before_forest
    diff_urban = avg_after_urban-avg_before_urban
    
    diff_crop = pd.DataFrame(diff_crop)
    diff_crop = diff_crop.rename(columns={0:'diff'})
    diff_crop['change'] = x
    
    diff_forest = pd.DataFrame(diff_forest)
    diff_forest = diff_forest.rename(columns={0:'diff'})
    diff_forest['change'] = x
    
    diff_urban = pd.DataFrame(diff_urban)
    diff_urban = diff_urban.rename(columns={0:'diff'})
    diff_urban['change'] = x
    
    # Add a new row to the dataframe
    
    bj_crop = bj_crop.append(diff_crop, ignore_index=True)
    bj_forest = bj_forest.append(diff_forest, ignore_index=True)
    bj_urban = bj_urban.append(diff_urban, ignore_index=True)
    
bj_crop.to_csv('bj_crop_regre.csv')
bj_forest.to_csv('bj_forest_regre.csv')
bj_urban.to_csv('bj_urban_regre.csv')

import pandas as pd
import matplotlib.pyplot as plt

# Concatenate dataframes and convert 'change' column to categorical
#df = pd.concat([df2008, df2009, df2010, df2011, df2012, df2013, df2014, df2015])
bj_crop['change'] = bj_crop['change'].astype('category')
bj_forest['change'] = bj_forest['change'].astype('category')
bj_urban['change'] = bj_forest['change'].astype('category')

print(bj)
# Create a boxplot based on the dataframe
ax = bj_forest.boxplot(column='diff', by='change')

# Set the size of the figure
fig = plt.gcf()
fig.set_size_inches(12, 6)

# Set the rotation of the x-axis label and xticks
ax.set_xlabel('change', rotation=0, fontsize=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_title("Regression by Forest")

plt.ylim(-5,5)
# Show the plot
plt.show()


# In[11]:


#JJJ Regression method
import os

#####region needs to be modified #####
folder_path = "C:/Users/zhaoj/Desktop/umclst/dataimage/jjj/"
filenames = os.listdir(folder_path)
print(filenames)

for name in filenames:
    
    fullname = folder_path+name
    with rasterio.open(fullname) as src:
    # Read the bands as numpy arrays
        lcz2003 = src.read(1)
        lcz2004 = src.read(2)
        lcz2005 = src.read(3)
        lcz2006 = src.read(4)
        lcz2007 = src.read(5)
        lcz2008 = src.read(6)
        lcz2009 = src.read(7)
        lcz2010 = src.read(8)
        lcz2011 = src.read(9)
        lcz2012 = src.read(10)
        lcz2013 = src.read(11)
        lcz2014 = src.read(12)
        lcz2015 = src.read(13)
        lcz2016 = src.read(14)
        lcz2017 = src.read(15)
        lcz2018 = src.read(16)
        lcz2019 = src.read(17)
        lcz2020 = src.read(18)
        data = src.read()
        data_3d = np.moveaxis(data, 0, -1)
        # Extract land cover and land surface temperature data
        lcz = data_3d[..., :18]
        lst = data_3d[..., 18:36]

        lcz1 = lcz2008
        lcz2 = lcz2015

        year1 = 2008
        year2 = 2015

        new_data = {'change': [], 'diff': []}
        bj_crop = pd.DataFrame(new_data)
        bj_forest = pd.DataFrame(new_data)
        bj_urban = pd.DataFrame(new_data)

        ##### calculation #####
        change = lcz1*100+lcz2
        change_l = change[(lcz1!=lcz2)&(lcz2<=10)]
        change_list = np.unique(change_l)


        # Compute the mean land surface temperature of stable land cover areas
        crop = (lcz1 == lcz2) & (lcz2 == 14)
        crop_data = lst[crop, :]
        crop_mean_data = np.mean(crop_data, axis=0) / 100

        forest = (lcz1 == lcz2) & (lcz2 == 11)
        forest_data = lst[forest, :]
        forest_mean_data = np.mean(forest_data, axis=0) / 100

        urban = (lcz1 == lcz2) & (lcz1 <= 10)
        urban_data = lst[urban, :]
        urban_mean_data = np.mean(urban_data, axis=0) / 100

        # Create an array of x values to use for the scatter plot
        x_values = np.arange(2003, 2021)
        #plt.scatter(x_values,crop_mean_data)

        for x in change_list:

            # Compute the mean land surface temperature of change areas in 201
            change_type = change == x
            changetype_data = lst[change_type, :]
            changetype_mean_data = changetype_data/ 100

            # Fit line for stable mean data
            m1, b1 = np.polyfit(x_values, crop_mean_data, 1)
            #m2, b2 = np.polyfit(x_values, forest_mean_data, 1)
            m3, b3 = np.polyfit(x_values, urban_mean_data, 1)

            # Calculate residuals
            resi_crop = changetype_mean_data - m1*x_values-b1
            #resi_forest = changetype_mean_data - m2*x_values-b2
            resi_urban = changetype_mean_data - m3*x_values-b3

            # Create a mask for the data before and after 2010
            mask_before = x_values < year1
            mask_after = x_values > year2
            # Add a line for the average value of the two groups
            avg_before_crop = np.mean(resi_crop[:,mask_before],axis=1)
            #avg_before_forest = np.mean(resi_forest[:,mask_before],axis=1)
            avg_after_crop = np.mean(resi_crop[:,mask_after],axis=1) 
            #avg_after_forest = np.mean(resi_forest[:,mask_after],axis=1) 

            avg_before_urban = np.mean(resi_urban[:,mask_before],axis=1)
            avg_after_urban = np.mean(resi_urban[:,mask_after],axis=1)

            diff_crop = avg_after_crop-avg_before_crop
            #diff_forest = avg_after_forest-avg_before_forest
            diff_urban = avg_after_urban-avg_before_urban

            diff_crop = pd.DataFrame(diff_crop)
            diff_crop = diff_crop.rename(columns={0:'diff'})
            diff_crop['change'] = x

            #diff_forest = pd.DataFrame(diff_forest)
            #diff_forest = diff_forest.rename(columns={0:'diff'})
            #diff_forest['change'] = x

            diff_urban = pd.DataFrame(diff_urban)
            diff_urban = diff_urban.rename(columns={0:'diff'})
            diff_urban['change'] = x

            # Add a new row to the dataframe

            bj_crop = bj_crop.append(diff_crop, ignore_index=True)
            #bj_forest = bj_forest.append(diff_forest, ignore_index=True)
            bj_urban = bj_urban.append(diff_urban, ignore_index=True)

        #bj_crop.to_csv('C:/Users/zhaoj/Desktop/umclst/result/jjj/'+name+'_crop_regre.csv')
        #bj_forest.to_csv('C:/Users/zhaoj/Desktop/umclst/result/'+name+'_forest_regre.csv')
        bj_urban.to_csv('C:/Users/zhaoj/Desktop/umclst/result/jjj/'+name+'_urban_regre.csv')


# In[6]:


#JJJ Regression method unchanged
import os

#####region needs to be modified #####
folder_path = "C:/Users/zhaoj/Desktop/umclst/dataimage/jjj/"
filenames = os.listdir(folder_path)
print(filenames)

for name in filenames:
    
    fullname = folder_path+name
    with rasterio.open(fullname) as src:
    # Read the bands as numpy arrays
        lcz2003 = src.read(1)
        lcz2004 = src.read(2)
        lcz2005 = src.read(3)
        lcz2006 = src.read(4)
        lcz2007 = src.read(5)
        lcz2008 = src.read(6)
        lcz2009 = src.read(7)
        lcz2010 = src.read(8)
        lcz2011 = src.read(9)
        lcz2012 = src.read(10)
        lcz2013 = src.read(11)
        lcz2014 = src.read(12)
        lcz2015 = src.read(13)
        lcz2016 = src.read(14)
        lcz2017 = src.read(15)
        lcz2018 = src.read(16)
        lcz2019 = src.read(17)
        lcz2020 = src.read(18)
        data = src.read()
        data_3d = np.moveaxis(data, 0, -1)
        # Extract land cover and land surface temperature data
        lcz = data_3d[..., :18]
        lst = data_3d[..., 18:36]

        lcz1 = lcz2008
        lcz2 = lcz2015

        year1 = 2008
        year2 = 2015

        new_data = {'change': [], 'diff': []}
        bj_crop = pd.DataFrame(new_data)
        bj_forest = pd.DataFrame(new_data)
        bj_urban = pd.DataFrame(new_data)

        ##### calculation #####
        change = lcz1*100+lcz2
        change_l = change[(lcz1==lcz2)&(lcz1<=10)&(lcz2<=10)]
        change_list = np.unique(change_l)
        change_list = change_list[change_list>0]

        # Compute the mean land surface temperature of stable land cover areas
        crop = (lcz1 == lcz2) & (lcz2 == 14)
        crop_data = lst[crop, :]
        crop_mean_data = np.mean(crop_data, axis=0) / 100

        forest = (lcz1 == lcz2) & (lcz2 == 11)
        forest_data = lst[forest, :]
        forest_mean_data = np.mean(forest_data, axis=0) / 100

        urban = (lcz1 == lcz2) & (lcz1 <= 10)
        urban_data = lst[urban, :]
        urban_mean_data = np.mean(urban_data, axis=0) / 100

        # Create an array of x values to use for the scatter plot
        x_values = np.arange(2003, 2021)
        #plt.scatter(x_values,crop_mean_data)

        for x in change_list:

            # Compute the mean land surface temperature of change areas in 201
            change_type = change == x
            changetype_data = lst[change_type, :]
            changetype_mean_data = changetype_data/ 100

            # Fit line for stable mean data
            m1, b1 = np.polyfit(x_values, crop_mean_data, 1)
            #m2, b2 = np.polyfit(x_values, forest_mean_data, 1)
            m3, b3 = np.polyfit(x_values, urban_mean_data, 1)

            # Calculate residuals
            resi_urban = changetype_mean_data - m3*x_values-b3

            # Create a mask for the data before and after 2010
            mask_before = x_values < year1
            mask_after = x_values > year2
            # Add a line for the average value of the two groups
            avg_before_urban = np.mean(resi_urban[:,mask_before],axis=1)
            avg_after_urban = np.mean(resi_urban[:,mask_after],axis=1)
            
            diff_urban = avg_after_urban-avg_before_urban

            diff_urban = pd.DataFrame(diff_urban)
            diff_urban = diff_urban.rename(columns={0:'diff'})
            diff_urban['change'] = x

            # Add a new row to the dataframe
            bj_urban = bj_urban.append(diff_urban, ignore_index=True)

        bj_urban.to_csv('C:/Users/zhaoj/Desktop/umclst/result/jjjunchange/'+name+'_urban_regre_unchanged.csv')


# In[ ]:





# In[7]:


#JJJ UHI method
import os

#####region needs to be modified #####
folder_path = "C:/Users/zhaoj/Desktop/umclst/dataimage/jjj/"
filenames = os.listdir(folder_path)
print(filenames)

for name in filenames:
    
    fullname = folder_path+name
    with rasterio.open(fullname) as src:
    # Read the bands as numpy arrays
        lcz2003 = src.read(1)
        lcz2004 = src.read(2)
        lcz2005 = src.read(3)
        lcz2006 = src.read(4)
        lcz2007 = src.read(5)
        lcz2008 = src.read(6)
        lcz2009 = src.read(7)
        lcz2010 = src.read(8)
        lcz2011 = src.read(9)
        lcz2012 = src.read(10)
        lcz2013 = src.read(11)
        lcz2014 = src.read(12)
        lcz2015 = src.read(13)
        lcz2016 = src.read(14)
        lcz2017 = src.read(15)
        lcz2018 = src.read(16)
        lcz2019 = src.read(17)
        lcz2020 = src.read(18)
        data = src.read()
        data_3d = np.moveaxis(data, 0, -1)
        # Extract land cover and land surface temperature data
        lcz = data_3d[..., :18]
        lst = data_3d[..., 18:36]

        lcz1 = lcz2008
        lcz2 = lcz2015

        year1 = 2008
        year2 = 2015

        new_data = {'change': [], 'diff': []}
        bj_crop = pd.DataFrame(new_data)
        bj_forest = pd.DataFrame(new_data)
        bj_urban = pd.DataFrame(new_data)

        ##### calculation #####
        change = lcz1*100+lcz2
        change_l = change[(lcz1==lcz2)&(lcz1<=10)&(lcz2<=10)]
        change_list = np.unique(change_l)
        change_list = change_list[change_list>0]


        # Compute the mean land surface temperature of stable land cover areas
        crop = (lcz1 == lcz2) & (lcz2 == 14)
        crop_data = lst[crop, :]
        crop_mean_data = np.mean(crop_data, axis=0) / 100

        #forest = (lcz1 == lcz2) & (lcz2 == 11)
        #forest_data = lst[forest, :]
        #forest_mean_data = np.mean(forest_data, axis=0) / 100

        urban = (lcz1 == lcz2) & (lcz1 <= 10)
        urban_data = lst[urban, :]
        urban_mean_data = np.mean(urban_data, axis=0) / 100

        # Create an array of x values to use for the scatter plot
        x_values = np.arange(2003, 2021)
        #plt.scatter(x_values,crop_mean_data)

        for x in change_list:

            # Compute the mean land surface temperature of change areas in 201
            change_type = change == x
            changetype_data = lst[change_type, :]
            changetype_mean_data = changetype_data/ 100

            # Fit line for stable mean data
            #m1, b1 = np.polyfit(x_values, crop_mean_data, 1)
            #m2, b2 = np.polyfit(x_values, forest_mean_data, 1)
            #m3, b3 = np.polyfit(x_values, urban_mean_data, 1)

            # Calculate residuals
            resi_crop = changetype_mean_data - crop_mean_data
            #resi_forest = changetype_mean_data - forest_mean_data
            resi_urban = changetype_mean_data - urban_mean_data

            # Create a mask for the data before and after 2010
            mask_before = x_values < year1
            mask_after = x_values > year2
            # Add a line for the average value of the two groups
            avg_before_crop = np.mean(resi_crop[:,mask_before],axis=1)
            #avg_before_forest = np.mean(resi_forest[:,mask_before],axis=1)
            avg_after_crop = np.mean(resi_crop[:,mask_after],axis=1) 
            #avg_after_forest = np.mean(resi_forest[:,mask_after],axis=1) 

            avg_before_urban = np.mean(resi_urban[:,mask_before],axis=1)
            avg_after_urban = np.mean(resi_urban[:,mask_after],axis=1)

            diff_crop = avg_after_crop-avg_before_crop
            #diff_forest = avg_after_forest-avg_before_forest
            diff_urban = avg_after_urban-avg_before_urban

            diff_crop = pd.DataFrame(diff_crop)
            diff_crop = diff_crop.rename(columns={0:'diff'})
            diff_crop['change'] = x

            #diff_forest = pd.DataFrame(diff_forest)
            #diff_forest = diff_forest.rename(columns={0:'diff'})
            #diff_forest['change'] = x

            diff_urban = pd.DataFrame(diff_urban)
            diff_urban = diff_urban.rename(columns={0:'diff'})
            diff_urban['change'] = x

            # Add a new row to the dataframe

            bj_crop = bj_crop.append(diff_crop, ignore_index=True)
            #bj_forest = bj_forest.append(diff_forest, ignore_index=True)
            bj_urban = bj_urban.append(diff_urban, ignore_index=True)

        
        bj_urban.to_csv('C:/Users/zhaoj/Desktop/umclst/result/jjjunchange/'+name+'_urban_uhi.csv')


# In[10]:


#JJJ UHI method
import os

#####region needs to be modified #####
folder_path = "C:/Users/zhaoj/Desktop/umclst/dataimage/jjj/"
filenames = os.listdir(folder_path)
print(filenames)

for name in filenames:
    
    fullname = folder_path+name
    with rasterio.open(fullname) as src:
    # Read the bands as numpy arrays
        lcz2003 = src.read(1)
        lcz2004 = src.read(2)
        lcz2005 = src.read(3)
        lcz2006 = src.read(4)
        lcz2007 = src.read(5)
        lcz2008 = src.read(6)
        lcz2009 = src.read(7)
        lcz2010 = src.read(8)
        lcz2011 = src.read(9)
        lcz2012 = src.read(10)
        lcz2013 = src.read(11)
        lcz2014 = src.read(12)
        lcz2015 = src.read(13)
        lcz2016 = src.read(14)
        lcz2017 = src.read(15)
        lcz2018 = src.read(16)
        lcz2019 = src.read(17)
        lcz2020 = src.read(18)
        data = src.read()
        data_3d = np.moveaxis(data, 0, -1)
        # Extract land cover and land surface temperature data
        lcz = data_3d[..., :18]
        lst = data_3d[..., 18:36]

        lcz1 = lcz2008
        lcz2 = lcz2015

        year1 = 2008
        year2 = 2015

        new_data = {'change': [], 'diff': []}
        bj_crop = pd.DataFrame(new_data)
        bj_forest = pd.DataFrame(new_data)
        bj_urban = pd.DataFrame(new_data)

        ##### calculation #####
        change = lcz1*100+lcz2
        change_l = change[(lcz1!=lcz2)&(lcz2<=10)]
        change_list = np.unique(change_l)


        # Compute the mean land surface temperature of stable land cover areas
        crop = (lcz1 == lcz2) & (lcz2 == 14)
        crop_data = lst[crop, :]
        crop_mean_data = np.mean(crop_data, axis=0) / 100

        #forest = (lcz1 == lcz2) & (lcz2 == 11)
        #forest_data = lst[forest, :]
        #forest_mean_data = np.mean(forest_data, axis=0) / 100

        urban = (lcz1 == lcz2) & (lcz1 <= 10)
        urban_data = lst[urban, :]
        urban_mean_data = np.mean(urban_data, axis=0) / 100

        # Create an array of x values to use for the scatter plot
        x_values = np.arange(2003, 2021)
        #plt.scatter(x_values,crop_mean_data)

        for x in change_list:

            # Compute the mean land surface temperature of change areas in 201
            change_type = change == x
            changetype_data = lst[change_type, :]
            changetype_mean_data = changetype_data/ 100

            # Fit line for stable mean data
            #m1, b1 = np.polyfit(x_values, crop_mean_data, 1)
            #m2, b2 = np.polyfit(x_values, forest_mean_data, 1)
            #m3, b3 = np.polyfit(x_values, urban_mean_data, 1)

            # Calculate residuals
            resi_crop = changetype_mean_data - crop_mean_data
            #resi_forest = changetype_mean_data - forest_mean_data
            resi_urban = changetype_mean_data - urban_mean_data

            # Create a mask for the data before and after 2010
            mask_before = x_values < year1
            mask_after = x_values > year2
            # Add a line for the average value of the two groups
            avg_before_crop = np.mean(resi_crop[:,mask_before],axis=1)
            #avg_before_forest = np.mean(resi_forest[:,mask_before],axis=1)
            avg_after_crop = np.mean(resi_crop[:,mask_after],axis=1) 
            #avg_after_forest = np.mean(resi_forest[:,mask_after],axis=1) 

            avg_before_urban = np.mean(resi_urban[:,mask_before],axis=1)
            avg_after_urban = np.mean(resi_urban[:,mask_after],axis=1)

            diff_crop = avg_after_crop-avg_before_crop
            #diff_forest = avg_after_forest-avg_before_forest
            diff_urban = avg_after_urban-avg_before_urban

            diff_crop = pd.DataFrame(diff_crop)
            diff_crop = diff_crop.rename(columns={0:'diff'})
            diff_crop['change'] = x

            #diff_forest = pd.DataFrame(diff_forest)
            #diff_forest = diff_forest.rename(columns={0:'diff'})
            #diff_forest['change'] = x

            diff_urban = pd.DataFrame(diff_urban)
            diff_urban = diff_urban.rename(columns={0:'diff'})
            diff_urban['change'] = x

            # Add a new row to the dataframe

            bj_crop = bj_crop.append(diff_crop, ignore_index=True)
            #bj_forest = bj_forest.append(diff_forest, ignore_index=True)
            bj_urban = bj_urban.append(diff_urban, ignore_index=True)

        bj_crop.to_csv('C:/Users/zhaoj/Desktop/umclst/result/jjj/'+name+'_crop_uhi.csv')
        #bj_forest.to_csv('C:/Users/zhaoj/Desktop/umclst/result/'+name+'_forest_uhi.csv')
        bj_urban.to_csv('C:/Users/zhaoj/Desktop/umclst/result/jjj/'+name+'_urban_uhi.csv')


# In[13]:


#YRD Regression method
import os

#####region needs to be modified #####
folder_path = "C:/Users/zhaoj/Desktop/umclst/dataimage/yrd/"
filenames = os.listdir(folder_path)
print(filenames)

for name in filenames:
    
    fullname = folder_path+name
    with rasterio.open(fullname) as src:
    # Read the bands as numpy arrays
        lcz2003 = src.read(1)
        lcz2004 = src.read(2)
        lcz2005 = src.read(3)
        lcz2006 = src.read(4)
        lcz2007 = src.read(5)
        lcz2008 = src.read(6)
        lcz2009 = src.read(7)
        lcz2010 = src.read(8)
        lcz2011 = src.read(9)
        lcz2012 = src.read(10)
        lcz2013 = src.read(11)
        lcz2014 = src.read(12)
        lcz2015 = src.read(13)
        lcz2016 = src.read(14)
        lcz2017 = src.read(15)
        lcz2018 = src.read(16)
        lcz2019 = src.read(17)
        lcz2020 = src.read(18)
        data = src.read()
        data_3d = np.moveaxis(data, 0, -1)
        # Extract land cover and land surface temperature data
        lcz = data_3d[..., :18]
        lst = data_3d[..., 18:36]

        lcz1 = lcz2008
        lcz2 = lcz2015

        year1 = 2008
        year2 = 2015

        new_data = {'change': [], 'diff': []}
        bj_crop = pd.DataFrame(new_data)
        bj_forest = pd.DataFrame(new_data)
        bj_urban = pd.DataFrame(new_data)

        ##### calculation #####
        change = lcz1*100+lcz2
        change_l = change[(lcz1!=lcz2)&(lcz2<=10)]
        change_list = np.unique(change_l)


        # Compute the mean land surface temperature of stable land cover areas
        crop = (lcz1 == lcz2) & (lcz2 == 14)
        crop_data = lst[crop, :]
        crop_mean_data = np.mean(crop_data, axis=0) / 100

        forest = (lcz1 == lcz2) & (lcz2 == 11)
        forest_data = lst[forest, :]
        forest_mean_data = np.mean(forest_data, axis=0) / 100

        urban = (lcz1 == lcz2) & (lcz1 <= 10)
        urban_data = lst[urban, :]
        urban_mean_data = np.mean(urban_data, axis=0) / 100

        # Create an array of x values to use for the scatter plot
        x_values = np.arange(2003, 2021)
        #plt.scatter(x_values,crop_mean_data)

        for x in change_list:

            # Compute the mean land surface temperature of change areas in 201
            change_type = change == x
            changetype_data = lst[change_type, :]
            changetype_mean_data = changetype_data/ 100

            # Fit line for stable mean data
            m1, b1 = np.polyfit(x_values, crop_mean_data, 1)
            #m2, b2 = np.polyfit(x_values, forest_mean_data, 1)
            m3, b3 = np.polyfit(x_values, urban_mean_data, 1)

            # Calculate residuals
            resi_crop = changetype_mean_data - m1*x_values-b1
            #resi_forest = changetype_mean_data - m2*x_values-b2
            resi_urban = changetype_mean_data - m3*x_values-b3

            # Create a mask for the data before and after 2010
            mask_before = x_values < year1
            mask_after = x_values > year2
            # Add a line for the average value of the two groups
            avg_before_crop = np.mean(resi_crop[:,mask_before],axis=1)
            #avg_before_forest = np.mean(resi_forest[:,mask_before],axis=1)
            avg_after_crop = np.mean(resi_crop[:,mask_after],axis=1) 
            #avg_after_forest = np.mean(resi_forest[:,mask_after],axis=1) 

            avg_before_urban = np.mean(resi_urban[:,mask_before],axis=1)
            avg_after_urban = np.mean(resi_urban[:,mask_after],axis=1)

            diff_crop = avg_after_crop-avg_before_crop
            #diff_forest = avg_after_forest-avg_before_forest
            diff_urban = avg_after_urban-avg_before_urban

            diff_crop = pd.DataFrame(diff_crop)
            diff_crop = diff_crop.rename(columns={0:'diff'})
            diff_crop['change'] = x

            #diff_forest = pd.DataFrame(diff_forest)
            #diff_forest = diff_forest.rename(columns={0:'diff'})
            #diff_forest['change'] = x

            diff_urban = pd.DataFrame(diff_urban)
            diff_urban = diff_urban.rename(columns={0:'diff'})
            diff_urban['change'] = x

            # Add a new row to the dataframe

            bj_crop = bj_crop.append(diff_crop, ignore_index=True)
            #bj_forest = bj_forest.append(diff_forest, ignore_index=True)
            bj_urban = bj_urban.append(diff_urban, ignore_index=True)

        bj_crop.to_csv('C:/Users/zhaoj/Desktop/umclst/result/yrd/'+name+'_crop_regre.csv')
        #bj_forest.to_csv('C:/Users/zhaoj/Desktop/umclst/result/yrd/'+name+'_forest_regre.csv')
        bj_urban.to_csv('C:/Users/zhaoj/Desktop/umclst/result/yrd/'+name+'_urban_regre.csv')


# In[12]:


#YRD UHI method
import os

#####region needs to be modified #####
folder_path = "C:/Users/zhaoj/Desktop/umclst/dataimage/yrd/"
filenames = os.listdir(folder_path)
print(filenames)

for name in filenames:
    
    fullname = folder_path+name
    with rasterio.open(fullname) as src:
    # Read the bands as numpy arrays
        lcz2003 = src.read(1)
        lcz2004 = src.read(2)
        lcz2005 = src.read(3)
        lcz2006 = src.read(4)
        lcz2007 = src.read(5)
        lcz2008 = src.read(6)
        lcz2009 = src.read(7)
        lcz2010 = src.read(8)
        lcz2011 = src.read(9)
        lcz2012 = src.read(10)
        lcz2013 = src.read(11)
        lcz2014 = src.read(12)
        lcz2015 = src.read(13)
        lcz2016 = src.read(14)
        lcz2017 = src.read(15)
        lcz2018 = src.read(16)
        lcz2019 = src.read(17)
        lcz2020 = src.read(18)
        data = src.read()
        data_3d = np.moveaxis(data, 0, -1)
        # Extract land cover and land surface temperature data
        lcz = data_3d[..., :18]
        lst = data_3d[..., 18:36]

        lcz1 = lcz2008
        lcz2 = lcz2015

        year1 = 2008
        year2 = 2015

        new_data = {'change': [], 'diff': []}
        bj_crop = pd.DataFrame(new_data)
        bj_forest = pd.DataFrame(new_data)
        bj_urban = pd.DataFrame(new_data)

        ##### calculation #####
        change = lcz1*100+lcz2
        change_l = change[(lcz1!=lcz2)&(lcz2<=10)]
        change_list = np.unique(change_l)


        # Compute the mean land surface temperature of stable land cover areas
        crop = (lcz1 == lcz2) & (lcz2 == 14)
        crop_data = lst[crop, :]
        crop_mean_data = np.mean(crop_data, axis=0) / 100

        #forest = (lcz1 == lcz2) & (lcz2 == 11)
        #forest_data = lst[forest, :]
        #forest_mean_data = np.mean(forest_data, axis=0) / 100

        urban = (lcz1 == lcz2) & (lcz1 <= 10)
        urban_data = lst[urban, :]
        urban_mean_data = np.mean(urban_data, axis=0) / 100

        # Create an array of x values to use for the scatter plot
        x_values = np.arange(2003, 2021)
        #plt.scatter(x_values,crop_mean_data)

        for x in change_list:

            # Compute the mean land surface temperature of change areas in 201
            change_type = change == x
            changetype_data = lst[change_type, :]
            changetype_mean_data = changetype_data/ 100

            # Fit line for stable mean data
            #m1, b1 = np.polyfit(x_values, crop_mean_data, 1)
            #m2, b2 = np.polyfit(x_values, forest_mean_data, 1)
            #m3, b3 = np.polyfit(x_values, urban_mean_data, 1)

            # Calculate residuals
            resi_crop = changetype_mean_data - crop_mean_data
            resi_forest = changetype_mean_data - forest_mean_data
            resi_urban = changetype_mean_data - urban_mean_data

            # Create a mask for the data before and after 2010
            mask_before = x_values < year1
            mask_after = x_values > year2
            # Add a line for the average value of the two groups
            avg_before_crop = np.mean(resi_crop[:,mask_before],axis=1)
            #avg_before_forest = np.mean(resi_forest[:,mask_before],axis=1)
            avg_after_crop = np.mean(resi_crop[:,mask_after],axis=1) 
            #avg_after_forest = np.mean(resi_forest[:,mask_after],axis=1) 

            avg_before_urban = np.mean(resi_urban[:,mask_before],axis=1)
            avg_after_urban = np.mean(resi_urban[:,mask_after],axis=1)

            diff_crop = avg_after_crop-avg_before_crop
            #diff_forest = avg_after_forest-avg_before_forest
            diff_urban = avg_after_urban-avg_before_urban

            diff_crop = pd.DataFrame(diff_crop)
            diff_crop = diff_crop.rename(columns={0:'diff'})
            diff_crop['change'] = x

            #diff_forest = pd.DataFrame(diff_forest)
            #diff_forest = diff_forest.rename(columns={0:'diff'})
            #diff_forest['change'] = x

            diff_urban = pd.DataFrame(diff_urban)
            diff_urban = diff_urban.rename(columns={0:'diff'})
            diff_urban['change'] = x

            # Add a new row to the dataframe

            bj_crop = bj_crop.append(diff_crop, ignore_index=True)
            #bj_forest = bj_forest.append(diff_forest, ignore_index=True)
            bj_urban = bj_urban.append(diff_urban, ignore_index=True)

        bj_crop.to_csv('C:/Users/zhaoj/Desktop/umclst/result/yrd/'+name+'_crop_uhi.csv')
        #bj_forest.to_csv('C:/Users/zhaoj/Desktop/umclst/result/yrd/'+name+'_forest_uhi.csv')
        bj_urban.to_csv('C:/Users/zhaoj/Desktop/umclst/result/yrd/'+name+'_urban_uhi.csv')


# In[15]:


#PRD Regression method
import os

#####region needs to be modified #####
folder_path = "C:/Users/zhaoj/Desktop/umclst/dataimage/prd/"
filenames = os.listdir(folder_path)
print(filenames)

for name in filenames:
    
    fullname = folder_path+name
    with rasterio.open(fullname) as src:
    # Read the bands as numpy arrays
        lcz2003 = src.read(1)
        lcz2004 = src.read(2)
        lcz2005 = src.read(3)
        lcz2006 = src.read(4)
        lcz2007 = src.read(5)
        lcz2008 = src.read(6)
        lcz2009 = src.read(7)
        lcz2010 = src.read(8)
        lcz2011 = src.read(9)
        lcz2012 = src.read(10)
        lcz2013 = src.read(11)
        lcz2014 = src.read(12)
        lcz2015 = src.read(13)
        lcz2016 = src.read(14)
        lcz2017 = src.read(15)
        lcz2018 = src.read(16)
        lcz2019 = src.read(17)
        lcz2020 = src.read(18)
        data = src.read()
        data_3d = np.moveaxis(data, 0, -1)
        # Extract land cover and land surface temperature data
        lcz = data_3d[..., :18]
        lst = data_3d[..., 18:36]

        lcz1 = lcz2008
        lcz2 = lcz2015

        year1 = 2008
        year2 = 2015

        new_data = {'change': [], 'diff': []}
        bj_crop = pd.DataFrame(new_data)
        bj_forest = pd.DataFrame(new_data)
        bj_urban = pd.DataFrame(new_data)

        ##### calculation #####
        change = lcz1*100+lcz2
        change_l = change[(lcz1!=lcz2)&(lcz2<=10)]
        change_list = np.unique(change_l)


        # Compute the mean land surface temperature of stable land cover areas
        crop = (lcz1 == lcz2) & (lcz2 == 14)
        crop_data = lst[crop, :]
        crop_mean_data = np.mean(crop_data, axis=0) / 100

        forest = (lcz1 == lcz2) & (lcz2 == 11)
        forest_data = lst[forest, :]
        forest_mean_data = np.mean(forest_data, axis=0) / 100

        urban = (lcz1 == lcz2) & (lcz1 <= 10)
        urban_data = lst[urban, :]
        urban_mean_data = np.mean(urban_data, axis=0) / 100

        # Create an array of x values to use for the scatter plot
        x_values = np.arange(2003, 2021)
        #plt.scatter(x_values,crop_mean_data)

        for x in change_list:

            # Compute the mean land surface temperature of change areas in 201
            change_type = change == x
            changetype_data = lst[change_type, :]
            changetype_mean_data = changetype_data/ 100

            # Fit line for stable mean data
            m1, b1 = np.polyfit(x_values, crop_mean_data, 1)
            #m2, b2 = np.polyfit(x_values, forest_mean_data, 1)
            m3, b3 = np.polyfit(x_values, urban_mean_data, 1)

            # Calculate residuals
            resi_crop = changetype_mean_data - m1*x_values-b1
            #resi_forest = changetype_mean_data - m2*x_values-b2
            resi_urban = changetype_mean_data - m3*x_values-b3

            # Create a mask for the data before and after 2010
            mask_before = x_values < year1
            mask_after = x_values > year2
            # Add a line for the average value of the two groups
            avg_before_crop = np.mean(resi_crop[:,mask_before],axis=1)
            #avg_before_forest = np.mean(resi_forest[:,mask_before],axis=1)
            avg_after_crop = np.mean(resi_crop[:,mask_after],axis=1) 
            #avg_after_forest = np.mean(resi_forest[:,mask_after],axis=1) 

            avg_before_urban = np.mean(resi_urban[:,mask_before],axis=1)
            avg_after_urban = np.mean(resi_urban[:,mask_after],axis=1)

            diff_crop = avg_after_crop-avg_before_crop
            #diff_forest = avg_after_forest-avg_before_forest
            diff_urban = avg_after_urban-avg_before_urban

            diff_crop = pd.DataFrame(diff_crop)
            diff_crop = diff_crop.rename(columns={0:'diff'})
            diff_crop['change'] = x

            #diff_forest = pd.DataFrame(diff_forest)
            #diff_forest = diff_forest.rename(columns={0:'diff'})
            #diff_forest['change'] = x

            diff_urban = pd.DataFrame(diff_urban)
            diff_urban = diff_urban.rename(columns={0:'diff'})
            diff_urban['change'] = x

            # Add a new row to the dataframe

            bj_crop = bj_crop.append(diff_crop, ignore_index=True)
            #bj_forest = bj_forest.append(diff_forest, ignore_index=True)
            bj_urban = bj_urban.append(diff_urban, ignore_index=True)

        bj_crop.to_csv('C:/Users/zhaoj/Desktop/umclst/result/prd/'+name+'_crop_regre.csv')
        #bj_forest.to_csv('C:/Users/zhaoj/Desktop/umclst/result/prd/'+name+'_forest_regre.csv')
        bj_urban.to_csv('C:/Users/zhaoj/Desktop/umclst/result/prd/'+name+'_urban_regre.csv')


# In[14]:


#PRD UHI method
import os

#####region needs to be modified #####
folder_path = "C:/Users/zhaoj/Desktop/umclst/dataimage/prd/"
filenames = os.listdir(folder_path)
print(filenames)

for name in filenames:
    
    fullname = folder_path+name
    with rasterio.open(fullname) as src:
    # Read the bands as numpy arrays
        lcz2003 = src.read(1)
        lcz2004 = src.read(2)
        lcz2005 = src.read(3)
        lcz2006 = src.read(4)
        lcz2007 = src.read(5)
        lcz2008 = src.read(6)
        lcz2009 = src.read(7)
        lcz2010 = src.read(8)
        lcz2011 = src.read(9)
        lcz2012 = src.read(10)
        lcz2013 = src.read(11)
        lcz2014 = src.read(12)
        lcz2015 = src.read(13)
        lcz2016 = src.read(14)
        lcz2017 = src.read(15)
        lcz2018 = src.read(16)
        lcz2019 = src.read(17)
        lcz2020 = src.read(18)
        data = src.read()
        data_3d = np.moveaxis(data, 0, -1)
        # Extract land cover and land surface temperature data
        lcz = data_3d[..., :18]
        lst = data_3d[..., 18:36]

        lcz1 = lcz2008
        lcz2 = lcz2015

        year1 = 2008
        year2 = 2015

        new_data = {'change': [], 'diff': []}
        bj_crop = pd.DataFrame(new_data)
        bj_forest = pd.DataFrame(new_data)
        bj_urban = pd.DataFrame(new_data)

        ##### calculation #####
        change = lcz1*100+lcz2
        change_l = change[(lcz1!=lcz2)&(lcz2<=10)]
        change_list = np.unique(change_l)


        # Compute the mean land surface temperature of stable land cover areas
        crop = (lcz1 == lcz2) & (lcz2 == 14)
        crop_data = lst[crop, :]
        crop_mean_data = np.mean(crop_data, axis=0) / 100

        #forest = (lcz1 == lcz2) & (lcz2 == 11)
        #forest_data = lst[forest, :]
        #forest_mean_data = np.mean(forest_data, axis=0) / 100

        urban = (lcz1 == lcz2) & (lcz1 <= 10)
        urban_data = lst[urban, :]
        urban_mean_data = np.mean(urban_data, axis=0) / 100

        # Create an array of x values to use for the scatter plot
        x_values = np.arange(2003, 2021)
        #plt.scatter(x_values,crop_mean_data)

        for x in change_list:

            # Compute the mean land surface temperature of change areas in 201
            change_type = change == x
            changetype_data = lst[change_type, :]
            changetype_mean_data = changetype_data/ 100

            # Fit line for stable mean data
            #m1, b1 = np.polyfit(x_values, crop_mean_data, 1)
            #m2, b2 = np.polyfit(x_values, forest_mean_data, 1)
            #m3, b3 = np.polyfit(x_values, urban_mean_data, 1)

            # Calculate residuals
            resi_crop = changetype_mean_data - crop_mean_data
            resi_forest = changetype_mean_data - forest_mean_data
            resi_urban = changetype_mean_data - urban_mean_data

            # Create a mask for the data before and after 2010
            mask_before = x_values < year1
            mask_after = x_values > year2
            # Add a line for the average value of the two groups
            avg_before_crop = np.mean(resi_crop[:,mask_before],axis=1)
            #avg_before_forest = np.mean(resi_forest[:,mask_before],axis=1)
            avg_after_crop = np.mean(resi_crop[:,mask_after],axis=1) 
            #avg_after_forest = np.mean(resi_forest[:,mask_after],axis=1) 

            avg_before_urban = np.mean(resi_urban[:,mask_before],axis=1)
            avg_after_urban = np.mean(resi_urban[:,mask_after],axis=1)

            diff_crop = avg_after_crop-avg_before_crop
            #diff_forest = avg_after_forest-avg_before_forest
            diff_urban = avg_after_urban-avg_before_urban

            diff_crop = pd.DataFrame(diff_crop)
            diff_crop = diff_crop.rename(columns={0:'diff'})
            diff_crop['change'] = x

            #diff_forest = pd.DataFrame(diff_forest)
            #diff_forest = diff_forest.rename(columns={0:'diff'})
            #diff_forest['change'] = x

            diff_urban = pd.DataFrame(diff_urban)
            diff_urban = diff_urban.rename(columns={0:'diff'})
            diff_urban['change'] = x

            # Add a new row to the dataframe

            bj_crop = bj_crop.append(diff_crop, ignore_index=True)
            #bj_forest = bj_forest.append(diff_forest, ignore_index=True)
            bj_urban = bj_urban.append(diff_urban, ignore_index=True)

        bj_crop.to_csv('C:/Users/zhaoj/Desktop/umclst/result/prd/'+name+'_crop_uhi.csv')
        #bj_forest.to_csv('C:/Users/zhaoj/Desktop/umclst/result/prd/'+name+'_forest_uhi.csv')
        bj_urban.to_csv('C:/Users/zhaoj/Desktop/umclst/result/prd/'+name+'_urban_uhi.csv')

