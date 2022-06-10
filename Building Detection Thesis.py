#!/usr/bin/env python
# coding: utf-8

# Problem Statement:
# Abandoned buildings and property is a common plight that urban centers face  
# 
# According to the existing literature, abandoned buildings and land in a city are responsible for:
# 
#     increasing crime rates (drug use, prostitution, etc.) 
#     increasing danger for public health and safety (collapsing parts of buildings, fires, etc.)
#     depressing nearby property values 
#     generating low property taxes; increasing costs for local governments (to secure, inspect, provide additional police and fire services, etc.) 
#     
# In other words, abandoned buildings and land in the city contribute to the downgrade of the quality of life, 
# creating an unattractive urban environment for the citizens and visitors, as well as for future investors
# 
# 
# Business Potential:
#                     
#     Real Estate Agencies can identify and proceed to buy these buildings
#     Municipalities may want to transform them for other purposes.
#     Investors can get the upper hand when searching for vacant buildings to buy.
#     Businesses can easily assess the surrounding area for possible dangers when census data are scarse or outdated.
#                     
# Anyone can locate an abandonded building just by having a walk around but naturally this is inefficient.
# 

# In[1]:


# google Housing Vacancy


# https://www.tandfonline.com/doi/abs/10.1080/01431161.2019.1615655?journalCode=tres20
#     
# https://www.mdpi.com/2072-4292/10/12/1920    
# 
# https://www.researchgate.net/publication/350617115_Detecting_individual_abandoned_houses_from_google_street_view_A_hierarchical_deep_learning_approach

# https://www.arcgis.com/home/item.html?id=d3da5dd386d140cf93fc9ecbf8da5e31
# 
# https://www.arcgis.com/home/item.html?id=d3da5dd386d140cf93fc9ecbf8da5e31

# https://www.nerdwallet.com/article/small-business/business-location
#     
# https://journalistsresource.org/politics-and-government/abandoned-buildings-revitalization/
#     
# https://www.fortunebuilders.com/vacant-property/

# ### Can freely available AND easily accessible satellite data detect abandonded buildings?

# We are going to approach this topic from different angles
# 
# 1) Use Google Earth Engine since it offers a great variety of data, has a relatively good python api, and because anyone can access it
# 
# 2) Use Sentinel Hub since it has a great python API and there are existing libraries for a lot of tasks. However it's not free

# Methodology
# 
# Data Collection:
#     
#     Get satellite data from Sentinel-2 (10m spatial resolution) and Landsat-8(30m spatial resolution)
#     
#     Get satellite data from VIIRS(750m spatial resolution-yep bad) since absence of light may indicate abandonment
#     
#     Possibly get google satellite images(bigger spatial resolution but outdated), can be used to identify characteristics of an abandonded building
#     
#     OSM geojson for buildings
#     
# Preprocessing: 
#     
#     Cloud Masks for all data
#     
#     Use neural network in order to improve spatial resolution of Sentinel-2 images (
#     https://up42.com/blog/tech/sentinel-2-superresolution
#     https://github.com/lanha/DSen2
#     https://github.com/up42/DSen2)
#     
#     Possibly deblur VIIRs, or use them along with DMSP-OLS
# 
# Models:
#     
#     Map NDVI,NDWI since abandonded places tend to have low vegetation or water
#     
#     Measure Mean Radiance of Lights over a particular area
# 

# NPP-VIIRS PREPROCESSING from housing_vacancy-npp.pdf
# The minimum value of NPP-VIIRS data
# should be 0, representing regions without light intensity. However, values of a few pixels
# were lower than 0, caused by imaging error. In our study, these negative values were
# reset to 0. (2) Some abruptly large pixel also existed which might be extraordinary noises
# or pixels associated with the weak light reflected by high reflectance surfaces (e.g. snowcapped mountains). To distinguish these pixels, the maximum radiance value derived
# from the city centre artificially was first set as the upper threshold and then used to
# distinguish pixels with larger values. A Max Filter was used in these abnormal pixels to
# fix their values. In this way, the background noises of NPP-VIIRS NTL data was eliminated
# effectively.

# https://www.esri.com/about/newsroom/arcnews/start-up-fights-urban-blight/

# https://github.com/d-smit/sentinel2-deep-learning
# 
# https://github.com/sentinel-hub/multi-temporal-super-resolution
# 
# 
# https://machinelearningmastery.com/a-gentle-introduction-to-particle-swarm-optimization/
# 
# https://www.tandfonline.com/doi/abs/10.1080/01431161.2017.1331060
# 
# https://github.com/KlemenKozelj/sentinel2-earth-observation

# https://code.earthengine.google.com/?scriptPath=Examples%3ADatasets%2FCOPERNICUS_S2_SR
# 
# https://code.earthengine.google.com/?scriptPath=Examples%3ADatasets%2FSKYSAT_GEN-A_PUBLIC_ORTHO_MULTISPECTRAL

# In[2]:


#Firstly, we will do the analysis, with almost zero preprocessing


# In[1]:


import rasterio
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import requests
import json
import datetime

from pprint import pprint
from rasterio.features import Window
from rasterio.windows import bounds
from shapely.geometry import MultiPolygon, box
from PIL import Image
from rasterio.features import Window
from subprocess import call
from IPython import display


# In[2]:


# reminder that if you are installing libraries in a Google Colab instance you will be prompted to restart your kernal

try:
    import geemap, ee
except ModuleNotFoundError:
    if 'google.colab' in str(get_ipython()):
        print("package not found, installing w/ pip in Google Colab...")
        get_ipython().system('pip install geemap')
    else:
        print("package not found, installing w/ conda...")
        get_ipython().system('conda install mamba -c conda-forge -y')
        get_ipython().system('mamba install geemap -c conda-forge -y')
    import geemap, ee


# Get Sentinel-2 Data though Google Earth Engine

# We focus on Thessaly

# In[3]:


try:
        ee.Initialize()
except Exception as e:
        ee.Authenticate()
        ee.Initialize()

# get our Nepal boundary
aoi = ee.FeatureCollection("FAO/GAUL/2015/level1").filter(ee.Filter.eq('ADM1_NAME','Thessalia')).geometry()

# Sentinel-2 image filtered on 2019 and on Nepal
se2 = ee.ImageCollection('COPERNICUS/S2').filterDate("2015-01-01","2022-01-01").filterBounds(aoi).median().divide(10000)

rgb = ['B4','B3','B2']

# set some thresholds
rgbViz = {"min":0.0, "max":0.3,"bands":rgb}


# initialize our map
map1 = geemap.Map()
map1.centerObject(aoi, 7)
map1.addLayer(se2.clip(aoi), rgbViz, "S2")

map1.addLayerControl()
map1


# The image is not clipped near coastal areas which may be problematic for our analysis as we go further.

# Get VIIRS Data through Google Earth Engine

# I select data between 2015 and 2022(latest available trhough the engine) since I want to capture the whole covid situation as well as changes possibly related to politics etc.

# In[6]:


viirs = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG").filterDate('2015-01-01','2022-01-01').select('avg_rad')


# In[7]:


viirs


# In[8]:


viirsJan2015= ee.Image(viirs)

# or equivalently

viirsJan2015 = viirs.first()


# In[9]:


scaleFactor=100

greece_pref = ee.FeatureCollection(viirsJan2015.reduceRegions(reducer=ee.Reducer.mean(),
                                        collection=aoi,
                                        scale=scaleFactor))


# In[10]:


radiance_img = greece_pref.reduceToImage(properties=['mean'],reducer=ee.Reducer.first());


# In[11]:


#map1 = geemap.Map()
#map1.centerObject(aoi, zoom=7)
viz_params = {'min':1,
              'max':17,
              'palette':['2C105C','711F81','B63679','EE605E','FDAE78','FCFDBF']}
map1.addLayer(greece_pref, {}, "Prefecture boundaries", opacity=.5)
map1.addLayer(radiance_img, viz_params, 'VIIRS Jan 2015 avg rad by prefecture',opacity=.6)
map1.addLayerControl()
map1

# file for Greece
greece0 = ee.Feature(ee.FeatureCollection("FAO/GAUL/2015/level0").filter(ee.Filter.eq('ADM0_NAME', 'Greece')).first()).geometry()
# thessalia = ee.FeatureCollection("FAO/GAUL/2015/level2").filter(ee.Filter.eq('ADM1_NAME', 'Thessalia'))
# 
# print(f"There are {thessalia.size().getInfo()} level one admin units in Thessalia.")
# thessalia.getInfo()

# In[12]:


# file for Greece
magnisia = ee.Feature(ee.FeatureCollection("FAO/GAUL/2015/level0").filter(ee.Filter.eq('ADM2_NAME', 'Magnisias'))).geometry()


# A region that has lower-levels of light may be affected by background light that the VIIRS instrument is sensitive to can influence interpretation. 
# 
# As we look at this scene, you can see the relatively high levels of “noise” present.
# 
# As discussed earlier, one approach to increase the signal / noise ratio would be to reduce data over time.
# 
# But if the noise levels persist throughout the time period, that may not reduce the noise much. And what if your analysis is specifically to look at December 2017?
# 
# Or what if you’re looking to conduct comparative analysis on these data or use them as inputs for a model for statistical inference?
# 
# In this case, you will very likely want to reduce the noise levels in your data in order for your algorithm to learn your data without over-fitting (in other words, a more sensitive model might “learn” the noise…which is generally bad). Additionally, many loss functions are subject to “exploding” or “vanishing” gradients if your data are not close to zero and scale

# In[13]:


scaleFactor=500

greeceSOL500 = viirsJan2015.reduceRegion(reducer=ee.Reducer.sum(),
                                     geometry=aoi,
                                     scale=scaleFactor,
                                     maxPixels=2e9)
print(f"The SOL for Greece at {scaleFactor}m scale is: {greeceSOL500.get('avg_rad').getInfo():.2f}")


# In[14]:


scaleFactor=100

greeceSOL100 = viirsJan2015.reduceRegion(reducer=ee.Reducer.sum(),
                                     geometry=aoi,
                                     scale=scaleFactor,
                                     maxPixels=2e9)

print(f"The SOL for Greece at {scaleFactor}m scale is: {greeceSOL100.get('avg_rad').getInfo():.2f}")


# In[15]:


# Sum of Lights is prone to erros and outliers, and we can see that depending on the scale factor the results vary greatly
# Thus we will try another metric, mean lights


# In[16]:


scaleFactor=500

greeceSOL = viirsJan2015.reduceRegion(reducer=ee.Reducer.mean(),
                                     geometry=aoi,
                                     scale=scaleFactor,
                                     maxPixels=2e9)

print(f"The avg radiance for Greece in Jan 2019 (per {scaleFactor}m grid) is: {greeceSOL.get('avg_rad').getInfo():.4f}")


# In[17]:


scaleFactor=100

greeceSOL = viirsJan2015.reduceRegion(reducer=ee.Reducer.mean(),
                                     geometry=aoi,
                                     scale=scaleFactor,
                                     maxPixels=2e9)

print(f"The avg radiance for Greece in Jan 2019 (per {scaleFactor}m grid) is: {greeceSOL.get('avg_rad').getInfo():.4f}")


# In[18]:


greece_pref.aggregate_stats('mean').getInfo()

poi = ee.Geometry.Polygon([[[ 39.65925368635945,21.11034311414164],
      [39.62786608248593,21.121745024122855],
      [39.60034000685686,21.12322992837611],
      [39.58496058866589,21.137953857024677],
      [39.580256262903916,21.15488061550989],
      [39.56527812229155,21.166081893846382],
      [39.538599346735936,21.16055706829246],
      [39.51027513801262,21.169011461923077],
      [39.498155244777976,21.155781358741176],
      [39.48480473493292,21.153034556881146],
      [39.46943863854366,21.167700546861106],
      [21.174852870112257, 39.452195322381776],
      [21.171553152754843, 39.43618716001877],
      [21.178718950237318, 39.41894827384953],
      [21.20235665616196, 39.40387871570699],
      [21.225994300925528, 39.38880920157895],
      [21.240258983703445, 39.35432255291319],
      [21.275209471620975, 39.3337972870931],
      [21.327398793001453, 39.32761248053104],
      [21.345921841660804, 39.330867660350165],
      [21.345239645649674, 39.34462841269293],
      [21.35150913959843, 39.35755092753244],
      [21.361265596364863, 39.37005871386407],
      [21.380355009240894, 39.37598482735699],
      [21.39831180086926, 39.376568988722624],
      [21.432009284114198, 39.36707552911248],
      [21.463998840482606, 39.34957357665826],
      [21.48271370534117, 39.351972598832454],
      [21.50454993836163, 39.37162830392784],
      [21.526386131613272, 39.39128401640999],
      [21.552302393079845, 39.4291640940549],
      [21.570865670478067, 39.43240141531394],
      [21.601695906461664, 39.40951728703932],
      [21.63006019969542, 39.425204379637655],
      [21.666602569137673, 39.42896343594438],
      [21.68632514659234, 39.43750703730765],
      [21.707751199729085, 39.45404588469823],
      [21.72518180764335, 39.45190991651593],
      [21.73276224290483, 39.43731082748466],
      [21.749015695688772, 39.429841851309035],
      [21.768154175528267, 39.435696671321],
      [21.777402306073682, 39.461911683720636],
      [21.79074842463069, 39.47395575603591],
      [21.807585985667654, 39.4855672084231],
      [21.83081344946508, 39.49364269733661],
      [21.8709007384975, 39.49688890551485],
      [21.887742770255013, 39.508495953001734],
      [21.924343072358383, 39.512165745502706],
      [21.955157671897908, 39.52201820293759],
      [21.985972263102518, 39.53187054444512],
      [22.019094507602635, 39.519509912853344],
      [22.036511721511346, 39.51732050722246],
      [22.078986902962743, 39.53114819344666],
      [22.121462078170254, 39.544975817348536],
      [22.16449687341818, 39.544998152350395],
      [22.16694049895755, 39.57209160766255],
      [22.155971086520093, 39.587185720353965],
      [22.12226475056408, 39.596888728129265],
      [22.114840292925066, 39.62795527388407],
      [22.100334882309216, 39.6434773979652],
      [22.082890902607616, 39.64568021330115],
      [22.064242942582307, 39.64254544701414],
      [22.04855574641285, 39.65272108930522],
      [22.043985174581813, 39.680670780985714],
      [22.028864407802136, 39.693530796745065],
      [21.973495779671108, 39.68673512667089],
      [21.956586815127977, 39.675141484000854],
      [21.939686862605406, 39.67999299192468],
      [21.932137558888062, 39.694609928246024],
      [21.92403095050353, 39.73942833729418],
      [21.917123780914032, 39.82242555776728],
      [21.889102800529045, 39.8422997537275],
      [21.876256092029305, 39.83294462350949],
      [21.823687800363164, 39.839419205903106],
      [21.786318279925528, 39.83304048273634],
      [21.748948829700513, 39.82666167669445],
      [21.71331610770214, 39.82825804889596],
      [21.663358593285334, 39.83841592302598],
      [21.613401082428158, 39.84857376595426],
      [21.57642613284989, 39.85232498268147],
      [21.539451332907234, 39.85607620595456],
      [21.502476455018908, 39.85982741510876],
      [21.465501544543507, 39.863578611245856],
      [21.427010578610453, 39.85174416081156],
      [21.4066548217214, 39.840471567500366],
      [21.383462989304785, 39.815888547086985],
      [21.36367799107608, 39.807286951729054],
      [21.321744575162338, 39.7958403897652],
      [21.30366294520401, 39.79523847184679],
      [21.29130234101267, 39.80487897631912],
      [21.26384318023154, 39.810849727309105],
      [21.244615571241848, 39.80490132468022],
      [21.23480104596234, 39.792375711605345],
      [21.2150606616099, 39.78372947534038],
      [21.2047155354438, 39.76853734574027],
      [21.1984370514637, 39.75560592794437],
      [21.214775204257904, 39.69628654407656],
      [21.216224414597786, 39.66878279935999],
      [21.19816060165989, 39.66815402476963],
      [21.182344156264634, 39.67819598370166],
      [21.14330479486066, 39.68002416912158],
      [21.123586600269782, 39.6713869283674],
      [21.11034311414164, 39.65925368635945]]]).buffer(1600)

# In[19]:


#Let's check the activity at T.Oikonomaki and the neighbouring areas that comprise a part of Volos' city center.


# In[4]:


poi_oikonomaki= ee.Geometry.Polygon([[[
              22.949098348617554,39.35979978548018],
            [22.951193153858185,39.35979978548018 ],
            [22.951193153858185,39.36196275350115],
            [22.949098348617554,39.36196275350115],
            [22.949098348617554,39.35979978548018]]])


# In[5]:


poi_rozou =ee.Geometry.Polygon(  [
          [
            [
              22.944881916046143,
              39.36383326244071
            ],
            [
              22.947140336036682,
              39.36383326244071
            ],
            [
              22.947140336036682,
              39.36550879775804
            ],
            [
              22.944881916046143,
              39.36550879775804
            ],
            [
              22.944881916046143,
              39.36383326244071
            ]
          ]
        ])


# In[22]:


#poi = ee.Geometry.Point(22.9416,39.3725).buffer(1600)


# In[6]:


poi_fil_kounta =ee.Geometry.Point([22.95246586203575, 39.358972323625814])


# In[24]:


def poi_mean(img):
    mean = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=poi_rozou, scale=30).get('avg_rad')
    return img.set('date', img.date().format()).set('mean',mean)
#this function takes as input the image and our desired point of interest as geometry


# In[25]:


def poi(poi):
    selected_poi =  poi
    return  selected_poi
# function to use in case we want a lot of pois in a particular area, in order to automate it


# In[26]:



poi_reduced_imgs = viirs.map(poi_mean)


# In[27]:


nested_list = poi_reduced_imgs.reduceColumns(ee.Reducer.toList(2), ['date','mean']).values().get(0)


# In[28]:


import pandas as pd
# dont forget we need to call the callback method "getInfo" to retrieve the data
magnisia_lights = pd.DataFrame(nested_list.getInfo(), columns=['date','mean_rad'])

magnisia_lights


# In[29]:


volosMap = geemap.Map()
volosMap.centerObject(poi_rozou, zoom=13)
volosMap.add_basemap("SATELLITE")
volosMap.addLayer(poi_rozou, {}, "T.Rozou")
volosMap


# In[30]:


pd. set_option('display.max_rows', None)

magnisia_lights.date


# In[ ]:




# Sklearn
from sklearn.linear_model import LinearRegression

cols = magnisia_lights['mean']
x= magnisia_lights.index.values.reshape(-1,1)
y = magnisia_lights['mean']

# instantiate and fit
lm_2 = LinearRegression()
lm_2.fit(x, y)

# print the coefficients
print('Intercept: ', lm_2.intercept_)
print('mean: ', lm_2.coef_[0])import datetime as dt
magnisia_lights['date'] = pd.to_datetime(magnisia_lights['date'])
magnisia_lights['date']=magnisia_lights['date'].map(dt.datetime.toordinal)

x= magnisia_lights['date'].values.reshape(-1,1)
y = magnisia_lights['mean']

# instantiate and fit
lm_2 = LinearRegression()
lm_2.fit(x, y)

#lm_2.score(x,y)
# print the coefficients
print('Intercept: ', lm_2.intercept_)
print('mean: ', lm_2.coef_[0])
# In[31]:


#from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
def arima_model(series):
    log_series = series.apply(lambda x: np.log(x))
    log_series_diff=log_series-log_series.shift()

    model=ARIMA(log_series, order=(1,1,1))
    results_ARIMA = model.fit(disp=-1)
    
    RSS = np.nansum((results_ARIMA.fittedvalues-log_series_diff)**2)
    return RSS, log_series_diff, results_ARIMA


RSS, log_series_diff, arima = arima_model(magnisia_lights['mean_rad'])
plt.plot(log_series_diff)
plt.plot(arima.fittedvalues, color='red')
plt.title('RSS: %.4f'% RSS)
plt.legend(['Lights', 'Light Prediction'])


# In[32]:


#magnisia_lights['date'] = pd.to_datetime(magnisia_lights['date'])
from datetime import datetime,date
magnisia_lights['date'] = pd.to_datetime(magnisia_lights['date']).dt.date
df_ntl = magnisia_lights.set_index('date')

df_ntl


# In[33]:


import seaborn as sns
import matplotlib.pyplot as plt
# we create a figure with pyplot and set the dimensions to 15 x 7
fig, ax = plt.subplots(figsize=(15,7))

# we'll create the plot by setting our dataframe to the data argument
sns.lineplot(data=df_ntl, ax=ax)

# we'll set the labels and title
ax.set_ylabel('mean radiance',fontsize=20)
ax.set_xlabel('date',fontsize=20)
ax.set_title('Monthly mean radiance for T.Oikonomaki (Jan 2015 to Jan 2021)',fontsize=20);


# In[34]:


fig, ax = plt.subplots(figsize=(15,7))

# we'll plot the moving averate using ".rolling" and set a window of 12 months
window=12
sns.lineplot(data=df_ntl.rolling(window).mean(), ax=ax)

# we'll set the labels and title
ax.set_ylabel('mean radiance',fontsize=20)
ax.set_xlabel('date',fontsize=20)
ax.set_title(f'Monthly mean radiance ({window} moving avg.) for T.Oikonomaki (Jan 2014 to May 2021)',fontsize=20);


# In[35]:


df_ntl['mean_rad'].idxmax() 


# In[36]:


#### otan to ooi einai h rozou mou kanei entypwsh


# #### Otan poi einai h t.oik
# This may be due to led lights in the city centre during the christmas period.
# We are using mean which can be affected from outliers, i.e the leds are the outlier

# Drop in 2019 could be due to COVID-19

# In[ ]:





# In[37]:


#import data
from simple_ndvi.importData import importData

#manual Module
#folium
from simple_ndvi.myfolium import *


#indexes, topography and covariates
#from simple_ndvi.GetIndexes import GetIndexes
from simple_ndvi.GetIndexes import getNDVI

print("Initialized")


# ### Visualization of ndvi over poi

# In[38]:


# Create a folium map object.
#location = [22.9416,39.3725]
#mapObject = foliumInitialize(poi_rozou,6,600)
#my_map = mapObject.Initialize()
# initialize our map

map1.centerObject(poi_rozou, 7)
map1.addLayer(se2.clip(poi_rozou), rgbViz, "S2rozou")

map1.addLayerControl()

print('folium map Initialized')


# In[39]:


startyear = 2015
endyear = 2022

startDate = ee.Date.fromYMD(startyear,1,1)
endDate = ee.Date.fromYMD(endyear,1,1)

#gee assets to get the study area
studyArea = poi_rozou



print("getting images")
ss2 = importData(studyArea,startDate,endDate)

ss2 = ss2.mean().clip(studyArea)

print(str(ss2.bandNames().getInfo()))

#get Indexes
print("getting indexes")
ss2 = getNDVI(ss2)
print(str(ss2.bandNames().getInfo()))

ss2 = ss2.select('ndvi')
print(str(ss2.bandNames().getInfo()))

ndviParams = {min: -1, max: 1, 'palette': ['blue', 'white', 'green']};
map1.add_ee_layer(ss2,ndviParams,'ndvi')
                    
print('done')
map1


# ### Time Series Analysis of NDVI over our poi

# In[40]:


print(str(ss2.bandNames().getInfo()))


# In[41]:


ss2.getInfo()


# In[ ]:





# In[42]:


#get sentile-2 as collection
se2_ndvi = ee.ImageCollection('COPERNICUS/S2_SR').filterDate("2015-01-01","2022-01-01").filterBounds(poi_rozou)
#collection = ee.ImageCollection('LANDSAT/LC08/C01/T1')
startyear = 2015
endyear = 2022

startDate = ee.Date.fromYMD(startyear,1,1)
endDate = ee.Date.fromYMD(endyear,1,1)
ss2 = importData(poi_rozou,startDate,endDate)
ss2


# In[43]:


#fuction to add ndvi column on sentinel-2 collection
def addNDVI(image):
    ndvi = image.normalizedDifference(['nir', 'red']).rename('ndvi')
    return image.addBands(ndvi)
S2 = ss2.map(addNDVI);    
S2 = S2.select('ndvi')
#print(S2.getInfo())


# In[44]:


def poi_mean(image):
    mean_ndvi = image.reduceRegion(reducer=ee.Reducer.mean(), geometry=poi_rozou, scale=30).get('ndvi')
    return image.set('date', image.date().format()).set('mean_ndvi',mean_ndvi)
poi_ndvi = S2.map(poi_mean)


# In[45]:


nested_list = poi_ndvi.reduceColumns(ee.Reducer.toList(2), ['date','mean_ndvi']).values().get(0)


# In[46]:


mean_ndvi = pd.DataFrame(nested_list.getInfo(), columns=['date','mean_ndvi'])

mean_ndvi


# In[47]:


mean_ndvi['date'].nunique()


# In[48]:


non_dup_date = mean_ndvi['date'].drop_duplicates()
non_dup_date.reset_index()


# In[49]:


mean_ndvi['date'] = pd.to_datetime(mean_ndvi['date']).dt.date
ndvi_df = mean_ndvi.set_index('date')


# In[50]:


import seaborn as sns
import matplotlib.pyplot as plt
# we create a figure with pyplot and set the dimensions to 15 x 7
fig, ax = plt.subplots(figsize=(15,7))

# we'll create the plot by setting our dataframe to the data argument
sns.lineplot(data=ndvi_df, ax=ax)


# we'll set the labels and title
ax.set_ylabel('mean ndvi',fontsize=20)
ax.set_xlabel('date',fontsize=20)
ax.set_title('Monthly mean ndvi for Rozou Vegetation (Jan 2015 to Jan 2022)',fontsize=20);


# These changes are probably due to seasons' changing.

# In[51]:


fig, ax = plt.subplots(figsize=(15,7))

# we'll plot the moving averate using ".rolling" and set a window of 12 months
window=12
sns.lineplot(data=ndvi_df.rolling(window).mean(), ax=ax)

# we'll set the labels and title
ax.set_ylabel('mean ndvi',fontsize=20)
ax.set_xlabel('date',fontsize=20)
ax.set_title(f'Monthly mean ndvi ({window} moving avg.) for Rozou Vegetation (Jan 2014 to Jan 2022)',fontsize=20);


# In[52]:


ndvi_df['mean_ndvi'].idxmax()


# In[53]:


#from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
def arima_model(series):
    log_series = series.apply(lambda x: np.log(abs(x))) #to evala absolute giati to ndvi exei kai arnhtikes times
    log_series_diff=log_series-log_series.shift()

    model=ARIMA(log_series, order=(1,1,1))
    results_ARIMA = model.fit(disp=-1)
    
    RSS = np.nansum((results_ARIMA.fittedvalues-log_series_diff)**2)
    return RSS, log_series_diff, results_ARIMA


RSS, log_series_diff, arima = arima_model(mean_ndvi['mean_ndvi'])
plt.plot(log_series_diff)
plt.plot(arima.fittedvalues, color='red')
plt.title('RSS: %.4f'% RSS)
plt.legend(['Lights', 'Light Prediction'])


# In[54]:


concat_df  = [df_ntl,ndvi_df]
concat_df = pd.concat(concat_df,ignore_index=False)
concat_df


# In[55]:


a = concat_df['mean_rad'].fillna(concat_df['mean_rad'].mean())

b = concat_df['mean_ndvi'].fillna(concat_df['mean_ndvi'].mean())
c_df  = [a,b]
c_df = pd.concat(c_df,ignore_index=False)
c_df


# At this point we have obtained  the ndvi and avg radiance over our area of interest.
# Ideally we want to create a model that can assess the amount of damage on a specified building\
# Unfortunately there are some limitations mainly due to data avalability.

# In[56]:


from sklearn.model_selection import train_test_split


# In[61]:





# In[60]:


import datetime as dt

new_df['date'] = magnisia_lights['date'] 
new_df['date'] = pd.to_datetime(new_df['date'])
new_df['date']=new_df.map(dt.datetime.toordinal)
new_df['date']


# In[ ]:





# In[58]:


y = magnisia_lights['mean_rad']
x = magnisia_lights['date']

X_train,X_test,Y_train,Y_test = train_test_split(x,y_2,test_size=0.3, random_state=42)


# In[ ]:


yy= np.array(y).flatten()
xx=np.array(x)


# In[ ]:


import datetime as dt

# .flatten converts numpy arrays into pandas df columns
df = pd.DataFrame(yy.flatten(),xx.flatten())  

# creates a new index (as pd.Dataframe made x_full_month the index initially)
df.reset_index(inplace=True) 
    
# meaningful column names
df = df.rename(columns = {'index':'ord_date',0:'cumul_DN'}) 
    
# Convert oridinal date to yyyy-mm-dd
df['date']=df['ord_date'].map(dt.datetime.toordinal) 


# In[ ]:


x=np.arange(0,len(df['date']),1)
df['date']=x #
df


# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
scaler = StandardScaler()
df['date'] = scaler.fit_transform(df['date'].values.reshape(-1,1))
df['cumul_DN'] = scaler.fit_transform(df['cumul_DN'].values.reshape(-1,1))


# In[ ]:


X_train,X_test,Y_train,Y_test = train_test_split(df['date'].values.reshape(-1,1), df['cumul_DN'],test_size=0.3, random_state=42)

# instantiate and fit
lm_2 = LinearRegression()
lm_2.fit(X_train, Y_train)

# print the coefficients
print('Intercept: ', lm_2.intercept_)
print('mean: ', lm_2.coef_[0])


# In[ ]:


y_pred = lm_2.predict(X_test)


# In[ ]:


lm_2.score(X_test,Y_test)


# In[ ]:


from sklearn.metrics import r2_score,mean_squared_error


# In[ ]:


r2_score(Y_test,y_pred)


# In[ ]:


mean_squared_error(Y_test, y_pred)


# In[ ]:


# Plot outputs
plt.scatter(X_test, Y_test, color="black")
plt.plot(X_test, y_pred, color="blue", linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()


# In[ ]:


import numpy as np
import statsmodels.api as sm
import pylab

test = df['date']

sm.qqplot(test, line='45')
pylab.show()


# In[ ]:


import numpy as np
import statsmodels.api as sm
import pylab

test = df['cumul_DN']

sm.qqplot(test, line='45')
pylab.show()


# In[ ]:


a = concat_df['mean_rad'].fillna(concat_df['mean_rad'].mean())

b = concat_df['mean_ndvi'].fillna(concat_df['mean_ndvi'].mean())


# In[ ]:


# summarize
from numpy import mean
from numpy import std
print('data1: mean=%.3f stdv=%.3f' % (mean(a), std(a)))
print('data2: mean=%.3f stdv=%.3f' % (mean(b), std(b)))
plt.scatter(a, b)
plt.show()


# In[ ]:


df.date


# In[ ]:


from numpy import cov
covariance = cov(a, b)
covariance


# In[ ]:


from scipy.stats import pearsonr

corr, _ = pearsonr(a, b)
print('Pearsons correlation: %.3f' % corr)


# In[ ]:


map1


# In[ ]:





# Find Abandoned Buildings

# https://www.researchgate.net/publication/268816675_Assessment_of_Concrete_Surfaces_Using_Multi-Spectral_Image_Analysis
# 
# https://www.researchgate.net/publication/236941973_Intelligent_Concrete_Health_Monitoring_ICHM_An_Innovative_Method_for_Monitoring_Concrete_Structures_using_Multi_Spectral_Analysis_and_Image_Processing

# In[ ]:


# mean radiance with fills
# we may want to only use monthly data in order to avoid filling with mean
# same goes for ndvi
a


# In[ ]:


# mean ndvi with fills
b


# Methodology
# 
# We hypothesise that when there is low avg radiance and higher mean ndvi there is a probability that a building is abandoned,
# since vacant buildings shouldn't have any lights open and since they are unmaintained grass etc should be present.

# We set up a threshold for these values
# rad_thresh < mean_rad
# ndvi_thresh > mean_ndvi

# https://github.com/awesome-spectral-indices/awesome-spectral-indices

# In[ ]:


for i in range(len(a)):
    if(a[i] < 80.403156 and b[i] > 0.083002):
        print(i)
        print(a[i])
        print(b[i])
        print("chance for abandoned building")


# In[ ]:


if((a[-2] < 80.403156 and b[-2] > 0.083002)):
    print('true')
else:
    print('false')


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


labels=['abandoned','not_abandoned']


# In[ ]:


b=ndvi_df


# In[ ]:


ndvi_df.describe()


# In[ ]:


a=df_ntl
a


# In[ ]:


df_ntl.describe()


# In[ ]:


labels


# In[ ]:


concat = pd.merge(df_ntl, ndvi_df, left_on='mean_rad', right_on='mean_ndvi')
concat


# In[ ]:


len(b)-len(a)


# In[ ]:


b


# 

# https://www.isprs.org/proceedings/XXXI/congress/part7/321_XXXI-part7.pdf #ui
# 
# https://www.tandfonline.com/doi/abs/10.1080/01431161.2019.1615655?journalCode=tres20 #viirs house vacancy

# In[63]:


import eemont
Ss2 = (ee.ImageCollection('COPERNICUS/S2_SR')
   .filterBounds(poi_rozou)
   .filterDate('2020-01-01','2021-01-01')
   .maskClouds()
   .scale()
   .index(['EVI','NDVI','NDBI','UI','BLFEI']))

# By Region
ts = Ss2.getTimeSeriesByRegion(reducer = [ee.Reducer.mean(),ee.Reducer.median()],
                              geometry = poi_rozou,
                              bands = ['EVI','NDVI','NDBI','UI','BLFEI'],
                              scale = 10)
ts


# In[ ]:





# In[8]:


import pandas as pd
import numpy as np
import eemont
f1 = ee.Feature(ee.Geometry.Point([22.951893210411072,39.35932487749404]).buffer(50),{'ID':'A'})
f2 = ee.Feature(ee.Geometry.Point([22.947140336036682,39.36547976654773]).buffer(50),{'ID':'B'})
fc = ee.FeatureCollection([f1,f2])

Ss2 = (ee.ImageCollection('COPERNICUS/S2_SR')
   .filterBounds(fc)
   .filterDate('2020-01-01','2022-01-01')
   .maskClouds()
   .scaleAndOffset()
   .spectralIndices(['EVI','NDVI','NDBI','UI','BLFEI']))
N = Ss2.select('B8')
R = Ss2.select('B4')
B = Ss2.select('B2')


ts = Ss2.getTimeSeriesByRegion(reducer = [ee.Reducer.mean()],
                              geometry = fc,
                              bands = ['EVI','NDVI','NDBI','UI','BLFEI'],
                              scale = 10,
                              bestEffort = True,
                              maxPixels = 1e13,
                              dateFormat = 'YYYYMMdd',
                              tileScale = 2)

tsPandas = geemap.ee_to_pandas(ts)

tsPandas[tsPandas == -9999] = np.nan
tsPandas['date'] = pd.to_datetime(tsPandas['date'],infer_datetime_format = True)


# In[10]:


tsPandas


# In[11]:


viirs2 = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG").filterDate('2020-01-01','2022-01-01').select('avg_rad')
ts2=viirs2.getTimeSeriesByRegion(reducer = [ee.Reducer.mean()],
                              geometry = fc,
                              scale = 10,
                              bestEffort = True,
                              maxPixels = 1e13,
                              dateFormat = 'YYYYMMdd',
                              tileScale = 2)

tsPandas2 = geemap.ee_to_pandas(ts2)
tsPandas2['date'] = pd.to_datetime(tsPandas2['date'],infer_datetime_format = True)


# In[12]:


tsPandas2

VNP09GA = (ee.ImageCollection("NOAA/VIIRS/001/VNP09GA")          
          .filterDate("2020-01-01","2021-01-01")
          .maskClouds()
          .scaleAndOffset()
       )
          
ts3 = VNP09GA.getTimeSeriesByRegion(reducer = [ee.Reducer.mean()],
                              geometry = fc,
                              scale = 10,
                              bestEffort = True,
                              maxPixels = 1e13,
                              dateFormat = 'YYYYMMdd',
                              tileScale = 2)

tsPandas3 = geemap.ee_to_pandas(ts3)
tsPandas3['date'] = pd.to_datetime(tsPandas3['date'],infer_datetime_format = True)tsPandas3
# In[87]:



# initialize our map
map2 = geemap.Map()
map2.centerObject(aoi, 7)
map2.addLayer(Ss2.first().clip(aoi), rgbViz, "S2")
map2.addLayer(viirs2.first().clip(aoi),radiance_img, viz_params,'Viirs')
map2.addLayerControl()
map2


# In[88]:


g=tsPandas
g['month'] = g['date'].dt.month
g


# In[89]:


tsPandas['EVI'] = tsPandas['EVI'].fillna(tsPandas['EVI'].mean())
tsPandas['NDVI'] = tsPandas['NDVI'].fillna(tsPandas['NDVI'].mean())
tsPandas


# In[90]:


c=tsPandas2
c['month'] = c['date'].dt.month
c


# In[92]:


len(g[:142])


# In[98]:


g['avg_rad'] = c['avg_rad']


# In[ ]:


# auto to kanw gia na valei to radiance sto ypoloipo df analoga me to mhna(epeidh exoume monthly composites sta viirs)

#to oloklhro to kanw  epeidh kateutheian de diabazei to  g['avg_rad'].loc[j] = c['avg_rad'].loc[i]
#otan ta pairnw misa misa

# genika auto den einai kan optimal
# tha protimousa na to kanw me kapoio merge alla den jerw pws
# h sto nested loop na exw kapoio condition gia na allazei
# giati auto pou ginetai einai oti otan perasoun oi 12 mhnes(j=12) meta vriskei pali month=1 kai vazei thn i=1 timh
# to tsekaroume an egine swsta me ta unique
# epishs isws htan kalo na exoume nan opou den exoume data wste na kanoune kapoio allo fill(trial and error)


# for i in range(len(c)):
#     for j in range(len(g)):
#        # print("i",i)
#        # print("j",j)    
#         if((g['month'].loc[j])==c['month'].loc[i]):
#             #print('true')
#             g['avg_rad'].loc[j] = c['avg_rad'].loc[i]
#            # print(c['avg_rad'].loc[i])

# In[100]:


for i in range(len(c[:12])):
    for j in range(len(g[:142])):
       # print("i",i)
       # print("j",j)    
        if((g['month'].loc[j])==c['month'].loc[i]):
            #print('true')
            g['avg_rad'].loc[j] = c['avg_rad'].loc[i]
           # print(c['avg_rad'].loc[i])
            
            
#g


# In[101]:


for i in range(len(c[12:])):
    for j in range(len(g[142:])):
       # print("i",i)
       # print("j",j)    
        if((g['month'].loc[j])==c['month'].loc[i]):
            #print('true')
            g['avg_rad'].loc[j] = c['avg_rad'].loc[i]
           # print(c['avg_rad'].loc[i])
            
            
g


# In[102]:


print(c['avg_rad'])
print(g['avg_rad'])


# In[103]:


g['avg_rad'].unique()


# In[104]:


c['avg_rad'].unique()


# In[105]:


print(len(c))
print(len(g))


# In[ ]:





# In[ ]:





# In[106]:


combined=g
combined


# In[107]:


combined['EVI'] = combined['EVI'].fillna(combined['EVI'].mean())
combined['NDVI'] = combined['NDVI'].fillna(combined['NDVI'].mean())
combined['NDBI'] = combined['NDBI'].fillna(combined['NDBI'].mean())
combined['BLFEI'] = combined['BLFEI'].fillna(combined['BLFEI'].mean())
combined['UI'] = combined['UI'].fillna(combined['UI'].mean())


# In[ ]:


combined['avg_rad']


# In[108]:


combined


# for i in range(len(combined)):
#     if(combined['EVI'].loc[i] > combined['EVI'].mean() and combined['NDVI'].loc[i]>combined['NDVI'].mean() and combined['avg_rad'].loc[i]<combined['avg_rad'].mean()):
#         combined['label'].loc[i] = labels[0]
#     else:
#         combined['label'].loc[i] = labels[1]

# for i in range(len(combined)):
#     print(combined['EVI'].loc(i))
#     if(combined['EVI'].loc(i) > combined['EVI'].mean()):
#         print('true')

# In[ ]:





# In[113]:


import seaborn as sns
import matplotlib.pyplot as plt
# we create a figure with pyplot and set the dimensions to 15 x 7
fig, ax = plt.subplots(figsize=(15,7))

# we'll create the plot by setting our dataframe to the data argument
sns.lineplot(data=combined,y=combined['EVI'],x=combined['date'], ax=ax)


# we'll set the labels and title
ax.set_ylabel('mean evi',fontsize=20)
ax.set_xlabel('date',fontsize=20)
ax.set_title('EVI to Jan  2020 Jan 2022)',fontsize=20);


# In[114]:


fig, ax = plt.subplots(figsize=(15,7))

# we'll plot the moving averate using ".rolling" and set a window of 12 months
window=12
sns.lineplot(data=combined,y=combined['EVI'].rolling(window).mean(),x=combined['date'], ax=ax)

# we'll set the labels and title
ax.set_ylabel('mean evi',fontsize=20)
ax.set_xlabel('date',fontsize=20)
ax.set_title(f'Monthly mean evi ({window} moving avg.) for  to Jan  2020 Jan 2022)',fontsize=20);


# In[115]:


fig, ax = plt.subplots(figsize=(15,7))

# we'll plot the moving averate using ".rolling" and set a window of 12 months
window=12
sns.lineplot(data=combined,y=combined['NDVI'].rolling(window).mean(),x=combined['date'], ax=ax)

# we'll set the labels and title
ax.set_ylabel('mean ndvi',fontsize=20)
ax.set_xlabel('date',fontsize=20)
ax.set_title(f'Monthly mean ndvi ({window} moving avg.) for  to Jan  2020 Jan 2022)',fontsize=20);


# In[116]:


fig, ax = plt.subplots(figsize=(15,7))

# we'll plot the moving averate using ".rolling" and set a window of 12 months
window=12
sns.lineplot(data=combined,y=combined['NDBI'].rolling(window).mean(),x=combined['date'], ax=ax)

# we'll set the labels and title
ax.set_ylabel('mean ndbi',fontsize=20)
ax.set_xlabel('date',fontsize=20)
ax.set_title(f'Monthly mean ndbi ({window} moving avg.) for  to Jan  2020 Jan 2022)',fontsize=20);


# In[117]:


fig, ax = plt.subplots(figsize=(15,7))

# we'll plot the moving averate using ".rolling" and set a window of 12 months
window=12
sns.lineplot(data=combined,y=combined['UI'].rolling(window).mean(),x=combined['date'], ax=ax)

# we'll set the labels and title
ax.set_ylabel('mean UI',fontsize=20)
ax.set_xlabel('date',fontsize=20)
ax.set_title(f'Monthly mean UI ({window} moving avg.) for  to Jan  2020 Jan 2022)',fontsize=20);


# In[119]:


fig, ax = plt.subplots(figsize=(15,7))

# we'll plot the moving averate using ".rolling" and set a window of 12 months
window=12
sns.lineplot(data=combined,y=combined['BLFEI'].rolling(window).mean(),x=combined['date'], ax=ax)

# we'll set the labels and title
ax.set_ylabel('mean blfei',fontsize=20)
ax.set_xlabel('date',fontsize=20)
ax.set_title(f'Monthly mean blfei ({window} moving avg.) for  to Jan  2020 Jan 2022)',fontsize=20);


# In[134]:


fig, ax = plt.subplots(figsize=(15,7))

# we'll plot the moving averate using ".rolling" and set a window of 12 months
window=12
sns.lineplot(data=combined,y=combined['avg_rad'].rolling(window).mean(),x=combined['date'], ax=ax)

# we'll set the labels and title
ax.set_ylabel('mean avg_rad',fontsize=20)
ax.set_xlabel('DATE',fontsize=20)
ax.set_title(f'Monthly mean blfei ({window} moving avg.) for  to Jan  2020 Jan 2022)',fontsize=20);


# In[ ]:





# In[227]:


rgbScaled = {'min':0, 'max':0.3, 'bands':['B4','B3','B2']}
rgbUnscaled = {'min':0, 'max':3000, 'bands':['B4','B3','B2']}
ndvi = {'min':-1, 'max':1, 'bands':['NDVI']}
smead=Ss2.first()
# initialize our map
map5 = geemap.Map()
map5.centerObject(aoi, 7)
map5.addLayer(smead.clip(aoi), rgbUnscaled, "S2")
map5.addLayer(smead.clip(aoi), rgbScaled, "S2-scaled")
map5.addLayer(smead.clip(aoi), ndvi, "S2-NDVI")

map5.addLayerControl()
map5

https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/ndvi/
NDVI < -0.2	#000000	
-.2 < NDVI ≤ 0	#a50026	
0 < NDVI ≤ .1	#d73027	
.1 < NDVI ≤ .2	#f46d43	
.2 < NDVI ≤ .3	#fdae61	
.3 < NDVI ≤ .4	#fee08b	
.4 < NDVI ≤ .5	#ffffbf	
.5 < NDVI ≤ .6	#d9ef8b	
.6 < NDVI ≤ .7	#a6d96a	
.7 < NDVI ≤ .8	#66bd63	
.8 < NDVI ≤ .9	#1a9850	
.9 < NDVI ≤ 1.0	#006837
# In[229]:


# Make the training dataset.
training = Ss2.first().sample(
    **{
        #'region': region,
        'scale': 30,
        'numPixels': 5000,
        'seed': 0,
        'geometries': True,  # Set this to False to ignore geometries
    }
)

map5.addLayer(training, {}, 'training', False)
map5


# In[230]:


# Instantiate the clusterer and train it.
n_clusters = 5
clusterer = ee.Clusterer.wekaKMeans(n_clusters).train(training)


# In[231]:


# Cluster the input using the trained clusterer.
resultcl= Ss2.first().cluster(clusterer)

# # Display the clusters with random colors.
map5.addLayer(resultcl.randomVisualizer(), {}, 'clusters')
map5


# In[233]:


legend_keys = ['One', 'Two', 'Three', 'Four', 'ect']
legend_colors = ['#8DD3C7', '#FFFFB3', '#BEBADA', '#FB8072', '#80B1D3']

# Reclassify the map
resultcl = resultcl.remap([0, 1, 2, 3, 4], [1, 2, 3, 4, 5])

map5.addLayer(
    resultcl, {'min': 1, 'max': 5, 'palette': legend_colors}, 'Labelled clusters'
)
map5.add_legend(
    legend_keys=legend_keys, legend_colors=legend_colors, position='bottomright'
)
map5


# # AYTO DES TO
# https://github.com/Toroitich6783/classification-using-GEE
# 
# https://github.com/afrozalopa/Project_Report_Group-1

# https://geohackweek.github.io/GoogleEarthEngine/05-classify-imagery/

# https://geemap.org/workshops/GeoPython_2021/#zonal-statistics

# In[9]:


labels=['abandoned','not-abandoned']

abandoned_vis = {
    'min': 0,
    'max': 4000,
    'palette': ['006633', 'E5FFCC'],
}


# In[201]:


S_sentinel_bands = Ss2.select('[B1-B12]')
S_sentinel_bands


# In[205]:


s2tes = ee.Image("COPERNICUS/S2/20160111T112432_20160111T113311_T28PDT").tasseledCap()


# In[206]:



s2tes.bandNames().getInfo()


# In[210]:


colors = ['tomato', 'navy', 'MediumSpringGreen', 'lightblue', 'orange', 'blue',
          'maroon', 'purple', 'yellow', 'olive', 'brown', 'cyan']

ep.hist(s2tes.bandNames(), 
         colors = colors,
        title=[f'Band-{i}' for i in range(1, 13)], 
        cols=3, 
        alpha=0.5, 
        figsize = (12, 10)
        )

plt.show()


# In[208]:


Map4 = geemap.Map()
Map4.addLayer(s2tes, {"min": [-1000, -1000, -100], "max": [9000, 2000, 800], "bands": ["TCB", "TCG", "TCW"]}, "S2 TC")
Map4.centerObject(s2tes, 10)

Map4


# # papers pou einai xrhsima
# https://github.com/DominiquePaul/GDP-Satellite-Prediction

# https://towardsdatascience.com/hyperspectral-image-analysis-getting-started-74758c12f2e9

# https://github.com/worldbank/OpenNightLights/tree/master/onl/tutorials

# https://github.com/konkyrkos/hyperspectral-image-classification

# https://github.com/yohman/workshop-python-spatial-stats

# https://github.com/holderbp/pwpd

# https://www.frontiersin.org/articles/10.3389/fbuil.2018.00032/full
#     
# https://icaarconcrete.org/wp-content/uploads/2020/11/15ICAAR-SanchezL-2.pdf
#     
# https://www.sciencedirect.com/science/article/pii/S2666549220300013
#     
# https://www.researchgate.net/publication/268816675_Assessment_of_Concrete_Surfaces_Using_Multi-Spectral_Image_Analysis

# There are a few basic boolean operations that Google Earth Engine includes as built-ins for Images. The output is a binary file that sets a pixel value to 1 if it meets the condition and 0 if it doesnt. Those operations include:
# 
#     lt: "less than"
#     lte: "less than or equal to"
#     eq: "equal to"
#     gt: "greater than or equal to"
#     gte: "greater than
# 
# The method compares the Image object the method is called on as the left-hand side of the comparison with the value passed as the input argument to the function on the right-hand side. This input can be a scalar value that will be compared to all pixels in the image, or another image that will be used as an element-wise / pixel-wise comparison.

# In[175]:


# create a 200 km buffer around the center of Catalonia
aoi_2 = ee.Geometry.Point( [
          22.949130535125732,
          39.364330950394994
        ]).buffer(200000);
viirs2019_12 = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG").filterDate("2021-12-01","2021-12-31").select('avg_rad').median()

# extract a number array from this region
arr = geemap.ee_to_numpy(viirs2019_12, region=aoi_2)

# create a histogram
fig, ax = plt.subplots(figsize=(15,5))
sns.kdeplot(arr.flatten(), label='region',legend=True, ax=ax)
ax.axvline(2, color='indianred', label='suggested threshold')
plt.legend(fontsize=20)
plt.title('Distribution of VIIRS-DNB from sample(smoothed w/ Gaussian kernel)', fontsize=20);
plt.ylabel('avg radiance')
plt.legend();


# Based on our histogram of radiance in the sample region, it might be interesting mask all values that are not greater than or equal to 4.
# 
# 

# In[182]:


viirs2019_12_mask = viirs2019_12.gte(2)

# initialize our map
map3 = geemap.Map(center=[ 39.364330950394994,22.949130535125732],zoom=7)
map3.add_basemap('SATELLITE')

# we'll mask the image with itself to keep the visualization clean
map3.addLayer(viirs2019_12_mask.mask(viirs2019_12_mask), {}, "Avg rad >=2")
map3.addLayerControl()
map3


# In[184]:


zones = viirs2019_12.gt(1.5).add(viirs2019_12.gt(2)).add(viirs2019_12.gt(4))

# initialize our map
map3 = geemap.Map(center=[ 39.364330950394994,22.949130535125732],zoom=7)
map3.add_basemap('SATELLITE')

map3.addLayer(zones.mask(zones), {'palette':['#cc0909','#e67525','#fff825']}, 'zones')

map3.addLayerControl()
map3


# Now we can see variation in radiance in a way that sheds "light" (apologies for the pun!) on activity around denser urban areas.
# 
# Later in this tutorial, we'll look at calculating the difference in two Images -- and this is a another potential for leveraging conditional operators.M

# ## Ayta apo katw den exoun toso nohma pros to paron

# In[157]:


import os
out_dir = os.path.join(os.path.expanduser('~'), 'Downloads')
filename = os.path.join(out_dir, 'viirsfirst.tif')
imageti = viirs2.first().clip(poi_rozou).unmask()
geemap.ee_export_image(
    imageti, filename=filename, scale=90, region=poi_rozou, file_per_band=False
)


# In[161]:


#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Lei Dong
# Email: arch.dongl@gmail.com


import numpy as np
import rasterio
import sys
from osgeo import gdal, osr, ogr
from math import radians, cos, sin, asin, sqrt
from rasterio.features import shapes
from shapely.geometry import shape


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers.
    return c * r


# 按一定band的阈值选择坐标点
def filter(data, thre):
    rst = {}
    for row in range(len(data)):
        for col in range(len(data[row])):
            lon = xOrigin + (col + 0.5) * pixelWidth #pixal center
            lat = yOrigin + (row + 0.5) * pixelHeight
            if data[row][col] > thre:
                rst[(row, col)] = [lon, lat, data[row][col]]
    return rst


# 每个选择出来的点，计算和周围一定距离阈值内的点的球面距离
def neighbor(rst, grid):
    foo = {}
    for key in rst:
        originX = key[0] - grid
        originY = key[1] - grid
        extend_key = []
        for i in range(2 * grid + 1):
            extend_key.append((originX, originY))
            originX += 1
            originY += 1
        for j in extend_key:
            if j in rst:
                lon1, lat1, value1 = rst[key]
                lon2, lat2, value2 = rst[j]
                distance = haversine(lon1, lat1, lon2, lat2)
                if (key, j) not in foo and (j, key) not in foo:
                    foo[(key, j)] = [lon1, lat1, value1, lon2, lat2, value2, distance]
    return foo


# 点对之间的筛选条件
def cluster(foo, dist):
    final = {}
    for key in foo:
        if foo[key][6] <= dist:
            if key[0] not in final:
                final[key[0]] = 1
            if key[1] not in final:
                final[key[1]] = 1
    return final


def array2raster(newRasterfn, rasterOrigin, pixelWidth, pixelHeight, array):
    array = array[::-1] # reverse array so the tif looks like the array
    cols = array.shape[1]
    rows = array.shape[0]
    originX = rasterOrigin[0]
    originY = rasterOrigin[1]

    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Byte)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(4326)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()


# test
if __name__ == "__main__":
    # open geotiff
    try:
        dataset = gdal.Open("viirsfirst.tif")
    except Exception as e:
        print(e)
    
    # select tiff band
    try:
        band_num = 1
        band = dataset.GetRasterBand(band_num) 
    except Exception as e:
        print(e)
    
    # get coord information
    transform = dataset.GetGeoTransform()
    xOrigin = transform[0] # top left x
    yOrigin = transform[3] # top left y
    pixelWidth = transform[1] # width pixal resolution
    pixelHeight = transform[5] # hight pixal resolution (negative value)
    print(xOrigin, yOrigin, pixelWidth, pixelHeight)


    # Transform the band value to array
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    data = band.ReadAsArray(0, 0, cols, rows)
    
    # filter data
    rst = filter(data, thre = 10)
    foo = neighbor(rst, grid = 100)
    final = cluster(foo, dist = 50)
    print(len(rst), len(foo), len(final))
    
    # export
    rasterOrigin = (xOrigin, xOrigin)
    newRasterfn = 'test.tif'
    array = np.zeros((rows, cols))
    for row in range(len(data)):
        for col in range(len(data[row])):
            if (row, col) in final:
                array[row][col] = 1
    array2raster(newRasterfn, rasterOrigin, pixelWidth, pixelHeight, array)

    
# raster to vector (polygon)
mask = None
with rasterio.drivers():
    with rasterio.open('test.tif') as src:
        image = src.read(1) # first band
        results = (
        {'properties': {'raster_val': v}, 'geometry': s}
        for i, (s, v) in enumerate(shapes(image, mask=mask, transform=src.affine)))

        
geoms = list(results)
print (geoms[0])
print (shape(geoms[0]['geometry']))


# In[160]:


rasterio.open('test.tif')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# get South Korea national boundary geometry
sovolos= ee.Feature(ee.FeatureCollection("FAO/GAUL/2015/level2").filter(ee.Filter.eq('ADM2_NAME', 'Magnisias')).first()).geometry()

# revise our reducer function to be to get SOL for South Korea
def get_sovolos_sol(img):
    sol = img.reduceRegion(reducer=ee.Reducer.sum(), geometry=sovolos, scale=500, maxPixels=2e9).get('avg_rad')
    return img.set('date', img.date().format()).set('SOL',sol)


# In[ ]:


# reduce collection
sovolos_sol = viirs.map(get_sovolos_sol)

# get lists
nested_list = sovolos_sol.reduceColumns(ee.Reducer.toList(2), ['date','SOL']).values().get(0)

# convert to dataframe
soldf = pd.DataFrame(nested_list.getInfo(), columns=['date','SOL'])
#soldf['date'] = pd.to_datetime(soldf['date'])
#soldf = soldf.set_index('date')


# In[ ]:


volosMap2 = geemap.Map()
volosMap2.centerObject(sovolos, zoom=7)
volosMap2.add_basemap("SATELLITE")
volosMap2.addLayer(sovolos, {}, "Greece")
volosMap2.addLayer(viirs.select('avg_rad').median(), {'min':0,'max':10}, "VIIRS 2014-2021 mean")
volosMap2.addLayerControl()
volosMap2


# In[ ]:


fig, ax = plt.subplots(figsize=(15,7))
sns.lineplot(data=soldf, ax=ax)
ax.set_ylabel('SOL',fontsize=20)
ax.set_xlabel('Date',fontsize=20)
ax.set_title('Monthly Sum Of Lights (SOL) for Volos (Jan 2014 to May 2021)',fontsize=20);


# In[ ]:





# In[ ]:


magnisia_lights


# In[ ]:


# Import the packages and classes needed in this example:
import numpy as np
from sklearn.linear_model import LinearRegression

#x = pd.Timestamp(soldf['date']).to_pydatetime()
# Create a numpy array of data:
x=magnisia_lights.index.values.reshape(-1,1)
#x=x.map(dt.datetime.toordinal)
y =magnisia_lights['mean']
# Create an instance of a linear regression model and fit it to the data with the fit() function:
model = LinearRegression().fit(x, y) 

# The following section will get results by interpreting the created instance: 

# Obtain the coefficient of determination by calling the model with the score() function, then print the coefficient:
r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)

# Print the Intercept:
print('intercept:', model.intercept_)

# Print the Slope:
print('slope:', model.coef_) 

# Predict a Response and print it:
y_pred = model.predict(x)
print('Predicted response:', y_pred, sep='\n')


# In[ ]:


magnisia_lights.index.values


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#https://github.com/Digdgeo/Ndvi2Gif


# In[ ]:


from ndvi2gif import NdviSeasonality

# You could need a first login to sart with python Earth Engine login
Map = geemap.Map()
Map.add_basemap('Google Satellite')
Map

# You can use any of these depending on your input
MyClass = NdviSeasonality(poi)
#MyClass = ndvi_seasonality(shp)
#MyClass = ndvi_seasonality(geojson)

wintermax = MyClass.get_year_composite().select('winter').max()
median = MyClass.get_year_composite().median()
Map.addLayer(wintermax, {'min': 0, 'max': 0.6}, 'winterMax')
Map.addLayer(median, {'min': 0.1, 'max': 0.8}, 'median') 

MyClass.get_gif()


# %%html
# <img src="mygif.gif">

# In[ ]:


## reading GeoTiff file

import geopandas as gpd
from rasterio.transform import xy
import rioxarray as rxr
from shapely.geometry.multipolygon import MultiPolygon
from shapely import wkt
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.mask import mask
import matplotlib.colors as colors
from adjustText import adjust_text


# load shapefile in geopandas dataframe
# source: https://maps.princeton.edu/catalog/stanford-cp683xh3648
Lebanon_regions = gpd.read_file('data/stanford-cp683xh3648-shapefile.zip')

#Lebanon_regions.plot()

years = ['2012','2015', '2020']
############### using VIIRS
# need to put /vsigzip/ in front of path with .gz
raster20 = rasterio.open('/vsigzip/data/VNL_v2_npp_2020_global_vcmslcfg_c202102150000.median_masked.tif.gz')
raster12 = rasterio.open('/vsigzip/data/VNL_v2_npp_201204-201303_global_vcmcfg_c202102150000.median_masked.tif.gz')
raster15 = rasterio.open('/vsigzip/data/VNL_v2_npp_2015_global_vcmslcfg_c202102150000.median_masked.tif.gz')


#raster.meta

def getFeatures(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    import json
    return [json.loads(gdf.to_json())['features'][0]['geometry']]


raster_dic = {years[0] :raster12, years[1]: raster15,years[2]:raster20}
Lebanon_regions[f'light{years[0]}'] = 0.0
Lebanon_regions[f'light{years[2]}'] = 0.0
for y in years:
    raster = raster_dic[f'{y}']
    for i in range(1,len(Lebanon_regions)+1):
        coords = getFeatures(Lebanon_regions.iloc[i-1:i,:])
        out_img, out_transform = mask(raster, shapes=coords, crop=True)
        light = np.average(out_img)
        Lebanon_regions.loc[i-1,f'light{y}'] = light

# get difference
for y in years[0:2]:
    Lebanon_regions[f'light_diff_{y}_20'] = Lebanon_regions.light2020 - Lebanon_regions.loc[:,f'light{y}']
    Lebanon_regions[f'light_reldiff_{y}_20'] = Lebanon_regions.loc[:,f'light_diff_{y}_20'] / Lebanon_regions.loc[:,f'light{y}']


# 
minx, miny, maxx, maxy = Lebanon_regions.geometry.total_bounds

# plot in levels
columns = [f'light{y}' for y in years]
vmin = Lebanon_regions[columns].min().min()
vmax = Lebanon_regions[columns].max().max()

for y in years:
    fig, ax = plt.subplots()
    Lebanon_regions.plot(column = f'light{y}', legend=True, cmap='PRGn', ax = ax, vmin=vmin, vmax=vmax, alpha=.7)
    texts = []
    for i in range(len(Lebanon_regions)):
        point = Lebanon_regions.geometry[i].centroid
        lab = np.round(Lebanon_regions.loc[i,f'light{y}'],1)
        txt = ax.annotate(lab, xy=(point.x, point.y), xytext=(-3, 8), textcoords="offset points", size = 8, fontweight = 'bold', arrowprops={'arrowstyle':'-'})
        texts.append(txt)
    adjust_text(texts)
    ax.set_title(f'average light in {y}')
    ax.text(x= maxx, y = miny, s='unit: nW/cm2/sr', fontsize=8, ha='right')
    ax.axis('off')    
    fig.savefig(f'out/Lebanon_nightlight_{y}.png')


# plot relative difference
columns = [f'light_reldiff_{y}_20' for y in years[0:2]]
vmin = Lebanon_regions[columns].min().min()
vmax = Lebanon_regions[columns].max().max()

for y in years[0:2]:
    fig, ax = plt.subplots()
    divnorm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    Lebanon_regions.plot(column = f'light_reldiff_{y}_20', legend=True, cmap='PiYG', norm=divnorm, vmin=vmin, vmax=vmax, ax= ax, alpha=.8)
    texts = []
    for i in range(len(Lebanon_regions)):
        point = Lebanon_regions.geometry[i].centroid
        lab = np.round(Lebanon_regions.loc[i,f'light_reldiff_{y}_20'],1)
        txt = ax.annotate(lab, xy=(point.x, point.y), xytext=(-3, 8), textcoords="offset points", size = 8, fontweight = 'bold', arrowprops={'arrowstyle':'-'})
        texts.append(txt)
    adjust_text(texts)
    ax.set_title(f'relative change in average light from {y}-2020')
    ax.axis('off')
    fig.savefig(f'out/Lebanon_nightlight_diff{y}.png')

