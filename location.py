import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import json
import time
import pickle
from datetime import datetime
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score, make_scorer
from sklearn.utils import compute_class_weight
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

print("Loading data sets...")
train = load_dataset("ieuniversity/competition_ai_ml_24_train")
test = load_dataset("ieuniversity/competition_ai_ml_24_test")
df_train = train["train"].to_pandas()


#Create a color map for each unique class.
classes= df_train['CLASE'].unique

#Create a scatterplot using plotly
fig = px.scatter(
    df_train,
    x= 'X',
    y= 'Y',
    color = 'CLASE',   #Plotly automatically assigns colors.
    title = 'X (Longitud) vs Y (Latitude) Distribution by Class'

)

#Customize layout
fig.update_layout(
    xaxis_title='X: Longitud',
    yaxis_title='Y: Latitud',
    legend_title='CLASE',
    template ='plotly_white',
    height = 650
)

#Show interactive plot