# Dependencies and Setup
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
import numpy as np
import json
from scipy.stats import linregress
from collections import Counter
from pprint import pprint
import itertools


#open json
f = open('imdb.json')
data = json.load(f)

#Get all countries into a single list
countrylist = []
for i in range(len(data['country'])):
   countrylist.append(data['country'][f'{i}']) 

countrylist2 = filter(None, countrylist)

#Get count for each country
#I printed this in jupyter notebook and then calculated all the values for the pie chart by hand
seq = countrylist2
Counter(x for xs in seq for x in set(xs))

#set up for chart
countries = ['United States', 'United Kingdom', 'Other', 'Japan', 'France', 'Canada', 'Germany', 'Hong Kong', 'India', 'Italy', 'Australia', 'Spain']
count = [5339, 1477, 1258, 724, 680, 618, 333, 311, 300, 295, 180, 164]
colors = ['firebrick', 'lightcoral', 'darkorange', 'goldenrod', 'olive', 'lightseagreen', 'deepskyblue', 'royalblue', 'blueviolet', 'violet', 'purple', 'black']
explode = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)


#PIE CHARTS
#create pie chart
plt.pie(count, explode=explode, labels=countries, colors=colors, shadow=True, startangle=8)
plt.title('Number of Movies by Country')