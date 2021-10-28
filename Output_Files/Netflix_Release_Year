# Dependencies and Setup
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
import numpy as np
from scipy.stats import linregress

###GET AND SET UP DATAFRAME
# make paths and read csv raw data
data_path = "Resources/imdb.csv"
data = pd.read_csv(data_path)

#get needed columns
year_data_df = data[['title','year','rating','vote']]

#drop any with non-existant values
year_data_df = year_data_df.dropna()

# dropping ALL duplicate values
year_data_df.drop_duplicates(subset ="title", keep = 'first', inplace = True)

#Used this in jupyter to check earliest year
earliestyear = year_data_df['year'].min()

##BINNING INTO YEARS FOR GRAPHING
#create the bins to separate the age group and label the bins
bins = [0, 1939.9, 1949.9, 1959.9, 1969.9, 1979.9, 1989.9, 1999.9, 2009.9, 2019.9, 3000]
bin_names = ["Before 1940", "1940-1950", "1950-1960", "1960-1970", "1970-1980", "1980-1990", "1990-2000", "2000-2010", "2010-2020", "Post 2020"]

#add column to dataframe that describes that shows bins
year_data_df["Year Group"] = pd.cut(year_data_df["year"], bins, labels=bin_names, include_lowest=True)

##GRAPH OF DISTRIBUTION OF YEARS BY GROUP
#group by year group
yeargroup_df = year_data_df.groupby(['Year Group'])

#create dataframe to graph
ygcount_df = pd.DataFrame(yeargroup_df['year'].count())

#create bar chart
yeargroup_bar = ygcount_df.plot(kind="bar", title="Distribution of Movies by Year", color="red", legend=False)

#label the chart
yeargroup_bar.set_xlabel("Years")
yeargroup_bar.set_ylabel("Number of Movie")
plt.tight_layout


#sort data by rating
topmovies_df = year_data_df.sort_values("rating", ascending=False)

##GRAPH FOR AVERAGE VOTES PER YEAR GROUP
topyear_df = topmovies_df.groupby(['Year Group'])
tycount_df = pd.DataFrame(topyear_df['vote'].mean())

#create bar chart
top25_bar = tycount_df.plot(kind="bar", title="Average Votes per Movie by Year", color="red", legend=False)

#label the chart
top25_bar.set_xlabel("Years")
top25_bar.set_ylabel("Average Votes per Movie")
plt.tight_layout

#the dataframe for the top 25
top25_df = topmovies_df[:25]





