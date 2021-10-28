#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Netflix Release Year


# In[1]:


# Dependencies and Setup
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
import numpy as np
from scipy.stats import linregress
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go


# In[2]:


from api_keys import netflix_api_key


# In[3]:


# make paths and read csv raw data
data_path = "imdb.csv"

data = pd.read_csv(data_path)


# In[4]:


data.head()


# In[5]:


#get needed columns
year_data_df = data[['title','year','rating','vote']]

year_data_df.head()


# In[6]:



#drop and null values
year_data_df = year_data_df.dropna()


year_data_df.head(20)


# In[7]:



# dropping ALL duplicate values
year_data_df.drop_duplicates(subset ="title", keep = 'first', inplace = True)


# In[8]:


#check for earliest date
earliestyear = year_data_df['year'].min()
earliestyear


# In[9]:


#create the bins to separate the age group and label the bins
bins = [0, 1939.9, 1949.9, 1959.9, 1969.9, 1979.9, 1989.9, 1999.9, 2009.9, 2019.9, 3000]
bin_names = ["Before 1940", "1940-1950", "1950-1960", "1960-1970", "1970-1980", "1980-1990", "1990-2000", "2000-2010", "2010-2020", "Post 2020"]


# In[10]:



#add column to dataframe that describes that shows bins
year_data_df["Year Group"] = pd.cut(year_data_df["year"], bins, labels=bin_names, include_lowest=True)
year_data_df.head(20)


# In[11]:



#GRAPH FOR DISTRIBUTION OF YEARS
#group by year group
yeargroup_df = year_data_df.groupby(['Year Group'])

ygcount_df = pd.DataFrame(yeargroup_df['year'].count())

#create bar chart
yeargroup_bar = ygcount_df.plot(kind="bar", title="Distribution of Movies by Year", color="crimson", legend=False)

#label the chart
yeargroup_bar.set_xlabel("Years")
yeargroup_bar.set_ylabel("Number of Movie")
plt.tight_layout

plt.show


# In[12]:


#sort data by rating
topmovies_df = year_data_df.sort_values("rating", ascending=False)

topmovies_df.head()


# In[13]:


#GRAPH FOR AVERAGE VOTES PER MOVIE BY YEAR
topyear_df = topmovies_df.groupby(['Year Group'])

tycount_df = pd.DataFrame(topyear_df['vote'].mean())

#create bar chart
top25_bar = tycount_df.plot(kind="bar", title="Average Votes per Movie by Year", color="crimson", legend=False)

#label the chart
top25_bar.set_xlabel("Years")
top25_bar.set_ylabel("Average Votes per Movie")
plt.tight_layout

plt.show


# In[14]:


top25_df = topmovies_df[:25]

top25_df


# In[15]:


## Netflix Kind Values


# In[16]:


import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
import numpy as np
from scipy.stats import linregress
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go


# In[17]:


netflix_pd = pd.read_csv("imdb.csv")


# In[18]:


# remove duplicate movies
netflix_pd.drop_duplicates(subset ="title", keep = "first", inplace = True)


# In[19]:


# clean and drop NAN in reduced data set
netflix_reduced = netflix_pd.loc[:, ["kind", "rating", "vote"]]
netflix_clean = netflix_reduced.dropna(how="any")


# In[20]:


# assign kind values and unique labels
kind_unique = netflix_clean["kind"].unique()
kind_values = netflix_clean["kind"].value_counts()


# In[21]:


# graph all the kinds distribution
# define parameters of the graph
labels = ['movie', 'tv short', 'video movie', 'tv movie', 'tv series', 'episode', 'tv mini series', 'video game']
sizes = [5213, 10, 1191, 744, 583, 469, 255, 17]
seperate = (0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01)


# In[22]:


# plot the pie chart
plt.figure(figsize=(10,10))
plt.pie(sizes, explode=seperate, autopct="%1.1f%%", labels=labels)
plt.legend(loc="upper right")
plt.axis("equal")
plt.title("Netflix Kind Distribution")
plt.show()


# In[23]:


movie_rating = netflix_clean.loc[netflix_clean["kind"] == "movie", "rating"]
video_rating = netflix_clean.loc[netflix_clean["kind"] == "video movie", "rating"]
movie_rating_mean = movie_rating.mean()
video_rating_mean = video_rating.mean()


# In[24]:


# define parameters
labels = ["Digital Movies", "Video Movies"]
ratings = [movie_rating_mean, video_rating_mean]


# In[30]:


# graph bar chart
plt.bar(labels, ratings, color="red", alpha=0.5, align="center")
plt.title("Digital Movie vs Video Movie Ratings")
plt.xlabel("Movie Type")
plt.ylabel("IMDB Ratings")
plt.ylim(0,10)
plt.show()


# In[31]:


## Netflix Genre Types


# In[32]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go


# In[33]:


netflix_pd = pd.read_csv("Netflix_Data.csv")


# In[34]:


data_path = "Netflix_Data.csv"
data = pd.read_csv(data_path)


# In[35]:


df = pd.read_csv("Netflix_Data.csv")
df.head()


# In[36]:


df.isnull().sum()


# In[37]:


df = df.drop(columns = [ 'Metacritic Score', 'Boxoffice', 'Production House', 'Netflix Link', 'IMDb Link',
        'Poster', 'TMDb Trailer', 'Trailer Site'], axis = 1)


# In[38]:


df['Release Date']= pd.to_datetime(df['Release Date'])
df['Netflix Release Date']= pd.to_datetime(df['Netflix Release Date'])


# In[39]:


df['Released_Year'] = pd.DatetimeIndex(df['Release Date']).year
df['Released_Year_Net'] = pd.DatetimeIndex(df['Netflix Release Date']).year


# In[38]:


colors = ['black',] * 2
colors[0] = 'crimson'

count = df['Series or Movie'].value_counts()

fig = go.Figure(data=[go.Bar(
    x = df["Series or Movie"],
    y = count,
    text = count,
    textposition='auto',
    marker_color=colors # marker color can be a single color value or an iterable
)])
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout(title_text= 'Movie or Tv Series ?',
                  uniformtext_minsize=8, uniformtext_mode='hide',
                  barmode='group', xaxis_tickangle=-45,
                  yaxis=dict(
                  title='Quantity',
                  titlefont_size=14),
                  xaxis=dict(
                  title='Category',
                  titlefont_size=14))


# In[40]:


df_movie = df[df['Series or Movie']=='Movie']
df_movie.head(1)


# In[41]:


df_series = df[df["Series or Movie"] == "Series"]
df_series.head(1)


# In[42]:


df_series_gen = df_series.dropna(subset=['Genre'])


# In[43]:


colors_10 = ['DarkRed', 'FireBrick','Red', 'Crimson', 'IndianRed', 'slategray', 'gray', 'dimgrey', 'DarkSlateGrey', 'black']
series_gen_list = df_series_gen.Genre.str.split(',') #split the list into names
s_gen_list = {} #create an empty list
for genres in series_gen_list: # for any names in series_gen_list
    for genre in genres: # for any genre in genres
        if (genre in s_gen_list): #if this genre is already present in the s_gen_list
            s_gen_list[genre]+=1 # increase his value
        else:  # else
            s_gen_list[genre]=1 # Create his index in the list
s_gen_df = pd.DataFrame(s_gen_list.values(),index = s_gen_list.keys(),
                        columns = {'Counts of Genres in Tv Series'}) #Create a s_gen_df
s_gen_df.sort_values(by = 'Counts of Genres in Tv Series',ascending = False,inplace = True) #Sort the dataframe in ascending order
top_10_s_gen = s_gen_df[0:10] 


# In[45]:


fig = go.Figure(data=[go.Bar(
    x = top_10_s_gen.index,
    y = top_10_s_gen['Counts of Genres in Tv Series'],
    text = top_10_s_gen['Counts of Genres in Tv Series'],
    textposition='auto',
    marker_color=colors_10 # marker color can be a single color value or an iterable
)])
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout(title_text= 'Most Popular in TV Genre',
                  uniformtext_minsize=8, uniformtext_mode='hide',
                  yaxis=dict(
                  title='Quantity',
                  titlefont_size=14),
                  xaxis=dict(
                  title='Genres',
                  titlefont_size=14))


# In[46]:


df_movie_gen = df_movie.dropna(subset=['Genre'])


# In[47]:


movie_gen_list = df_movie_gen.Genre.str.split(', ') #split the list into names
m_gen_list = {} #create an empty list
for genres in movie_gen_list: # for any genres in movie_gen_list
    for genre in genres: # for any genre in genres
        if (genre in m_gen_list): #if this name is already present in the m_gen_list
            m_gen_list[genre]+=1 # increase his value
        else:  # else
            m_gen_list[genre]=1 # Create his index in the list
m_gen_df = pd.DataFrame(m_gen_list.values(),index = m_gen_list.keys(),
                        columns = {'Counts of Genres in Movies'}) #Create a m_gen_df
m_gen_df.sort_values(by = 'Counts of Genres in Movies',ascending = False,inplace = True) #Sort the dataframe in ascending order
top_10_m_gen = m_gen_df[0:10] 


# In[49]:


fig = go.Figure(data=[go.Bar(
    x = top_10_m_gen.index,
    y = top_10_m_gen['Counts of Genres in Movies'],
    text = top_10_m_gen['Counts of Genres in Movies'],
    textposition='auto',
    marker_color=colors_10 # marker color can be a single color value or an iterable
)])
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout(title_text= 'Most Popular Movie Genre',
                  uniformtext_minsize=8, uniformtext_mode='hide',
                  yaxis=dict(
                  title='Quantity',
                  titlefont_size=14),
                  xaxis=dict(
                  title='Genres',
                  titlefont_size=14))


# In[50]:


df_series_imdb = df_series.dropna(subset=['IMDb Score'])
df_series_imdb = df_series_imdb.sort_values(by = 'IMDb Score', ascending = False)
top_s_imdb_10_list =df_series_imdb[:10]


# In[54]:


fig = go.Figure(data=[go.Bar(
    x = top_s_imdb_10_list['Title'],
    y = top_s_imdb_10_list['IMDb Score'],
    text = top_s_imdb_10_list['IMDb Score'],
    textposition='auto',
    marker_color=colors_10 # marker color can be a single color value or an iterable
)])
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout(title_text= 'Top Rated Tv Series Rated by IMDB Score?',
                  uniformtext_minsize=8, uniformtext_mode='hide',
                  yaxis=dict(
                  title='IMDb Score',
                  titlefont_size=14),
                  xaxis=dict(
                  title='Titles',
                  titlefont_size=14))


# In[52]:


df_movie_imdb = df_movie.dropna(subset=['IMDb Score'])
df_movie_imdb = df_movie_imdb.sort_values(by = 'IMDb Score', ascending = False)
top_m_imdb_10_list = df_movie_imdb[:10]


# In[55]:


fig = go.Figure(data=[go.Bar(
    x = top_m_imdb_10_list['Title'],
    y = top_m_imdb_10_list['IMDb Score'],
    text = top_m_imdb_10_list['IMDb Score'],
    textposition='auto',
    marker_color=colors_10 # marker color can be a single color value or an iterable
)])
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout(title_text= 'Top Rated Movies Rated by IMDB Rating',
                  uniformtext_minsize=8, uniformtext_mode='hide',
                  yaxis=dict(
                  title='IMDb Score',
                  titlefont_size=14),
                  xaxis=dict(
                  title='Titles',
                  titlefont_size=14))


# In[56]:


##Country of Orgin


# In[63]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AffinityPropagation
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")
import plotly as py
import plotly.graph_objs as go
import os
py.offline.init_notebook_mode(connected = True)
#print(os.listdir("../input"))
import datetime as dt
import missingno as msno
plt.rcParams['figure.dpi'] = 140


# In[64]:


df = pd.read_csv('netflix_titles.csv')
df.head(3)


# In[65]:


# Missing data

for i in df.columns:
    null_rate = df[i].isna().sum() / len(df) * 100 
    if null_rate > 0 :
        print("{} null rate: {}%".format(i,round(null_rate,2)))


# In[66]:


# Replacments

df['country'] = df['country'].fillna(df['country'].mode()[0])


df['cast'].replace(np.nan, 'No Data',inplace  = True)
df['director'].replace(np.nan, 'No Data',inplace  = True)

# Drops

df.dropna(inplace=True)

# Drop Duplicates

df.drop_duplicates(inplace= True)


# In[67]:


df.isnull().sum()


# In[68]:


df.info()


# In[69]:


df["date_added"] = pd.to_datetime(df['date_added'])

df['month_added']=df['date_added'].dt.month
df['month_name_added']=df['date_added'].dt.month_name()
df['year_added'] = df['date_added'].dt.year

df.head(3)


# In[70]:


# Helper column for various plots
df['count'] = 1

# Many productions have several countries listed - this will skew our results , we'll grab the first one mentioned

# Lets retrieve just the first country
df['first_country'] = df['country'].apply(lambda x: x.split(",")[0])
df['first_country'].head()

# Rating ages from this notebook: https://www.kaggle.com/andreshg/eda-beginner-to-expert-plotly (thank you!)

ratings_ages = {
    'TV-PG': 'Older Kids',
    'TV-MA': 'Adults',
    'TV-Y7-FV': 'Older Kids',
    'TV-Y7': 'Older Kids',
    'TV-14': 'Teens',
    'R': 'Adults',
    'TV-Y': 'Kids',
    'NR': 'Adults',
    'PG-13': 'Teens',
    'TV-G': 'Kids',
    'PG': 'Older Kids',
    'G': 'Kids',
    'UR': 'Adults',
    'NC-17': 'Adults'
}

df['target_ages'] = df['rating'].replace(ratings_ages)
df['target_ages'].unique()

# Genre

df['genre'] = df['listed_in'].apply(lambda x :  x.replace(' ,',',').replace(', ',',').split(',')) 

# Reducing name length

df['first_country'].replace('United States', 'USA', inplace=True)
df['first_country'].replace('United Kingdom', 'UK',inplace=True)
df['first_country'].replace('South Korea', 'S. Korea',inplace=True)


# In[71]:


data = df.groupby('first_country')['count'].sum().sort_values(ascending=False)[:10]

# Plot

color_map = ['#f5f5f1' for _ in range(10)]
color_map[0] = color_map[1] = color_map[2] =  '#b20710' # color highlight

fig, ax = plt.subplots(1,1, figsize=(12, 6))
ax.bar(data.index, data, width=0.5, 
       edgecolor='darkgray',
       linewidth=0.6,color=color_map)

#annotations
for i in data.index:
    ax.annotate(f"{data[i]}", 
                   xy=(i, data[i] + 150), #i like to change this to roughly 5% of the highest cat
                   va = 'center', ha='center',fontweight='light', fontfamily='serif')



# Remove border from plot

for s in ['top', 'left', 'right']:
    ax.spines[s].set_visible(False)
    
# Tick labels

ax.set_xticklabels(data.index, fontfamily='serif', rotation=0)

# Title and sub-title

fig.text(0.09, 1, 'Top 10 countries on Netflix', fontsize=15, fontweight='bold', fontfamily='serif')
fig.text(0.09, 0.95, 'The three most frequent countries have been highlighted.', fontsize=12, fontweight='light', fontfamily='serif')

fig.text(1.1, 1.01, 'Insight', fontsize=15, fontweight='bold', fontfamily='serif')

fig.text(1.1, 0.67, '''
The most prolific producers of
content for Netflix are, primarily,
the USA, with India and the UK
a significant distance behind.

It makes sense that the USA produces 
the most content as, afterall, 
Netflix is a US company.
'''
         , fontsize=12, fontweight='light', fontfamily='serif')

ax.grid(axis='y', linestyle='-', alpha=0.4)   

grid_y_ticks = np.arange(0, 4000, 500) # y ticks, min, max, then step
ax.set_yticks(grid_y_ticks)
ax.set_axisbelow(True)

#Axis labels

#plt.xlabel("Country", fontsize=12, fontweight='light', fontfamily='serif',loc='left',y=-1.5)
#plt.ylabel("Count", fontsize=12, fontweight='light', fontfamily='serif')
 #plt.legend(loc='upper right')
    
# thicken the bottom line if you want to
plt.axhline(y = 0, color = 'black', linewidth = 1.3, alpha = .7)

ax.tick_params(axis='both', which='major', labelsize=12)


import matplotlib.lines as lines
l1 = lines.Line2D([1, 1], [0, 1], transform=fig.transFigure, figure=fig,color='black',lw=0.2)
fig.lines.extend([l1])

ax.tick_params(axis=u'both', which=u'both',length=0)

plt.show()


# In[75]:


country_order = df['first_country'].value_counts()[:11].index
data_q2q3 = df[['type', 'first_country']].groupby('first_country')['type'].value_counts().unstack().loc[country_order]
data_q2q3['sum'] = data_q2q3.sum(axis=1)
data_q2q3_ratio = (data_q2q3.T / data_q2q3['sum']).T[['Movie', 'TV Show']].sort_values(by='Movie',ascending=False)[::-1]




###
fig, ax = plt.subplots(1,1,figsize=(15, 8),)

ax.barh(data_q2q3_ratio.index, data_q2q3_ratio['Movie'], 
        color='#b20710', alpha=0.8, label='Movie')
ax.barh(data_q2q3_ratio.index, data_q2q3_ratio['TV Show'], left=data_q2q3_ratio['Movie'], 
        color='#221f1f', alpha=0.8, label='TV Show')


ax.set_xlim(0, 1)
ax.set_xticks([])
ax.set_yticklabels(data_q2q3_ratio.index, fontfamily='serif', fontsize=11)

# male percentage
for i in data_q2q3_ratio.index:
    ax.annotate(f"{data_q2q3_ratio['Movie'][i]*100:.3}%", 
                   xy=(data_q2q3_ratio['Movie'][i]/2, i),
                   va = 'center', ha='center',fontsize=12, fontweight='light', fontfamily='serif',
                   color='white')

for i in data_q2q3_ratio.index:
    ax.annotate(f"{data_q2q3_ratio['TV Show'][i]*100:.3}%", 
                   xy=(data_q2q3_ratio['Movie'][i]+data_q2q3_ratio['TV Show'][i]/2, i),
                   va = 'center', ha='center',fontsize=12, fontweight='light', fontfamily='serif',
                   color='white')
    

fig.text(0.13, 0.93, 'Top 10 countries Movie & TV Show split', fontsize=15, fontweight='bold', fontfamily='serif')   
fig.text(0.131, 0.89, 'Percent Stacked Bar Chart', fontsize=12,fontfamily='serif')   

for s in ['top', 'left', 'right', 'bottom']:
    ax.spines[s].set_visible(False)
    
#ax.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.06))

fig.text(0.75,0.9,"Movie", fontweight="bold", fontfamily='serif', fontsize=15, color='#b20710')
fig.text(0.81,0.9,"|", fontweight="bold", fontfamily='serif', fontsize=15, color='black')
fig.text(0.82,0.9,"TV Show", fontweight="bold", fontfamily='serif', fontsize=15, color='#221f1f')


fig.text(1.1, 0.93, 'Insight', fontsize=15, fontweight='bold', fontfamily='serif')

fig.text(1.1, 0.44, '''
Interestingly, Netflix in India
is made up nearly entirely of Movies. 

Bollywood is big business, and perhaps
the main focus of this industry is Movies
and not TV Shows.

South Korean Netflix on the other hand is 
almost entirely TV Shows.

The underlying resons for the difference 
in content must be due to market research
conducted by Netflix.
'''
         , fontsize=12, fontweight='light', fontfamily='serif')



import matplotlib.lines as lines
l1 = lines.Line2D([1, 1], [0, 1], transform=fig.transFigure, figure=fig,color='black',lw=0.2)
fig.lines.extend([l1])




ax.tick_params(axis='both', which='major', labelsize=12)
ax.tick_params(axis=u'both', which=u'both',length=0)

plt.show()


# In[76]:


##Language


# In[77]:


netflix_pd = pd.read_csv("NetflixOriginals.csv")


# In[78]:


data_path = "NetflixOriginals.csv"
data = pd.read_csv(data_path)


# In[79]:


df = pd.read_csv("NetflixOriginals.csv")
df.head()


# In[80]:


common_languages=netflix_pd['Language'].value_counts().reset_index(name='total')
language_list=common_languages[common_languages['total']>=3]['index']


# In[82]:


common_languages[common_languages['total']>3].plot.bar(x='index', y='total',rot=90)


# In[83]:


## Netflix Runtime


# In[84]:


netflix_pd = pd.read_csv("IMDB-Movie-Data.csv")
netflix_pd.head()


# In[85]:


netflix_pd.drop_duplicates(subset ="Title", keep = "first", inplace = True)


# In[86]:


netflix_runtime = netflix_pd[["Title","Year","Rating","Runtime (Minutes)"]]


# In[87]:


netflix_runtime = netflix_runtime.dropna(how="any")


# In[88]:


bins = [0, 59.99, 74.99, 89.99, 104.99, 119.99, 134.99, 149.99, 300]
bin_names = ["Less than 60", "60-75", "75-90", "90-105", "105-120", "120-135", "135-150", "More than 150"]


# In[89]:


netflix_pd["Runtime Group (Minutes)"] = pd.cut(netflix_runtime["Runtime (Minutes)"], bins, labels=bin_names, include_lowest=True)


# In[90]:


# theres a better way to do this BUT...
# get the mean rating of each bin
less_hour_rating = netflix_pd.loc[netflix_pd["Runtime Group (Minutes)"] == "Less than 60", "Rating"]
hour_rating = netflix_pd.loc[netflix_pd["Runtime Group (Minutes)"] == "60-75", "Rating"]
hour_15_rating = netflix_pd.loc[netflix_pd["Runtime Group (Minutes)"] == "75-90", "Rating"]
hour_30_rating = netflix_pd.loc[netflix_pd["Runtime Group (Minutes)"] == "90-105", "Rating"]
hour_45_rating = netflix_pd.loc[netflix_pd["Runtime Group (Minutes)"] == "105-120", "Rating"]
two_hour_rating = netflix_pd.loc[netflix_pd["Runtime Group (Minutes)"] == "120-135", "Rating"]
two_hour_15_rating = netflix_pd.loc[netflix_pd["Runtime Group (Minutes)"] == "135-150", "Rating"]
over_hour_rating = netflix_pd.loc[netflix_pd["Runtime Group (Minutes)"] == "More than 150", "Rating"]


# In[91]:


less_hour_rating_mean = less_hour_rating.mean()
hour_rating_mean = hour_rating.mean()
hour_15_rating_mean = hour_15_rating.mean()
hour_30_rating_mean = hour_30_rating.mean()
hour_45_rating_mean = hour_45_rating.mean()
two_hour_rating_mean = two_hour_rating.mean()
two_hour_15_rating_mean = two_hour_15_rating.mean()
over_hour_rating_mean = over_hour_rating.mean()


# In[92]:


# define parameters
labels = ["Less than 60", "60-75", "75-90", "90-105", "105-120", "120-135", "135-150", "More than 150"]
ratings = [0, hour_rating_mean, hour_15_rating_mean, hour_30_rating_mean, hour_45_rating_mean, two_hour_rating_mean, two_hour_15_rating_mean, over_hour_rating_mean]


# In[93]:


# graph bar chart
plt.bar(labels, ratings, color="red", alpha=0.5, align="center")
plt.title("Runtime Ratings")
plt.xlabel("Runtime Bins (Minutes)")
plt.ylabel("Ratings")
plt.ylim(0, 10)
plt.xticks(rotation=75)


# In[ ]:




