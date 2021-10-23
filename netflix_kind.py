# Dependencies
import pandas as pd
import matplotlib.pyplot as plt

# Import imdb.csv file as DataFrame
netflix_pd = pd.read_csv("../Resources/archive/imdb.csv")

# remove duplicate movies
netflix_pd.drop_duplicates(subset ="title", keep = "first", inplace = True)

# clean and drop NAN in reduced data set
netflix_reduced = netflix_pd.loc[:, ["kind", "rating", "vote"]]
netflix_clean = netflix_reduced.dropna(how="any")

# assign kind values and unique labels
kind_unique = netflix_clean["kind"].unique()
kind_values = netflix_clean["kind"].value_counts()

# graph all the kinds distribution
# define parameters of the graph
labels = ['movie', 'tv short', 'video movie', 'tv movie', 'tv series', 'episode', 'tv mini series', 'video game']
sizes = [5213, 10, 1191, 744, 583, 469, 255, 17]
seperate = (0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01)

# plot the pie chart
plt.figure(figsize=(10,10))
plt.pie(sizes, explode=seperate, autopct="%1.1f%%", labels=labels)
plt.legend(loc="upper right")
plt.axis("equal")
plt.title("Netflix Kind Distribution")
plt.show()

#-----------------------------------------------------------------#

# THIS IS FOR GRAPH #2 IF WE WANT IT
# rename rows
netflix_clean.loc[netflix_clean["kind"] == "video movie", "kind"]="movie"
netflix_clean.loc[netflix_clean["kind"] == "tv movie", "kind"]="movie"
netflix_clean.loc[netflix_clean["kind"] == "tv mini series", "kind"]="tv series"
netflix_clean.loc[netflix_clean["kind"] == "episode", "kind"]="tv series"

# redefine kind_values
unique_kind_count = netflix_clean["kind"].value_counts()

# define parameters for new graph
labels = ["movie","tv short","tv series","video game"]
sizes = [7148, 10, 1307, 17]
seperate = (0.01, 0.01, 0.01, 0.01)

# plot new pie chart
plt.figure(figsize=(10,10))
plt.pie(sizes, explode=seperate, autopct="%1.1f%%", labels=labels)
plt.legend(loc="upper right")
plt.axis("equal")
plt.title("Netflix Kind Distribution")
plt.show()