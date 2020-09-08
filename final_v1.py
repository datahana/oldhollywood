import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import string
import spacy

from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

from sklearn import metrics
import matplotlib.pyplot as plt

nlp = spacy.load("en_core_web_sm")

#thank you to geeksforgeeks for their introduction to text preprocessing
#source: https://www.geeksforgeeks.org/text-preprocessing-in-python-set-1/
#https://www.geeksforgeeks.org/text-preprocessing-in-python-set-2/
def remove_numbers(text):
    return re.sub(r'\d+','', text)

def count_length(text):
    return len(text)

#remove actors name and any comments in parentheses
def remove_parentheses(text):
    return re.sub(r'\([^()]*\)', '', text)

def remove_punc(text):
    return text.translate(str.maketrans('','', string.punctuation))

def remove_whitespaces(text):
    return " ".join(text.split())

def change_lower(text):
    return text.lower()

#stopwords
def remove_stopwords(text):
    sw = set(stopwords.words("english"))
    word_tokens = word_tokenize(text)
    return ' '.join([w for w in word_tokens if w not in sw])

#lemmatization
lemmatizer = WordNetLemmatizer()

def change_stem(text):
    word_tokens = word_tokenize(text)
    return ' '.join([lemmatizer.lemmatize(word, pos = 'v') for word in word_tokens])

#sentiment analysis (don't clean yet)
def score_senti(text):
    sia = SentimentIntensityAnalyzer()
    p_scores = sia.polarity_scores(text)
    return p_scores

#parts of speech
def tag_pos(text):
    word_tokens = word_tokenize(text)
    return pos_tag(word_tokens)

def clean_plot(text):
    #identify names via Spacy
    doc = nlp(text)
    stor = set()
    for entity in doc.ents:
        if entity.label_ == 'PERSON':
            stor.add(entity.text)
    
    #remove proper pronouns and names from Plot
    de_noun = ' '.join([w[0] for w in tag_pos(text) if w[1] not in ['NNP', "PRP", "PRP$", "POS"]])
    nltk_text = remove_whitespaces(remove_numbers(remove_punc(de_noun)))
    return remove_stopwords(change_stem(change_lower(' '.join([x for x in nltk_text.split(' ') if x not in stor]))))

#############################################
#wikipedia dataset pre-processing
#############################################
#change column names
wiki_df = pd.read_csv('wikimovieplots.csv')
wiki_df.rename({'Release Year': 'year'}, axis=1, inplace=True)
wiki_df['title'] = wiki_df.Title.str.lower()
wiki_df.drop('Title', axis=1, inplace=True)

#filter wikidata by years of interest
wiki_main = wiki_df[((wiki_df.year >= 1930) & (wiki_df.year <= 1960)) & (wiki_df['Origin/Ethnicity'] == 'American')][['year', 'title', 'Genre', 'Plot']].copy()
wiki_new = wiki_df[((wiki_df.year >= 2000) & (wiki_df.year <= 2019)) & (wiki_df['Origin/Ethnicity'] == 'American')][['year', 'title', 'Genre', 'Plot']].copy()

#clean up wikidata of 'thethe's and spaces
thethe_indn = wiki_new[wiki_new.title.str.contains('thethe')].index
wiki_new.loc[thethe_indn, 'title'] = wiki_new.loc[wiki_new.title.str.contains('thethe'), 'title'].apply(lambda x: 'the' + x.split('thethe')[1])

wiki_main.title = wiki_main.title.apply(remove_whitespaces)
wiki_new.title = wiki_new.title.apply(remove_whitespaces)

#keep movie with earliest year
wiki_dedup = wiki_main.sort_values(['year','title']).drop_duplicates(subset=['title', 'Plot'], keep='first')
wiki_dedup_new = wiki_new.sort_values(['year','title']).drop_duplicates(subset=['title', 'Plot'], keep='first').drop_duplicates(subset=['title', 'year'], keep='first')

#############################################
#imdb dataset pre-processing
#############################################
#repeat above data preprocessing but for imdbdata: change column names, filter by years of interest, clean up spaces, and keep movie with earliest year for duplicates
imdb_df = pd.read_csv('imdbmovies.csv')
imdb_df.title = imdb_df.title.str.lower()
imdb_main = imdb_df[((imdb_df.year >= 1930) & (imdb_df.year <= 1960))][['imdb_title_id', 'title', 'year', 'genre', 'description', 'avg_vote', 'votes']].copy()
imdb_main.title = imdb_main.title.apply(remove_whitespaces)
imdb_dedup = imdb_main.sort_values(['year','title', 'votes']).drop_duplicates(subset=['title', 'year'], keep='last')

imdb_new = imdb_df[((imdb_df.year >= 2000) & (imdb_df.year <= 2019))][['imdb_title_id', 'title', 'year', 'genre', 'description', 'avg_vote', 'votes']].copy()
imdb_new.title = imdb_new.title.apply(remove_whitespaces)

#############################################
#combine datasets
#############################################
#take top 22 most voted imdb movies from 2000-2019 and combine with movies from 1930-1960
top20 = imdb_new.sort_values('votes', ascending=False).head(20)
combo_df = imdb_dedup.merge(wiki_dedup, on= ['title','year'], how='inner')
combo_new = top20.merge(wiki_dedup_new, on = ['title','year'], how='inner')
df = pd.concat([combo_new[['imdb_title_id', 'title', 'year', 'avg_vote', 'votes', 'genre', 'Plot']], combo_df[['imdb_title_id', 'title', 'year', 'avg_vote', 'votes', 'genre', 'Plot']]]).reset_index(drop=True)

#############################################
#feature extraction
#############################################
#count vectorizer on genre
cvect = CountVectorizer()
cvect_genre = cvect.fit_transform(df.genre)
df_genre = pd.concat([df, pd.DataFrame(cvect_genre.toarray(), columns=cvect.get_feature_names())], axis=1).drop('genre', axis=1)

#apply sentiment analysis on plot
df_senti_add = df_genre.Plot.apply(score_senti).apply(pd.Series)
df_senti = pd.concat([df_genre, df_senti_add], axis=1)

#add plot length column
df_length_add = df_senti.Plot.apply(lambda x: len(x))
df_length_add.name = 'plot_length'
df_len = pd.concat([df_senti, df_length_add], axis = 1)

#restrict to movies with plot_length >= 500 
df_len = df_len[df_len.plot_length >= 500]

#tfidf the processed plot
tf_test = TfidfVectorizer(ngram_range=(1,1), lowercase=False)
tfidf_test = tf_test.fit_transform(df_len['Plot'].apply(clean_plot))

#perform LSA (3900 selected from analyzing explained_variance_ratio)
svd = TruncatedSVD(n_components = 3900, random_state = 42)
normalizer = Normalizer(copy = False)
lsa = make_pipeline(svd, normalizer)
lsa_df = lsa.fit_transform(tfidf_test)

#analyzed inertia and silhouette scores; selected cluster sizes of 84, 100, 154, 180
#future work includes exploring other clustering options
km84 = KMeans(n_clusters=84, init='k-means++', random_state = 42)
km84.fit(lsa_df)

km100 = KMeans(n_clusters=100, init='k-means++', random_state = 42)
km100.fit(lsa_df)

km154 = KMeans(n_clusters=154, init='k-means++', random_state = 42)
km154.fit(lsa_df)

km180 = KMeans(n_clusters=180, init='k-means++', random_state = 42)
km180.fit(lsa_df)

#assign cluster group labels
df_len['km84'] = km84.labels_
df_len['km100'] = km100.labels_
df_len['km154'] = km154.labels_
df_len['km180'] = km180.labels_

#convert to indicator features
df_km = pd.concat([df_len, pd.get_dummies(df_len.km84, prefix='km84'), pd.get_dummies(df_len.km100, prefix='km100'), pd.get_dummies(df_len.km154, prefix='km154'), pd.get_dummies(df_len.km180, prefix='km180')], axis = 1).drop(['km84', 'km100', 'km154', 'km180'], axis = 1)

#############################################
#recommendations
#############################################
#select movie from top 20 most-rated films of the 21st century (as of 2017)
input_movie_title = 'batman begins'

#subset data frame to look at specific movie title
df_subset = df_km[(df_km.title == input_movie_title) | (df_km.year <= 1960)].reset_index(drop = True)

#setting up data frame for KNN
df_final = df_subset.set_index(['imdb_title_id', 'title', 'year']).drop(['votes', 'Plot', 'plot_length'], axis = 1).copy()

#scaled to values from 0 to 1
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df_final)

#k-nearest-neighbors to get top 5 similar movies (recommmendations)
knn = NearestNeighbors(n_neighbors = 5)
knn.fit(df_scaled[1:])
k_ind = knn.kneighbors([df_scaled[0]])[1]

print([(df_final[1:].reset_index().iloc[i]['title'], df_final[1:].reset_index().iloc[i]['year']) for i in list(k_ind.flatten())])