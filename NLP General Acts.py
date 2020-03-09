# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 11:49:55 2020

@author: btgl1e14
"""

# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('allukprimaryacts edited.csv')

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1728):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Text >> '][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()


# Remove the columns
dataset = dataset.drop(columns=['web-scraper-order', 'web-scraper-start-url', 'Link-href', 'LinkWhole', 'LinkWhole-href', 'LinkPlain', 'LinkPlain-href'], axis=1)

# Load the regular expression library
import re
# Remove punctuation
dataset['text_processed'] = dataset['Text >> '].map(lambda x: re.sub('([,\.!?>+])', '', x))
# Convert the titles to lowercase
dataset['text_processed'] = dataset['text_processed'].map(lambda x: x.lower())


# Import the wordcloud library
from wordcloud import WordCloud
# Join the different processed titles together.
long_string = ','.join(list(papers['paper_text_processed'].values))
# Create a WordCloud object
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
# Generate a word cloud
wordcloud.generate(long_string)
# Visualize the word cloud
wordcloud.to_image()