#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import re
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


def max_length_text(df):
    max_length = 0
    for i in range(df.shape[0]):
        length = np.size(word_tokenize(df.at[i, 'text']))
        if length > max_length: max_length = length
    return max_length

def text_normalization(data):
    data['text'] = data['text'].apply(lambda x: x.lower())
    data['text'] = data['text'].apply((lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x)))


def remove_stop_words(data, language):
    stopwords = nltk.corpus.stopwords.words(language)
    for i in range(data.shape[0]):
        data.at[i, 'text'] = [word for word in nltk.word_tokenize(data.at[i, 'text']) if word not in stopwords]
        data.at[i, 'text'] = ' '.join(data.at[i, 'text'])


def apply_stemming(data, language):
    stemmer = SnowballStemmer(language)
    for i in range(data.shape[0]):
         data.at[i, 'text'] = (' '.join([stemmer.stem(word) for word in data.at[i, 'text'].split()]))


def get_matrix(data, representation, vocabulary_length, stop_words_flag, language):

    df = data.copy(deep = True)
    
    text_normalization(df) # Text normalization
    
    # Stop_words
    if stop_words_flag:
        remove_stop_words(df, language)
    
    apply_stemming(df, language) # Stemming
    
    # Word representation
    if representation == 'BoW':
        count_vectorizer = CountVectorizer(max_features = vocabulary_length)
        matrix = count_vectorizer.fit_transform(df.text)
        
    elif representation == 'tf-idf':
        tfidf_vectorizer = TfidfVectorizer(max_features = vocabulary_length, use_idf=True)
        matrix = tfidf_vectorizer.fit_transform(df.text)
    
    return matrix, df
