#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import re
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt


def max_length_text(df):
    max_length = 0
    for i in range(df.shape[0]):
        length = np.size(word_tokenize(df.at[i, 'text']))
        if length > max_length: max_length = length
    return max_length


def sequence_length_histogram(df):
    lengths = []
    for i in range(df.shape[0]):
        length = np.size(word_tokenize(df.at[i, 'text']))
        lengths.append(length)
    
    plt.hist(lengths, bins = 20)
    plt.show()
    return lengths


# add n_samples from data_2 to data_1
def add_data_portion(data_1, data_2, n_samples):
    df_slice = data_2.sample(n_samples)
    df_rest = data_2.loc[~data_2.index.isin(df_slice.index)]
    df_extended = data_1.append(df_slice, ignore_index = True)
    
    df_rest.reset_index(inplace = True)
    del(df_rest['index'])
    df_extended.reset_index(inplace = True)
    del(df_extended['index'])
    
    return df_extended, df_rest


def text_normalization(data):
    data['text'] = data['text'].apply(lambda x: x.lower())
    data['text'] = data['text'].apply((lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x)))

    
def remove_stop_words(data, language, get_tokenize):
    stopwords = nltk.corpus.stopwords.words(language)
    if get_tokenize:
        for i in range(data.shape[0]):
            data.at[i, 'text'] = [word for word in nltk.word_tokenize(data.at[i, 'text']) if word not in stopwords]
    else:
        for i in range(data.shape[0]):
            data.at[i, 'text'] = [word for word in nltk.word_tokenize(data.at[i, 'text']) if word not in stopwords]
            data.at[i, 'text'] = ' '.join(data.at[i, 'text'])


def apply_stemming(data, language):
    stemmer = SnowballStemmer(language)
    for i in range(data.shape[0]):
         data.at[i, 'text'] = (' '.join([stemmer.stem(word) for word in data.at[i, 'text'].split()]))


# #### get_matrix representation | BoW and Tf-idf for Classic ML

def get_matrix(data, representation, vocabulary_length, stemming, remove_stopwords, language):

    df = data.copy(deep = True)
    
    text_normalization(df) # Text normalization
    
    # Stop_words
    if remove_stopwords:
        remove_stop_words(df, language, False)
    
    # Stemming
    if stemming:
        apply_stemming(df, language)
    
    # Word representation
    if representation == 'BoW':
        count_vectorizer = CountVectorizer(max_df = 0.9, max_features = vocabulary_length, min_df = 0)
        #count_vectorizer = CountVectorizer(max_features = vocabulary_length)
        matrix = count_vectorizer.fit_transform(df.text)
        
    elif representation == 'tf-idf':
        tfidf_vectorizer = TfidfVectorizer(max_df = 0.9, max_features = vocabulary_length, min_df = 0, use_idf = True)
        #tfidf_vectorizer = TfidfVectorizer(max_features = vocabulary_length, use_idf=True)
        matrix = tfidf_vectorizer.fit_transform(df.text)
    
    return matrix, df


# #### Preprocessing for RNN - LSTM; CNN

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def get_input(data, stemming, remove_stopwords, vocabulary_length, max_length_sequence, language):
    
    df = data.copy(deep = True)
    
    text_normalization(df) # Text normalization
    
    # Stemming
    if stemming:
        apply_stemming(df, language)
    
    # Stop_words
    if remove_stopwords:
        remove_stop_words(df, language, True)
        
    # Tokenizer
    tokenizer = Tokenizer(num_words = vocabulary_length)
    tokenizer.fit_on_texts(df.text)
    X = tokenizer.texts_to_sequences(df.text)
    
    # Padding
    X = pad_sequences(X, maxlen = max_length_sequence, padding = 'post', truncating = 'post')
    
    return X, df


def get_input_share_tokenizer(data_1, data_2, stemming, remove_stopwords, vocabulary_length, max_length_sequence, language):
    
    df1 = data_1.copy(deep = True)
    df2 = data_2.copy(deep = True)
    
    text_normalization(df1) # Text normalization
    text_normalization(df2)
    
    # Stemming
    if stemming:
        apply_stemming(df1, language)
        apply_stemming(df2, language)
    
    # Stop_words
    if remove_stopwords:
        remove_stop_words(df1, language, True)
        remove_stop_words(df2, language, True)
        
    # Tokenizer
    tokenizer = Tokenizer(num_words = vocabulary_length)
    tokenizer.fit_on_texts(df1.text)
    X_1 = tokenizer.texts_to_sequences(df1.text)
    X_2 = tokenizer.texts_to_sequences(df2.text)
    
    # Padding
    X_1 = pad_sequences(X_1, maxlen = max_length_sequence, padding = 'post', truncating = 'post')
    X_2 = pad_sequences(X_2, maxlen = max_length_sequence, padding = 'post', truncating = 'post')
    
    return X_1, X_2, df1, df2
