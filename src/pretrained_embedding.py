#!/usr/bin/env python
# coding: utf-8

# In[42]:


from string import punctuation
from os import listdir
from numpy import array
from numpy import asarray
from numpy import zeros

from data_preprocessing import text_normalization, remove_stop_words

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import itertools


# In[2]:


# load embedding as a dict
def load_embedding(file_path):
	# load embedding into memory
	file = open(file_path,'r', encoding="utf8")
	lines = file.readlines()
	file.close()
	# create a map of words to vectors
	embedding = dict()
	for line in lines:
		parts = line.split()
		# key is string word, value is numpy array for vector
		embedding[parts[0]] = asarray(parts[1:], dtype='float32')
	return embedding


# In[45]:


# create a weight matrix for the Embedding layer from a loaded embedding
def get_weight_matrix(embedding, vocab, vocabulary_length, embedding_dim): # vocab --> tokenizer word_index
    # define weight matrix dimensions with all 0
    weight_matrix = zeros((vocabulary_length, embedding_dim))
    # step vocab, store vectors using the Tokenizer's integer mapping
    for word, i in itertools.islice(vocab.items(), 0, vocabulary_length - 1):
        vector = embedding.get(word)
        if vector is not None:
            weight_matrix[i] = vector
    return weight_matrix


# In[39]:


def get_input_plus_embedding_vectors(data, file_path, vocabulary_length, max_length_sequence, emb_dim, language):
    
    df = data.copy(deep = True)
    text_normalization(df) # Text normalization
    remove_stop_words(df, language, True)
    
    # Tokenizer
    tokenizer = Tokenizer(num_words = vocabulary_length)
    tokenizer.fit_on_texts(df.text)
    X = tokenizer.texts_to_sequences(df.text)
    
    # Padding
    X = pad_sequences(X, maxlen = max_length_sequence, padding = 'post', truncating = 'post')
    
    # load embedding from file
    raw_embedding = load_embedding(file_path)
    # get vectors in the right order
    embedding_vectors = get_weight_matrix(raw_embedding, tokenizer.word_index, vocabulary_length, emb_dim)
    
    return X, df, embedding_vectors

