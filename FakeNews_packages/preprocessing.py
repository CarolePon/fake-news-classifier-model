# imports
import numpy as np
import pandas as pd
import string


import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import stopwords
from nltk import word_tokenize


# Function dropna to remove empty text or empty title
def drop_na(df):
    cleaned_df= df.dropna()
    return cleaned_df

# Strip text to remove empty spaces at beginning and end of rows
def strip_text(df,col_title,col_text):
    df[col_title] = df[col_title].apply(lambda x: x.strip())
    df[col_text] = df[col_text].apply(lambda x: x.strip())
    return df

# Punctuation 1/2 : function to remove punction from string.punctuation
# # ('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
def remove_punctuation(text_to_clean):
    for punctuation in string.punctuation:
        text_to_clean = text_to_clean.replace(punctuation, ' ')
    return text_to_clean

# Punctuation 2/2 :apply the punctuation function to the dataframe (df)
def remove_punctuation_df(df,col_title,col_text):
    df[col_text] = df[col_text].apply(remove_punctuation)
    df[col_title] = df[col_title].apply(remove_punctuation)
    return df

# remove numbers from title and text
# remove numbers 1/2 : create function for each string
def remove_numbers_text(text):
    words_only = ''.join([i for i in text if not i.isdigit()])
    return words_only

# remove numbers 2/2 : apply function to dataframe
def remove_numbers(df,col_title,col_text):
    df[col_title] = df[col_title].apply(remove_numbers_text)
    df[col_text] = df[col_text].apply(remove_numbers_text)
    return df
