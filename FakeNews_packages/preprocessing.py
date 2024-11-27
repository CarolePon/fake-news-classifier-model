# imports:
from polyglot.detect import Detector
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



# Variables for the functions
col_title = 'title'
col_text = 'text'
high_lim_word = 5000
low_lim_word = 0



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
def remove_punctuation_text(text_to_clean):
    for punctuation in string.punctuation:
        text_to_clean = text_to_clean.replace(punctuation, ' ')
    return text_to_clean

# Punctuation 2/2 :apply the punctuation function to the dataframe (df)
def remove_punctuation_df(df,col_title,col_text):
    df[col_text] = df[col_text].apply(remove_punctuation_text)
    df[col_title] = df[col_title].apply(remove_punctuation_text)
    return df

# remove numbers from title and text
# remove numbers 1/2 : create function for each string
def remove_numbers_text(text):
    words_only = ''.join([i for i in text if not i.isdigit()])
    return words_only

# remove numbers 2/2 : apply function to dataframe
def remove_numbers_df(df,col_title,col_text):
    df[col_title] = df[col_title].apply(remove_numbers_text)
    df[col_text] = df[col_text].apply(remove_numbers_text)
    return df

# move to lower case
def lowercase_text(text):
    lowercased = text.lower()
    return lowercased

def lower_case_df(df,col_title,col_text):
    df[col_title] = df[col_title].apply(lowercase_text)
    df[col_text] = df[col_text].apply(lowercase_text)
    return df


# function to check the number of words in the string in a column 'col'
def nb_words_col(dataframe, col_text):
    dataframe[f"word_count_{col_text}"] = dataframe[col_text].fillna("").str.count(r'\S+')
    return dataframe

# create high and low limits to remove outliers in dataset dor a column word_count_'col'
def limit_nb_words(dataframe, col_text, high_lim_word, low_lim_word):
    # boolean mask on the datafreme removing rows where number of words < low_limit for the column 'col'
    df_low = dataframe[dataframe[f"word_count_{col_text}"]>=low_lim_word]

    # df sans les articles ou le nombre de mot est au dessus de la lim haute
    df_low_high = df_low[df_low[f"word_count_{col_text}"]<= high_lim_word]

    return df_low_high


# function to detect and take out all http and https
def remove_regex(df, col_text, col_title):
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    df[col_text] = df[col_text].str.replace(url_pattern, '', regex=True)
    df[col_title] = df[col_title].str.replace(url_pattern, '', regex=True)
    return df

# function to remove the recurring words in our dataframe
# .com, etc


""" STOP ENLEVER APRES """
FILE= '/Users/macpro/code/CarolePon/fake-news-classifier-model/raw_data/Fake_News_kaggle_english.csv'
""" STOP ENLEVER APRES """


if __name__=="__main__":
    df= pd.read_csv(FILE, nrows= 1000)

    temp_df_1 = drop_na(df)
    print(f"drop na OK")

    temp_df_2 = strip_text(temp_df_1,col_title,col_text)
    print("strip_text OK")

    temp_df_3 = remove_punctuation_df(temp_df_2,col_title,col_text)
    print("remove_punctuation_df OK")

    temp_df_4 = remove_numbers_df(temp_df_3,col_title,col_text)
    print('remove_numbers OK')

    temp_df_4_bis = lower_case_df(temp_df_4, col_title,col_text)

    temp_df_5 = nb_words_col(temp_df_4_bis, col_title)
    print('colonne_longeur_article_mot for title ok')

    temp_df_6 = nb_words_col(temp_df_5, col_text)
    print('colonne_longeur_article_mot for text ok')

    temp_df_7 = limit_nb_words(temp_df_6, col_text,high_lim_word, low_lim_word)
    print('limit words')

    temp_df_8 = limit_nb_words(temp_df_7, col_text,high_lim_word, low_lim_word)
    print('limit words')

    print(temp_df_8.head(3))
    print(temp_df_8.shape)
    print(temp_df_8.columns)
