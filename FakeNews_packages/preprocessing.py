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



# Variables: booléens ou limites numéraires e.g. nb de mots/charactères
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





# fonction de preprocessing pour détecter les langues
def lang_detector(texte):
    try:
        detector = Detector(texte)
        return detector.language.name
    except Exception as e:
        return 'unknown / too short'


#création de colonnes pr la langue du texte et du titre de l'article
def df_full_eng(dataframe, col_title, col_text):

    # applique fct de détection pour les titres/et textes de chacund de nos articles
    dataframe['title_language'] = dataframe[col_title].apply(lang_detector)
    dataframe['text_language'] = dataframe[col_text].apply(lang_detector)

    # crée un df temporaire pr garder que les articles dont le texte est en anglais
    temp_df = dataframe[dataframe['text_language'] == 'English']

    # # crée le df de retour pr garder que les articles dont le texte et le titre sont en anglais
    new_df = temp_df[dataframe['title_language'] == 'English']

    # drop colonnes de langues de titre et de texte
    final_df = new_df.drop(columns=['title_language', 'text_language'])

    return new_df

# création de colonne nb de mots par article
def colonne_longeur_article_mot(dataframe, col_text):
    dataframe['word_count_text'] = dataframe[col_text].fillna("").str.count(r'\S+')
    return dataframe

# création de colonne nb de mots par titre d'article
def colonne_longeur_article_mot(dataframe, col_title):
    dataframe['word_count_title'] = dataframe[col_title].fillna("").str.count(r'\S+')
    return dataframe


# création barrière haute et basse pour nb de mots
def limite_h_b_mots(dataframe, high_lim_word, low_lim_word):
    # df sans les articles ou le compte de mot est en dessous de la limite basse
    df_low = dataframe[dataframe['word_count_text']>=low_lim_word]

    # df sans les articles ou le nombre de mot est au dessus de la lim haute
    df_low_high = df_low[df_low['word_count_text']<= high_lim_word]

    return df_low_high
