# imports:
from polyglot import polyglot.Detector
#from polyglot.detect import Detector
import numpy as np
import pandas as pd

url = '/home/baptiste/code/fake_news_classifier_model/FNCM/raw_data/WELFake_Dataset.csv'
df = pd.read_csv(url)

# Variables: booléens ou limites numéraires e.g. nb de mots/charactères
col_title = 'title'
col_text = 'text'
high_lim_word = 5000
low_lim_word = 0



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
    new_df = temp_df[temp_df['title_language'] == 'English']

    # drop colonnes de langues de titre et de texte
    final_df = new_df.drop(columns=['title_language', 'text_language'])

    return new_df

# création de colonne nb de mots par article
def colonne_longeur_article_mot(dataframe, col_text):
    dataframe['word_count_text'] = dataframe[col_text].fillna("").str.count(r'\S+')
    return dataframe

# création de colonne nb de mots par titre d'article
def colonne_longeur_titre_mot(dataframe, col_title):
    dataframe['word_count_title'] = dataframe[col_title].fillna("").str.count(r'\S+')
    return dataframe


# création barrière haute et basse pour nb de mots
def limite_h_b_mots(dataframe, high_lim_word, low_lim_word):
    # df sans les articles ou le compte de mot est en dessous de la limite basse
    df_low = dataframe[dataframe['word_count_text']>=low_lim_word]

    # df sans les articles ou le nombre de mot est au dessus de la lim haute
    df_low_high = df_low[df_low['word_count_text']<= high_lim_word]

    return df_low_high

# df_anglais = df_full_eng(df, col_title, col_text)
# df_col_longueur_article = colonne_longeur_article_mot(df_anglais, col_text)
# df_col_titre_mot = colonne_longeur_titre_mot(df_col_longueur_article, col_title)
# df_limite_h_b = limite_h_b_mots(df_col_titre_mot, high_lim_word, low_lim_word)

# print(df_anglais, df_col_longueur_article, df_col_titre_mot, df_limite_h_b, df_limite_h_b.shape)
