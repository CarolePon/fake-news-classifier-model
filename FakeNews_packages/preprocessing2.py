import pandas as pd
import string
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List
from FakeNews_packages import params
from FakeNews_packages.data import get_data_text_title_df
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')


""" This function takes a string and apply preprocessing functions """
""" This function takes a string and apply preprocessing functions

"""


# OPERATIONS ON DFS:
# order: drop na, removes news for which word count is higher and lower than limits, selects the columns to apply the preprocess to


# Function dropna to remove empty IF dataframe
def drop_na(df):
    cleaned_df= df.dropna()
    return cleaned_df

# of words in string (NEW TO ADD), returns a number to integrate to a new column of a df
def word_count(txt: str) -> str:
    if txt is None:
        txt = ""
    return len(re.findall(r'\S+', txt))

# removes the lines that contain more words than high lim and less words than low lim (NEW TO ADD)
def lim_nb_of_words_title(df, column_title, high_lim, low_lim):
    df['word_count'] = df[column_title].apply(word_count)
    df_filtered_t = df[df['word_count'] <= high_lim].copy()  # Keep rows with word_count <= limit
    df_filtered = df_filtered_t[df_filtered_t['word_count'] > low_lim].copy()
    df_filtered = df_filtered.drop(columns=['word_count'])

    return df_filtered

def lim_nb_of_words_article(df, column_article, high_lim, low_lim):
    df['word_count'] = df[column_article].apply(word_count)
    df_filtered_t = df[df['word_count'] <= high_lim].copy()  # Keep rows with word_count <= limit
    df_filtered = df_filtered_t[df_filtered_t['word_count'] > low_lim].copy()
    df_filtered = df_filtered.drop(columns=['word_count'])

    return df_filtered


# choose columns to feed to the model (NEW TO ADD)
def column_choice(df, title: bool, article: bool, both: bool):
    if title:
        return pd.DataFrame(df[['title', 'label']])
    elif article:
        return pd.DataFrame(df[['text', 'label']])
    elif both:
        df_concat = df.copy()
        df_concat['text_and_title'] = df['title'] + ' ' + df['text']
        df_concat['label'] = df['label']
        return df_concat[['text_and_title', 'label']]
    else:
        return ValueError('at least one of the arguments must be true')



def strip_txt(txt: str) -> str:
    return txt.strip()

def lower_txt(txt: str) -> str:
    return txt.lower()

# remove all internet links (NEW TO ADD)
def remove_links(txt: str) -> str:
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    new_string = re.sub(url_pattern, '', txt)
    return new_string

# remove selected words from list (NEW TO ADD)
def remove_selected_words(txt: str, word_list: List[str]) -> str:
    pattern =r'\b(?:' + '|'.join(map(re.escape, word_list)) + r')\b'
    return re.sub(pattern, '', txt)

def remove_digits_txt(txt: str) -> str:
    return  ''.join(char for char in txt if not char.isdigit())

def remove_punctuation_txt(txt: str, keep_hashtags: bool=False) -> str:
    """"couldn't use the replace method because articles and title had some
    'fake accents' not part of the string.punctuation list.
    So instead we create a list of acceptable characters, which are the
    alphabet letters plus a regular space ' ' (plus '#' if we want to keep
    the hashtags) and create a txt made of joins of only acceptable chars"""

    allowed_chars = string.ascii_lowercase + string.ascii_uppercase + ' '
    if keep_hashtags == True:
        allowed_chars += '#'

    txt= ''.join(c for c in txt if c in allowed_chars)

    return txt

def basic_txt_cleaning(txt: str, strip: bool=True, remove_links_bool: bool=True,
                       remove_selected_words_bool: bool=False, list_words_to_remove: List[str]=[''], lower: bool=True,
                       remove_digits: bool=True, remove_punctuation: bool=True,
                       keep_hashtags: bool=False) -> str:
    if strip == True: txt = strip_txt(txt)
    if remove_links_bool == True: txt = remove_links(txt)
    if remove_selected_words_bool == True: txt=remove_selected_words(txt,list_words_to_remove)
    if lower == True: txt = lower_txt(txt)
    if remove_digits == True: txt = remove_digits_txt(txt)
    if remove_punctuation == True: txt = remove_punctuation_txt(txt, keep_hashtags)
    return txt

def tokenize_txt(txt: str) -> List[str]:
    return word_tokenize(txt)

def remove_stop_words_from_list(tokens_list: List[str],
                                language: str='english') -> List[str]:
    stop_words = set(stopwords.words(language))
    return [w for w in tokens_list if not w in stop_words]

def lemmatize_tokens_list(tokens_list: List[str],nouns: bool=False,
                          verbs: bool=False, adjectives: bool=False,
                          adverbs: bool=False,
                          satellite_adjectives: bool=False) -> List[str]:
    if nouns==True:
        tokens_list= [WordNetLemmatizer().lemmatize(word, pos = "n") for word in tokens_list]

    if verbs==True:
        tokens_list= [WordNetLemmatizer().lemmatize(word, pos = "v") for word in tokens_list]

    if adjectives==True:
        tokens_list= [WordNetLemmatizer().lemmatize(word, pos = "a") for word in tokens_list]

    if adverbs==True:
        tokens_list= [WordNetLemmatizer().lemmatize(word, pos = "r") for word in tokens_list]

    if satellite_adjectives==True:
        tokens_list= [WordNetLemmatizer().lemmatize(word, pos = "s") for word in tokens_list]

    return tokens_list

def reassemble_txt_post_lemmatization(lem_tokens: List[str]) -> str:
    return  ' '.join(word for word in lem_tokens)

def preproc_txt(txt: str, clean_txt: bool=True, strip: bool=True, remove_links_bool: bool=True,
                remove_selected_words_bool: bool=False, list_words_to_remove: List[str]=[''],lower: bool=True,
                remove_digits: bool=True, remove_punctuation: bool=True,
                keep_hashtags: bool=False, tokenize: bool=True,
                stopwords: bool=True, language: str='english',
                lemmatize: bool=True, nouns: bool=False, verbs: bool=False,
                adjectives: bool=False, adverbs: bool=False,
                satellite_adjectives: bool=False,
                reassemble_txt: bool=True) -> str:


    #print error if lemmatize true but tokenize false as we want to lemmatize tokens only
    if lemmatize == True and tokenize == False:
        print("can't lemmatize without tokenizing first, please change params")
        return

    #same for stopwords
    if stopwords == True and tokenize == False:
        print("can't remove stopwords without tokenizing first, please change params")
        return

    #same for reassembling
    if reassemble_txt == True and tokenize == False:
        print("can't reassemble text without tokenizing first, please change params")
        return

    if clean_txt == True:
        txt = basic_txt_cleaning(txt, strip=strip, remove_links_bool=remove_links_bool,remove_selected_words_bool=remove_selected_words_bool,lower=lower, remove_digits=remove_digits, remove_punctuation=remove_punctuation, keep_hashtags=keep_hashtags)

    if tokenize == True:
        txt = tokenize_txt(txt)

    if stopwords == True:
        txt = remove_stop_words_from_list(txt, language=language)

    if lemmatize == True:
        txt = lemmatize_tokens_list(txt, nouns=nouns, verbs=verbs, adjectives=adjectives, adverbs=adverbs, satellite_adjectives=satellite_adjectives)

    if reassemble_txt == True:
        txt = reassemble_txt_post_lemmatization(txt)

    return txt




""" STOP ENLEVER APRES """
FILE= '/home/toji/code/CarolePon/fncm/raw_data/Fake_News_kaggle_english.csv'
""" STOP ENLEVER APRES """

# if __name__=="__main__":
# takes as input the boolean desiging if we want the text, the title or the text and title concatenated
def preproc_training(text:bool, title:bool, both:bool): #text, or title, or both takes 1 and the other takes zeros
    df= get_data_text_title_df()
    df = drop_na(df)
    df = lim_nb_of_words_title(df, params.column_title, params.high_lim_title, params.low_lim_title)
    df = lim_nb_of_words_article(df, params.column_title, params.high_lim_title, params.low_lim_title)

    if title:
        df = column_choice(df, 1, 0, 0)
    elif text:
        df = column_choice(df, 0, 1, 0)
    elif both:
        df = column_choice(df, 0, 0, 1)

    """make a list of the df columns to preprocess"""
    preproc_columns = df.columns.drop('label')
    preproc_params={'clean_txt': params.clean_text,
                    'strip': params.Strip,
                    'remove_links_bool': params.rem_links,
                    'remove_selected_words_bool': params.rem_sel_words,
                    'list_words_to_remove': params.word_list,
                    'lower': params.lower,
                    'remove_digits': params.rem_dig,
                    'remove_punctuation': params.rem_pun,
                    'keep_hashtags': params.keep_h,
                    'tokenize': params.tokens,
                    'stopwords': params.stpwords,
                    'language': params.language,
                    'lemmatize': params.lemmat,
                    'nouns': params.nouns,
                    'verbs': params.verbs,
                    'adjectives': params.adjectives,
                    'satellite_adjectives': params.sat_adj,
                    'reassemble_txt': params.reassemble_txt
                    }


    for col in preproc_columns:
        df[col] = df[col].apply(preproc_txt, **preproc_params)
    """the '**' before preproc params will unpack the dictionary and pass key=value"""
    return df


# the input is the text and the booleans refer to the type you want as entries
# the function return the preprocessed input to feed the model for the prediction
def preproc_predict(input, text:bool, title:bool, both:bool):
    preproc_params={'clean_txt': params.clean_text,
                    'strip': params.Strip,
                    'remove_links_bool': params.rem_links,
                    'remove_selected_words_bool': params.rem_sel_words,
                    'list_words_to_remove': params.word_list,
                    'lower': params.lower,
                    'remove_digits': params.rem_dig,
                    'remove_punctuation': params.rem_pun,
                    'keep_hashtags': params.keep_h,
                    'tokenize': params.tokens,
                    'stopwords': params.stpwords,
                    'language': params.language,
                    'lemmatize': params.lemmat,
                    'nouns': params.nouns,
                    'verbs': params.verbs,
                    'adjectives': params.adjectives,
                    'satellite_adjectives': params.sat_adj,
                    'reassemble_txt': params.reassemble_txt
                    }
    if type(input)==str:
        if text:
            X_predict = preproc_txt(text, **preproc_params)
            return X_predict
        elif title:
            X_predict = preproc_txt(text, **preproc_params)
            return X_predict
    elif type(input)==list:
        text_and_title = input[0] + input[1]
        X_predict = preproc_txt(text_and_title, **preproc_params)
        return X_predict
