from from data import get_data_text_title_df

"""
PARAMS PREPROCESSING
"""

# words limits:
df = get_data_text_title_df
column_name = '' # "title" or "text"
high_lim = 5000  # high lim of words
low_lim = # low lim of words

# column choice: for the function to return either a df with only the title + label; only the text + label or text + title concatenated + label
title = False
text = True
both = False


# BASIC_TEXT_CLEANING
# Strip :
Strip = True

# remove internet links:
rem_links = True

# remove selected words
word_list = ['']
rem_sel_words = False

# list words to remove???

# lowercase
lower = True

# remove digits:
rem_dig = True

# remove punctuation
rem_pun = True

# keep hashtags:
keep_h = False

# PREPROC_TEXT

tokens = True # do not change

Stopwords = True

language = 'english' #for lemmatizing

lemmat = True

# selecting the lematization for each type of word
nouns = True
verbs = True
adjectives = True
sat_adj = True

reassemble_txt = True



def preproc_txt(txt: str, clean_txt: bool=True, strip: bool=True, remove_links_bool: bool=True,
                remove_selected_words_bool: bool=False, list_words_to_remove: List[str]=[''],lower: bool=True,
                remove_digits: bool=True, remove_punctuation: bool=True,
                keep_hashtags: bool=False, tokenize: bool=True,
                stopwords: bool=True, language: str='english',
                lemmatize: bool=True, nouns: bool=False, verbs: bool=False,
                adjectives: bool=False, adverbs: bool=False,
                satellite_adjectives: bool=False,
                reassemble_txt: bool=True) -> str:
