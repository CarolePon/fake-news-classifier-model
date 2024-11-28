from data import get_data_text_title_df

"""
PARAMS PREPROCESSING
"""

df = get_data_text_title_df

# words limits title:
column_title = 'title' # "title" or "text"
high_lim_title = 250  # high lim of words
low_lim_title =  0 # low lim of words

# words limits title:
column_article = 'text' # "title" or "text"
high_lim_article = 5000  # high lim of words
low_lim_article =  0 # low lim of words


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
clean_text = True
tokens = True # do not change

stpwords = True

language = 'english' #for lemmatizing

lemmat = True

# selecting the lematization for each type of word
nouns = True
verbs = True
adjectives = True
sat_adj = True

reassemble_txt = True
