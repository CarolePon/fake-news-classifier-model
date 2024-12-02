
"""
PARAMS PREPROCESSING
"""


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


# get the data: either from a local csv or a csv in a bucket in CGP
# if from locally saved file:
SOURCE_DATA = "local"   # =  "gcs" or "local"
#file nalme for data=
DATA_FILE = "Fake_News_kaggle_english.csv"
# file path where the data is locally saved:
LOCAL_FILE_PATH = f"../raw_data/{DATA_FILE}"
# bucket where the data is saved on gcs:
BUCKET_NAME = "fnsm"
#name of the file in the bucket = blob name
SOURCE_BLOB_NAME = DATA_FILE
# destination_file_name: The path and name where the file will be saved locally on the VM:
DESTINATION_FILE_NAME = "raw_data/Temp_raw_data_model"
columns = ["text","label"]


# ML parameters found with 30,000 samples
tfidfvectorizer__ngram_range = (2, 3)
multinomialnb__alpha = 0.1
