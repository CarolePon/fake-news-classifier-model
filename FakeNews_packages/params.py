
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
DATA_FILE = "FN_preproc_text.csv"
# file path where the data is locally saved:
LOCAL_FILE_PATH = f"../raw_data/{DATA_FILE}"
# bucket where the data is saved on gcs:
BUCKET_NAME = "fnsm"
#name of the file in the bucket = blob name
SOURCE_BLOB_NAME = DATA_FILE
# destination_file_name: The path and name where the file will be saved locally on the VM:
DESTINATION_FILE_NAME = "../raw_data/Temp_raw_data_model.csv"

# File path and file name to store the fitted and trained model
TRAINED_MODEL_DESTINATION_FILE_NAME = "../raw_data/Trained_ML_model.pkl"

# which column to select
columns = ["text","label"]  # or "title"

MODEL_FILE = "Trained_ML_model_FN_preproc_text.pkl"
LOCAL_FILE_PATH_MODEL_ML = f"../raw_data/{MODEL_FILE}"


# ML parameters found from the grid search
tfidfvectorizer__ngram_range = (1,3)
multinomialnb__alpha = 0.221076
