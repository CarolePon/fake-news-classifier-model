import pandas as pd
import seaborn as sns
import string
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List

""" This fucntion takes a string and apply:

"""



# Function dropna to remove empty IF dataframe
def drop_na(df):
    cleaned_df= df.dropna()
    return cleaned_df


# Strip text to remove empty spaces at beginning and end of rows
def strip_txt(txt: str) -> str:
    return txt.strip()

# remove all internet links (NEW TO ADD)
def remove_links(txt: str) -> str:
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    new_string = re.sub(url_pattern, '', txt)
    return new_string

# remove selected words from list (NEW TO ADD)
def remove_selected_words(txt: str, word_list: list):
    pattern =r'\b(?:' + '|'.join(map(re.escape, word_list)) + r')\b'
    return re.sub(pattern, '', txt)

# of words in string (NEW TO ADD), returns a number to integrate to a new column of a df
def word_count(txt: str) -> str:
    if text is None:
        text = ""
    return len(re.findall(r'\S+', text))


# move to lower case
def lower_txt(txt: str) -> str:
    return txt.lower()

# remove the digits
def remove_digits_txt(txt: str) -> str:
    return  ''.join(char for char in txt if not char.isdigit())

# remove punctuation
def remove_punctuation_txt(txt: str, keep_hashtags: bool=False) -> str:
    if keep_hashtags == False:
        for punctuation in string.punctuation:
            txt = txt.replace(punctuation, '')
    else:
        for punctuation in string.punctuation.replace('#',''):
            txt = txt.replace(punctuation, '')

    return txt

# cleaning fucntion taking the above functions
def basic_txt_cleaning(txt: str, strip: bool=True, lower: bool=True,
                       remove_digits: bool=True, remove_punctuation: bool=True,
                       keep_hashtags: bool=False) -> str:
    if strip == True: txt = strip_txt(txt)
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


def preproc_txt(txt: str, clean_txt: bool=True, strip: bool=True, lower: bool=True,
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
        txt = basic_txt_cleaning(txt, strip=strip, lower=lower, remove_digits=remove_digits, remove_punctuation=remove_punctuation, keep_hashtags=keep_hashtags)

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
FILE= '/Users/macpro/code/CarolePon/fake-news-classifier-model/raw_data/Fake_News_kaggle_english.csv'
""" STOP ENLEVER APRES """


if __name__=="__main__":
    df= pd.read_csv(FILE, nrows= 1000)
    df = drop_na(df)
    preprocessed_df = pd.DataFrame()
    df['titlepreproc'] = df['title'].apply(preproc_txt,nouns=True)

    print(df.head(10))
