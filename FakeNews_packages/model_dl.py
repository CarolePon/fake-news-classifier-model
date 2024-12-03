import os.path
import pandas as pd
import seaborn as sns
import string
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from FakeNews_packages import params
from FakeNews_packages.data import get_data

raw_data_path = os.fspath('/home/toji/code/CarolePon/fncm/raw_data')
dataset_path=os.path.join(raw_data_path,'Fake_News_kaggle_english.csv')
col_to_analyze = 'text'
max_words=1000
test_size=0.2
val_size=0.2
dl_dir_path = os.fspath('/home/toji/code/CarolePon/fncm/raw_data')
embedding_vector_size=30
embedding_window=5
vocab_size=10000
padding_max_length=1000
batch_size=32
epochs=10
patience=5


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

def get_clean_df(df, col, max_words):
    """get a dataframe, the column we'll make our analysis on later, and the maximum words to drop
    article that are too long.
    Then we make a clean df that we'll pass to next functions to make our X and y"""

    clean_df=df[[col,'label']].copy().dropna()

    clean_df = clean_df[clean_df[col].str.split().str.len()<max_words]

    clean_df[col]=clean_df[col].apply(basic_txt_cleaning)
    clean_df.reset_index(drop=True, inplace=True)
    return clean_df


def get_embedding_matrix(X,y,test_size=0.2,embedding_vector_size=300,window=5, vocab_size=10000, padding_max_length=1000):
    """make the embedding matrix we'll pass to the model"""

    """split X,y to train/test, tokenize words & make the embedding using word2vec"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    sentences = [word_tokenize(text) for text in X_train]
    w2v = Word2Vec(sentences=sentences, vector_size=embedding_vector_size, window=window, min_count=1, workers=4)

    """assign an index to each word of corpus"""
    tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)
    #embedding_dim = embedding_vector_size
    embedding_matrix = np.zeros((vocab_size, embedding_vector_size))

    """fill the embedding matrix with weights"""
    for word, i in tokenizer.word_index.items():
        if i >= vocab_size:
            continue
        if word in w2v.wv.key_to_index:
            embedding_matrix[i] = w2v.wv[word]
        else:
            # Initialize with random vectors or zeros
            embedding_matrix[i] = np.random.normal(size=(embedding_vector_size,))

    """get the sequences we'll work on"""
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_train_padded = pad_sequences(X_train_seq, maxlen=padding_max_length, padding='post', truncating='post')
    X_test_padded = pad_sequences(X_test_seq, maxlen=padding_max_length, padding='post', truncating='post')

    return embedding_matrix, X_train_padded, X_test_padded, y_train, y_test

def create_model(embedding_matrix, vocab_size, embedding_vector_size, padding_max_length):
    model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_vector_size, weights=[embedding_matrix],
              input_length=padding_max_length, trainable=False),
    Bidirectional(LSTM(256)),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def train_and_save_model(model, X, y, model_dir='model_checkpoint', batch_size=32, epochs=10, patience=3, val_size=0.2):
    """Train the model with checkpoints and save the final model"""

    """set ES/Checkpoint params"""
    early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    checkpoint_path = os.path.join(model_dir, 'checkpoints')
    checkpoint_path = os.path.join(model_dir, 'model_epoch{epoch:02d}_valacc{val_accuracy:.2f}.hdf5')
    checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max')

    """make validation set"""
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size)


    """train model"""
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stop, checkpoint]
    )

    """save it"""
    final_model_file_path = os.path.join(model_dir, 'final_model.h5')
    model.save(final_model_file_path)
    print(f'Model saved to {final_model_file_path}')

    return history


def main():
    """get data, clean it, assign X,y"""
    data = get_data(params.SOURCE_DATA, params.BUCKET_NAME, params.SOURCE_BLOB_NAME,params.DESTINATION_FILE_NAME, params.columns)

    data_clean=get_clean_df(data,col_to_analyze,max_words)
    X=data_clean[col_to_analyze][:128]
    y=data_clean['label'][:128]

    """embedding, model, training, save model"""
    embedding_matrix, X_train_pad, X_test_pad, y_train, y_test = get_embedding_matrix(X,y,embedding_vector_size=embedding_vector_size,window=embedding_window, vocab_size=vocab_size, padding_max_length=padding_max_length)
    model = create_model(embedding_matrix, vocab_size, embedding_vector_size, padding_max_length)
    history = train_and_save_model(model, X_train_pad, y_train, model_dir=dl_dir_path, batch_size=batch_size, epochs=epochs, patience=patience)


    """test model"""
    loss, accuracy = model.evaluate(X_test_pad, y_test)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

if __name__ == '__main__':
    main()
