import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import set_config; set_config("diagram")
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from FakeNews_packages.data import get_data_from_gcs, get_data_text_title_df, get_data_text_df
from preprocessing2 import preproc_txt
import numpy as np





#fonction qui prend un échantillon de sample_nb élements remove later
def sample_2(data_cleaned,sample_nb):
    data_cleaned_sample=data_cleaned.sample(sample_nb)
    return data_cleaned_sample

#definition des X et y
def variable_X(data_cleaned_sample, column_name):
    X=data_cleaned_sample[column_name]
    return X
def variable_y(data_cleaned_sample, column_name):
    y=data_cleaned_sample[column_name]
    return(y)


def vectorize(X,y, vect_fitted):
    vectorizer = vect_fitted
    X_bow = vectorizer.transform(X)
    X_bow.toarray()

    vectorizer.get_feature_names_out()
    vectorized_texts = pd.DataFrame(
        X_bow.toarray(),
        columns = vectorizer.get_feature_names_out(),
        index = X)

    return vectorized_texts, y



#recherche des meilleurs parametres
def hyperparams(X, y):

    # Pipe
    pipeline_naive_bayes = make_pipeline(
        TfidfVectorizer(max_df=0.6),
        MultinomialNB()
        )

    #parameters
    parameters = {
    'tfidfvectorizer__ngram_range': ((1,2),(1,3),(2,3),(2,2)),
    'multinomialnb__alpha': (0.1,1)
                }

    # GridSearch
    grid_search = RandomizedSearchCV(
        pipeline_naive_bayes,
        parameters,
        scoring = "accuracy",
        cv = 5,
        n_jobs=-1,
        verbose=1
        )

    #Fit grid
    grid_search.fit(X,y)

    # Best score
    print(f"Best Score = {grid_search.best_score_}",f"Best params = {grid_search.best_params_}")

    ngrams = grid_search.best_estimator_.get_params()['tfidfvectorizer__ngram_range']
    vect = TfidfVectorizer(ngram_range=ngrams,max_df=0.6)
    vect_fitted=vect.fit(X,y)

    print(f"vectorized X shape = {vect_fitted[0]}")


    return grid_search.best_estimator_, vect_fitted #, grid_search.best_score_





if __name__ == "__main__":

    # Get the dataframe to run model with title and text
    #data_cleaned = get_data_text_df()

    """ test vm"""
    SOURCE_DATA = "gcs"   # =  "gcs" or "local"
    # file path where the data is locally saved:
    LOCAL_FILE_PATH = "../raw_data/Fake_News_kaggle_english.csv"
    # bucket where the data is saved on gcs:
    BUCKET_NAME = "fnsm"
    #name of the file in the bucket = blob name
    SOURCE_BLOB_NAME = "Fake_News_kaggle_english.csv"
    # destination_file_name: The path and name where the file will be saved locally on the VM:
    DESTINATION_FILE_NAME = "../raw_data/Temp_raw_data_model.csv"


    data_cleaned_vm = get_data_from_gcs(SOURCE_DATA, BUCKET_NAME, SOURCE_BLOB_NAME,DESTINATION_FILE_NAME)

    print(f"data being used : {data_cleaned_vm}")
    print(f"data shape : {data_cleaned_vm.shape}")
    """test vm fin """

    sample_nb = 1000
    sample_data_cleaned = sample_2(data_cleaned_vm,sample_nb)
    #print(sample_data_cleaned)

    preproc_params={'nouns':True,'verbs':True}

    sample_data_cleaned['preproc_text'] = sample_data_cleaned['text'].apply(preproc_txt, **preproc_params)

    preprocessed_data = sample_data_cleaned[['preproc_text','label']]


    X=variable_X(preprocessed_data,'preproc_text')
    #print(X)

    y=variable_y(preprocessed_data,'label')
    #print(y)

    X_train, X_test, y_train, y_test= train_test_split(X, y,test_size = 0.3)

    best_pipeline,vect_fitted=hyperparams(X_train,y_train)


    vectorize_text,y=vectorize(X_test,y_test,vect_fitted)
    print(f"vectorize_text_test shape : {vectorize_text.shape}")

    y_test_predict=(best_pipeline.predict(vectorize_text))

    print(f"sample number ={sample_nb}")
    print(best_pipeline.score(X_test, y_test))




    #print(X_train)

    #vecteurs_text,y=vectorize(X,y)
    #print(vecteurs_text,y)


    #hyperparams

    #X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, random_state = 42)
    #cross_validate(MultinomialNB(), X_train, y_train, cv=5)["test_score"].mean()
