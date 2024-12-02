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
from FakeNews_packages.data import get_data
from preprocessing2 import preproc_txt
import numpy as np

from params import *


# for the timer to be displayed:
# Import the time library
import time

# Calculate the start time
start = time.time()


# Remove later = sample of the whole data
def sample_2(data_cleaned,sample_nb):
    data_cleaned_sample=data_cleaned.sample(sample_nb)
    return data_cleaned_sample

#definition of X et y
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
def hyperparams(X, y, tfidfvectorizer__ngram_range, multinomialnb__alpha):

    min_df = 20
    max_df = 0.5
    max_features = X.shape[0]


    # Pipe
    pipeline_naive_bayes = make_pipeline(
        TfidfVectorizer(ngram_range=tfidfvectorizer__ngram_range,min_df= min_df, max_df=max_df),
        MultinomialNB(alpha = multinomialnb__alpha)
        )
    #print("starting grid search")
    print("grid esearch bypassed")

    fitted_pipe = pipeline_naive_bayes.fit(X,y)


    #parameters
    # parameters = {
    # 'tfidfvectorizer__ngram_range': ((1,2),(1,3),(2,3),(2,2)),
    # 'multinomialnb__alpha': (0.1,1)
    #             }

    # best params from 30,000 samples:

    # # GridSearch
    # grid_search = RandomizedSearchCV(
    #     pipeline_naive_bayes,
    #     parameters,
    #     scoring = "accuracy",
    #     cv = 5,
    #     n_jobs=-1,
    #     verbose=1
    #     )

    #Fit grid
    #grid_search.fit(X,y)

    # Best score
    #print(f"Best Score = {grid_search.best_score_}",f"Best params = {grid_search.best_params_}")
    print("best params were from sample =30,000: \n    tfidfvectorizer__ngram_range = (2, 3), \n  multinomialnb__alpha = 0.1")

    #ngrams = grid_search.best_estimator_.get_params()['tfidfvectorizer__ngram_range']

    vect = TfidfVectorizer(ngram_range=tfidfvectorizer__ngram_range,min_df= min_df, max_df=max_df)
    vect_fitted=vect.fit(X,y)
    print('fit is over')


    X_transformed = vect.fit_transform(X)
    print(f"vectorized X shape = {X_transformed.shape}")

    # for i in range(len(X)+1):
    #     X_transformed = vect.fit_transform(X[i:i+1])
    #     print(f"vectorized X shape = {X_transformed.shape}")

    print("returns fitted_pipe, vect_fitted :")
    #return grid_search.best_estimator_, vect_fitted #, grid_search.best_score_
    return fitted_pipe, vect_fitted #, grid_search.best_score_





if __name__ == "__main__":


    data_cleaned = get_data(SOURCE_DATA, BUCKET_NAME, SOURCE_BLOB_NAME,DESTINATION_FILE_NAME, columns)

    print(f"data being used : {data_cleaned}")
    print(f"data shape : {data_cleaned.shape}")


    #sample_nb = data_cleaned.shape[0]
    sample_nb = 1000
    sample_data_cleaned = sample_2(data_cleaned,sample_nb)
    print(f"data shape : {sample_data_cleaned}")

    # not needed if we take the preprocessed data
    #preproc_params={'nouns':True,'verbs':True}

    #sample_data_cleaned['preproc_text'] = sample_data_cleaned['text'].apply(preproc_txt, **preproc_params)

    # preprocessed_data = sample_data_cleaned[['preproc_text','label']]


    # X=variable_X(sample_data_cleaned,'preproc_text')
    X=variable_X(sample_data_cleaned,'text')
    #print(X)

    y=variable_y(sample_data_cleaned,'label')
    #print(y)

    X_train, X_test, y_train, y_test= train_test_split(X, y,test_size = 0.3)

    best_pipeline,vect_fitted=hyperparams(X_train,y_train, tfidfvectorizer__ngram_range, multinomialnb__alpha)


    vectorize_text,y=vectorize(X_test,y_test,vect_fitted)
    print(f"vectorize_text_test shape : {vectorize_text.shape}")

    y_test_predict=(best_pipeline.predict(vectorize_text))

    print(f"sample number ={sample_nb}")
    print(best_pipeline.score(X_test, y_test))



    # Calculate the end time and time taken
    end = time.time()
    length = end - start
    # Show the results : this can be altered however you like
    print("It took: ", length, "seconds")
