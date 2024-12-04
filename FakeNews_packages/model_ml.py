import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import set_config; set_config("diagram")
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from FakeNews_packages.data import get_data
import pickle  # to save the best model once fitted
from scipy import stats
import random
from FakeNews_packages import params


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


#looking for the best parameters: randomized grid search
def hyperparams(X, y, min_df, max_df, max_features):
    # Pipe
    pipeline_naive_bayes = make_pipeline(
        TfidfVectorizer(min_df= min_df, max_df=max_df, max_features=max_features),
        MultinomialNB()
        )
    #parameters
    parameters = {
    'tfidfvectorizer__ngram_range': ((1,3),(2,3),(2,2),(3,3)),
    'multinomialnb__alpha': stats.uniform(0.1, 1)
                }
    # GridSearch
    grid_search = RandomizedSearchCV(
        pipeline_naive_bayes,
        parameters,
        scoring = "accuracy",
        cv = 5,
        n_jobs=-1,
        verbose=1,
        n_iter=100
        )

    #Fit grid
    grid_search.fit(X,y)

    # Best score
    print(f"Best Score = {grid_search.best_score_}",f"Best params = {grid_search.best_params_}")

    # getting the parameters from this best score
    ngrams = grid_search.best_estimator_.get_params()['tfidfvectorizer__ngram_range']

    # fitting the vectorized X_train from the vectorizer with the best ngrams to return it
    vect = TfidfVectorizer(ngram_range=ngrams, max_features=max_features)
    vect_fitted= vect.fit(X,y)

    return grid_search.best_estimator_, vect_fitted #, grid_search.best_score_


#recherche des meilleurs parametres
def model_ml(X, y, tfidfvectorizer__ngram_range, multinomialnb__alpha, min_df, max_df, max_features):

    # Pipe
    pipeline_naive_bayes = make_pipeline(
        TfidfVectorizer(ngram_range=tfidfvectorizer__ngram_range,min_df= min_df, max_df=max_df,max_features=max_features),
        MultinomialNB(alpha = multinomialnb__alpha)
        )

    fitted_pipe = pipeline_naive_bayes.fit(X,y)

    print("model is fitted, returns fitted model")

    return fitted_pipe#, vect_fitted


#save the trained model in the destination folder
def saving_model(X, y,
                 tfidfvectorizer__ngram_range,
                 multinomialnb__alpha,
                 min_df, max_df, max_features,
                 TRAINED_MODEL_DESTINATION_FILE_NAME):

    #instantiate model
    pipeline_naive_bayes = make_pipeline(
        TfidfVectorizer(ngram_range=tfidfvectorizer__ngram_range,
                        min_df= min_df,
                        max_df=max_df,
                        max_features=max_features),
        MultinomialNB(alpha = multinomialnb__alpha)
        )

    fitted_pipe = pipeline_naive_bayes.fit(X,y)

    print(f"model saved as : {TRAINED_MODEL_DESTINATION_FILE_NAME}")

    pickle.dump(fitted_pipe, open(TRAINED_MODEL_DESTINATION_FILE_NAME, 'wb'))  # where we store the model weights

    return fitted_pipe






if __name__ == "__main__":


    data_cleaned = get_data(params.SOURCE_DATA, params.BUCKET_NAME, params.SOURCE_BLOB_NAME,params.DESTINATION_FILE_NAME, params.columns)

    print(f"data source for this run : {params.SOURCE_DATA}, source file name:  {params.DATA_FILE}")
    print(f"data being used (head) : {data_cleaned.head(2)}")
    print(f"data shape : {data_cleaned.shape}")


    sample_nb = data_cleaned.shape[0]

    #choose sample size or whole dataset
    #sample_nb = 1000
    sample_data_cleaned = sample_2(data_cleaned,sample_nb)


    # print info if the model is run on the whole datset for a sample
    if sample_nb < data_cleaned.shape[0]:
        print(f"Sample size for this run: {sample_nb}, sample_data shape: {sample_data_cleaned.shape}")
    else:
        print(f"run on the whole dataset: data shape = {sample_data_cleaned.shape}")



    """preprocessing step not needed anymore as we now have csv that are preprocessed """
    #preproc_params={'nouns':True,'verbs':True}
    #sample_data_cleaned['preproc_text'] = sample_data_cleaned['text'].apply(preproc_txt, **preproc_params)
    # preprocessed_data = sample_data_cleaned[['preproc_text','label']]
    """preprocessing step not needed anymore as we now have csv that are preprocessed """


    X = variable_X(sample_data_cleaned,'text')
    y = variable_y(sample_data_cleaned,'label')
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3)

    # Parameters of the vectorizer to limit the number of vectors created
    min_df = 20
    max_df = 0.5
    max_features = int(X.shape[0]/2)

    print(f"max_feature = {max_features}")


    # 3 options: run grid search, train model (and compare y_test and y_predict) or Save model
    #action = "gridsearch"
    # action = "model"
    action = "save_model"


    if action == "gridsearch":
        #Grid search with the above parameters
        print(f"Running Grid Search pipeline to get the best model parameters")

        best_pipeline,vect_fitted = hyperparams(X_train,y_train, min_df, max_df, max_features)
        vectorize_text,y = vectorize(X_test, y_test,vect_fitted)

        print(f"vectorize_X_test shape : {vectorize_text.shape}")

        y_test_predict=(best_pipeline.predict(vectorize_text))

        print(f"for sample number ={sample_nb}")
        print(f"result of gridsearch: best pipeline score: {best_pipeline.score(X_test, y_test)}")


    if action == "model":
        #running model with parameters chosen after grid search on X_train and y_train
        print(f"""Running model with best params to test it on train/test data (with parameters found from grid search on the whole dataset):
              tfidfvectorizer__ngram_range = {params.tfidfvectorizer__ngram_range},
              multinomialnb__alpha = {params.multinomialnb__alpha}
              """)

        # run the model
        fitted_pipe = model_ml(X_train, y_train,
                               params.tfidfvectorizer__ngram_range,
                               params.multinomialnb__alpha,
                               min_df,
                               max_df,
                               max_features)

        # apply to  the test data
        vectorize_text,y = vectorize(X_test, y_test, fitted_pipe.named_steps['tfidfvectorizer'])
        print(f"vectorize_text_test shape : {vectorize_text.shape}")

        # get prediction on y_test
        y_test_predict=(fitted_pipe.predict(vectorize_text))

        print(f"for sample number ={sample_nb}")
        # look at model performance
        print(f"result of model run with the gridsearch best parameters: {fitted_pipe.score(X_test, y_test)}")


    if action == "save_model":
    # Train and save the model
        print(f"""Running model with best parameters save it on the whole dataset (with parameters found from grid search the whole data):
              tfidfvectorizer__ngram_range = {params.tfidfvectorizer__ngram_range},
              multinomialnb__alpha = {params.multinomialnb__alpha}
              """)

        # run the model on the whole dataset (X,y) and to store it
        trained_model = saving_model(X, y,
                                     params.tfidfvectorizer__ngram_range,
                                     params.multinomialnb__alpha, min_df,
                                     max_df,
                                     max_features,
                                     params.TRAINED_MODEL_DESTINATION_FILE_NAME)
        print(f"Model saved in {params.TRAINED_MODEL_DESTINATION_FILE_NAME}")


    # Calculate the end time and time taken
    end = time.time()
    length = end - start
    # Show the results : this can be altered however you like
    print(f"It took: {round(length,0)} seconds")



# while we wait for trained models, models that randomly returns fake or real
def model_text_only():
    return random.randint(0, 1)
def model_title_only():
    return random.randint(0, 1)
def model_both():
    return random.randint(0, 1)
def model_vote():
    return random.randint(0, 1)
