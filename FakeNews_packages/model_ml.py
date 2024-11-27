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
import numpy as np






#fonction qui prend un échantillon de 1000 élements remove later
def sample (data_cleaned):
    data_cleaned_sample=data_cleaned.sample(1000,random_state=42)
    return data_cleaned_sample

def vectorize(X,y):
    count_vectorizer = TfidfVectorizer()
    X_bow = count_vectorizer.fit_transform(X['text'])
    X_bow.toarray()
    
    count_vectorizer.get_feature_names_out()
    vectorized_texts = pd.DataFrame(
        X_bow.toarray(),
        columns = count_vectorizer.get_feature_names_out(),
        index = X['text'])

    return vectorized_texts, y




#recherche des meilleurs parametres
def hyperparams(X, y):

    # Pipe
    pipeline_naive_bayes = make_pipeline(
        TfidfVectorizer(),
        MultinomialNB()
        )
    
    #parameters
    parameters = {
    'tfidfvectorizer__ngram_range': ((1,1), (1,2),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5),(2,2)),
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
    return grid_search.best_estimator_, grid_search.best_score_ 


    #sample
    #vectorize

    #X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, random_state = 42)
    #cross_validate(MultinomialNB(), X_train, y_train, cv=5)["test_score"].mean()

    #hyperparams