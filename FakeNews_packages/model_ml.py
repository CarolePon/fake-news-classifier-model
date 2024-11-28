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
from data import get_data_text_title_df, get_data_text_df
import numpy as np


# Get the dataframe to run model with title and text
data_cleaned = get_data_text_df()




#fonction qui prend un échantillon de 1000 élements remove later
def sample(data_cleaned):
    data_cleaned_sample=data_cleaned.sample(1000,random_state=42)
    return data_cleaned_sample

#definition des X et y
def variable_X(data_cleaned_sample):
    X=data_cleaned_sample['text']
    return X
def variable_y(data_cleaned_sample):
    y=data_cleaned_sample['label']
    return(y)

def train_test(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, random_state = 42)
    return X_train, X_test, y_train, y_test

def vectorize(X,y):
    count_vectorizer = TfidfVectorizer()
    X_bow = count_vectorizer.fit_transform(X)
    X_bow.toarray()
    
    count_vectorizer.get_feature_names_out()
    vectorized_texts = pd.DataFrame(
        X_bow.toarray(),
        columns = count_vectorizer.get_feature_names_out(),
        index = X)

    return vectorized_texts, y



#recherche des meilleurs parametres
def hyperparams(X, y):

    # Pipe
    pipeline_naive_bayes = make_pipeline(
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
    return grid_search.best_estimator_, grid_search.best_score_ 



if __name__ == "__main__":
    sample_data_cleaned=sample(data_cleaned)
    #print(sample_data_cleaned)
    
    X=variable_X(sample_data_cleaned)
    #print(X)
    
    y=variable_y(sample_data_cleaned)
    #print(y)
    
    X_train, X_test, y_train, y_test=train_test(X,y)
    print(X_train)
    
    #vecteurs_text,y=vectorize(X,y)
    #print(vecteurs_text,y)

    
    #hyperparams

    #X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, random_state = 42)
    #cross_validate(MultinomialNB(), X_train, y_train, cv=5)["test_score"].mean()

    