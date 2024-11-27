import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import set_config; set_config("diagram")
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
import numpy as np


#lien vers les raw_data
file_path = "/home/ludovic/code/CarolePon/fake-news-classifier-model/raw_data/WELFake_Dataset.csv"
data = pd.read_csv(file_path, index_col=0)

#fonction qui élimine les NAN
data_cleaned=data.dropna()

#fonction qui prend un échantillon de 1000 élements
data_cleaned_sample=data_cleaned.sample(1000,random_state=42)
data_cleaned_sample

#Cross-val
y = data_cleaned_sample.iloc[:,2]
X = data_cleaned_sample["text"]

count_vectorizer = TfidfVectorizer()
X_bow = count_vectorizer.fit_transform(data_cleaned_sample['text'])
X_bow.toarray()

count_vectorizer.get_feature_names_out()
vectorized_texts = pd.DataFrame(
    X_bow.toarray(),
    columns = count_vectorizer.get_feature_names_out(),
    index = data_cleaned_sample['text'])

vectorized_texts.head(1)

y = data_cleaned_sample.iloc[:,2]
X =vectorized_texts

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, random_state = 42)
cross_validate(MultinomialNB(), X_train, y_train, cv=5)["test_score"].mean()


#GridSearch

