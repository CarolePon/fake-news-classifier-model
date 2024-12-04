import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def data_sourcing():
    # Load all datasets
    url_both = '/home/baptiste/code/fake_news_classifier_model/FNCM/raw_data/FN_preproc_both.csv'
    url_text = '/home/baptiste/code/fake_news_classifier_model/FNCM/raw_data/FN_preproc_text.csv'
    url_title = '/home/baptiste/code/fake_news_classifier_model/FNCM/raw_data/FN_preproc_title.csv'

    df_both_only = pd.read_csv(url_both).drop(columns=['label', 'Unnamed: 0'])
    df_text_only = pd.read_csv(url_text).drop(columns=['label', 'Unnamed: 0'])
    df_title_only = pd.read_csv(url_title).drop(columns=['label', 'Unnamed: 0'])

    # Find common indices across all datasets
    common_indices = (
        df_both_only.index.intersection(df_text_only.index)
        .intersection(df_title_only.index)
    )

    # Filter datasets to include only common indices
    df_both_com = df_both_only.loc[common_indices]
    df_text_com = df_text_only.loc[common_indices]
    df_title_com = df_title_only.loc[common_indices]

    # Identify rows with NaN in any dataset
    nan_indices = (
        df_both_com[df_both_com.isna().any(axis=1)].index |
        df_text_com[df_text_com.isna().any(axis=1)].index |
        df_title_com[df_title_com.isna().any(axis=1)].index
    )

    # Drop rows with NaN from all datasets
    df_both_cl = df_both_com.drop(index=nan_indices)
    df_text_cl = df_text_com.drop(index=nan_indices)
    df_title_cl = df_title_com.drop(index=nan_indices)

    return df_both_cl, df_text_cl, df_title_cl

def data_sourcing_text():
    _, df_text_cl, _ = data_sourcing()
    return df_text_cl

def data_sourcing_title():
    _, _, df_title_cl = data_sourcing()
    return df_title_cl

def data_sourcing_both():
    df_both_cl, _, _ = data_sourcing()
    return df_both_cl

def label_sourcing():
    # Clean the label dataset based on aligned indices
    _, df_text_cl, _ = data_sourcing()
    url_title = '/home/baptiste/code/fake_news_classifier_model/FNCM/raw_data/FN_preproc_title.csv'
    df_label_only = pd.read_csv(url_title).drop(columns=['title', 'Unnamed: 0'])

    # Align indices with cleaned datasets
    common_indices = df_text_cl.index.intersection(df_label_only.index)
    df_label_cl = df_label_only.loc[common_indices]

    return df_label_cl

def df_preds_and_label():
    df_text_cl = data_sourcing_text()
    df_title_cl = data_sourcing_title()
    df_both_cl = data_sourcing_both()

    # Chemin vers le modele text, charger le modèle plus predictions
    model_path_text = "/home/baptiste/code/fake_news_classifier_model/FNCM/raw_data/Trained_ML_model_FN_preproc_text.pkl"

    with open(model_path_text, 'rb') as file:
        model_text = pickle.load(file)

    df_pred_text = df_text_cl.apply(model_text.predict)

    # Chemin vers le modele titre, charger le modèle plus predictions
    model_path_title = "/home/baptiste/code/fake_news_classifier_model/FNCM/raw_data/Trained_ML_model_FN_preproc_title.pkl"

    with open(model_path_title, 'rb') as file:
        model_title = pickle.load(file)

    df_pred_title = df_title_cl.apply(model_title.predict)

    # Chemin vers le modele both, charger le modèle plus predictions
    model_path_both = "/home/baptiste/code/fake_news_classifier_model/FNCM/raw_data/Trained_ML_model_FN_preproc_both.pkl"

    with open(model_path_both, 'rb') as file:
        model_both = pickle.load(file)

    df_pred_both = df_title_cl.apply(model_both.predict)

    df_label_cl = label_sourcing()

    df_preds_labels = pd.concat([df_pred_text, df_pred_title, df_pred_both, df_label_cl], axis=1)
    return df_preds_labels

def train_meta_model(df_preds_labels, test_size=0.3, random_state=42):
    """
    Train a meta-model using predictions from base models and true labels.

    Parameters:
    - df_preds_labels: DataFrame containing predictions from base models and true labels.
    - test_size: Proportion of the dataset to include in the test split.
    - random_state: Seed for reproducibility.

    Returns:
    - meta_model: Trained meta-model.
    - X_test, y_test: Test set features and labels for further evaluation.
    - predictions: Predictions made by the meta-model on the test set.
    """
    # Separate features (base model predictions) and labels
    X = df_preds_labels.iloc[:, :-1]  # All columns except the last one
    y = df_preds_labels.iloc[:, -1]   # Last column is the label

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Train a meta-model (Random Forest in this case)
    meta_model = RandomForestClassifier(random_state=random_state)
    meta_model.fit(X_train, y_train)

    # Predict on the test set
    predictions = meta_model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    print("Meta-Model Accuracy:", accuracy)
    print("\nClassification Report:\n", classification_report(y_test, predictions))

    return meta_model


def export_pickle():
    voting_model_path = '/home/baptiste/code/fake_news_classifier_model/FNCM/raw_data/voting_ml_model.pkl'

    with open(voting_model_path, 'wb') as file:
        pickle.dump(train_meta_model(df_preds_and_label()), file)

    print(f'Model saved as {voting_model_path}')

if __name__ == "__main__":
    export_pickle()
