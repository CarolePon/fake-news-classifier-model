import pandas as pd
import os


def  get_data_text_title_df():
    #lien vers les raw_data
    file_path = "../raw_data/Fake_News_kaggle_english.csv"

    #get current working directory with os library
    rootdir=(os.path.dirname(__file__))
    #agreger les paths de l'endroit ou l'on se trouve avec le csv
    csv_path=(os.path.join(rootdir,file_path))

    #lire le csv depuis ce path relatif
    data = pd.read_csv(csv_path)

    print(f"✅ Data loaded into dataframe with title and text, shape {data.shape}")

    return data


def  get_data_text_df():
    #lien vers les raw_data
    file_path = "../raw_data/Fake_News_kaggle_english.csv"

    #get current working directory with os library
    rootdir=(os.path.dirname(__file__))
    #agreger les paths de l'endroit ou l'on se trouve avec le csv
    csv_path=(os.path.join(rootdir,file_path))

    #lire le csv depuis ce path relatif
    data = pd.read_csv(csv_path)

    data_text_only = data[['text', 'label']]

    print(f"✅ Data loaded into dataframe with text only, shape {data_text_only.shape}")

    return data_text_only
