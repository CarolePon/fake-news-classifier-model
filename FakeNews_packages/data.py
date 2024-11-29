import pandas as pd
import os
from google.cloud import storage



# function to get the data either from local file or from the GCP bucket
def data_source(SOURCE_DATA, BUCKET_NAME, SOURCE_BLOB_NAME,DESTINATION_FILE_NAME):

    # if SOURCE_DATA = "local", error as this function should not run
    if SOURCE_DATA == "local":
        print("Error, data is from local source")

    # if data from the GCP bucket
    if SOURCE_DATA == "gcs":

        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)

        # name of the file in the bucket = SOURCE_BLOB_NAME
        blob = bucket.blob(SOURCE_BLOB_NAME)

        # Get the path for the locally saved file in the VM (in folder raw_data)
        rootdir=(os.path.dirname(__file__))
        path_for_temp_csv = os.path.join(rootdir,DESTINATION_FILE_NAME)

        blob.download_to_filename(path_for_temp_csv)

        print(f"data_path = {path_for_temp_csv}")

        return path_for_temp_csv





def  get_data_text_title_df():
    #lien vers les raw_data
    file_path = "../raw_data/Fake_News_kaggle_english.csv"

    #get current working directory with os library
    rootdir=(os.path.dirname(__file__))
    #agreger les paths de l'endroit ou l'on se trouve avec le csv
    csv_path=(os.path.join(rootdir,file_path))

    #lire le csv depuis ce path relatif
    data = pd.read_csv(csv_path)

    print(f"✅ Data loaded into dataframe with title and text, shape {data.shape} and it has {data.columns} columns")

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


if __name__ == "__main__":
    SOURCE_DATA = "gcs"   # =  "gcs" or "local"
    # file path where the data is locally saved:
    LOCAL_FILE_PATH = "../raw_data/Fake_News_kaggle_english.csv"
    # bucket where the data is saved on gcs:
    BUCKET_NAME = "fnsm"
    #name of the file in the bucket = blob name
    SOURCE_BLOB_NAME = "Fake_News_kaggle_english.csv"
    # destination_file_name: The path and name where the file will be saved locally on the VM:
    DESTINATION_FILE_NAME = "../raw_data/Temp_raw_data_model.csv"


    print(data_source(SOURCE_DATA, BUCKET_NAME, SOURCE_BLOB_NAME,DESTINATION_FILE_NAME))
