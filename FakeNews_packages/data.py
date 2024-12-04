import pandas as pd
import os
from google.cloud import storage
from FakeNews_packages import params
from raw_data import


# function to get the data either from local file or from the GCP bucket
def get_data(SOURCE_DATA, BUCKET_NAME, SOURCE_BLOB_NAME, DESTINATION_FILE_NAME, columns):

    # if SOURCE_DATA = "local", error as this function should not run
    if SOURCE_DATA == "local":
        return get_data_local(columns)


    # if data from the GCP bucket
    if SOURCE_DATA == "gcs":

        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)

        # name of the file in the bucket = SOURCE_BLOB_NAME
        blob = bucket.blob(SOURCE_BLOB_NAME)

        # Get the path for the locally saved file in the VM (in folder raw_data)
        rootdir = (os.path.dirname(__file__))
        path_for_temp_csv = os.path.join(rootdir, DESTINATION_FILE_NAME)

        blob.download_to_filename(path_for_temp_csv)

        print(f"data_path = {path_for_temp_csv}")

        data_from_gcs = pd.read_csv(path_for_temp_csv, index_col=0)

        data_selected_col = data_from_gcs[columns]

        data_selected_col = data_selected_col.dropna()

        return data_selected_col





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


def get_data_local(columns):
    #lien vers les raw_data

    # #get current working directory with os library
    # rootdir=(os.path.dirname(__file__))
    # #agreger les paths de l'endroit ou l'on se trouve avec le csv
    # csv_path=(os.path.join(rootdir,file_path))

    #lire le csv depuis ce path relatif

    data = pd.read_csv(params.LOCAL_FILE_PATH, index_col=0)


    data_selected_col = data[columns]

    data_selected_col = data_selected_col.dropna()


    print(f"✅ Data loaded into dataframe with selected column only, shape {data_selected_col.shape}")

    return data_selected_col




"""to delete, just for testing"""

if __name__ == "__main__":

    data = get_data(params.SOURCE_DATA, params.BUCKET_NAME, params.SOURCE_BLOB_NAME, params.DESTINATION_FILE_NAME, params.columns)
    print(f"number of nan : \n {data.isna().sum()}")
