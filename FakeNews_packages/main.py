from preprocessing2 import preproc_training, preproc_predict
from data import get_data_text_title_df, get_data_text_df
# Get the dataframe to run model with text only
data_title_text = get_data_text_title_df()
# Get the dataframe to run model with title and text
data_text_only = get_data_text_df()
#if __name__ == "__main__":
    # print(data_title_text.head())
    # print(data_text_only.head())
