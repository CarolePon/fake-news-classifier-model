# imports:
from polyglot.detect import Detector




# Variables: booléens ou limites numéraires e.g. nb de mots/charactères






# fonction de preprocessing pour détecter les langues
def lang_detector(texte):
    try:
        detector = Detector(texte)
        return detector.language.name
    except Exception as e:
        return 'unknown / too short'


#création de colonnes pr la langue du texte et du titre de l'article
def df_full_eng(dataframe):

    # applique fct de détection pour les titres/et textes de chacund de nos articles
    dataframe['title_language'] = dataframe['title'].apply(lang_detector)
    dataframe['text_language'] = dataframe['text'].apply(lang_detector)

    # crée un df temporaire pr garder que les articles dont le texte est en anglais
    temp_df = dataframe[dataframe['text_language'] == 'English']

    # # crée le df de retour pr garder que les articles dont le texte et le titre sont en anglais
    new_df = temp_df[dataframe['title_language'] == 'English']

    # drop colonnes de langues de titre et de texte
    final_df = new_df.drop(columns=['title_language', 'text_language']

    return new_df
