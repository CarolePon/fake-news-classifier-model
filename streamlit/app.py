import streamlit as st
import requests


page_bg_img = '''
<style>
[data-testid="stAppViewContainer"] {
background-image: url("https://plus.unsplash.com/premium_photo-1707774568376-b146c6bf79f0?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
background-size: cover;
}
</style>
'''


st.markdown(page_bg_img, unsafe_allow_html=True)

# Titre de l'application
st.title("Fake News Classifier")

# Interface utilisateur pour entrer une chaîne de caractères
user_input = st.text_input("Entrez un texte :", "")

# Bouton pour soumettre la chaîne de caractères
if st.button("Verification de la véracité de l'article"):
    if user_input.strip():  # Vérifier que l'entrée n'est pas vide
        # URL de l'API (remplacez cette URL par l'URL réelle de votre API)
        api_url = "https://example.com/api"
        
        # Préparer les données à envoyer
        payload = {"input_string": user_input}
        
        try:
            # Envoyer une requête POST à l'API
            response = requests.post(api_url, json=payload)
            
            # Vérifier si la requête a réussi
            if response.status_code == 200:
                # Extraire la réponse JSON de l'API
                api_response = response.json()
                st.success(f"Réponse de l'API : {api_response.get('result', 'Aucun résultat retourné')}")
            else:
                st.error(f"Erreur de l'API : {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Une erreur s'est produite : {e}")
    else:
        st.warning("Veuillez entrer une chaîne de caractères avant de soumettre.")
   
