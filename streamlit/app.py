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
user_title_input = st.text_input("Titre :", "")
user_text_input = st.text_input("Article :", "")

if st.button("Verification de la véracité de l'article"):
    # Check user inputs
    if user_title_input.strip() or user_text_input.strip():
        # Prepare payload
        payload = {
            "title_input_string": user_title_input if user_title_input.strip() else None,
            "text_input_string": user_text_input if user_text_input.strip() else None,
        }

        # Display which models will be used
        if payload["title_input_string"] and payload["text_input_string"]:
            st.info("Les modèles utilisant le titre, le texte, et les deux combinés seront utilisés.")
        elif payload["title_input_string"]:
            st.info("Seul le modèle basé sur le titre sera utilisé.")
        elif payload["text_input_string"]:
            st.info("Seul le modèle basé sur le texte sera utilisé.")

        # Send the request to the API
        api_url = "https://example.com/api"  # Replace with the real API URL
        try:
            response = requests.post(api_url, json=payload)
            if response.status_code == 200:
                api_response = response.json()
                st.success(f"Réponse de l'API : {api_response.get('result', 'Aucun résultat retourné')}")
            else:
                st.error(f"Erreur de l'API : {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Une erreur s'est produite : {e}")
    else:
        st.warning("Veuillez entrer une chaîne de caractères avant de soumettre.")
