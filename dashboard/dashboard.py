#  TO RUN : python -m streamlit run dashboard/dashboard.py
# Local URL: http://localhost:8501
# Network URL: http://192.168.0.50:8501
# Online URL : http://15.188.179.79

import streamlit as st
from PIL import Image
import requests
import json
import base64
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt

api_url = "http://localhost:5000/api/"


# Get list of index (cached)
@st.cache_data
def get_index_list():
    # URL of the index API
    index_api_url = api_url + "index/"
    # Requesting the API and saving the response
    response = requests.get(index_api_url)
    # Convert from JSON format to Python dict
    content = json.loads(response.content)
    # Getting the values of index from the content
    return content['data']


# Get list of features (cached)
@st.cache_data
def get_features_list():
    # URL of the index API
    features_api_url = api_url + "features/"
    # Requesting the API and saving the response
    response = requests.get(features_api_url)
    # Convert from JSON format to Python dict
    content = json.loads(response.content)
    # Getting the values of index from the content
    return content['data']


# Get personal data (cached)
@st.cache_data
def show_personal_data(select_index_):
    # URL of the data API
    data_api_url = api_url + "data/?index=" + str(select_index_)
    # Requesting the API and save the response
    response = requests.get(data_api_url)
    # convert from JSON format to Python dict
    content = json.loads(response.content.decode('utf-8'))
    # getting the values from the content
    j_data = json.loads(content['data'])
    return pd.DataFrame(j_data, index=[select_index_]).transpose()


# Get probability (cached)
@st.cache_data
def personal_probability(select_index_):
    # URL of the probability API
    probability_api_url = api_url + "predict_proba/?index=" + str(select_index_)
    # Requesting the API and save the response
    response = requests.get(probability_api_url)
    # convert from JSON format to Python dict
    content = json.loads(response.content.decode('utf-8'))
    # getting the values from the content
    return content['probability']


# Get prediction (cached)
@st.cache_data
def personal_prediction(select_index_):
    # URL of the probability API
    prediction_api_url = api_url + "predict/?index=" + str(select_index_)
    # Requesting the API and save the response
    response = requests.get(prediction_api_url)
    # convert from JSON format to Python dict
    content = json.loads(response.content.decode('utf-8'))
    # getting the values from the content
    return content['prediction']


@st.cache_data
def get_shap_image_global(number_features_):
    # URL of the probability API
    url = api_url + "shap_global/?number_features=" + str(number_features_)
    r = requests.get(url)
    return Image.open(BytesIO(base64.b64decode(r.content)))


@st.cache_data
def get_shap_image_individual(select_index_, number_features_):
    # URL of the probability API
    url = api_url + "shap_local/?index=" + str(select_index_) + "&" + "number_features=" + str(number_features_)
    r = requests.get(url)
    return Image.open(BytesIO(base64.b64decode(r.content)))


@st.cache_data
def get_distribution_feature(select_index_, feature_):
    # URL of the probability API
    url = api_url + "distribution_feature/?index=" + str(select_index_) + "&" + "feature_name=" + str(feature_)
    r = requests.get(url)
    return Image.open(BytesIO(base64.b64decode(r.content)))


@st.cache_data
def get_bivariate_plot(feature_x_, feature_y_):
    # URL of the probability API
    url = api_url + "bivariate_plot/?feature_name_x=" + str(feature_x_) + "&" + "feature_name_y=" + str(feature_y_)
    r = requests.get(url)
    return Image.open(BytesIO(base64.b64decode(r.content)))


########################
# FONCTION PRINCIPALE #
########################
def main():
    # Logo "Prêt à dépenser"

    image = Image.open('dashboard/logo.png')
    st.sidebar.image(image, width=280)

    st.title('Tableau de bord - "Prêt à dépenser"')
    st.header('Prédiction')
    index = get_index_list()

    features = get_features_list()

    # Selecting applicant ID
    select_index = st.sidebar.selectbox('Sélectionner l\'index du client :', index, key=1)
    st.write('Client index: ', select_index)

    st.sidebar.dataframe(show_personal_data(select_index))

    ##################################################
    # Prediction
    ##################################################

    probability = personal_probability(select_index)
    prediction = personal_prediction(select_index)

    st.markdown("* Probabilité de rembousement du client sélectioné: **%0.2f %%**" % (probability))

    st.markdown("* La réponse suggérée pour la demande de prêt du client sélectioné est : ")
    # Si la prediction vaut 1, on affiche "crédit refusé" sur bandeau rouge,
    # si prediction vaut 0, on affiche "crédit accordé" sur bandeau vert
    if prediction == 1:
        st.error("Crédit refusé !")
    elif prediction == 0:
        st.success("Crédit accordé !")

    number_features = st.slider("Sélectionner le nombre de paramètres pour expliquer la prédiction",
                                min_value=2,
                                max_value=30,
                                step=1)

    st.header('Importance globale des paramètres')
    st.image(get_shap_image_global(number_features))

    st.header('Importance des paramètres pour un client')
    st.image(get_shap_image_individual(select_index, number_features))

    st.header('Comparer le client sélectionné avec les autres clients')
    feature_x = st.selectbox("Sélectionner un paramètre x:", features)
    feature_y = st.selectbox("Sélectionner un paramètre y:", features)

    st.image(get_distribution_feature(select_index, feature_x))
    st.image(get_distribution_feature(select_index, feature_y))
    st.image(get_bivariate_plot(feature_x, feature_y))


if __name__ == '__main__':
    main()