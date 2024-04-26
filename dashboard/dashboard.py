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


def main():
    api_url = "http://localhost:5000/api/"

    # Logo "Prêt à dépenser"

    image = Image.open('dashboard/logo.png')
    st.sidebar.image(image, width=280)

    st.title('Tableau de bord - "Prêt à dépenser"')

    #################################################
    # LIST OF index_CURR

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

    index = get_index_list()

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

    features = get_features_list()

    ##################################################
    #  Selecting applicant ID
    select_index = st.sidebar.selectbox('Select index from list:', index, key=1)
    st.write('You selected: ', select_index)

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

    st.sidebar.dataframe(show_personal_data(select_index))

    ##################################################
    # Prediction
    ##################################################
    st.header('Prediction')

    probability = personal_probability(select_index)
    prediction = personal_prediction(select_index)
    # Display score (default probability)
    st.write('Default probability:', probability, '%')
    st.write('Prediction:', prediction)

    number_features = st.slider("Please select the number of features",
                           min_value=2,
                           max_value=30,
                           step=1)

    @st.cache_data
    def get_shap_image_global(number_features_):
        # URL of the probability API
        url = api_url + "shap_global/?number_features=" + str(number_features_)
        r = requests.get(url)
        return Image.open(BytesIO(base64.b64decode(r.content)))

    st.header('Importance globale des paramètres')
    st.image(get_shap_image_global(number_features))

    @st.cache_data
    def get_shap_image_individual(select_index_, number_features_):
        # URL of the probability API
        url = api_url + "shap_local/?index=" + str(select_index_) + "&" + "number_features=" + str(number_features_)
        r = requests.get(url)
        return Image.open(BytesIO(base64.b64decode(r.content)))

    st.header('Importance des paramètres pour un client')
    st.image(get_shap_image_individual(select_index, number_features))

    feature_x = st.selectbox("Which feature on x?", features)
    feature_y = st.selectbox("Which feature on y?", features)

    @st.cache_data
    def get_distribution_feature(feature_):
        # URL of the probability API
        url = api_url + "distribution_feature/?feature_name=" + str(feature_)
        r = requests.get(url)
        return Image.open(BytesIO(base64.b64decode(r.content)))

    @st.cache_data
    def get_bivariate_plot(feature_x_, feature_y_):
        # URL of the probability API
        url = api_url + "bivariate_plot/?feature_name_x=" + str(feature_x_) + "&" + "feature_name_y=" + str(feature_y_)
        r = requests.get(url)
        return Image.open(BytesIO(base64.b64decode(r.content)))

    st.image(get_distribution_feature(feature_x))
    st.image(get_distribution_feature(feature_y))
    st.image(get_bivariate_plot(feature_x, feature_y))

    # ##################################################
    # # FEATURES' IMPORTANCE
    # ##################################################
    # st.header('GLOBAL INTERPRETATION')
    #
    #
    # ##################################################
    # # PERSONAL DATA
    # ##################################################
    # st.header('PERSONAL DATA')
    #
    # #  Personal data (cached)
    # @st.cache
    # def get_personal_data(select_index):
    #     # URL of the scoring API (ex: index_CURR = 100005)
    #     PERSONAL_DATA_api_url = api_url + "personal_data/?index_CURR=" + str(select_index)
    #
    #     # save the response to API request
    #     response = requests.get(PERSONAL_DATA_api_url)
    #
    #     # convert from JSON format to Python dict
    #     content = json.loads(response.content.decode('utf-8'))
    #
    #     # convert data to pd.Series
    #     personal_data = pd.Series(content['data']).rename("index {}".format(select_index))
    #
    #     return personal_data
    #
    # # Aggregations of all applicants (train set, cached)
    # @st.cache
    # def get_aggregate():
    #     # URL of the aggregations API
    #     AGGREGATIONS_api_url = api_url + "aggregations"
    #
    #     # Requesting the API and save the response
    #     response = requests.get(AGGREGATIONS_api_url)
    #
    #     # convert from JSON format to Python dict
    #     content = json.loads(response.content.decode('utf-8'))
    #
    #     # convert data to pd.Series
    #     data_agg = pd.Series(content['data']["0"]).rename("Population (mean/mode)")
    #
    #     return data_agg
    #
    # if st.sidebar.checkbox('Show personal data'):
    #
    #     # Get personal data
    #     personal_data = get_personal_data(select_index)
    #
    #     if st.checkbox('Show population data'):
    #         #  Get aggregated data
    #         data_agg = get_aggregate()
    #         # Concatenation of the information to display
    #         df_display = pd.concat([personal_data, data_agg], axis=1)
    #
    #     else:
    #         #  Display only personal_data
    #         df_display = personal_data
    #
    #     st.dataframe(df_display)
    #

    #
    # # Get local interpretation of the score (surrogate model, cached)
    # @st.cache
    # def score_explanation(select_index):
    #     # URL of the scoring API
    #     SCORING_EXP_api_url = api_url + "local_interpretation?index_CURR=" + str(select_index)
    #
    #     # Requesting the API and save the response
    #     response = requests.get(SCORING_EXP_api_url)
    #
    #     # convert from JSON format to Python dict
    #     content = json.loads(response.content.decode('utf-8'))
    #
    #     # getting the values from the content
    #     prediction = content['prediction']
    #     bias = content['bias']
    #     contribs = pd.Series(content['contribs']).rename("Feature contributions")
    #
    #     return (prediction, bias, contribs)
    #
    # if st.sidebar.checkbox('Show default probability'):
    #     #  Get score
    #     score = personal_scoring(select_index)
    #     # Display score (default probability)
    #     st.write('Default probability:', score, '%')
    #
    #     if st.checkbox('Show explanations'):
    #         # Get prediction, bias and features contribs from surrogate model
    #         (_, bias, contribs) = score_explanation(select_index)
    #         # Display the bias of the surrogate model
    #         st.write("Population mean (bias):", bias * 100, "%")
    #         #  Remove the features with no contribution
    #         contribs = contribs[contribs != 0]
    #         #  Sorting by descending absolute values
    #         contribs = contribs.reindex(contribs.abs().sort_values(ascending=False).index)
    #
    #         st.dataframe(contribs)
    #
    # ##################################################
    # # FEATURES DESCRIPTIONS
    # ##################################################
    # st.header("FEATURES' DESCRIPTIONS")
    #
    # #  Get the list of features
    # @st.cache
    # def get_features_descriptions():
    #     # URL of the aggregations API
    #     FEAT_DESC_api_url = api_url + "features_desc"
    #
    #     # Requesting the API and save the response
    #     response = requests.get(FEAT_DESC_api_url)
    #
    #     # convert from JSON format to Python dict
    #     content = json.loads(response.content.decode('utf-8'))
    #
    #     # convert back to pd.Series
    #     features_desc = pd.Series(content['data']['Description']).rename("Description")
    #
    #     return features_desc
    #
    # features_desc = get_features_descriptions()
    #
    # if st.sidebar.checkbox('Show features descriptions'):
    #     # Display features' descriptions
    #     st.table(features_desc)
    #
    # ################################################


if __name__ == '__main__':
    main()