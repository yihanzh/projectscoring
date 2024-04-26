from flask import Flask, request, redirect, url_for, flash, jsonify
import numpy as np
import pickle as p
import pandas as pd
import json
from api.model import Model

app = Flask(__name__)
model = Model()


@app.route('/api/ping/', methods=['GET'])
def ping():
    return jsonify('pong')


@app.route('/api/index/', methods=['GET'])
def index():
    index = list(model.df.index)
    # Returning the processed data
    return jsonify({
        'status': 'ok',
        'data': index
     })


@app.route('/api/features/', methods=['GET'])
def features():
    features = list(model.df.columns)
    # Returning the processed data
    return jsonify({
        'status': 'ok',
        'data': features
     })


@app.route('/api/data/', methods=['GET'])
def data():
    #  Parsing the http request to get arguments
    index = int(request.args.get('index'))
    data = model.get_data(index)
    print(data)
    # Returning the processed data
    return jsonify({
        'status': 'ok',
        'index': index,
        'data': data
     })


@app.route('/api/predict/', methods=['GET'])
def predict():
    #  Parsing the http request to get arguments
    index = int(request.args.get('index'))
    prediction = model.predict(index)
    print('prediction: ' + str(prediction))
    # Returning the processed data
    return jsonify({
        'status': 'ok',
        'index': index,
        'prediction': prediction
     })


@app.route('/api/predict_proba/', methods=['GET'])
def predict_proba():
    #  Parsing the http request to get arguments
    index = int(request.args.get('index'))
    probability = model.predict_proba(index)
    print('probability: ' + str(probability))
    # Returning the processed data
    return jsonify({
        'status': 'ok',
        'index': index,
        'probability': probability
    })


@app.route('/api/shap_global/', methods=['GET'])
def shap_global():
    #  Parsing the http request to get arguments
    number_features = int(request.args.get('number_features'))
    shap_image = model.shap_chart_global(number_features)
    # Returning the processed data
    return shap_image


@app.route('/api/shap_local/', methods=['GET'])
def shap_local():
    #  Parsing the http request to get arguments
    index = int(request.args.get('index'))
    number_features = int(request.args.get('number_features'))
    shap_image = model.shap_chart_individual(index, number_features)
    # Returning the processed data
    return shap_image


@app.route('/api/distribution_feature/', methods=['GET'])
def distribution_feature():
    #  Parsing the http request to get arguments
    feature_name = request.args.get('feature_name')
    image = model.distribution_feature(feature_name)
    # Returning the processed data
    return image


@app.route('/api/bivariate_plot/', methods=['GET'])
def bivariate_plot():
    #  Parsing the http request to get arguments
    feature_name_x = request.args.get('feature_name_x')
    feature_name_y = request.args.get('feature_name_y')
    image = model.bivariate_plot(feature_name_x, feature_name_y)
    # Returning the processed data
    return image


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')


