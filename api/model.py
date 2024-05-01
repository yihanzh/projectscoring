import pandas as pd
import numpy as np
import pickle as p
import matplotlib.pyplot as plt
import shap
import io
import base64
import json
import seaborn as sns
from matplotlib.lines import Line2D


class Model:
    def __init__(self):
        self.df = self.load_data()
        self.model = self.load_model()
        self.features = self.df.columns
        clf = self.model["model"]
        scaler = self.model["scaler"]
        scaled_test_x = scaler.transform(self.df)
        explainer = shap.Explainer(clf, scaled_test_x, feature_names=self.features)
        self.shap_values = explainer(scaled_test_x)
        self.predictions= [self.predict(i) for i in range(len(self.df))]

    @staticmethod
    def load_data():
        df = pd.read_csv('./data/test_x_1000.csv', header=0)
        df.replace({'FALSE': 0, 'TRUE': 1}, inplace=True)
        return df

    @staticmethod
    def load_model():
        model_file = './models/model.pkl'
        model = p.load(open(model_file, 'rb'))
        return model

    def get_data(self, index):
        data = self.df.iloc[index, :].to_json()
        return data

    def predict(self, index):
        data = [self.df.iloc[index, :].values.tolist()]
        prediction = self.model.predict(data)[0]
        return prediction

    def predict_proba(self, index):
        data = [self.df.iloc[index, :].values.tolist()]
        prediction = self.model.predict_proba(data)[0][0] * 100
        return prediction

    def shap_chart_individual(self, index, number_feature):
        values = self.shap_values[index].values[:, 0]
        fig, ax = plt.subplots()
        plt.title("Importance des paramètres pour un client")
        shap.bar_plot(values, feature_names=self.features, max_display=number_feature, show=False)
        #shap.plots.waterfall(values, max_display=number_feature)
        buffer = io.BytesIO()
        # fig.savefig('shap.png')
        fig.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        image = base64.b64encode(buffer.getvalue())
        return image

    def shap_chart_global(self, number_feature):
        features = self.df.columns
        clf = self.model["model"]
        scaler = self.model["scaler"]
        scaled_test_x = scaler.transform(self.df)
        fig, ax = plt.subplots()
        plt.title("Importance globale des paramètres")
        explainer = shap.Explainer(clf, scaled_test_x, feature_names=features)
        shap_values = explainer(scaled_test_x)
        values = shap_values.values[:, :, 0]
        #print(values)
        shap.summary_plot(values, feature_names=features, max_display=number_feature, show=False)
        # shap.plots.waterfall(values, max_display=number_feature)
        buffer = io.BytesIO()
        # fig.savefig('shap.png')
        fig.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        image = base64.b64encode(buffer.getvalue())
        return image

    # def distribution_feature(self, feature_name):
    #     fig, ax = plt.subplots()
    #     plt.title("Distribution du paramètre " + feature_name)
    #     plt.hist(self.df[feature_name], color='lightgreen', ec='black', bins='auto')
    #     # shap.bar_plot(values, feature_names=self.features, max_display=number_feature, show=False)
    #     buffer = io.BytesIO()
    #     # fig.savefig('shap.png')
    #     fig.savefig(buffer, format='png', bbox_inches='tight')
    #     buffer.seek(0)
    #     image = base64.b64encode(buffer.getvalue())
    #     return image

    def distribution_feature(self, index, feature_name):
        selected_client = self.df.iloc[index, :]
        fig, ax = plt.subplots()
        mean = self.df[feature_name].mean()
        std = self.df[feature_name].std()
        sns.histplot(self.df, x=feature_name, color='grey')
        print(selected_client[feature_name])
        print(mean)
        ax.axvline(selected_client[feature_name], color="blue", linestyle='--', linewidth=2,
                   label='Client n° %s : %f' %(index, selected_client[feature_name]))
        ax.axvline(mean, color="black", linestyle='--', linewidth=2, label='Moyenne des clients : %f' % mean)
        ax.set(title='Distribution du paramètre %s' % feature_name, ylabel='')
        # ax.set_xlim(mean - 5 * std, mean + 5 * std)
        plt.grid(axis='y')
        plt.legend()
        buffer = io.BytesIO()
        # fig.savefig('test.png')
        fig.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        image = base64.b64encode(buffer.getvalue())
        return image

    def bivariate_plot(self, index, feature_name_x, feature_name_y):
        selected_client = self.df.iloc[index, :]
        fig, ax = plt.subplots()
        plt.title("analyse bi-variée entre " + feature_name_x + " et " + feature_name_y)
        plt.xlabel(feature_name_x)
        plt.ylabel(feature_name_y)
        colors = ['red' if result == 1 else 'green' for result in self.predictions]
        plt.scatter(self.df[feature_name_x], self.df[feature_name_y], s=20, color=colors)
        ax.scatter(selected_client[feature_name_x], selected_client[feature_name_y], s=200, marker='X', color='red' if self.predictions[index] == 1 else 'green',
                   label='Client n° %s' % index)
        custom_legend = [Line2D([0], [0], marker='o', color='red', label='Crédit refusé', markersize=5),
                        Line2D([0], [0], marker='o', color='green', label='Crédit accordé', markersize=5),
                        Line2D([0], [0], marker='X', color='red' if self.predictions[index] == 1 else 'green',
                               label='Client n° %s : Crédit refusé' % index if self.predictions[index] == 1 else 'Client n° %s : Crédit accordé' % index,
                                markersize=10)]
        ax.legend(handles=custom_legend, loc='upper right')
        buffer = io.BytesIO()
        # fig.savefig('test.png')
        fig.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        image = base64.b64encode(buffer.getvalue())
        return image

# test
# model = Model()
# # image = model.shap_chart_individual(0, 10)
# # print(image)
# # model.shap_chart_global(10)
# # model.distribution_feature(0, 'TOTALAREA_MODE')
# model.bivariate_plot(0, 'TOTALAREA_MODE', 'CNT_CHILDREN')