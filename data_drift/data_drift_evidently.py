import evidently
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics.base_metric import generate_column_metrics
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import *
import pandas as pd

# Data de "reference" utilisée pour l'entraînement du modèle
app_train_df = pd.read_csv('./data/application_train.csv', header=0)
# "current_data", données en production, correspondant à des nouveaux clients
app_test_df = pd.read_csv('./data/application_test.csv', header=0)

feats = [f for f in app_train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
app_train_df=app_train_df.filter(feats)
app_test_df=app_test_df.filter(feats)

# Column mapping
column_mapping = ColumnMapping()
column_mapping.numerical_features = app_train_df.select_dtypes(exclude=['object']).columns.tolist() #list of numerical features
column_mapping.categorical_features = app_train_df.select_dtypes(include=['object']).columns.tolist() # list of categ features

# Création du rapport de dataDrift
report = Report(metrics=[DataDriftPreset(), ]) # choix du rapport "DataDriftPreset"
report.run(column_mapping = column_mapping, reference_data=app_train_df, current_data=app_test_df) # Création du rapport
report.save_html("data_drift_report.html")