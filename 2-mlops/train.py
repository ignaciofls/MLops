# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license.

from sklearn.externals import joblib
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn_pandas import DataFrameMapper
from sklearn.metrics import accuracy_score
import os
import pandas as pd

from azureml.core import Run, Dataset, Workspace

ws = Run.get_context().experiment.workspace
os.makedirs('./outputs', exist_ok=True)
#comment

attritionData = Dataset.get_by_name(ws,'IBM-Employee-Attrition2').to_pandas_dataframe()

# Dropping Employee count as all values are 1 and hence attrition is independent of this feature
attritionData = attritionData.drop(['EmployeeCount'], axis=1)
# Dropping Employee Number since it is merely an identifier
attritionData = attritionData.drop(['EmployeeNumber'], axis=1)
attritionData = attritionData.drop(['Over18'], axis=1)
# Since all values are 80
attritionData = attritionData.drop(['StandardHours'], axis=1)

attritionData["Attrition_numerical"] = attritionData["Attrition"]
target = attritionData["Attrition_numerical"]

attritionXData = attritionData.drop(['Attrition_numerical', 'Attrition'], axis=1)

# Creating dummy columns for each categorical feature
categorical = []
for col, value in attritionXData.iteritems():
    if value.dtype == 'object':
        categorical.append(col)

# Store the numerical columns in a list numerical
numerical = attritionXData.columns.difference(categorical)

numeric_transformations = [([f], Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])) for f in numerical]
    
categorical_transformations = [([f], OneHotEncoder(handle_unknown='ignore', sparse=False)) for f in categorical]

transformations = numeric_transformations + categorical_transformations

# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.
clf = Pipeline(steps=[('preprocessor', DataFrameMapper(transformations)),
                      ('classifier', LogisticRegression(solver='lbfgs'))])

# Split data into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(attritionXData, 
                                                    target, 
                                                    test_size = 0.35,
                                                    random_state=0,
                                                    stratify=target)

# write x_text out as a pickle file for later visualization
x_test_pkl = 'x_test.pkl'
with open(x_test_pkl, 'wb') as file:
    joblib.dump(value=x_test, filename=os.path.join('./outputs/', x_test_pkl))

# preprocess the data and fit the classification model
clf.fit(x_train, y_train)
model = clf.steps[-1][1]

y_pred = clf.predict(x_test)
accu = accuracy_score(y_test, y_pred)

model_file_name = 'log_reg.pkl'

# save model in the outputs folder so it automatically get uploaded
with open(model_file_name, 'wb') as file:
    joblib.dump(value=clf, filename=os.path.join('./outputs/',
                                                 model_file_name))

run = Run.get_context()
run.upload_file('x_test_ibm.pkl', os.path.join('./outputs/', x_test_pkl))
run.log("accuracy", accu)

# Register the model
run.upload_file('original_model.pkl', os.path.join('./outputs/', model_file_name))
original_model = run.register_model(model_name='Attrition_model', model_path='original_model.pkl')

#Interpret steps
from azureml.contrib.interpret.explanation.explanation_client import ExplanationClient
from azureml.core.run import Run
from interpret.ext.blackbox import TabularExplainer
from azureml.contrib.interpret.visualize import ExplanationDashboard

client = ExplanationClient.from_run(run)

# write code to get and split your data into train and test sets here
# write code to train your model here 

# explain predictions on your local machine
# "features" and "classes" fields are optional
# Using SHAP TabularExplainer
explainer = TabularExplainer(clf.steps[-1][1], 
                             initialization_examples=x_train, 
                             features=attritionXData.columns, 
                             classes=["Not leaving", "leaving"], 
                             transformations=transformations)

# explain overall model predictions (global explanation)
global_explanation = explainer.explain_global(x_test)

# Sorted SHAP values
print('ranked global importance values: {}'.format(global_explanation.get_ranked_global_values()))
# Corresponding feature names
print('ranked global importance names: {}'.format(global_explanation.get_ranked_global_names()))
# Feature ranks (based on original order of features)
print('global importance rank: {}'.format(global_explanation.global_importance_rank))

# uploading global model explanation data for storage or visualization in webUX
# the explanation can then be downloaded on any compute
# multiple explanations can be uploaded
client.upload_model_explanation(global_explanation, comment='global explanation: all features',model_id=original_model.id)
# or you can only upload the explanation object with the top k feature info
#client.upload_model_explanation(global_explanation, top_k=2, comment='global explanation: Only top 2 features')


# Note: PFIExplainer does not support local explanations
# You can pass a specific data point or a group of data points to the explain_local function

# E.g., Explain the first data point in the test set
#instance_num = 1
#local_explanation = explainer.explain_local(x_test[:instance_num])

# Get the prediction for the first member of the test set and explain why model made that prediction
#prediction_value = clf.predict(x_test)[instance_num]

#sorted_local_importance_values = local_explanation.get_ranked_local_values()[prediction_value]
#sorted_local_importance_names = local_explanation.get_ranked_local_names()[prediction_value]

#print('local importance values: {}'.format(sorted_local_importance_values))
#print('local importance names: {}'.format(sorted_local_importance_names))

#ExplanationDashboard(global_explanation, model, x_test)
