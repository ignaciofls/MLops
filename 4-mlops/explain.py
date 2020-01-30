from azureml.contrib.interpret.explanation.explanation_client import ExplanationClient
from azureml.core.run import Run
from interpret.ext.blackbox import TabularExplainer
from azureml.contrib.interpret.visualize import ExplanationDashboard

run = Run.get_context()
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
client.upload_model_explanation(global_explanation, comment='global explanation: all features')
# or you can only upload the explanation object with the top k feature info
#client.upload_model_explanation(global_explanation, top_k=2, comment='global explanation: Only top 2 features')


# Note: PFIExplainer does not support local explanations
# You can pass a specific data point or a group of data points to the explain_local function

# E.g., Explain the first data point in the test set
instance_num = 1
local_explanation = explainer.explain_local(x_test[:instance_num])

# Get the prediction for the first member of the test set and explain why model made that prediction
prediction_value = clf.predict(x_test)[instance_num]

sorted_local_importance_values = local_explanation.get_ranked_local_values()[prediction_value]
sorted_local_importance_names = local_explanation.get_ranked_local_names()[prediction_value]

print('local importance values: {}'.format(sorted_local_importance_values))
print('local importance names: {}'.format(sorted_local_importance_names))

ExplanationDashboard(global_explanation, model, x_test)