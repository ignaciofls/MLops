# MLops example

Note: This repo is a fork from https://github.com/danielsc/azureml-workshop-2019/, all the credit to Daniel.

This repo provides a curated example of a MLops cycle where we train a model while showing main capabilities of AzureML and automate tedious tasks like model tracking, metric logging, explainabiliy, unit test, etc

Main addons on top of danielsc's repo:
- Fix of Azure Devops build pipeline to use AzureML pipeline (and avoid calling the python script straightaway)
- Addon of metric logging, to make them available in the model registry
- Addon of interpretability section, viewable in the AzureML unit
![](2-mlops/media/UIExpl.png)
- Azure Function to trigger a build devops pipeline whenever our dataset changes (see blobtrigger code)
- Placeholder for Unit testing steps in AzureML pipeline
- Linkage with Azure Monitoring

E2E this is the MLops architecture we implement:
![](2-mlops/media/MLops.png)