# A Tabnet Pipeline for predicting customer ratings

![image](https://github.com/viraj-s15/zenml-rating-tabnet-pipeline/assets/79002760/eab936df-5b5b-45fd-bdc8-536f438621d5)

## Table of Contents

- [About](#about)
- [Getting Started](#getting_started)

## About <a name = "about"></a>

A tabnet pipeline consisiting of:
  - Data Ingestion 
  - Data Cleaning
  - Feature Engineering
  - Model Training
  - Evaluation

MlFlow is being used for tracking the model (training and evaluation)

## Getting Started <a name = "getting_started"></a>


**DEPLOYMENT TO BE ADDED**

What things you need to install the software and how to install them.

```bash
pip install -r requirements.txt
zenml integration install mlflow -y
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow --flavor=mlflow
zenml stack register mlflow_stack -a default -o default -d mlflow -e mlflow_tracker --set 
python run_pipelines.py
```

Once the pipeline has finished running, you can view the pipeline using
`zenml up`
