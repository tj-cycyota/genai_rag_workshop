# Databricks notebook source
# MAGIC %md 
# MAGIC # Scaling up hyperparameter tuning with Optuna and MLflow
# MAGIC
# MAGIC [Optuna](https://github.com/optuna/optuna) is a Python library for hyperparameter tuning. It is lightweight, and integrates state-of-the-art hyperparameter optimization algorithms.
# MAGIC
# MAGIC This notebook is adapted from the [Github README of Optuna](https://github.com/optuna/optuna), with additional tips on how to scale up hyperparameter tuning for a single-machine Python ML algorithm and track the results using MLflow.
# MAGIC - In part 1, you create a single-machine Optuna workflow.
# MAGIC - In part 2, you learn to use `joblib` to parallelize the workflow calculations across the Spark cluster, and use the `MLflowCallback` provided by Optuna's integration with MLflow to track the hyperparameter and metrics.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install required packages

# COMMAND ----------

# MAGIC %pip install optuna
# MAGIC %pip install optuna-integration

# COMMAND ----------

# MAGIC %md ## Import required packages and load dataset

# COMMAND ----------

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import joblib
import optuna

# COMMAND ----------

# MAGIC %md ## Part 1. Getting started with Optuna in a single machine
# MAGIC
# MAGIC Here are the steps in a Optuna workflow:  
# MAGIC 1. Define an *objective function* to optimize. Within the *objective function*, define the hyperparameter search space.
# MAGIC 3. Create a `Study` via the `optuna.create_study()` function.
# MAGIC 4. Run the tuning algorithm by calling the [optimize](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize) function of the `Study` object.
# MAGIC
# MAGIC For more information, see the [Optuna documentation](https://optuna.readthedocs.io/en/stable/tutorial/index.html).

# COMMAND ----------

# MAGIC %md ### Define a function and search space to optimize
# MAGIC The search space here is defined by calling functions such as `suggest_categorical`, `suggest_float`, `suggest_int` for the `Trial` object that is passed to the objective function.
# MAGIC
# MAGIC A special feature of Optuna is that you can define the search space dynamically. In this example, you can define different hyperparameter spaces depending on the classifier, i.e., `svc_c` for `SVC` and `rf_max_depth` for `RandomForestClassifier`.
# MAGIC
# MAGIC Refer to the documentation of [optuna.trial.Trial](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html) for a full list of functions supported to define a hyperparameter search space.

# COMMAND ----------

def objective(trial):
    iris = load_iris()
    x, y = iris.data, iris.target

    classifier_name = trial.suggest_categorical("classifier", ["SVC", "RandomForest"])

    if classifier_name == "SVC":
        svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
        classifier_obj = SVC(C=svc_c, gamma="auto")
    else:
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
        classifier_obj = RandomForestClassifier(
            max_depth=rf_max_depth, n_estimators=10
        )

    score = cross_val_score(classifier_obj, x, y, n_jobs=-1, cv=3)
    accuracy = score.mean()

    return accuracy

# COMMAND ----------

# MAGIC %md ### Create an Optuna Study and run it
# MAGIC Since the `objective` function returns the `accuracy`, set the optimization direction as `maximize`, otherwise Optuna minimizes the objective function by default.

# COMMAND ----------

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

# COMMAND ----------

trial = study.best_trial

print(f"Best trial accuracy: {trial.value}")
print("Best trial params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# COMMAND ----------

# MAGIC %md ## Part 2. Distributed tuning using Joblib and MLflow
# MAGIC Optuna has a wide set of integrations, which makes it easy to distribute hyperparameter tuning to multiple machines, and tracking the hyperparameters and metrics.
# MAGIC
# MAGIC - Distributed tuning via Joblib: You can distribute Optuna trials to multiple machines in a Databricks cluster by selecting the Spark backend of Joblib.
# MAGIC - MLflow integration: the `MLflowCallback` helps automatically logging the hyperparameters and metrics.

# COMMAND ----------

from joblibspark import register_spark

register_spark() # register Spark backend for Joblib

# COMMAND ----------

import mlflow
from optuna.integration.mlflow import MLflowCallback

experiment_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

mlflow_callback = MLflowCallback(
    tracking_uri="databricks",
    metric_name="accuracy",
    create_experiment=False,
    mlflow_kwargs={
        "experiment_id": experiment_id
    }
)

# COMMAND ----------

# MAGIC %md
# MAGIC Run an Optuna hyperparameter optimization study with Joblib parallelization and MLflow logging.
# MAGIC - Wrap the `optimize` function with Joblib (Spark backend)
# MAGIC - Pass the `MLflowCallback` object to the `optimize` function.

# COMMAND ----------

study2 = optuna.create_study(direction="maximize")
with joblib.parallel_backend("spark", n_jobs=-1):
    study2.optimize(objective, n_trials=100, callbacks=[mlflow_callback])

# COMMAND ----------

trial = study2.best_trial

print(f"Best trial accuracy: {trial.value}")
print("Best trial params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# COMMAND ----------

# MAGIC %md To view the MLflow experiment associated with the notebook, click the **Experiment** icon in the notebook context bar on the upper right.  There, you can view all runs.
