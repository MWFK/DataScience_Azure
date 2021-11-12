#!/usr/bin/env python
# coding: utf-8

# Train Models

get_ipython().system('pip install --upgrade azureml-sdk azureml-widgets')

import azureml.core
from azureml.core import Workspace

# Load the workspace from the saved config file
ws = Workspace.from_config()
print('Ready to use Azure ML {} to work with {}'.format(azureml.core.VERSION, ws.name))


### Create a training script

import os, shutil

# Create a folder for the experiment files
training_folder = 'diabetes-training'
os.makedirs(training_folder, exist_ok=True)

# Copy the data file into the experiment folder
shutil.copy('data/diabetes.csv', os.path.join(training_folder, "diabetes.csv"))


### Now you're ready to create the training script and save it in the folder.

get_ipython().run_cell_magic('writefile', '$training_folder/diabetes_training.py', '# Import libraries\nfrom azureml.core import Run\nimport pandas as pd\nimport numpy as np\nimport joblib\nimport os\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import roc_auc_score\nfrom sklearn.metrics import roc_curve\n\n# Get the experiment run context\nrun = Run.get_context()\n\n# load the diabetes dataset\nprint("Loading Data...")\ndiabetes = pd.read_csv(\'diabetes.csv\')\n\n# Separate features and labels\nX, y = diabetes[[\'Pregnancies\',\'PlasmaGlucose\',\'DiastolicBloodPressure\',\'TricepsThickness\',\'SerumInsulin\',\'BMI\',\'DiabetesPedigree\',\'Age\']].values, diabetes[\'Diabetic\'].values\n\n# Split data into training set and test set\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)\n\n# Set regularization hyperparameter\nreg = 0.01\n\n# Train a logistic regression model\nprint(\'Training a logistic regression model with regularization rate of\', reg)\nrun.log(\'Regularization Rate\',  np.float(reg))\nmodel = LogisticRegression(C=1/reg, solver="liblinear").fit(X_train, y_train)\n\n# calculate accuracy\ny_hat = model.predict(X_test)\nacc = np.average(y_hat == y_test)\nprint(\'Accuracy:\', acc)\nrun.log(\'Accuracy\', np.float(acc))\n\n# calculate AUC\ny_scores = model.predict_proba(X_test)\nauc = roc_auc_score(y_test,y_scores[:,1])\nprint(\'AUC: \' + str(auc))\nrun.log(\'AUC\', np.float(auc))\n\n# Save the trained model in the outputs folder\nos.makedirs(\'outputs\', exist_ok=True)\njoblib.dump(value=model, filename=\'outputs/diabetes_model.pkl\')\n\nrun.complete()')


# Run the training script as an experiment

from azureml.core import Experiment, ScriptRunConfig, Environment
from azureml.core.conda_dependencies import CondaDependencies
from azureml.widgets import RunDetails

# Create a Python environment for the experiment
sklearn_env = Environment("sklearn-env")

# Ensure the required packages are installed (we need scikit-learn and Azure ML defaults)
sklearn_env = Environment.from_conda_specification(name = 'sklearn-env', file_path = './environment.yml')
# packages = CondaDependencies.create(pip_packages=['scikit-learn','azureml-defaults'])
# sklearn_env.python.conda_dependencies = packages

# Create a script config
script_config = ScriptRunConfig(source_directory=training_folder,
                                script='diabetes_training.py',
                                environment=sklearn_env) 

# submit the experiment run
experiment_name = 'mslearn-train-diabetes'
experiment = Experiment(workspace=ws, name=experiment_name)
run = experiment.submit(config=script_config)

### Show the running experiment run in the notebook widget
RunDetails(run).show()

# Block until the experiment run has completed
run.wait_for_completion()

### Get logged metrics and files
metrics = run.get_metrics()
for key in metrics.keys():
        print(key, metrics.get(key))
print('\n')
for file in run.get_file_names():
    print(file)

### Register the trained model
from azureml.core import Model

# Register the model
run.register_model(model_path='outputs/diabetes_model.pkl', model_name='diabetes_model',
                   tags={'Training context':'Script'},
                   properties={'AUC': run.get_metrics()['AUC'], 'Accuracy': run.get_metrics()['Accuracy']})

# List registered models
for model in Model.list(ws):
    print(model.name, 'version:', model.version)
    for tag_name in model.tags:
        tag = model.tags[tag_name]
        print ('\t',tag_name, ':', tag)
    for prop_name in model.properties:
        prop = model.properties[prop_name]
        print ('\t',prop_name, ':', prop)
    print('\n')


### Create a parameterized training script
import os, shutil

# Create a folder for the experiment files
training_folder = 'diabetes-training-params'
os.makedirs(training_folder, exist_ok=True)

# Copy the data file into the experiment folder
shutil.copy('data/diabetes.csv', os.path.join(training_folder, "diabetes.csv"))


# Now let's create a script with an argument for the regularization rate hyperparameter. The argument is read using a Python **argparse.ArgumentParser** object.

get_ipython().run_cell_magic('writefile', '$training_folder/diabetes_training.py', '# Import libraries\nfrom azureml.core import Run\nimport pandas as pd\nimport numpy as np\nimport joblib\nimport os\nimport argparse\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import roc_auc_score\nfrom sklearn.metrics import roc_curve\n\n# Get the experiment run context\nrun = Run.get_context()\n\n# Set regularization hyperparameter\nparser = argparse.ArgumentParser()\nparser.add_argument(\'--reg_rate\', type=float, dest=\'reg\', default=0.01)\nargs = parser.parse_args()\nreg = args.reg\n\n# load the diabetes dataset\nprint("Loading Data...")\n# load the diabetes dataset\ndiabetes = pd.read_csv(\'diabetes.csv\')\n\n# Separate features and labels\nX, y = diabetes[[\'Pregnancies\',\'PlasmaGlucose\',\'DiastolicBloodPressure\',\'TricepsThickness\',\'SerumInsulin\',\'BMI\',\'DiabetesPedigree\',\'Age\']].values, diabetes[\'Diabetic\'].values\n\n# Split data into training set and test set\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)\n\n# Train a logistic regression model\nprint(\'Training a logistic regression model with regularization rate of\', reg)\nrun.log(\'Regularization Rate\',  np.float(reg))\nmodel = LogisticRegression(C=1/reg, solver="liblinear").fit(X_train, y_train)\n\n# calculate accuracy\ny_hat = model.predict(X_test)\nacc = np.average(y_hat == y_test)\nprint(\'Accuracy:\', acc)\nrun.log(\'Accuracy\', np.float(acc))\n\n# calculate AUC\ny_scores = model.predict_proba(X_test)\nauc = roc_auc_score(y_test,y_scores[:,1])\nprint(\'AUC: \' + str(auc))\nrun.log(\'AUC\', np.float(auc))\n\nos.makedirs(\'outputs\', exist_ok=True)\njoblib.dump(value=model, filename=\'outputs/diabetes_model.pkl\')\n\nrun.complete()')


# Run the script with arguments

# Create a script config
script_config = ScriptRunConfig(source_directory=training_folder,
                                script='diabetes_training.py',
                                arguments = ['--reg_rate', 0.1],
                                environment=sklearn_env) 

# submit the experiment
experiment_name = 'mslearn-train-diabetes'
experiment = Experiment(workspace=ws, name=experiment_name)
run = experiment.submit(config=script_config)
RunDetails(run).show()
run.wait_for_completion()

# Once again, we can get the metrics and outputs from the completed run.

# Get logged metrics
metrics = run.get_metrics()
for key in metrics.keys():
        print(key, metrics.get(key))
print('\n')
for file in run.get_file_names():
    print(file)


# ## Register a new version of the model

from azureml.core import Model

# Register the model
run.register_model(model_path='outputs/diabetes_model.pkl', model_name='diabetes_model',
                   tags={'Training context':'Parameterized script'},
                   properties={'AUC': run.get_metrics()['AUC'], 'Accuracy': run.get_metrics()['Accuracy']})

# List registered models
for model in Model.list(ws):
    print(model.name, 'version:', model.version)
    for tag_name in model.tags:
        tag = model.tags[tag_name]
        print ('\t',tag_name, ':', tag)
    for prop_name in model.properties:
        prop = model.properties[prop_name]
        print ('\t',prop_name, ':', prop)
    print('\n')
