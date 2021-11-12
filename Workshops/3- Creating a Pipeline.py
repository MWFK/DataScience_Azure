#!/usr/bin/env python
# coding: utf-8

# Create a Pipeline

get_ipython().system('pip install --upgrade azureml-sdk azureml-widgets')


# Connect to your workspace

import azureml.core
from azureml.core import Workspace

# Load the workspace from the saved config file
ws = Workspace.from_config()
print('Ready to use Azure ML {} to work with {}'.format(azureml.core.VERSION, ws.name))


# Prepare data

from azureml.core import Dataset

default_ds = ws.get_default_datastore()

if 'diabetes dataset' not in ws.datasets:
    default_ds.upload_files(files=['./data/diabetes.csv', './data/diabetes2.csv'], # Upload the diabetes csv files in /data
                        target_path='diabetes-data/', # Put it in a folder path in the datastore
                        overwrite=True, # Replace existing files of the same name
                        show_progress=True)

    #Create a tabular dataset from the path on the datastore (this may take a short while)
    tab_data_set = Dataset.Tabular.from_delimited_files(path=(default_ds, 'diabetes-data/*.csv'))

    # Register the tabular dataset
    try:
        tab_data_set = tab_data_set.register(workspace=ws, 
                                name='diabetes dataset',
                                description='diabetes data',
                                tags = {'format':'CSV'},
                                create_new_version=True)
        print('Dataset registered.')
    except Exception as ex:
        print(ex)
else:
    print('Dataset already registered.')


# Create scripts for pipeline steps

import os
# Create a folder for the pipeline step files
experiment_folder = 'diabetes_pipeline'
os.makedirs(experiment_folder, exist_ok=True)

print(experiment_folder)


# Now let's create the first script, which will read data from the diabetes dataset and apply some simple pre-processing to remove any rows with missing data and normalize the numeric features so they're on a similar scale.

get_ipython().run_cell_magic('writefile', '$experiment_folder/prep_diabetes.py', '# Import libraries\nimport os\nimport argparse\nimport pandas as pd\nfrom azureml.core import Run\nfrom sklearn.preprocessing import MinMaxScaler\n\n# Get parameters\nparser = argparse.ArgumentParser()\nparser.add_argument("--input-data", type=str, dest=\'raw_dataset_id\', help=\'raw dataset\')\nparser.add_argument(\'--prepped-data\', type=str, dest=\'prepped_data\', default=\'prepped_data\', help=\'Folder for results\')\nargs = parser.parse_args()\nsave_folder = args.prepped_data\n\n# Get the experiment run context\nrun = Run.get_context()\n\n# load the data (passed as an input dataset)\nprint("Loading Data...")\ndiabetes = run.input_datasets[\'raw_data\'].to_pandas_dataframe()\n\n# Log raw row count\nrow_count = (len(diabetes))\nrun.log(\'raw_rows\', row_count)\n\n# remove nulls\ndiabetes = diabetes.dropna()\n\n# Normalize the numeric columns\nscaler = MinMaxScaler()\nnum_cols = [\'Pregnancies\',\'PlasmaGlucose\',\'DiastolicBloodPressure\',\'TricepsThickness\',\'SerumInsulin\',\'BMI\',\'DiabetesPedigree\']\ndiabetes[num_cols] = scaler.fit_transform(diabetes[num_cols])\n\n# Log processed rows\nrow_count = (len(diabetes))\nrun.log(\'processed_rows\', row_count)\n\n# Save the prepped data\nprint("Saving Data...")\nos.makedirs(save_folder, exist_ok=True)\nsave_path = os.path.join(save_folder,\'data.csv\')\ndiabetes.to_csv(save_path, index=False, header=True)\n\n# End the run\nrun.complete()')


# Now you can create the script for the second step, which will train a model. The script includes a argument named **--training-folder**, which references the folder where the prepared data was saved by the previous step.

get_ipython().run_cell_magic('writefile', '$experiment_folder/train_diabetes.py', '# Import libraries\nfrom azureml.core import Run, Model\nimport argparse\nimport pandas as pd\nimport numpy as np\nimport joblib\nimport os\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.metrics import roc_auc_score\nfrom sklearn.metrics import roc_curve\nimport matplotlib.pyplot as plt\n\n# Get parameters\nparser = argparse.ArgumentParser()\nparser.add_argument("--training-folder", type=str, dest=\'training_folder\', help=\'training data folder\')\nargs = parser.parse_args()\ntraining_folder = args.training_folder\n\n# Get the experiment run context\nrun = Run.get_context()\n\n# load the prepared data file in the training folder\nprint("Loading Data...")\nfile_path = os.path.join(training_folder,\'data.csv\')\ndiabetes = pd.read_csv(file_path)\n\n# Separate features and labels\nX, y = diabetes[[\'Pregnancies\',\'PlasmaGlucose\',\'DiastolicBloodPressure\',\'TricepsThickness\',\'SerumInsulin\',\'BMI\',\'DiabetesPedigree\',\'Age\']].values, diabetes[\'Diabetic\'].values\n\n# Split data into training set and test set\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)\n\n# Train adecision tree model\nprint(\'Training a decision tree model...\')\nmodel = DecisionTreeClassifier().fit(X_train, y_train)\n\n# calculate accuracy\ny_hat = model.predict(X_test)\nacc = np.average(y_hat == y_test)\nprint(\'Accuracy:\', acc)\nrun.log(\'Accuracy\', np.float(acc))\n\n# calculate AUC\ny_scores = model.predict_proba(X_test)\nauc = roc_auc_score(y_test,y_scores[:,1])\nprint(\'AUC: \' + str(auc))\nrun.log(\'AUC\', np.float(auc))\n\n# plot ROC curve\nfpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])\nfig = plt.figure(figsize=(6, 4))\n# Plot the diagonal 50% line\nplt.plot([0, 1], [0, 1], \'k--\')\n# Plot the FPR and TPR achieved by our model\nplt.plot(fpr, tpr)\nplt.xlabel(\'False Positive Rate\')\nplt.ylabel(\'True Positive Rate\')\nplt.title(\'ROC Curve\')\nrun.log_image(name = "ROC", plot = fig)\nplt.show()\n\n# Save the trained model in the outputs folder\nprint("Saving model...")\nos.makedirs(\'outputs\', exist_ok=True)\nmodel_file = os.path.join(\'outputs\', \'diabetes_model.pkl\')\njoblib.dump(value=model, filename=model_file)\n\n# Register the model\nprint(\'Registering model...\')\nModel.register(workspace=run.experiment.workspace,\n               model_path = model_file,\n               model_name = \'diabetes_model\',\n               tags={\'Training context\':\'Pipeline\'},\n               properties={\'AUC\': np.float(auc), \'Accuracy\': np.float(acc)})\n\n\nrun.complete()')


# Prepare a compute environment for the pipeline steps

from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

cluster_name = "dp100-cluster"

try:
    # Check for existing compute target
    pipeline_cluster = ComputeTarget(workspace=ws, name=cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    # If it doesn't already exist, create it
    try:
        compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_DS11_V2', max_nodes=2)
        pipeline_cluster = ComputeTarget.create(ws, cluster_name, compute_config)
        pipeline_cluster.wait_for_completion(show_output=True)
    except Exception as ex:
        print(ex)
    


# The compute will require a Python environment with the necessary package dependencies installed, so you'll need to create a run configuration.

from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.runconfig import RunConfiguration

# Create a Python environment for the experiment
diabetes_env = Environment("diabetes-pipeline-env")
diabetes_env.python.user_managed_dependencies = False # Let Azure ML manage dependencies
diabetes_env.docker.enabled = True # Use a docker container

# Create a set of package dependencies
diabetes_packages = CondaDependencies.create(conda_packages=['scikit-learn','ipykernel','matplotlib','pandas','pip'],
                                             pip_packages=['azureml-defaults','azureml-dataprep[pandas]','pyarrow'])

# Add the dependencies to the environment
diabetes_env.python.conda_dependencies = diabetes_packages

# Register the environment 
diabetes_env.register(workspace=ws)
registered_env = Environment.get(ws, 'diabetes-pipeline-env')

# Create a new runconfig object for the pipeline
pipeline_run_config = RunConfiguration()

# Use the compute you created above. 
pipeline_run_config.target = pipeline_cluster

# Assign the environment to the run configuration
pipeline_run_config.environment = registered_env

print ("Run configuration created.")


### Create and run a pipeline

from azureml.pipeline.core import PipelineData
from azureml.pipeline.steps import PythonScriptStep

# Get the training dataset
diabetes_ds = ws.datasets.get("diabetes dataset")

# Create a PipelineData (temporary Data Reference) for the model folder
prepped_data_folder = PipelineData("prepped_data_folder", datastore=ws.get_default_datastore())

# Step 1, Run the data prep script
train_step = PythonScriptStep(name = "Prepare Data",
                                source_directory = experiment_folder,
                                script_name = "prep_diabetes.py",
                                arguments = ['--input-data', diabetes_ds.as_named_input('raw_data'),
                                             '--prepped-data', prepped_data_folder],
                                outputs=[prepped_data_folder],
                                compute_target = pipeline_cluster,
                                runconfig = pipeline_run_config,
                                allow_reuse = True)

# Step 2, run the training script
register_step = PythonScriptStep(name = "Train and Register Model",
                                source_directory = experiment_folder,
                                script_name = "train_diabetes.py",
                                arguments = ['--training-folder', prepped_data_folder],
                                inputs=[prepped_data_folder],
                                compute_target = pipeline_cluster,
                                runconfig = pipeline_run_config,
                                allow_reuse = True)

print("Pipeline steps defined")


### OK, you're ready build the pipeline from the steps you've defined and run it as an experiment.

from azureml.core import Experiment
from azureml.pipeline.core import Pipeline
from azureml.widgets import RunDetails

# Construct the pipeline
pipeline_steps = [train_step, register_step]
pipeline = Pipeline(workspace=ws, steps=pipeline_steps)
print("Pipeline is built.")

# Create an experiment and run the pipeline
experiment = Experiment(workspace=ws, name = 'mslearn-diabetes-pipeline')
pipeline_run = experiment.submit(pipeline, regenerate_outputs=True)
print("Pipeline submitted for execution.")
RunDetails(pipeline_run).show()
pipeline_run.wait_for_completion(show_output=True)


### A graphical representation of the pipeline experiment will be displayed in the widget as it runs. keep an eye on the kernel indicator at the top right of the page, when it turns from **&#9899;** to **&#9711;**, the code has finished running. You can also monitor pipeline runs in the **Experiments** page in [Azure Machine Learning studio](https://ml.azure.com).

for run in pipeline_run.get_children():
    print(run.name, ':')
    metrics = run.get_metrics()
    for metric_name in metrics:
        print('\t',metric_name, ":", metrics[metric_name])


# Assuming the pipeline was successful, a new model should be registered with a *Training context* tag indicating it was trained in a pipeline. Run the following code to verify this.

from azureml.core import Model

for model in Model.list(ws):
    print(model.name, 'version:', model.version)
    for tag_name in model.tags:
        tag = model.tags[tag_name]
        print ('\t',tag_name, ':', tag)
    for prop_name in model.properties:
        prop = model.properties[prop_name]
        print ('\t',prop_name, ':', prop)
    print('\n')