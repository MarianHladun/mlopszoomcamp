import os
import pickle
import click
import mlflow
from math import sqrt

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Define experiment names
HPO_EXPERIMENT_NAME = "random-forest-hyperopt"  # Hyperparameter optimization experiment
EXPERIMENT_NAME = "random-forest-best-models"  # Best models experiment

# Parameters needed for RandomForestRegressor
RF_PARAMS = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'random_state']

# Set tracking server details and enable autologging
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # MLflow tracking server
mlflow.set_experiment(EXPERIMENT_NAME)  # Create/Get the best-models experiment
mlflow.sklearn.autolog()  # Automatically log metrics, params, and models

# Function to load datasets from pickle files
def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

# Train a model and log results to MLflow
def train_and_log_model(data_path, params):
    # Load train, validation, and test datasets
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

    # Start an MLflow run
    with mlflow.start_run():
        # Ensure parameter values are integers
        for param in RF_PARAMS:
            params[param] = int(params[param])

        # Train RandomForestRegressor with specified params
        rf = RandomForestRegressor(**params)
        rf.fit(X_train, y_train)

        # Evaluate model on validation and test datasets
        mse_val = mean_squared_error(y_val, rf.predict(X_val))  # Compute MSE for validation dataset
        val_rmse = sqrt(mse_val)  # Convert to RMSE
        mlflow.log_metric("val_rmse", val_rmse)  # Log validation RMSE

        mse_test = mean_squared_error(y_test, rf.predict(X_test))  # Compute MSE for test dataset
        test_rmse = sqrt(mse_test)  # Convert to RMSE
        mlflow.log_metric("test_rmse", test_rmse)  # Log test RMSE

# Main script functionality
@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--top_n",
    default=5,
    type=int,
    help="Number of top models that need to be evaluated to decide which one to promote"
)
def run_register_model(data_path: str, top_n: int):

    client = MlflowClient()

    # Retrieve the top_n model runs from the hyperopt experiment
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,  # Use Hyperopt experiment ID
        run_view_type=ViewType.ACTIVE_ONLY,  # Look at active runs only
        max_results=top_n,  # Limit to top_n runs
        order_by=["metrics.rmse ASC"]  # Sort runs by RMSE in ascending order
    )

    # Train and log the top_n models in the "random-forest-best-models" experiment
    for run in runs:
        train_and_log_model(data_path=data_path, params=run.data.params)

    # Select the model with the lowest test RMSE from the "random-forest-best-models" experiment
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)  # Get best-models experiment
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.test_rmse ASC"]  # Sort runs by test RMSE in ascending order
    )[0]  # Take the best model (lowest test RMSE)

    # Register the selected model
    run_id = best_run.info.run_id  # Retrieve the best run's ID
    model_uri = f"runs:/{run_id}/model"  # Construct the model URI
    model_name = "rf-best-model"  # Name for the registered model
    mlflow.register_model(model_uri, name=model_name)  # Register the model
    print(f"Model registered as '{model_name}' with URI '{model_uri}'")


if __name__ == '__main__':
    run_register_model()