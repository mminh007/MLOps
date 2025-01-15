from utils.func import registered_model, connect_mlflow
from config_args import setup_parse, update_config
import mlflow
from mlflow.tracking import MlflowClient


def main(args, **kwargs):

	"""
	Function used to register a model to the MLflow Models.
	"""
	
	run_id = kwargs["ti"].xcom_pull(task_id="train_model", key="run_id")
	test_loss = kwargs["ti"].xcom_pull(task_id="train_model", key="test_loss")

	connect_mlflow(args)

	client = MlflowClient()
	registered_name = args.registered_name
	model_alias = args.model_alias

	try:
		alias_mv = client.get_model_version_by_alias(registered_name, model_alias)
		print(f"Alias: {model_alias} found")

	except:
		print(f"Alias: {model_alias} not found")
		registered_model(client, registered_name, model_alias, run_id)

	else:
		print(f"Retrieving run: {alias_mv.run_id}")
		prod_metric = mlflow.get_run(alias_mv.run_id).data.metrics
		prod_test_loss = prod_metric["test_loss"]

		if prod_test_loss < test_loss:
			print(f"Current model is better: {prod_test_loss}")
		else:
			registered_model(client, registered_name, model_alias, run_id)


if __name__ == "__main__":
	parser = setup_parse()

	args = parser.parse_args()
	args = update_config(args)

	main(args)