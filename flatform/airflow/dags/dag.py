from airflow import DAG
from airflow.utils.dates import days_ago
from datetime import timedelta, datetime

from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from pathlib import Path
import os

def create_temp_dir():
	"""
	"""
	path_save = Path("./cls_tmp_dir")
	
	# if path_save.exists():
	# 	cmd_0  = f"rm -rf {path_save}"
	# 	os.system(cmd_0)

	path_save.mkdir(parents=True, exist_ok=True)

	if not path_save.exists():
		raise FileExistsError(f"Failed to create directory: {path_save}")
	print(path_save)

default_args = {
    "owner": "airflow",
    "depend_on_past": False,
    "start_date": datetime(2025, 1, 1),
    "retries": 3,
    "retry_delay": timedelta(minutes=1)
}
    
with DAG(
    "Classification-Task",
    default_args= default_args,
    description= "A simple classify pipeline",
) as dag:
	
	whoami_task = BashOperator(task_id="check_user",
    							bash_command="whoami",
    							dag=dag)
	
	create_temp_dir_task = PythonOperator(task_id="create_temp_dir",
								  python_callable=create_temp_dir,
								  dag=dag)
	
	processing_task = BashOperator(task_id="process_data",
								   bash_command=("cd / && cd opt/airflow && python3 src/processing.py" \
						 			" --config-file='./config/parameters.yaml'"),
									dag=dag)
	
	training_task = BashOperator(task_id="train_model",
								 bash_command=("cd / && cd opt/airflow && python3 src/train.py" \
					                " --config-file='./config/parameters.yaml'"),
									dag=dag)
	
	registered_task = BashOperator(task_id="registered_model",
								  bash_command=("cd / && cd opt/airflow && python3 src/register.py" \
						            " --config-file='./config/parameters.yaml'"),
									dag=dag)
	


whoami_task >> create_temp_dir_task >> processing_task >> training_task >> registered_task


							 
