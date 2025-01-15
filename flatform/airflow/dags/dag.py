from airflow import DAG
from airflow.utils.dates import days_ago
from datetime import timedelta, datetime

from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from pathlib import Path

def create_temp_dir():
	"""
	"""
	path_save = Path("./cls_temp_dir")
	path_save.mkdir(parents=True, exist_ok=True)
	if not path_save.exists():
		raise FileExistsError(f"Failed to create directory: {path_save}")

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

    create_temp_dir_task = PythonOperator(task_id="create_temp_dir",
									 python_callable=create_temp_dir,
									 dag=dag)
	
    processing_task = BashOperator(task_id="process_data",
								   bash_command=("python3 /tmp/processing.py" \
						 			" --config_file= /tmp/parameters.yaml"),
									dag=dag)
									
	
    training_task = BashOperator(task_id="train_model",
								 bash_command=("python3 /tmp/train.py" \
					                " --config_file= /tmp/parameters.yaml"),
									dag=dag)
	
    registered_task = BashOperator(task_id="registered_model",
								  bash_command=("python3 /tmp/register.py" \
						            " --config_file= /tmp/parameters.yaml"),
									dag=dag)
	

create_temp_dir_task >> processing_task >> training_task >> registered_task

							 
