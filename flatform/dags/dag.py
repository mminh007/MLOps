from airflow import DAG
from airflow.utils.dates import days_ago
from datetime import timedelta

from airflow.operators.python import PythonOperator
from flatform.src.ultis import create_temp_dir, split_data, logging_artifacts
from flatform.src.modeling import train_model


default_args = {
    "owner": "airflow",
    "depend_on_past": False,
    "start_date": days_ago(1),
    "retries": 3,
    "retry_delay": timedelta(minutes=1)
}
    
with DAG(
    "Multi-tasks_Predictions",
    default_args= default_args,
    description= "A simple ML pipeline",
) as dag:

    create_temp_dir_task = PythonOperator(
        task_id="create_temp_dir",
        python_callable=create_temp_dir,
        dag=dag,
    )

    processing_data_task = PythonOperator(
        task_id="processing_data",
        python_callable=split_data,
        dag=dag
    )
    train_model_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
        dag=dag,
    )

    logging_artifacts_task = PythonOperator(
        task_id="logging_artifacts",
        python_callable=logging_artifacts,
        dag=dag,
    )

    create_temp_dir_task >> processing_data_task >> \
          train_model_task >> logging_artifacts_task