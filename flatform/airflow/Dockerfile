FROM apache/airflow:2.10.3

#WORKDIR /tempt

USER root

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git unzip

USER airflow

COPY requirements.txt .

RUN pip install apache-airflow==${AIRFLOW_VERSION} -r requirements.txt
