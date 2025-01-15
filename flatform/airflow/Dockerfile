FROM apache/airflow:2.10.3

WORKDIR /tempt

COPY ./src /tempt/src

COPY ./config_args.py /tempt/config_args.py

COPY ./processing.py /tempt/processing.py

COPY ./register.py /tempt/register.py

COPY ./train.py /tempt/train.py

COPY ./config/parameters.yaml /tempt/parameters.yaml

COPY requirements.txt .

RUN pip install apache-airflow==${AIRFLOW_VERSION} -r requirements.txt