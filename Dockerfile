FROM apache/airflow:2.9.0rc3-python3.11

COPY requirements.txt /
RUN pip install --no-cache-dir -r /requirements.txt
