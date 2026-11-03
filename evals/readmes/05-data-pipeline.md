# rainfall-etl

Nightly pipeline that pulls weather-station readings, cleans them with pandas and loads them into BigQuery on Google Cloud.

Orchestration is an Airflow DAG (`dags/rainfall.py`). The bucket and dataset are created with Terraform; see `infra/`.

Questions: open an issue or email rain.etl.maintainer@example.com.
