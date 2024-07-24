# How to run the benchmark

## Install the dependencies

```bash
poetry install
```

## Export the environment variables

```bash
source .env
```

## Run the classification task

```bash
poetry run python -m tasks.classification classification --project_id --dataset_id --task_path --task_id
```