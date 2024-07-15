import asyncio
import click
from collections import deque
from collections.abc import Coroutine
import json
import re

from google.cloud import bigquery
from more_itertools import ichunked
from loguru import logger

from llms import BaseLLM, get_llm

ListInterleavedJobs = list[dict[str, list[asyncio.Task]]]


def clean_model_name(model_name: str) -> str:
    return re.sub(r'[-@/.]', '_', model_name)


class ClassifierBenchmark:
    @logger.catch
    def __init__(self, project_id: str, dataset_id: str, task: dict, batch_size: int = 30) -> None:
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.task = task
        self.batch_size = batch_size

        self.bq_client = bigquery.Client(project=project_id)
        self.models = self.init_models()
        self.init_tables()

    @logger.catch
    def get_table_schema(self) -> list[bigquery.SchemaField]:
        return [
            bigquery.SchemaField(self.task["ID_COLUMN_NAME"], self.task["ID_COLUMN_TYPE"], mode="NULLABLE"),
            bigquery.SchemaField(self.task["LABEL_COLUMN_NAME"], "STRING", mode="NULLABLE"),
            bigquery.SchemaField(self.task["COMPLETION_TOKENS_COLUMN_NAME"], "INTEGER", mode="NULLABLE"),
            bigquery.SchemaField(self.task["PROMPT_TOKENS_COLUMN_NAME"], "INTEGER", mode="NULLABLE"),
            bigquery.SchemaField("model", "STRING", mode="NULLABLE"),
        ]

    @logger.catch
    def init_models(self) -> list[BaseLLM]:
        models = self.task["MODELS"]
        return [get_llm(provider=provider,
                        model_name=model_name)
                for provider, model_list in models.items()
                for model_name in model_list]

    @logger.catch
    def init_tables(self) -> None:
        """Create BQ tables for results if not exist."""
        for model in self.models:
            # create table if not exists
            model_name = clean_model_name(model.model_name)
            table_id = f"{self.project_id}.{self.dataset_id}.{model_name}_{self.task['TASK_ID']}"
            try:
                self.bq_client.get_table(table_id)

            except Exception:
                logger.info(f"Model {model.model_name} : table {table_id} does not exist. Creating it.")
                schema = self.get_table_schema()
                table = bigquery.Table(table_id, schema=schema)
                self.bq_client.create_table(table)

            else:
                logger.info(f"Model {model} : table {table_id} already exists.")

    @logger.catch
    def write_to_bq(self, model_name: str, bq_rows: list[dict]) -> None:
        """Write rows to BQ table.

        Args:
            model_name (str): model name
            bq_rows (list[dict]): list of records to insert
        """
        table_id = f"{self.project_id}.{self.dataset_id}.{model_name}_{self.task['TASK_ID']}"
        self.bq_client.insert_rows_json(table=table_id,
                                        json_rows=bq_rows)
        logger.info(f"Model {model_name} : inserted {len(bq_rows)} rows to {table_id}.")

    @logger.catch
    async def agenerate(self, model: BaseLLM, input_row: dict) -> dict:
        """Generate predictions for a single row.

        Args:
            model (str): model name
            input_row (dict): row to predict
        """
        label, completion_tokens, prompt_tokens = model.classify(
            preprompt=self.task["PREPROMPT"],
            prompt=input_row[self.task["TEXT_COLUMN_NAME"]],
            labels=self.task["LABELS"],
            predict_labels_index=False,
            example_input=self.task["EXAMPLES"]["INPUT"],
            example_output=self.task["EXAMPLES"]["OUTPUT"],
            validation_error_label=self.task["VALIDATION_ERROR_LABEL"],
            response_blocked_error_label=self.task["RESPONSE_BLOCKED_ERROR_LABEL"],
            harmful_content_error_label=self.task["HARMFUL_CONTENT_ERROR_LABEL"]
        )

        # insert into BQ table
        insert_row = {
            self.task["ID_COLUMN_NAME"]: input_row[self.task["ID_COLUMN_NAME"]],
            "label": label,
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "model": model.model_name,
        }

        return insert_row

    @logger.catch
    def prepare_jobs(self) -> list[dict[str, list[Coroutine]]]:
        """Prepare jobs for parallel computing."""
        # import data
        data = self.bq_client.query(
            f"SELECT * FROM {self.project_id}.{self.dataset_id}.{self.task['SOURCE_TABLE']}").to_dataframe()
        logger.info(f"Loaded {len(data)} rows to predict from BigQuery.")

        jobs = []

        for i, model in enumerate(self.models):
            model_name = clean_model_name(model.model_name)
            table_id = f"{self.project_id}.{self.dataset_id}.{model_name}_{self.task['TASK_ID']}"
            predicted_data = self.bq_client.query(
                f"SELECT {self.task['ID_COLUMN_NAME']} FROM {table_id}").to_dataframe()
            data_to_predict = data[~data[self.task['ID_COLUMN_NAME']].isin(predicted_data[self.task['ID_COLUMN_NAME']])]
            logger.info(f"Model {model.model_name} : predicting {len(data_to_predict)} rows.")

            model_jobs = []
            # append job to list for given models
            for row in data_to_predict.to_dict(orient="records"):
                model_jobs.append(self.agenerate(model, row))

            # split into batches
            jobs.append(deque(ichunked(model_jobs, self.batch_size)))

        # return list of dict(model_name: batch[jobs])
        interleaved_jobs = []
        while max([len(j) for j in jobs]) > 0:
            dict_jobs = {}
            for i, model in enumerate(self.models):
                model_name = clean_model_name(model.model_name)
                try:
                    dict_jobs[model_name] = list(jobs[i].popleft())
                except IndexError:
                    pass
            interleaved_jobs.append(dict_jobs)

        return interleaved_jobs

    @logger.catch
    async def run_batch(self, model_jobs: dict[str, list[asyncio.Task]]) -> None:
        """Run all jobs for a given batch.

        Args:
            model_jobs (dict[str, list[asyncio.Task]]): dict(model_name: list[asyncio.Task])
        """
        # for a batch, first obtain results from different models in parallel
        models_results = {model_name: [] for model_name in model_jobs.keys()}
        async with asyncio.TaskGroup() as tg:
            for model_name, model_jobs in model_jobs.items():
                for j in model_jobs:
                    models_results[model_name].append(tg.create_task(j))

        # then write to BQ
        for model_name, results in models_results.items():
            rows_to_bq = [r.result() for r in results]
            self.write_to_bq(model_name, rows_to_bq)


@click.group()
def cli():
    pass


@click.command()
@click.option("--project_id", required=True)
@click.option("--dataset_id", required=True)
@click.option("--task_path", type=click.Path(exists=True), required=True)
@click.option("--task_id", required=True)
@click.option("--batch_size", default=50, help="Batch size")
def classification(project_id: str, dataset_id: str, task_path: str, task_id: str, batch_size: int):
    task = json.load(open(task_path, "r"))[task_id]

    bench = ClassifierBenchmark(project_id=project_id,
                                dataset_id=dataset_id,
                                task=task,
                                batch_size=batch_size)

    logger.info(f"Batch size for parallel computing: {batch_size}")

    jobs = bench.prepare_jobs()

    # run each batch
    for i, job in enumerate(jobs):
        logger.info(f"Computing batch {i + 1} out of {len(jobs)}")
        asyncio.run(bench.run_batch(job))


if __name__ == '__main__':
    cli.add_command(classification)
    cli()
