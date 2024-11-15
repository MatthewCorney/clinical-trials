import time

from src.settings import settings, client, logger
import ijson
from pydantic import BaseModel
from typing import List, Dict
import json


class ClinicalTrial(BaseModel):
    """
    Basic data object for the trials
    """
    nctId: str
    title: str
    conditions: List[Dict[str, str]]

    def __getitem__(self, item):
        return getattr(self, item)

    def get(self, item, default=None):
        return getattr(self, item, default)


def load_jsonl(path: str) -> list[dict]:
    """
    Loads a jsonl

    :param path: path to file
    :return:
    """
    batch_objects = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                batch_objects.append(json.loads(line))
    return batch_objects


def save_jsonl(path:str, obj: List[dict]):
    """
    OpenAi standard requires a jsonl rather than pure json

    :param path: path to file
    :param obj: object to save, nb must be a list of json objects
    :return:
    """
    with open(path, 'w') as f:
        for entry in obj:
            json.dump(entry, f)
            f.write('\n')
    logger.info(f"Saved {path}")
    return path


def build_dataset(file_path: str = f"{settings.data_dir}\\ctg-studies.json") -> List[ClinicalTrial]:
    """
    Parses clinical trials and return a smaller json object containing conditions, id and the title
    We use ijson here as loading all the json at once has issue for memory

    :param file_path:
    :return:
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        objects = ijson.items(file, 'item')
        return_object = []
        for obj in objects:
            nct_id = obj['protocolSection']['identificationModule']['nctId']
            # If the officialTitle is not present use the brief title
            title = obj['protocolSection']['identificationModule'].get('officialTitle')
            if not title:
                title = obj['protocolSection']['identificationModule'].get('briefTitle')
            if not title:
                continue
            if obj['derivedSection'].get('conditionBrowseModule') is not None:
                conditions = obj['derivedSection']['conditionBrowseModule'].get('meshes', [])
            else:
                conditions = []
            clinical_trial = ClinicalTrial(**{'nctId': nct_id,
                                              "title": title,
                                              "conditions": conditions
                                              }
                                           )

            return_object.append(clinical_trial)
        logger.info(f"Built dataset {len(return_object)}")
        return return_object


def batch_upload(return_object: List[ClinicalTrial],
                 path: str = 'batch_input.jsonl',
                 embedding_model: str = settings.embedding_model,
                 request_limit: int = 50000):
    """
    Batch uploads the required queries to the endpoint, Although we are limited to uploading 50,000 items per time,
    we are further limited by tokens

    :param return_object: Object of clinical trials
    :param path: Path for the response with the batch id
    :param embedding_model: The embedding model to use (set in settings by default)
    :param request_limit: Raises an exception if the number of items is above this
    :return:
    """
    batch_list = []
    if len(return_object) > request_limit:
        logger.error(f'OpenAI has a limit of {request_limit}')
        raise Exception
    for clinical_trial in return_object:
        batch_list.append({
            "url": '/v1/embeddings',
            "custom_id": clinical_trial["nctId"],
            "method": "POST",
            "body": {
                "input": clinical_trial["title"],
                "model": embedding_model,
                "encoding_format": "float"}
        })
    path = save_jsonl(path, obj=batch_list)
    batch_input_file = client.files.create(
        file=open(path, "rb"),
        purpose="batch"
    )
    logger.info(f"Uploaded dataset")
    return batch_input_file.id


def run_batch(batch_input_file_id: str, path="ct-embedd-test.json") -> dict:
    """
    Runs a submitted batch

    :param batch_input_file_id:
    :param path:
    :return:
    """
    batch_object = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/embeddings",
        completion_window="24h",
        metadata={
            "description": "ct-embedd-test"
        }
    )
    returned_batch_json = json.loads(batch_object.json())
    logger.info(f"{returned_batch_json}")

    save_jsonl(path, obj=[returned_batch_json])
    logger.info(f"Batch has been submitted")
    return returned_batch_json


def parse_embedding_file(batch_file_id: str) -> List[dict]:
    """
    Reads the Embedding file received from OpenAi

    :param batch_file_id: The file id string
    :return:
    """
    file_response = client.files.content(batch_file_id)
    text_response = file_response.text
    all_responses = []
    split_file = text_response.split('\n')
    for split_response in split_file:
        if len(split_response) > 0:
            response_object = json.loads(split_response)
            all_responses.append(response_object)
    logger.info(f'Embedding file read {len(all_responses)} embeddings')
    data = []
    for embedding_json in all_responses:
        data.append({"nctId": embedding_json['custom_id'],
                     'embedding': embedding_json['response']['body']['data'][0]['embedding']})
    return data


def check_status(batch_job_id: str, pause: int) -> dict:
    """
    Checks the status of the batch job for completion

    :param pause:
    :param batch_job_id: OpenAi batch job id
    :return:
    """
    completed = False
    retrieved = None
    while not completed:
        retrieved = client.batches.retrieve(batch_job_id)
        logger.info(f"{retrieved.status}")
        if retrieved.failed_at:
            logger.error("Job Failed")
            logger.info(retrieved.dict())
        if (retrieved.status == 'completed') and (retrieved.output_file_id is not None):
            completed = True
        else:
            time.sleep(pause)
    return retrieved.dict()
