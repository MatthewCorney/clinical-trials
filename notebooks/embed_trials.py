from pprint import pprint

from embed import build_dataset, save_jsonl, batch_upload, run_batch, load_jsonl, check_status, parse_embedding_file
from settings import settings


def main():
    """
    Runs the Embedding on the clinical trials dataset
    """
    maximum_requests = 50000
    data_object = build_dataset()
    save_jsonl(path=f'{settings.data_dir}//core_data_annotation.jsonl', obj=[x.dict() for x in data_object])
    for current in range(0, len(data_object), maximum_requests):
        start = current
        end = current + maximum_requests
        subset = data_object[start:end]
        file_id = batch_upload(return_object=subset, path=f'{settings.data_dir}\\batch_input_{end}.jsonl')
        run_batch(file_id, path=f"{settings.data_dir}\\ct-embedd-test_{end}.jsonl")
        batch_json = load_jsonl(f'{settings.data_dir}\\ct-embedd-test_{end}.jsonl')
        retrieved_obj = check_status(batch_job_id=batch_json[0]["id"], pause=60)
        pprint(retrieved_obj)
        result = parse_embedding_file(retrieved_obj.output_file_id)
        save_jsonl(path=f'{settings.data_dir}//core_data_embedding_{end}.jsonl', obj=result)
