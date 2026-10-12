import os
import sys
from os.path import abspath, dirname
from dotenv import load_dotenv

load_dotenv('../.env')
sys.path.insert(0, dirname(dirname(abspath(__file__))))

from occinteract.processing.parallel_processor import ParallelProcessor
from evaluation.vlm_eval.scripts import BevOccEvaluator, get_process_task


def main():
    tasks, dataset = get_process_task()
    handler = BevOccEvaluator(dataset)

    if not os.getenv('OUTPUT_PATH'):
        output_path = f"output/dataset_score__{os.getenv('OPENAI_MODEL')}.json"
    else:
        output_path = os.getenv('OUTPUT_PATH')

    print(output_path)

    processor = ParallelProcessor(
        text_rewriter=handler,
        texts_to_process=tasks,
        output_path=output_path,
        max_retries=3,
        result_schema={
            'input': 'task_cfg_str',
            'output': 'result'
        },
        max_workers=8,
    )
    processor.run()


if __name__ == '__main__':
    main()
