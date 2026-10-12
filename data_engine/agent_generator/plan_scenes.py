# Derived from https://github.com/basiclab/TTSG (Text-to-Traffic-Scene Generation, arXiv 2409.09575),
# modified for OccInteract-Agent occupancy collection.
import json
import os

import dotenv
import concurrent.futures
from tqdm import tqdm
from colorama import Fore, Style
from openai import OpenAI

from data_engine.agent_generator.misc import remove_thinking_content
from data_engine.agent_generator.misc.create_scene_from_json import create_seed
from data_engine.agent_generator.prompt import (
    SYSTEM_PROMPT,
    check_analysis_output,
    check_planning_output,
    check_retreival_output,
)
from data_engine.agent_generator.prompt.format import (
    ANALYSIS_FORMAT,
    ANALYSIS_FORMAT_WITH_ERROR,
    PLANNING_FORMAT,
    PLANNING_FORMAT_WITH_ERROR,
    ROAD_RETREIVAL_FORMAT,
    ROAD_RETREIVAL_FORMAT_WITH_ERROR,
)

dotenv.load_dotenv()


def text_to_scene(
        input_prompt: str,
        model_name: str = "gpt-4o",
        save_dir: str = "text_to_scene",
        return_ego: bool = True,
        max_retry: int = 3,
        print_mid_result=False,
        save_prompt=True
):
    if os.path.isfile(f'{save_dir}/agent_output.json'):
        if save_prompt:
            if os.path.isfile(f'{save_dir}/prompt.txt'):
                return
        else:
            return

    chat_client = OpenAI()
    analysis_success = False
    analysis_check_output = None
    analysis_output = None
    count_analysis_retry = 0
    while not analysis_success and count_analysis_retry < max_retry:
        if analysis_check_output is None or analysis_output is None:
            analysis_input = ANALYSIS_FORMAT.format(
                description=input_prompt,
                return_ego="True" if return_ego else "False",
            )
        else:
            analysis_input = ANALYSIS_FORMAT_WITH_ERROR.format(
                description=input_prompt,
                return_ego="True" if return_ego else "False",
                error=analysis_check_output,
                previous_output=analysis_output,
            )
        analysis_response = chat_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": analysis_input},
            ],
            max_completion_tokens=1024 * 8,
            temperature=1.0,
        )
        try:
            analysis_output = analysis_response.choices[0].message.content
            analysis_output = remove_thinking_content(analysis_output)
            analysis_success, analysis_check_output = check_analysis_output(
                analysis_output
            )
            if print_mid_result:
                print(f"{Style.BRIGHT}Analysis input{Style.RESET_ALL}: {analysis_input}")
                print(f"{Style.BRIGHT}Analysis output{Style.RESET_ALL}: {analysis_output}")
                print(
                    f"{Style.BRIGHT}Analysis check message{Style.RESET_ALL}: {Fore.GREEN + 'success' + Style.RESET_ALL if analysis_success else Fore.RED + analysis_check_output + Style.RESET_ALL}"
                )
        except Exception as e:
            print(str(e))
            analysis_output = None
        count_analysis_retry += 1

    analysis_output = json.dumps(analysis_check_output)
    retreival_success = False
    retreival_check_output = None
    retreival_output = None
    count_retreival_retry = 0
    while not retreival_success and count_retreival_retry < max_retry:
        if retreival_check_output is None or retreival_output is None:
            retreival_input = ROAD_RETREIVAL_FORMAT.format(
                description=input_prompt,
                analysis_context=analysis_output,
                return_ego="True" if return_ego else "False",
            )
        else:
            retreival_input = ROAD_RETREIVAL_FORMAT_WITH_ERROR.format(
                description=input_prompt,
                analysis_context=analysis_output,
                return_ego="True" if return_ego else "False",
                error=retreival_check_output,
                previous_output=retreival_output,
            )
        retreival_response = chat_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": retreival_input},
            ],
            max_completion_tokens=1024 * 8,
            temperature=1.0,
        )
        try:
            retreival_output = retreival_response.choices[0].message.content
            retreival_output = remove_thinking_content(retreival_output)
            retreival_success, retreival_check_output = check_retreival_output(
                retreival_output
            )
            if print_mid_result:
                print(f"{Style.BRIGHT}Retreival input{Style.RESET_ALL}: {retreival_input}")
                print(
                    f"{Style.BRIGHT}Retreival output{Style.RESET_ALL}: {retreival_output}"
                )
                print(
                    f"{Style.BRIGHT}Retreival check message{Style.RESET_ALL}: {Fore.GREEN + 'success' + Style.RESET_ALL if retreival_success else Fore.RED + retreival_check_output + Style.RESET_ALL}"
                )
        except Exception:
            retreival_output = None
        count_retreival_retry += 1

    planning_success = False
    planning_check_output = None
    planning_output = None
    count_planning_retry = 0
    while not planning_success and count_planning_retry < max_retry:
        if planning_check_output is None or planning_output is None:
            planning_input = PLANNING_FORMAT.format(
                description=input_prompt,
                analysis_context=analysis_output,
                return_ego={"True" if return_ego else "False"},
            )
        else:
            planning_input = PLANNING_FORMAT_WITH_ERROR.format(
                description=input_prompt,
                analysis_context=analysis_output,
                return_ego={"True" if return_ego else "False"},
                error=planning_check_output,
                previous_output=planning_output,
            )
        planning_response = chat_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": planning_input},
            ],
            temperature=1.0,
            max_completion_tokens=1024 * 8,
        )
        try:
            planning_output = planning_response.choices[0].message.content
            planning_output = remove_thinking_content(planning_output)
            planning_success, planning_check_output = check_planning_output(
                planning_output
            )
            if print_mid_result:
                print(f"{Style.BRIGHT}Planning input{Style.RESET_ALL}: {planning_input}")
                print(f"{Style.BRIGHT}Planning output{Style.RESET_ALL}: {planning_output}")
                print(
                    f"{Style.BRIGHT}Planning check message{Style.RESET_ALL}: {Fore.GREEN + 'success' + Style.RESET_ALL if planning_success else Fore.RED + planning_check_output + Style.RESET_ALL}"
                )
        except Exception as e:
            planning_output = None
        count_planning_retry += 1

    if planning_check_output:
        if planning_check_output['agents']:
            os.makedirs(save_dir, exist_ok=True)
            with open(f"{save_dir}/agent_output.json", "w") as f:
                json.dump(
                    {
                        "analysis": analysis_check_output,
                        "retreival": retreival_check_output,
                        "planning": planning_check_output,
                    },
                    f,
                    indent=2,
                )
            if save_prompt:
                with open(f"{save_dir}/prompt.txt", "w") as f:
                    f.write(str(input_prompt))


def batch_run(workers=4, prompt_bank=None, output_dir=None, tries=5):
    if prompt_bank is None:
        prompt_bank = os.path.join(os.path.dirname(__file__), 'data', 'prompt-20251225.json')
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(prompt_bank), 'prompt_bank_output')

    with open(prompt_bank, 'r') as f:
        data = json.load(f)

    tasks = []
    for scene_type, prompts in data.items():
        for i, prompt in enumerate(prompts):
            input_prompt = prompt
            model_name = os.getenv('OPENAI_MODEL')

            # 每个 prompt 重复多次
            for j in range(tries):
                save_dir = f"{output_dir}/{scene_type.replace(' ', '_').lower()}/prompt_{i}_try_{j + 1}"
                tasks.append((
                    input_prompt,
                    model_name,
                    save_dir
                ))

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(text_to_scene, *task): task
            for task in tasks
        }

        with tqdm(total=len(tasks)) as pbar:
            for _ in concurrent.futures.as_completed(futures):
                pbar.update(1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='LLM 3-stage planning (analysis -> retrieval -> planning)')
    parser.add_argument('--prompt-bank', default=None,
                        help='JSON prompt bank (default: bundled data/prompt-20251225.json)')
    parser.add_argument('--output-dir', default=None, help='output root for planned scenes')
    parser.add_argument('--tries', type=int, default=5, help='planning repeats per prompt')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    create_seed(42)
    batch_run(args.workers, args.prompt_bank, args.output_dir, args.tries)
