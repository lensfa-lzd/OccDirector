import json
import logging
import os
from typing import Optional, List, Dict

import cv2
import httpx
import numpy as np
from occinteract.core.scripts import encode_image_to_base64
from langchain.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from occinteract.core.base_handler import BaseHandler, BaseVisualizationHandler
from occinteract.loaders.supervised.core import OccCaptionDataset
from occinteract.loaders.supervised.get_dataset import get_test_dataset
from occinteract.type_models import OccCaptionDatasetInstance

# -- 1. 将 System Prompt 定义为类级别的常量，并添加 format_instructions 占位符 ---
#    LangChain 的 PydanticOutputParser 将会把 JSON schema 注入到这里
_BEV_EVALUATION_PROMPT_TEMPLATE = """
You are an expert evaluator for an autonomous driving scene generation system. Your task is to score a generated Bird's-Eye View (BEV) driving video frames based on three criteria: Scene Completeness, Structural Correctness, and Instruction Following.

**Input Data Description:**

1.  A text **Prompt** that was used to generate the scene.

2.  **Bird's-Eye View Image Sequence (video frames):**
    *   The images are arranged in strictly ascending chronological order.
    *   Frame IDs: Displayed as black numbers in the top-left corner. Use these to track temporal progression.
    *   The video has a frame rate of 1Hz, meaning that every 1 Frame ID corresponds to 1 second of real time.

    **--- SCENE PROJECTION LEGEND ---**
    The given BEV Image, which is a flat, color-coded projection of a full 3D semantic occupancy grid.
    Here is the color-coding key for the BEV image. Use it to understand the semantics of the generated scene.
    *   **Road:** Magenta `(255, 0, 255)`
    *   **Car:** Blue `(0, 150, 245)`
    *   **Pedestrian:** Red `(255, 0, 0)`
    *   **Building:** Lavender `(230, 230, 250)`
    *   **Vegetation:** Green `(0, 175, 0)`
    *   **Terrain:** Dark Gray `(100, 100, 100)` (e.g., sidewalks, curbs)
    *   **Undefined (White):** Represents "unseen" or empty areas where the model generated nothing.
    *   **Free (Transparent/Black):** Represents unoccupied drivable space.

    **--- EGO VEHICLE POSITIONING ---**
    A black point in the center indicates the ego vehicle's pose. In the image, the area above the black point represents the front of the vehicle, and the area below represents the rear. The left and right sides of the image correspond to the left and right sides of the vehicle, respectively.

**--- THE PROMPT USED FOR GENERATION ---**
`{text_prompt}`

**--- EVALUATION CRITERIA & SCORING RUBRICS (1-5 points) ---**

You must evaluate the generated image against the prompt based on the following three criteria.

**1. Scene Completeness:**
This assesses how much of the scene is filled versus being empty (white/undefined) (you may ignore the image edge areas). A good score means the scene is reasonably filled, without excessive and unnatural empty areas.
*   **1 (Bad):** The image is almost entirely white/empty. The model failed to generate any meaningful content.
*   **2 (Poor):** Large, unnatural white patches dominate the image, making the scene feel fragmented and incomplete.
*   **3 (Fair):** The scene is generated, but there are noticeable empty areas that detract from realism. The model seems to have 'given up' on parts of the scene.
*   **4 (Good):** The scene is mostly complete. White areas are minimal and appear only in distant or naturally occluded regions, which is acceptable.
*   **5 (Excellent):** The scene is fully rendered with no unnatural white/empty patches. The content fills the frame appropriately, reflecting a complete environment.

**2. Structural Correctness:**
This assesses the logical and physical plausibility of the scene's layout, independent of the prompt. Does this look like a real-world road layout?
*   **1 (Bad):** The layout is nonsensical. Roads lead to nowhere, buildings are on top of roads, or objects are arranged in a chaotic, physically impossible manner.
*   **2 (Poor):** The layout has major flaws. For example, roads have illogical breaks, or the relationship between road, terrain, and buildings is strange.
*   **3 (Fair):** The basic layout is plausible, but contains some minor unrealistic elements (e.g., slightly odd intersection shapes, unnatural object placement).
*   **4 (Good):** The road network and object placements are logical and consistent. The scene represents a common and believable driving environment.
*   **5 (Excellent):** The layout is highly realistic, coherent, and complex. It perfectly mimics a plausible real-world traffic scene (e.g., a well-formed intersection, a natural-looking highway curve).

**3. Instruction Following:**
This is the most important criterion. It assesses how well the generated image **matches the given text prompt**.
*   **1 (Bad):** The image completely ignores or contradicts the prompt. (e.g., prompt asks for a highway, image shows a small intersection).
*   **2 (Poor):** The image captures only a minor element of the prompt but fails on the main theme. (e.g., prompt asks for "a dense city intersection with pedestrians," image shows a sparse road with one car).
*   **3 (Fair):** The image reflects the general theme of the prompt (e.g., correct road type) but misses key details or objects mentioned.
*   **4 (Good):** The image successfully generates the scene described in the prompt, including the correct road layout and the primary objects mentioned. It might miss some minor nuances.
*   **5 (Excellent):** The image is a perfect visual representation of the text prompt. All specified elements, their quantities, and their spatial relationships are accurately depicted.

**--- ADDITIONAL INSTRUCTIONS ---**

During the EVALUATION process, since the given BEV Image is a color-coded semantic image, you must **IGNORE** color information mentioned in the text prompt. Specifically:

1.  If the prompt states that a traffic light turns from red to green: Since the semantic image cannot reflect traffic light colors, you should directly ignore this instruction.

2.  If the prompt describes a specific vehicle color (e.g., "a yellow car ahead"): Since all vehicles in the semantic image are rendered as Blue `(0, 150, 245)`, you should consider the generation correct as long as a corresponding blue block exists in the specified location.

**--- OUTPUT FORMAT ---**
You MUST provide your evaluation in a single JSON object. Do not add any text before or after the JSON block.

{format_instructions}

Now, evaluate the provided BEV image based on the prompt:
"""


# -- 2. 使用 Pydantic 定义期望的输出数据结构 ---
class EvaluationScores(BaseModel):
    """The scores for the three evaluation criteria."""
    completeness_score: int = Field(
        description="Score for scene completeness (1-5)."
    )
    structural_correctness_score: int = Field(
        description="Score for structural correctness (1-5)."
    )
    instruction_following_score: int = Field(
        description="Score for instruction following (1-5)."
    )


class EvaluationResult(BaseModel):
    """The final evaluation result including thought process and scores."""
    thought: str = Field(
        description="A brief, step-by-step reasoning of the evaluation process for each criterion."
    )
    scores: EvaluationScores = Field(
        description="An object containing the integer scores for the three criteria."
    )


class BevOccEvaluator(BaseHandler, BaseVisualizationHandler):
    """
    使用 LangChain 和 Pydantic 评估 BEV 图像的类。
    """

    def __init__(
            self,
            dataset: Dict[str, OccCaptionDataset],

            temperature: float = 1.0,
            max_tokens: int = 10240,
            max_retries: int = 3,
    ):
        """
        初始化 BEV 图像评估器。
        """
        super().__init__(data_path=None, dataset=dataset)

        if os.getenv('https_proxy'):
            http_client = httpx.Client(
                proxy=os.getenv('https_proxy'),
            )
        else:
            http_client = None

        # 初始化 LangChain ChatOpenAI 客户端
        self.llm = ChatOpenAI(
            model=os.getenv('OPENAI_MODEL'),
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            http_client=http_client,
            model_kwargs={"response_format": {"type": "json_object"}}  # 强制JSON输出
        )

        # 初始化 Pydantic 输出解析器
        self.parser = PydanticOutputParser(pydantic_object=EvaluationResult)
        self.system_prompt_template = _BEV_EVALUATION_PROMPT_TEMPLATE

        self.occ_render_kwargs = {
            'elev': 90,
            'radius_factor': 1.5,
            'center_point_args': {
                'radius': 15
            },
            'frame_id_args': {
                'org': (0, 80),
                'front_scale': 2.5,
                'thickness': 5,
            },
        }

    def run_evaluate(
            self,
            occ_image_list: List[np.ndarray],
            text_prompt: str
    ) -> Optional[EvaluationResult]:
        """
        调用 VLM 评估生成的 BEV 图像。。
        """
        format_instructions = self.parser.get_format_instructions()
        system_prompt = self.system_prompt_template.format(
            text_prompt=text_prompt,
            format_instructions=format_instructions
        )

        user_content = [
            {
                "type": "text",
                "text": "Please evaluate this generated BEV image based on the prompt and criteria provided."
            }
        ]
        for img in occ_image_list:
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image_to_base64(img)}",
                    "detail": "high"  # 细节很重要，特别是看红绿灯或路口结构
                }
            })

        # 2. 构建 LangChain 消息
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_content,
            }
        ]

        # 3. 调用 LLM
        # logging.info("Sending evaluation request to the model...")
        response = self.llm.invoke(messages)

        # 4. 解析输出
        # logging.info("Parsing model response...")
        parsed_result = self.parser.parse(response.content)

        return parsed_result

    def evaluate(self, scene_key: str):
        result = {
            'input': {
                'scene_key': scene_key,
                'data': None,
            },
            'analysis': None,

            'scene_caption': None,
            'occ_image_list': None,
        }

        dataset_name, dataset_index, _ = scene_key.split('__')
        data: OccCaptionDatasetInstance = self.dataset[dataset_name][int(dataset_index)]
        result['input']['data'] = data

        caption = data["scene_caption"]
        occ_array = data['occ_label']

        result['scene_caption'] = caption
        occ_image_list = []
        for i in range(occ_array.shape[0]):
            img = self.render_bev_image(
                frame_index=i,
                semantic_grid=occ_array[i],
                **self.occ_render_kwargs,
            )
            img = cv2.resize(img, (2048, 2048))
            occ_image_list.append(img)

        occ_image_list = occ_image_list[::2]  # 降采样到 1hz
        result['occ_image_list'] = occ_image_list
        analysis_result = self.run_evaluate(
            occ_image_list,
            caption
        )
        result['analysis'] = {
            'thought': analysis_result.thought,
            'scores': dict(analysis_result.scores)
        }
        return result

    def visualize_core_process(self, result):
        # 展示运行的核心结果
        caption = result['scene_caption']
        occ_image_list = result['occ_image_list']
        analysis = result['analysis']

        print('=' * 20)
        self.show_cv2_image_list(
            occ_image_list,
            axis='off',
            make_grid=True,
        )

        print('-' * 15)
        print(caption)
        print('-' * 15)
        print(analysis['thought'])
        print('-' * 15)
        print(json.dumps(analysis['scores'], indent=2, ensure_ascii=False))
        print('=' * 20)

    def pipeline(self, scene_key):
        try:
            result = self.evaluate(scene_key)
            return True, result['analysis']

        except Exception as e:
            logging.warning(f'vlm评测/scripts.py -> BevOccEvaluator -> pipeline: {e}')
            return False, None


def get_process_task():
    uniocc_dataset, occ_collect_dataset, ttsg_dataset = get_test_dataset(
        samples=int(os.getenv("NUM_TEST_SAMPLES", 100)),
        uni_occ_dataset_path=os.getenv('DATA_PATH_1'),
        occ_collect_dataset_path=os.getenv('DATA_PATH_2'),
        ttsg_dataset_path=os.getenv('DATA_PATH_3'),
        resize_cfg={
            'origin': 200,
            'after': 224,
        },
        load_occ=True,
    )

    try_count = int(os.getenv("NUM_TEST_COUNT", 5))  # 计算5次取平均值
    dataset_dict = {
        'uniocc_dataset': uniocc_dataset,
        'occ_collect_dataset': occ_collect_dataset,
        'ttsg_dataset': ttsg_dataset,
    }

    tasks = []
    for dataset_name, dataset in dataset_dict.items():
        for i in range(len(dataset)):
            for j in range(try_count):
                task_cfg = {
                    'key': f'{dataset_name}__{i}__{j}',
                }
                tasks.append(json.dumps(task_cfg))

    return tasks, dataset_dict
