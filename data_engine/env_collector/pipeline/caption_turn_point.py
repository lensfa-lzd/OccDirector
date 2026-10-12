import json
import logging
import os
import pickle

import cv2
import numpy as np
from openai import OpenAI

from occinteract.core.base_handler import BaseVisualizationHandler, BaseHandler
from occinteract.core.scripts import encode_image_to_base64, cover_model_output, extract_frames_to_numpy


class Handler(BaseHandler, BaseVisualizationHandler):
    def __init__(self, data_path, dataset):
        super().__init__(data_path=data_path, dataset=dataset)

        self.client = OpenAI(timeout=600.0)
        self.model = os.getenv('OPENAI_MODEL')
        self.temperature = 1.0

        self.frame_id_args = {
            'org': (0, 80),
            'front_scale': 2.5,
            'thickness': 5,
        }

        self.occ_road_render_kwargs = {
            'elev': 90,
            'radius_factor': 1.5,
            'line_args': {
                'thickness': 10
            },
            'frame_id_args': {
                'org': (0, 80),
                'front_scale': 2.5,
                'thickness': 5,
            },
            'extra_ignore_class': [1, 2, 3, 4, 5, 6, 8, 9]  # 只渲染路面， 排除 7
        }

    def analyze_turn_point(self, turn_point_frame_id, turn_point_occ_file, scene_name):
        result = {
            'input': {
                'turn_point_frame_id': turn_point_frame_id,
            },
            'frame_idx': None,
            'front_video': None,
            'analysis': None,
        }

        try:
            video_file = f'{self.data_path}/{scene_name}/front.mp4'

            # 加载以 turn_point_frame_id 为中心的 32 帧
            start = max(0, turn_point_frame_id - 4)
            end = turn_point_frame_id + 5
            result['frame_idx'] = list(range(start, end))

            front_video = extract_frames_to_numpy(
                video_path=video_file,
                start_frame=start * 5,  # 2Hz -> 10Hz
                end_frame=end * 5,
            )
            front_video = front_video[::5]

            for i, frame in enumerate(front_video):
                self.add_frame_id(
                    frame,
                    i + start,
                    color=(0, 0, 0),
                    frame_id_args=self.frame_id_args,
                )

            result['front_video'] = front_video

            occ_array = np.load(f'{self.data_path}/{turn_point_occ_file}')
            road_img = self.render_bev_image(
                turn_point_frame_id,
                occ_array,
                **self.occ_road_render_kwargs,
            )
            road_img = cv2.resize(road_img, (2048, 2048))
            front_video.append(road_img)

            analysis_result = self.analyze_turn_point_action(front_video)
            result['analysis'] = analysis_result

        except Exception as e:
            logging.warning(f'caption_turn_point.py -> Handler -> analyze_turn_point: {e}')

        return result

    def analyze_turn_point_action(self, front_video):
        system_prompt = """
**Role:**
You are an expert Image Analyst specialized in autonomous driving scenarios. You will receive a sequence of real world (or simulation) driving video frames captured from the ego vehicle's front windshield, followed by a final reference image.

**Task:**
Analyze the ego vehicle's temporal driving behavior and the surrounding spatial environment based on the visual sequence and the provided road structure map.

**Input Data Description:**

1.  **Image Sequence:**
    *   The images are arranged in strictly ascending chronological order.
    *   **Frame IDs:** Displayed as black numbers in the top-left corner. Use these to track temporal progression.

2.  **Road Structure Map (Bird's-Eye View):**
    *   **The Final Image:** The very last image in the input list is a BEV (Bird's-Eye View) map representing the static road structure around the ego vehicle.
    *   **Temporal Alignment:** The Frame ID in the top-left of this map corresponds to the frame in the video sequence where this road structure applies.
    *   **Ego Vehicle Positioning:** Two perpendicular black lines indicate the ego vehicle's pose:
        *   **Vertical Line:** Represents the vehicle's heading (pointing Up).
        *   **Horizontal Line:** Represents the vehicle's lateral axis (Left/Right).
        *   **Intersection:** The ego vehicle is located exactly at the intersection of these two lines.
    *   **Semantic Colors:** The map is simplified. Only the traversable road surface is highlighted in **Magenta (255, 0, 255)**.

---

**Guidelines**

1.  **Behavior Classification (Taxonomy):**
    Determine the ego vehicle's action based on the sequence:
    *   **STRAIGHT:** Vehicle proceeds forward without significant deviation.
    *   **LEFT:** Vehicle executes a left turn.
    *   **RIGHT:** Vehicle executes a right turn.

2.  **Environmental Description:**
    Generate a description of the driving environment. Focus on spatial layout, surrounding structures (buildings), vegetation, and road morphology (geometry/shape).

3. **Critical Constraints (For Occupancy Labeling):**
    The output description will be used as a text label for a Occupancy  training pipeline. Therefore, you must adhere to the following:
    *   **Ignore Simulation Artifacts:** Even if the images appear to be from a simulator, describe the scene as if it were the real world. Do not mention "simulation," "game engine," or "synthetic data."
    *   **Focus on Geometry/Structure:** Do not include details imperceptible to a geometric occupancy sensor (e.g., specific weather conditions, color, texture details, or lighting). Focus on *what* objects are present and *where* they are.

---

**Output Format:**
Return the result strictly in JSON format without Markdown code block encapsulation.

```json
{
    "reason": "Step-by-step reasoning explaining how the visual cues and the BEV map lines justify the selected action.",
    "action": "STRAIGHT | LEFT | RIGHT",
    "description_short": "A concise summary of the ego vehicle's behavior, including the shape and width of the road.",
    "description": "A detailed description of the environment and traffic situation, focusing on structural elements (buildings, vegetation layout) and road topology suitable for occupancy labeling."
}
```
"""

        # 构建 VLM 输入
        user_content = [{"type": "text", "text": "Analyze the provided frames."}]
        for img in front_video:
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image_to_base64(img)}",
                    "detail": "high"  # 细节很重要，特别是看红绿灯或路口结构
                }
            })

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                max_completion_tokens=1024 * 8,
                temperature=self.temperature
            )
            return cover_model_output(response.choices[0].message.content)
        except Exception as e:
            raise ValueError(f"caption_turn_point.py -> Handler ->  analyze_turn_point_action: {e}")

    def pipeline(self, scene_key):
        try:
            # print(index)
            turn_points_analysis = {}
            turn_points = self.dataset['turn_point'][scene_key]
            occ_files = self.dataset['dataset'][scene_key]['occ_in_scene_paths']
            for frame_id, dist in turn_points:
                occ_file = occ_files[frame_id]
                result = self.analyze_turn_point(frame_id, occ_file, scene_key)
                assert result['analysis'] is not None
                turn_points_analysis[frame_id] = result['analysis']

            return True, turn_points_analysis

        except Exception as e:
            logging.warning(f'caption_turn_point.py -> Handler -> pipeline: {e}')
            return False, None


def get_process_task(info_file, turn_point_file):
    with open(f'{info_file}', 'rb') as f:
        dataset = pickle.load(f)

    dataset_map = {}
    for it in dataset:
        scene_name = it['scene_name']
        dataset_map[scene_name] = it

    with open(turn_point_file, 'rb') as f:
        file_info = pickle.load(f)

    tasks = []
    for scene_name in file_info:
        task_cfg = {
            'key': scene_name,
        }
        tasks.append(json.dumps(task_cfg))

    return_dataset = {
        'dataset': dataset_map,
        'turn_point': file_info,
    }
    return tasks, return_dataset
