import json
import logging
import os

from openai import OpenAI
import pickle

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

    def _analyze(self, front_video, fps):
        system_prompt = """
**Role:**
You are an expert Image Analyst specialized in autonomous driving scenarios. You will receive a sequence of real world (or simulation) driving video frames captured from the ego vehicle's front windshield.

**Task:**
Analyze the driving behavior, environment, and potential risks based on the provided frames.

**Input Data Description:**
1.  **Image Sequence:** The images are arranged in ascending chronological order.
2.  **Meta Data:** Frame IDs are displayed as Black Numbers in the top-left corner (use these to understand the temporal progression).
3.  The video has a frame rate of {fps}Hz, meaning that every {fps} Frame IDs corresponds to 1 second of real time.

**Guidelines:**

1.  **Environment Analysis:** Analyze the situation in the sequence. Describe the environment, including surrounding vehicles, pedestrians, buildings, vegetation, road types.

2.  **Safety Scoring:** Evaluate the necessity for the ego vehicle (driver) to exercise caution or take emergency action. Assign a **score (0-10)** based on the following scale:
    *   **0-2:** Safe driving conditions. Good visibility, no immediate hazards (e.g., clear road).
    *   **3-5:** Caution required. Vulnerable road users (pedestrians, cyclists) or potential obstacles present in the field of view.
    *   **6-8:** Immediate action/Emergency required. High-risk situations (e.g., vehicle cutting in, sudden braking ahead, jaywalking).
    *   **9-10:** Collision inevitable or has occurred.

3.  **Critical Constraints (For Occupancy Labeling):**
    The output description will be used as a text label for a Occupancy  training pipeline. Therefore, you must adhere to the following:
    *   **Ignore Simulation Artifacts:** Even if the images appear to be from a simulator, describe the scene as if it were the real world. Do not mention "simulation," "game engine," or "synthetic data."
    *   **Focus on Geometry/Structure:** Do not include details imperceptible to a geometric occupancy sensor (e.g., specific weather conditions, color, texture details, or lighting). Focus on *what* objects are present and *where* they are.

**Output Format:**
Return the result strictly in JSON format. Do not output markdown code blocks alone, just the raw JSON string.

```json
{
    "reason": "Explain the reasoning behind the assigned safety score and action.",
    "description_short": "Briefly describe the ego vehicle's driving behavior and the cause of risks (if any).",
    "description": "Detailed description of the environment and traffic situation focused on the frames.",
    "score": <Integer between 0-10>
}
```
"""
        system_prompt = system_prompt.replace('{fps}', f'{fps}')

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
                max_completion_tokens=1024 * 16,
                temperature=self.temperature
            )
            return cover_model_output(response.choices[0].message.content)
        except Exception as e:
            raise ValueError(f"caption_scene_with_critical_score.py -> Handler ->  analyze: {e}")

    def analyze(self, scene_key):
        video_file = f'{self.data_path}/{scene_key}/front.mp4'

        fps = 2
        front_video = extract_frames_to_numpy(video_file)

        if fps == 2:
            front_video = front_video[::5]

        for i, frame in enumerate(front_video):
            self.add_frame_id(
                frame,
                i,
                color=(0, 0, 0),
                frame_id_args=self.frame_id_args,
            )

        result = self._analyze(front_video, fps=fps)

        return {
            'fps': fps,
            'front_video': front_video,
            'result': result
        }

    def pipeline(self, scene_key):
        try:
            result = self.analyze(scene_key)
            return True, result['result']

        except Exception as e:
            logging.warning(f'caption_scene_with_critical_score.py -> Handler -> pipeline: {e}')
            return False, None


def get_process_task(scene_file):
    with open(scene_file, 'rb') as f:
        file_info = pickle.load(f)

    tasks = []
    dataset = {}
    for it in file_info:
        scene_name = it['scene_name']
        task_cfg = {
            'key': scene_name,
        }
        tasks.append(json.dumps(task_cfg))
        dataset[scene_name] = it

    return tasks, dataset
