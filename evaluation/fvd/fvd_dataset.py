import os

import dotenv

dotenv.load_dotenv()

from model.trainer.data_module import TextToOccFVDDataModule

import torch
import pytorch_lightning as pl
from occinteract.tools.fvd.handler import OccFVDHandler


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def text_to_occ_data_warper_func(batch):
    result = []
    for i, scene_name in enumerate(batch['scene_name']):
        result.append((scene_name, batch['occ_label'][i]))
    return result


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')

    sample_rate = 0.1
    data_module = TextToOccFVDDataModule(
        sample_rate=sample_rate,
        uni_occ_dataset_path=os.getenv('DATA_PATH_1'),
        occ_collect_dataset_path=os.getenv('DATA_PATH_2'),
        ttsg_dataset_path=os.getenv('DATA_PATH_3'),
        resize_cfg={
            'origin': 200,
            'after': 224,
        },
        batch_size=4,
        num_workers=4,
    )

    pipeline = OccFVDHandler(
        output_file=f'text_to_occ_dataset_sample_rate_{sample_rate}_fvd.pkl',
        data_warper_func=text_to_occ_data_warper_func,
    )

    trainer = pl.Trainer(
        devices=torch.cuda.device_count(),
        logger=False,
        enable_checkpointing=False,
        precision='32',
        accelerator="gpu",
    )

    trainer.test(pipeline, datamodule=data_module)
