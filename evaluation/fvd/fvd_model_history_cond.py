import dotenv

dotenv.load_dotenv()

import os
import torch
import pytorch_lightning as pl

from model.occ_gen_test_fvd import OccGenTestFVD
from model.trainer.data_module import TextToOccFVDDataModule

torch.set_float32_matmul_precision('high')

test_frames = int(os.getenv('TEST_FRAMES', '16'))
data_size = int(os.getenv('DATA_SIZE'))

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
    batch_size=int(os.getenv('BATCH_SIZE', 4)),
    num_workers=0,
)

pipeline = OccGenTestFVD(
    save_file=os.getenv('SAVE_FILE', 'result.pkl'),
    test_frames=test_frames,

    # 占位
    lr=1e-4,
    max_frames=32,
)

trainer = pl.Trainer(
    devices=torch.cuda.device_count(),
    max_steps=int(os.getenv('DATA_SIZE')),
    logger=False,
    enable_checkpointing=False,
    precision='bf16-mixed',
    accelerator="gpu"
)

ckpt_path = os.getenv('LOAD_FROM_CKPT')
trainer.test(pipeline, datamodule=data_module, ckpt_path=ckpt_path)
