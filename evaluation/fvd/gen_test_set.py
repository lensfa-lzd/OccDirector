import dotenv

dotenv.load_dotenv()

import os
import torch
import pytorch_lightning as pl

from model.occ_gen_test_fvd import OccGenTest
from model.trainer.data_module import TextToOccTestSetDataModule

torch.set_float32_matmul_precision('high')

test_frames = int(os.getenv('TEST_FRAMES', '16'))

data_module = TextToOccTestSetDataModule(
    uni_occ_dataset_path=os.getenv('DATA_PATH_1'),
    occ_collect_dataset_path=os.getenv('DATA_PATH_2'),
    ttsg_dataset_path=os.getenv('DATA_PATH_3'),
    batch_size=int(os.getenv('BATCH_SIZE', 4)),
    num_workers=0,
)

pipeline = OccGenTest(
    test_frames=test_frames,

    # 占位
    lr=1e-4,
    max_frames=32,
)

trainer = pl.Trainer(
    devices=torch.cuda.device_count(),
    max_steps=int(os.getenv('DATA_SIZE', -1)),
    logger=False,
    enable_checkpointing=False,
    precision='bf16-mixed',
    accelerator="gpu"
)

ckpt_path = os.getenv('LOAD_FROM_CKPT')
trainer.test(pipeline, datamodule=data_module, ckpt_path=ckpt_path)
