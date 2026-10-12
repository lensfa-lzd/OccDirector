import dotenv

dotenv.load_dotenv()

import os
import torch
import pytorch_lightning as pl

from model.occ_gen_test_fvd import OccGenTestFVD
from model.trainer.data_module import UnCondictionOccFVDDataModule

torch.set_float32_matmul_precision('high')

test_frames = int(os.getenv('TEST_FRAMES', '16'))
data_size = int(os.getenv('DATA_SIZE'))

data_module = UnCondictionOccFVDDataModule(
    batch_size=int(os.getenv('BATCH_SIZE', 4)),
    data_size=data_size,
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
    logger=False,
    enable_checkpointing=False,
    precision='bf16-mixed',
    accelerator="gpu"
)

ckpt_path = os.getenv('LOAD_FROM_CKPT')
trainer.test(pipeline, datamodule=data_module, ckpt_path=ckpt_path)
