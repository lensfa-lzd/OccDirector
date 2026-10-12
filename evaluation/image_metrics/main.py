import os
import sys
from os.path import dirname, abspath

project_root = dirname(dirname(abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

import dotenv

dotenv.load_dotenv()

import pytorch_lightning as pl
import torch
from evaluation.image_metrics.datamodule import OccPairDataModule
from evaluation.image_metrics.handler import Occ2DMetricHandler

torch.set_float32_matmul_precision('high')

batch_size = int(os.getenv("BATCH_SIZE", 100))
data_module = OccPairDataModule(batch_size=batch_size)
pipeline = Occ2DMetricHandler(
    batch_size=batch_size,
    output_file=os.getenv('OUTPUT_FILE'),
)

trainer = pl.Trainer(
    devices=torch.cuda.device_count(),
    logger=False,
    enable_checkpointing=False,
    precision='bf16-mixed',
    accelerator="gpu"
)

trainer.test(pipeline, datamodule=data_module)
