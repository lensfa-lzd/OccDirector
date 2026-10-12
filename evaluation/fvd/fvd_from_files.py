import os

from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from model.occ_gen_test_fvd import OccTestFVD


class OccFileDataset(Dataset):
    def __init__(self, folder_path):
        """
        Args:
            folder_path (str): 包含 .npy 文件的文件夹路径
        """
        self.folder_path = Path(folder_path)
        self.files = sorted(list(self.folder_path.glob('*.npy')))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]

        data = np.load(file_path)
        tensor_data = torch.from_numpy(data)  # 适用于 dome 的 [bs, f, h, w, d] (bs=1)
        return tensor_data

    @staticmethod
    def collate_fn(batch):
        return torch.cat(batch, dim=0)


class UnCondictionOccFVDDataModule(pl.LightningDataModule):
    def __init__(
            self,
            batch_size,
            folder_path,
            num_worker=4,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.test_dataset = OccFileDataset(folder_path)
        self.num_worker = num_worker

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_worker,
            shuffle=False, drop_last=False,
            collate_fn=self.test_dataset.collate_fn,
        )


torch.set_float32_matmul_precision('high')

data_module = UnCondictionOccFVDDataModule(
    batch_size=int(os.getenv('BATCH_SIZE', 4)),
    folder_path=os.getenv('GEN_DATA_PATH', 'output/gen'),
)

pipeline = OccTestFVD(
    save_file=os.getenv('SAVE_FILE', 'output/fvd_feats.pkl'),
)

trainer = pl.Trainer(
    devices=torch.cuda.device_count(),
    logger=False,
    enable_checkpointing=False,
    precision='bf16-mixed',
    accelerator="gpu"
)

trainer.test(pipeline, datamodule=data_module)
