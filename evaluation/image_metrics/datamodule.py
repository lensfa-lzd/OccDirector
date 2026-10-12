import os
from pathlib import Path
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import Dataset, DataLoader

from model.trainer.data_module import TextToOccFVDDataModule


def get_gt_occ_list():
    sample_rate = 0.1
    data_module = TextToOccFVDDataModule(
        sample_rate=sample_rate,
        uni_occ_dataset_path=os.getenv('DATA_PATH_1'),
        occ_collect_dataset_path=os.getenv('DATA_PATH_2'),
        ttsg_dataset_path=os.getenv('DATA_PATH_3'),
        resize_cfg=None,
        load_occ=False
    )

    occ_list = []
    dataset = data_module.test_dataset
    for i in range(len(dataset)):
        # 一个视频取第一帧图像
        occ_list.append(dataset[i]['instance_path'][0])

    return occ_list


def get_gen_occ_list():
    data_path = os.getenv('GEN_DATA_PATH')
    files = sorted(list(Path(data_path).glob('*.npy')))
    return files


class OccPairDataset(Dataset):
    def __init__(self):
        self.gen_occ_list = get_gen_occ_list()

        full_gt_occ_list = get_gt_occ_list()

        # 等间距抽样到 gen_occ_list 相等的大小
        step = int(len(full_gt_occ_list) / len(self.gen_occ_list))
        self.gt_occ_list = full_gt_occ_list[::step]

        if len(self.gt_occ_list) > len(self.gen_occ_list):
            self.gt_occ_list = self.gt_occ_list[:len(self.gen_occ_list)]

        if len(self.gt_occ_list) < len(self.gen_occ_list):
            self.gt_occ_list += self.gt_occ_list[:len(self.gen_occ_list) - len(self.gt_occ_list)]

        assert len(self.gt_occ_list) == len(self.gen_occ_list)

    def __len__(self):
        return len(self.gt_occ_list)

    @staticmethod
    def load_occ(file_path):
        data = np.load(file_path)
        if len(data.shape) == 4:  # F, H, W, D
            return data[0]  # 取第一帧
        elif len(data.shape) == 5:  # 适用于 dome 的 [bs, f, h, w, d] (bs=1)
            return data[0, 0, ...]
        return data  # H, W, D

    def __getitem__(self, idx):
        gen = self.gen_occ_list[idx]
        gt = self.gt_occ_list[idx]

        gen = self.load_occ(gen)
        gt = self.load_occ(gt)
        return gt, gen

    @staticmethod
    def collate_fn(batch):
        x = []
        y = []
        for it in batch:
            x.append(it[0])
            y.append(it[1])

        return np.stack(x, axis=0), np.stack(y, axis=0)


class OccPairDataModule(pl.LightningDataModule):
    def __init__(
            self,
            batch_size=64,
            num_worker=2,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.test_dataset = OccPairDataset()
        self.num_worker = num_worker

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_worker,
            shuffle=False, drop_last=True,
            collate_fn=self.test_dataset.collate_fn,
        )
