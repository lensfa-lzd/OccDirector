import logging
import os
from os.path import dirname, abspath

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset, ConcatDataset

from occinteract.loaders.supervised.core import OccCaptionDataset
from occinteract.loaders.supervised.get_dataset import get_full_dataset, get_test_dataset

__project_path__ = dirname(dirname(abspath(__file__)))


class TextToOccDataModule(pl.LightningDataModule):
    def __init__(
            self,
            uni_occ_dataset_path,
            occ_collect_dataset_path,
            ttsg_dataset_path,
            resize_cfg,
            batch_size=4,
            num_workers=4,
            load_occ=True
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset = get_full_dataset(
            uni_occ_dataset_path=uni_occ_dataset_path,
            occ_collect_dataset_path=occ_collect_dataset_path,
            ttsg_dataset_path=ttsg_dataset_path,
            load_occ=load_occ,
            resize_cfg=resize_cfg,
        )

        # 换算成 训练时显示的 steps per epoch 需要 ÷ GPU 数量
        logging.warning(f'数据集一共 {len(self.dataset) // batch_size:,} batch')

    def train_dataloader(self):
        return DataLoader(
            self.dataset, batch_size=self.batch_size, num_workers=self.num_workers,
            shuffle=True, pin_memory=True,
            collate_fn=OccCaptionDataset.collate_fn,
        )


class TextToOccTestSetDataModule(pl.LightningDataModule):
    def __init__(
            self,
            uni_occ_dataset_path,
            occ_collect_dataset_path,
            ttsg_dataset_path,
            batch_size=4,
            num_workers=4,
            load_occ=True
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        uniocc_dataset, occ_collect_dataset, ttsg_dataset = get_test_dataset(
            samples=int(os.getenv("NUM_TEST_SAMPLES", 100)),
            uni_occ_dataset_path=uni_occ_dataset_path,
            occ_collect_dataset_path=occ_collect_dataset_path,
            ttsg_dataset_path=ttsg_dataset_path,
            resize_cfg={
                'origin': 200,
                'after': 224,
            },
            load_occ=load_occ,
        )

        cat_datasets = []

        if not os.getenv("DATA_NO_STATIC"):
            cat_datasets.append(uniocc_dataset)

        if not os.getenv("DATA_NO_INTERACT_ENV"):
            cat_datasets.append(occ_collect_dataset)

        if not os.getenv("DATA_NO_INTERACT_AGENT"):
            cat_datasets.append(ttsg_dataset)

        self.dataset = ConcatDataset(cat_datasets)
        logging.warning(f'数据集一共 {len(self.dataset) // batch_size:,} batch')

    def test_dataloader(self):
        return DataLoader(
            self.dataset, batch_size=self.batch_size, num_workers=self.num_workers,
            shuffle=False, pin_memory=True, drop_last=False,
            collate_fn=OccCaptionDataset.collate_fn,
        )


class TextToOccFVDDataModule(TextToOccDataModule):
    def __init__(
            self,
            sample_rate=0.01,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

        step = int(1 / sample_rate)
        indices = list(range(0, len(self.dataset), step))
        self.test_dataset = Subset(self.dataset, indices)

        logging.warning(f'TEST 设置采样率 {sample_rate}, 数据集一共 {len(self.test_dataset):,} 个实例')

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
            shuffle=False, pin_memory=True, drop_last=False,
            collate_fn=OccCaptionDataset.collate_fn,
        )


class UnCondictionOccFVDDataModule(pl.LightningDataModule):
    def __init__(
            self,
            batch_size,
            data_size
    ):
        super().__init__()

        self.batch_size = batch_size

        # 伪数据
        x_train = torch.randn(data_size, 1)  # 100个样本，每个样本1个特征
        y_train = torch.randint(0, 2, (data_size,))  # 100个标签 (0或1)
        self.test_dataset = TensorDataset(x_train, y_train)

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size,
            shuffle=False, drop_last=False,
        )
