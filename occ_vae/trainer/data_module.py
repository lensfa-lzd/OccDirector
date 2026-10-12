from torch.utils.data import DataLoader
import pytorch_lightning as pl


class BaseDataModuleV2(pl.LightningDataModule):
    def __init__(self, datasets, collate_fn, batch_size=32, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset, self.val_dataset, self.test_dataset = datasets

        self.collate_fn = collate_fn

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
            shuffle=True, pin_memory=True, collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
            pin_memory=True, collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
            pin_memory=True, collate_fn=self.collate_fn,
        )
