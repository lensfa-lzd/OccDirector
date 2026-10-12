import os
import random
from typing import Any

import numpy as np
import pytorch_lightning as pl

from occinteract.tools.fvd.handler import OccFVDHandler
from model.occ_gen import OccGen


class OccTestFVD(pl.LightningModule):
    """用于计算/导出 FVD 特征的测试模型包装。

    用于已经生成了完整文件的情况
    """

    def __init__(
            self,
            save_file: str,
    ) -> None:
        super().__init__()

        self.fvd_handler = OccFVDHandler(output_file=save_file)

    def test_step(self, batch: Any, batch_idx: int) -> float:
        batch_size = batch.shape[0]
        fvd_feats = self.fvd_handler.process_batch_occ_seq(batch)
        for i in range(batch_size):
            fvd_feat = fvd_feats[i][np.newaxis, :]
            self.fvd_handler.process_result.append(fvd_feat)

        return random.random()  # 占位符

    def on_test_epoch_end(self) -> None:
        """在 test epoch 结束时写出/汇总 FVD 结果。"""
        self.fvd_handler.on_test_epoch_end()


class OccGenTestFVD(OccGen):
    """用于计算/导出 FVD 特征的测试模型包装。

    通过调用 `OccGen.predict_step` 进行无条件生成（prompt 为空字符串），并将生成的 occupancy 序列
    交给 `dataset.fvd.handler.OccFVDHandler` 处理。
    """

    def __init__(
            self,
            save_file: str,
            test_frames: int,

            *args,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.test_frames = test_frames
        self.fvd_handler = OccFVDHandler(output_file=save_file)

        self.occ_save_path = None
        if os.getenv('SAVE_OCC_PATH'):
            self.occ_save_path = os.getenv('SAVE_OCC_PATH')

        self.rank = int(os.environ.get("LOCAL_RANK", 0))

    def save_batch_occ(self, batch_occ_seq, batch_idx):
        os.makedirs(self.occ_save_path, exist_ok=True)
        for i in range(batch_occ_seq.shape[0]):
            occ = batch_occ_seq[i].detach().cpu().numpy().astype(np.uint8)
            np.save(f'{self.occ_save_path}/rank_{self.rank}_batch_{batch_idx}_{i}.npy', occ)

    def test_step(self, batch: Any, batch_idx: int) -> float:
        """测试 step：无条件生成并累计 FVD 特征。

        Args:
            batch: dataloader 返回的 batch。
                dataset 可参考 `trainer.data_module.UnCondictionOccFVDDataModule`。
                由于这里测试的是无条件生成，执行时会将 dataloader 的 batch_size 设置为 1
                （多卡推理时更容易精确控制总生成数量），
                并通过单个 batch 的 `shape[0]` 来确定生成数量。
            batch_idx (int): batch 索引。

        Returns:
            float: 占位返回值（Lightning test_step 需要一个可记录的标量）。
                该值不代表真实性能指标；FVD 特征会由 `OccFVDHandler` 在 epoch end 时写出/汇总。
        """
        if isinstance(batch, dict):
            def get_history_occ_list(occ_list):
                history_occ_list = []
                valid_occ = None
                for occ in occ_list:
                    if occ.shape[0] > 4:
                        valid_occ = occ

                if valid_occ is None:
                    return history_occ_list

                for occ in occ_list:
                    if occ.shape[0] > 4:
                        history_occ_list.append(occ[:4])
                    else:
                        history_occ_list.append(valid_occ[:4])

                return history_occ_list

            # 历史帧控制生成
            batch_size = len(batch['occ_label'])
            prompts = [''] * batch_size
            history_occ = get_history_occ_list(batch['occ_label'])

            if not history_occ:
                return 0

            batch_occ_seq = self.predict_step(
                self.test_frames,
                prompts,
                history_occ_list=history_occ,
                restore=False
            )
        else:
            # 使用无条件生成进行测试
            batch_size = batch[0].shape[0]
            prompts = [''] * batch_size
            batch_occ_seq = self.predict_step(self.test_frames, prompts, restore=False)

        fvd_feats = self.fvd_handler.process_batch_occ_seq(batch_occ_seq)
        for i in range(batch_size):
            fvd_feat = fvd_feats[i][np.newaxis, :]
            self.fvd_handler.process_result.append(fvd_feat)

        if self.occ_save_path:
            self.save_batch_occ(batch_occ_seq, batch_idx)

        return random.random()  # 占位符

    def on_test_epoch_end(self) -> None:
        """在 test epoch 结束时写出/汇总 FVD 结果。"""
        self.fvd_handler.on_test_epoch_end()


class OccGenTest(OccGen):
    def __init__(
            self,
            test_frames: int,

            *args,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.test_frames = test_frames
        self.occ_save_path = os.getenv('SAVE_OCC_PATH')
        self.rank = int(os.environ.get("LOCAL_RANK", 0))
        self.with_his_cond = os.getenv("WITH_HIS_COND", False)

        assert self.occ_save_path is not None

    def save_batch_occ(self, batch_occ_seq, batch_idx, prompts):
        os.makedirs(self.occ_save_path, exist_ok=True)
        for i in range(batch_occ_seq.shape[0]):
            occ = batch_occ_seq[i].detach().cpu().numpy().astype(np.uint8)
            np.save(f'{self.occ_save_path}/rank_{self.rank}_batch_{batch_idx}_{i}.npy', occ)

            with open(f'{self.occ_save_path}/rank_{self.rank}_batch_{batch_idx}_{i}.txt', 'w') as f:
                f.write(prompts[i])

    def test_step(self, batch: Any, batch_idx: int) -> float:
        """测试 step：无条件生成并累计 FVD 特征。

        Args:
            batch: dataloader 返回的 batch。
                dataset 可参考 `trainer.data_module.UnCondictionOccFVDDataModule`。
                由于这里测试的是无条件生成，执行时会将 dataloader 的 batch_size 设置为 1
                （多卡推理时更容易精确控制总生成数量），
                并通过单个 batch 的 `shape[0]` 来确定生成数量。
            batch_idx (int): batch 索引。

        Returns:
            float: 占位返回值（Lightning test_step 需要一个可记录的标量）。
                该值不代表真实性能指标；FVD 特征会由 `OccFVDHandler` 在 epoch end 时写出/汇总。
        """

        def get_history_occ_list(occ_list):
            history_occ_list = []
            for occ in occ_list:
                if occ.shape[0] > 4:
                    history_occ_list.append(occ[:4])
                else:
                    return []

            return history_occ_list

        prompts = batch['scene_caption']

        history_occ = None
        if self.with_his_cond:
            history_occ = get_history_occ_list(batch['occ_label'])
            if not history_occ:
                return 0

        batch_occ_seq = self.predict_step(
            self.test_frames,
            prompts,
            history_occ_list=history_occ,
            restore=False
        )

        if self.occ_save_path:
            self.save_batch_occ(batch_occ_seq, batch_idx, prompts)

        return random.random()  # 占位符
