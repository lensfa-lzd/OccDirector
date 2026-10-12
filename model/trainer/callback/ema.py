import logging
from copy import deepcopy
from typing import Optional, Any, Dict, List

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn


class EmaProxyModel(nn.Module):
    """
    一个代理模型，用于持有对主模型中特定子模块的引用。
    AveragedModel 将作用于这个代理模型，从而间接只对指定的子模块进行 EMA。
    """

    def __init__(self, pl_module: "pl.LightningModule", submodule_names: Optional[List[str]]):
        super().__init__()
        if submodule_names is not None:
            # 使用 add_module 将子模块的引用添加到代理模型中
            for name in submodule_names:
                try:
                    module = pl_module.get_submodule(name)
                    self.add_module(name, module)
                except AttributeError:
                    raise AttributeError(f"Submodule '{name}' not found in the LightningModule.")
        else:
            # 添加 pl_module 的所有子模块
            for name, module in pl_module.named_children():
                self.add_module(name, module)


class EMACallback(pl.Callback):
    """
    指数移动平均 (EMA) Callback。

    该 Callback 通过封装 PyTorch 的 `torch.optim.swa_utils.AveragedModel` 来实现 EMA。
    它会在每个训练步后更新 EMA 权重，并在验证和测试时自动使用 EMA 权重。
    检查点会同时保存原始模型权重和 EMA 权重。
    """

    def __init__(
            self,
            submodule_names: Optional[List[str]] = None,
            decay: float = 0.999,
            use_ema_weights_for_validation: bool = True,
            device=None,
    ):
        """
        Args:
            submodule_names: 可以指定使用 ema 的子模块
            decay (float): EMA 的衰减率。值越接近 1，历史权重的影响越大。
            use_ema_weights_for_validation (bool): 是否在验证/测试时使用 EMA 权重。
        """
        if not (0.0 < decay <= 1.0):
            raise ValueError("EMA decay must be in (0, 1]")

        self.decay = decay
        self.submodule_names = submodule_names
        self.use_ema_weights_for_validation = use_ema_weights_for_validation
        self.ema_model: Optional[AveragedModel] = None
        self.original_model_state_dict: Optional[dict] = None
        self.device = device

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        """在训练开始时初始化 EMA 模型。"""
        if stage == "fit":
            # AveragedModel 需要一个模型作为参数
            # 我们不需要 DDP 来包裹 ema_model，因为它不直接参与梯度计算和反向传播
            # 它仅作为权重的容器存在
            if self.device is None:
                device = pl_module.device
            else:
                device = self.device

            # todo 兼容多模块
            self.ema_model = AveragedModel(
                EmaProxyModel(pl_module, ['mmdit', ]),
                device=device,
                multi_avg_fn=get_ema_multi_avg_fn(self.decay)
            )
            logging.warning(f"EMA Callback is active with decay={self.decay}.")

    def on_train_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: STEP_OUTPUT,
            batch: Any,
            batch_idx: int,
    ) -> None:
        """在每个训练步之后更新 EMA 权重。"""
        if self.ema_model is not None:
            self.ema_model.update_parameters(pl_module.mmdit)

    def _swap_weights(self, pl_module: "pl.LightningModule") -> None:
        """切换模型权重（原始权重 vs EMA 权重）。"""
        if self.ema_model is None:
            return

        # 确保在 DDP 环境下，所有进程都持有正确的权重副本
        # deepcopy 保证我们有一个独立的原始权重副本
        current_state = deepcopy(pl_module.state_dict())

        # 加载 EMA 权重到 pl_module
        pl_module.load_state_dict(self.ema_model.module.state_dict())

        # 保存原始权重，以便稍后恢复
        self.original_model_state_dict = current_state

    def _restore_weights(self, pl_module: "pl.LightningModule") -> None:
        """恢复模型的原始权重。"""
        if self.original_model_state_dict is not None:
            pl_module.load_state_dict(self.original_model_state_dict)
            self.original_model_state_dict = None

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """验证前，换上 EMA 权重。"""
        if self.use_ema_weights_for_validation:
            self._swap_weights(pl_module)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """验证后，换回原始权重。"""
        if self.use_ema_weights_for_validation and self.original_model_state_dict is not None:
            self._restore_weights(pl_module)

    # 为 test 和 predict 也添加同样逻辑
    on_test_epoch_start = on_validation_epoch_start
    on_test_epoch_end = on_validation_epoch_end
    on_predict_epoch_start = on_validation_epoch_start
    on_predict_epoch_end = on_validation_epoch_end

    def on_save_checkpoint(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]
    ) -> None:
        """保存检查点时，额外保存 EMA 模型的权重。"""
        if self.ema_model is not None:
            checkpoint['ema_state_dict'] = self.ema_model.module.state_dict()

    def on_load_checkpoint(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]
    ) -> None:
        """从检查点加载时，恢复 EMA 模型的状态。"""
        if self.ema_model is not None and "ema_state_dict" in checkpoint:
            logging.warning("Loading EMA weights from checkpoint.")
            self.ema_model.module.load_state_dict(checkpoint["ema_state_dict"])

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """ 训练结束后，更新 EMA 模型的 BN 统计数据。"""
        if self.ema_model is not None and trainer.datamodule is not None:
            logging.warning("Updating EMA model's batch normalization statistics...")
            # 确保使用 EMA 权重
            self._swap_weights(pl_module)
            # 使用训练数据的前几批来更新 BN
            torch.optim.swa_utils.update_bn(trainer.train_dataloader, pl_module, device=pl_module.device)
            # 恢复原始权重
            self._restore_weights(pl_module)
            logging.warning("EMA model's BN statistics updated.")
