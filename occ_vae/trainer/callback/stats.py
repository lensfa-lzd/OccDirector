import torch
import pytorch_lightning as pl
from torchmetrics import Metric


class MeanStdMetric(Metric):
    """
    一个用于计算均值和标准差的 TorchMetrics 度量类。
    """
    # 让 TorchMetrics 知道这个度量在 compute() 之后已经完成。
    full_state_update = False

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        # 定义需要累积的状态变量
        # sum: 所有元素的和
        # sum_sq: 所有元素平方的和
        # count: 元素的总数量
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_sq", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

        # 这个状态似乎是用来统计批次数量的，对于均值/标准差计算不是必需的，但我们保留它
        self.add_state("num_instance", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor):
        """
        用一个新的批次数据来更新状态。
        """

        self.sum += torch.sum(preds)
        self.sum_sq += torch.sum(preds ** 2)
        self.count += preds.numel()

        # self.num_instance 的更新逻辑取决于你的具体需求，这里假设 preds 的第一维是实例数
        # 如果 preds 可能是一维的, 直接用 preds.shape[0] 会有问题，所以先检查维度
        if preds.ndim > 1:
            self.num_instance += preds.shape[0]
        else:
            self.num_instance += 1  # 如果是一维张量，可以认为是一个实例

    def compute(self):
        """
        根据累积的状态计算最终的均值和标准差。
        """
        # 防止除以零
        if self.count == 0:
            return torch.tensor(0.0), torch.tensor(0.0)

        # 计算均值: E[X]
        mean = self.sum / self.count

        # 计算方差: Var(X) = E[X^2] - (E[X])^2
        var = (self.sum_sq / self.count) - mean ** 2

        # 【核心修改】计算标准差，即方差的平方根
        # 使用 torch.clamp 来防止因浮点精度问题导致 var 为微小负数，从而产生 NaN
        std = torch.sqrt(torch.clamp(var, min=0.0))

        return mean, std


class CalcScaleFactorCallback(pl.Callback):
    """
    一个在多 GPU 推理时流式计算统计信息的回调，以节省内存。
    这个例子计算了均值和标准差。
    """

    def __init__(self):
        self.stats_metric = MeanStdMetric()

    def on_predict_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: torch.Tensor,
            batch: dict,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        # 将 metric 移动到与数据相同的设备（只在第一次调用时发生）
        self.stats_metric.to(outputs.device)
        self.stats_metric.update(outputs)

    def on_predict_epoch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule"
    ) -> None:
        """在所有批次处理完毕后，聚合所有进程的统计信息"""
        # 1. 在所有进程上调用 compute()。
        #    这个方法会自动触发所有进程间的状态同步，然后计算最终结果。
        mean, std = self.stats_metric.compute()

        # 2. 只在主进程 (rank 0) 中打印结果。
        if trainer.is_global_zero:
            print("\n" + "=" * 40)
            print("           Global Statistics")
            print("=" * 40)
            # num_instance 也已经被同步过了，所以可以直接使用
            print(f"Total instances processed: {self.stats_metric.num_instance.item():,.0f}")
            print(f"Mean of all instances: {mean.item():.6f}")
            print(f"Standard Deviation of all instances: {std.item():.6f}")
            print("=" * 40 + "\n")

        # 3. (推荐) 在所有进程上重置 metric，为下一次可能的 predict 调用做准备。
        self.stats_metric.reset()
