import logging
import os
import dotenv

dotenv.load_dotenv()

import torch
import pytorch_lightning as pl

from occ_vae.trainer import find_latest_version
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import AsyncCheckpointIO
from occinteract.loaders.unsupervised.get_dataset import get_full_dataset
from occinteract.loaders.unsupervised.core import OccReconstructionDataset
from occ_vae.trainer.data_module import BaseDataModuleV2
from occ_vae.trainer.trainer_module import TrainerSeq as Trainer
from occ_vae.trainer.callback.visualizer_seq import ReconstructionVisualizerSeqByStep
from occ_vae.trainer.callback.ema import EMACallback

from occ_vae.vae import get_vae
from occ_vae.loss.recon_loss import ReconLoss
from occ_vae.loss.kl_loss import KLLoss
from occ_vae.loss.lovasz_loss import LovaszLoss


def main(
        exp_name: str,
        learning_rate: float,
):
    if os.getenv('BATCH_SIZE'):
        batch_size = int(os.getenv('BATCH_SIZE'))
    else:
        batch_size = 4

    # 1. 實例化數據模塊
    data_module = BaseDataModuleV2(
        datasets=get_full_dataset(
            uniocc_dataset_path=os.getenv('DATA_PATH_1'),
            collect_dataset_path=os.getenv('DATA_PATH_2'),
            ttsg_dataset_path=os.getenv('DATA_PATH_3'),
            resize_cfg={
                'origin': 200,
                'after': 224,
            },
        ),
        collate_fn=OccReconstructionDataset.collate_fn,
        batch_size=batch_size,
        num_workers=int(os.getenv('DATA_WORKER_NUM', 4)),
    )

    # 2. 實例化模型
    recon_loss = ReconLoss()
    kl_loss = KLLoss()
    lovasz_loss = LovaszLoss()

    if os.getenv('LOG_DIR'):
        log_dir = os.getenv('LOG_DIR')
    else:
        log_dir = "../pl_logs"

    experiment_name = exp_name

    if os.getenv('TOTAL_STEPS'):
        total_steps = int(os.getenv('TOTAL_STEPS'))
    else:
        total_steps = 1000_0000

    # 可视化与保存模型（小数据集冒烟时可调小，需小于训练 batch 总数）
    check_every_steps = int(os.getenv('CHECK_EVERY_STEPS', 5000))
    log_every_n_steps = 50

    # check_every_steps = 10  # 可视化与保存模型
    # log_every_n_steps = 1

    if os.getenv("LOSS_SETTING") == '1':
        loss_wright = [5.0, 1.0, 0.0]
    else:
        loss_wright = [10.0, 1.0, 0.0]

    model = get_vae()
    call_backs = [
        # pl.callbacks.EarlyStopping(monitor="val_IoU", mode="max", patience=10),
        EMACallback()
    ]

    call_backs += [
        ModelCheckpoint(
            monitor="val_IoU",  # 监控的指标名称
            filename="best-IoU-{step}-{val_IoU:.2f}-{val_mIoU:.2f}",  # 文件名格式
            save_top_k=2,  # 只保存最好的一个模型
            mode="max",  # 模式为 "max"
            save_last=True,
            every_n_train_steps=check_every_steps,
        ),

        # ModelCheckpoint(
        #     monitor="val_mIoU",
        #     filename="best-mIoU-{epoch:02d}-{val_IoU:.2f}-{val_mIoU:.2f}",
        #     save_top_k=1,
        #     mode="max",
        #     every_n_train_steps=check_every_steps,
        # ),

        # 因为同一个 batch 内部的图像相似
        ReconstructionVisualizerSeqByStep(
            test_samples=4,
            every_n_steps=check_every_steps,
            dataset=data_module.test_dataset,
        )
    ]

    loss_wright = [weight / sum(loss_wright) for weight in loss_wright]
    loss_cfg = [
        {
            'name': "ReconLoss",
            'weight': loss_wright[0],
            'function': recon_loss
        },
        {
            'name': "LovaszLoss",
            'weight': loss_wright[1],
            'function': lovasz_loss
        },
        {
            'name': "KLLoss",
            'weight': loss_wright[2],
            'function': kl_loss
        }
    ]

    if os.getenv('NOT_SPLIT_FRAMES'):
        split_frame = False
    else:
        split_frame = True

    pipeline = Trainer(
        model,
        # total_epochs=total_epochs,
        learning_rate=learning_rate,
        loss_cfg=loss_cfg,
        split_frame=split_frame,
        warmup_steps=100
    )

    # 3. 實例化 Trainer
    # 在這裡我們可以配置訓練的所有細節
    trainer = pl.Trainer(
        # 正常训练 | 使用 step 控制训练流程
        max_steps=int(total_steps),
        max_epochs=-1,  # 显式禁用 epoch 限制
        num_nodes=int(os.getenv('NODE_NUM', 1)),
        devices=torch.cuda.device_count(),
        callbacks=call_backs,

        limit_val_batches=1000,
        limit_test_batches=1000 if not os.getenv('TEST_ONLY') else None,
        val_check_interval=check_every_steps,
        # enable_checkpointing=False,

        # 测试全流程
        # fast_dev_run=10,
        # devices=torch.cuda.device_count(),
        # callbacks=call_backs,

        gradient_clip_val=1.0,
        precision='bf16-mixed',
        accelerator="gpu",
        log_every_n_steps=log_every_n_steps,
        logger=pl.loggers.TensorBoardLogger(
            log_dir,
            name=experiment_name,
        ),
        plugins=[
            AsyncCheckpointIO()
        ],
    )

    print("--- Starting training ---")
    print(f"Experiment Name: {exp_name}")
    print(f"Learning Rate:   {learning_rate}")

    # 自动查找最新的 checkpoint
    if not os.getenv('LOAD_FROM_CKPT'):
        latest_ckpt_path = find_latest_version(log_dir, experiment_name)
    else:
        latest_ckpt_path = os.getenv('LOAD_FROM_CKPT')

    if not os.getenv('TEST_ONLY'):
        trainer.fit(pipeline, datamodule=data_module, ckpt_path=latest_ckpt_path)

    # 運行測試
    if os.getenv('TEST_ONLY'):
        trainer.test(pipeline, datamodule=data_module, ckpt_path=latest_ckpt_path)
    else:
        trainer.test(pipeline, datamodule=data_module)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    torch.set_float32_matmul_precision('high')

    # os.environ['EXP_NAME'] = 'dome_dev'
    # os.environ['BATCH_SIZE'] = '8'

    # --- 通过环境变量设置实验参数 ---
    exp_name = os.getenv("EXP_NAME")
    default_lr = 1e-4
    learning_rate = float(os.getenv("LR", default_lr))
    if exp_name:
        main(
            exp_name=exp_name,
            learning_rate=learning_rate,
        )
    else:
        print(f'请设置有效的 exp_name 实验名称')
