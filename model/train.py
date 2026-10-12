import os

import dotenv

dotenv.load_dotenv()

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins.io import AsyncCheckpointIO

from model.occ_gen import OccGen
from model.trainer import find_latest_version
from model.trainer.callback.ema import EMACallback
from model.trainer.callback.visualizer import Occ4DGenVisualizer
from model.trainer.data_module import TextToOccDataModule


def main(
        exp_name: str,
        training_stage: str,
        learning_rate: float,
        load_from_ckpt: str | None
):
    if os.getenv('BATCH_SIZE'):
        batch_size = int(os.getenv('BATCH_SIZE'))
    else:
        batch_size = 4

    data_module = TextToOccDataModule(
        uni_occ_dataset_path=os.getenv('DATA_PATH_1'),
        occ_collect_dataset_path=os.getenv('DATA_PATH_2'),
        ttsg_dataset_path=os.getenv('DATA_PATH_3'),
        resize_cfg={
            'origin': 200,
            'after': 224,
        },
        batch_size=batch_size,
        num_workers=int(os.getenv('DATA_WORKER_NUM', 4)),
    )

    if os.getenv('LOG_DIR'):
        log_dir = os.getenv('LOG_DIR')
    else:
        log_dir = "pl_logs"

    experiment_name = exp_name

    total_steps = int(os.getenv('TOTAL_STEPS', 100_0000))
    check_every_steps = int(os.getenv('CHECK_EVERY_STEPS', 5000))
    log_every_n_steps = int(os.getenv('LOG_EVERY_STEPS', 50))

    # total_steps = 100
    # check_every_steps = 50
    # log_every_n_steps = 1

    call_backs = [
        EMACallback(
            submodule_names=['mmdit', ]
            # device='cpu',  # 如果显存不够则设置为 cpu
        ),
        ModelCheckpoint(
            filename='model-step={step}-ema',
            every_n_train_steps=check_every_steps,  # 每 n 步保存一次
            save_top_k=int(os.getenv('STEP_CKPT_SAVE_NUM', 1)),  # 保存多少个模型
            monitor='step',
            mode='max',
            save_last=True
        ),
        ModelCheckpoint(
            filename='model-step={step}-ema',
            every_n_epochs=int(os.getenv('EPOCH_CKPT_SAVE_EVERY_N_EPOCH', 1)),
            save_top_k=int(os.getenv('EPOCH_CKPT_SAVE_NUM', 2)),  # 保存多少个模型
            monitor='step',
            mode='max',
            save_last=False
        )
    ]

    if not os.getenv("DATA_NO_STATIC"):
        call_backs += [
            # 绘制给定的单帧场景
            Occ4DGenVisualizer(
                tag='_static',
                plot_mode='video',
                gen_frames=4,
                every_n_steps=check_every_steps,
                sample_prompts=[
                    '',  # cfg

                    # from 数据集
                    'The density of vehicles is very low, with only a single car present on the road ahead.',
                    'The density of vehicles is high in the rear-right quadrant, with cars closely packed together on the road surface.',

                    # 位置控制
                    # 'There is a bus parked on the left side of the road',
                    # 'There is a bus parked on the right side of the road',

                    # 数据集中的场景描描述
                    'A scene of a straight road in a mixed urban and natural environment.',
                    'A scene of a complex, multi-lane intersection in a dense urban environment.',
                    # 'A scene of a multi-lane road in a dense urban environment, with the ego-vehicle positioned at an intersection or a complex junction.',

                ],
            )
        ]

    if not os.getenv("DATA_NO_INTERACT_ENV"):
        call_backs += [
            # 绘制序列描述
            Occ4DGenVisualizer(
                tag='_interact_with_environment',
                plot_mode='video',
                gen_frames=16,
                every_n_steps=check_every_steps,
                sample_prompts=[
                    'The ego vehicle is passing through a large intersection.',
                    'The ego vehicle is executing a left turn at a wide, four-way urban intersection.',
                    'The ego vehicle executes a right turn from a curved road onto a straight, two-lane road',
                    'The ego vehicle turns left from a T intersection.',
                ],
            ),
        ]

    if not os.getenv("DATA_NO_INTERACT_AGENT"):
        call_backs += [
            Occ4DGenVisualizer(
                tag='_interact_with_agent',
                plot_mode='video',
                gen_frames=16,
                every_n_steps=check_every_steps,
                sample_prompts=[
                    'A pedestrian crosses the road in front of the ego vehicle.',
                    'The ego vehicle follows a large truck, necessitating caution to avoid a potential collision.',
                    'Two vehicles ahead collide.',
                    'A car suddenly cut in.',
                    'The ego vehicle was overtaken',
                ],
            ),
        ]

    # 将 training_stage 和 learning_rate 传递给模型
    pipeline = OccGen(
        max_frames=32,
        seed=42,
        training_stage=training_stage,
        lr=learning_rate
    )

    trainer = pl.Trainer(
        # 正常训练 | 使用 step 控制训练流程
        max_steps=int(total_steps),
        max_epochs=-1,  # 显式禁用 epoch 限制
        num_nodes=int(os.getenv('NODE_NUM', 1)),
        devices=torch.cuda.device_count(),
        callbacks=call_backs,

        # enable_checkpointing=False,

        # 测试全流程
        # fast_dev_run=2,
        # devices=torch.cuda.device_count(),
        # callbacks=call_backs,

        accumulate_grad_batches=int(os.getenv("ACCUMULATE_GRAD_BATCHES", 1)),
        gradient_clip_val=1.0,
        precision='bf16-mixed',
        accelerator="gpu",
        log_every_n_steps=log_every_n_steps,
        logger=pl.loggers.TensorBoardLogger(
            log_dir,
            name=experiment_name,
        ),
        plugins=[
            # 异步保存 checkpoint
            AsyncCheckpointIO()
        ],
        strategy="ddp_find_unused_parameters_true"
    )

    print("--- Starting training ---")
    print(f"Experiment Name: {exp_name}")
    print(f"Training Stage:  {training_stage.upper()}")
    print(f"Learning Rate:   {learning_rate}")

    # 自动查找最新的 checkpoint
    latest_ckpt_path = find_latest_version(log_dir, experiment_name)
    if not latest_ckpt_path and load_from_ckpt:
        print(f'加载权重 {load_from_ckpt}')
        ckpt = torch.load(
            load_from_ckpt,
            map_location="cpu"
        )
        pipeline.load_state_dict(ckpt['state_dict'])
    trainer.fit(pipeline, datamodule=data_module, ckpt_path=latest_ckpt_path)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')

    # --- 通过环境变量设置实验参数 ---
    exp_name = os.getenv("EXP_NAME")
    training_stage = os.getenv("TRAINING_STAGE", "stage1").lower()

    # 为不同阶段设置不同的学习率
    default_lr = 1e-3 if training_stage == 'stage1' else 5e-5
    learning_rate = float(os.getenv("LR", default_lr))
    # 显式指定要加载的 checkpoint 路径，用于启动第二阶段训练
    load_from_ckpt = os.getenv("LOAD_FROM_CKPT")

    if exp_name:
        main(
            exp_name=exp_name,
            training_stage=training_stage,
            learning_rate=learning_rate,
            load_from_ckpt=load_from_ckpt,
        )
    else:
        print(f'请设置有效的 exp_name 实验名称')
