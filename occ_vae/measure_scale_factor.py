"""Measure the latent scale factors of a trained OccSeqCausalVAE checkpoint.

Runs predict over the training set to accumulate mean/std of the latents
(CalcScaleFactorCallback) and a test pass for IoU metrics, once for the raw
state_dict and once for the EMA weights. The measured mean/std feed
export_weights.py as the scale factors attached to the released checkpoint
(and get_vae(use_scale=True, ...)).

Usage (from the repo root):

    python -m occ_vae.measure_scale_factor \
        --data-path /path/static_root --ckpt-path /path/to/best.ckpt
"""
import argparse

import torch
import pytorch_lightning as pl

from occinteract.loaders.uniocc_seq.get_dataset import get_full_uniocc_dataset
from occinteract.tools.uniocc_dataset import UniOcc
from occ_vae.trainer.data_module import BaseDataModuleV2
from occ_vae.trainer.trainer_module import TrainerSeq as Trainer
from occ_vae.trainer.callback.stats import CalcScaleFactorCallback
from occ_vae.vae import get_vae


def evaluation(pipeline, trainer, data_module):
    # 检查缩放系数
    trainer.predict(pipeline, dataloaders=data_module.train_dataloader())

    # 评估性能指标
    trainer.test(pipeline, datamodule=data_module)


def main():
    parser = argparse.ArgumentParser(description='Measure OccSeqCausalVAE latent scale factors')
    parser.add_argument('--data-path', required=True,
                        help='8-bit static root ({dataset}-{split} dirs)')
    parser.add_argument('--ckpt-path', required=True, help='Lightning checkpoint')
    parser.add_argument('--seq-len', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--devices', type=int, default=1)
    args = parser.parse_args()

    # 1. 實例化數據模塊
    data_module = BaseDataModuleV2(
        datasets=get_full_uniocc_dataset(
            seq_len=args.seq_len,
            uni_occ_path=args.data_path,
            resize_cfg={
                'origin': 200,
                'after': 224,
            },
        ),
        collate_fn=UniOcc.collate_fn,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    model = get_vae()
    ckpt = torch.load(args.ckpt_path, map_location='cpu')
    pipeline = Trainer.load_from_checkpoint(
        args.ckpt_path,
        model=model,
    )

    # 进行推理
    trainer = pl.Trainer(
        devices=args.devices,
        callbacks=[CalcScaleFactorCallback()],

        precision='bf16-mixed',
        accelerator="gpu",
        logger=False,
        enable_checkpointing=False,
    )

    print('评估 state_dict')
    pipeline.load_state_dict(ckpt['state_dict'])
    evaluation(pipeline, trainer, data_module)

    print('评估 ema')
    pipeline.load_state_dict(ckpt['ema_state_dict'])
    evaluation(pipeline, trainer, data_module)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    main()
