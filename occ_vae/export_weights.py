"""Export a trained OccSeqCausalVAE Lightning checkpoint for release.

Strips the ``model.`` prefix from the raw and EMA state dicts and attaches the
latent scale factors measured by ``measure_scale_factor.py``. The output is
the "整理好" format consumed by ``get_vae(use_scale=True, ckpt_path=...)`` and
shipped with the release.

Usage (from the repo root):

    python -m occ_vae.export_weights \
        --ckpt pl_logs/.../best-IoU-....ckpt \
        --mean -0.032686 --std 0.361309 \
        --ema-mean -0.031684 --ema-std 0.361601 \
        --output dome_vae.ckpt
"""
import argparse

import torch


def renew_state_dict(raw_state_dict):
    state_dict = {}
    for name, weight in raw_state_dict.items():
        # 移除 name 前面的 model
        name_new = name[6:]
        state_dict[name_new] = weight
    return state_dict


def main():
    parser = argparse.ArgumentParser(description='Export release VAE checkpoint with scale factors')
    parser.add_argument('--ckpt', required=True, help='Lightning checkpoint to export')
    parser.add_argument('--mean', type=float, required=True,
                        help='latent mean of the raw state_dict (from measure_scale_factor)')
    parser.add_argument('--std', type=float, required=True)
    parser.add_argument('--ema-mean', type=float, required=True,
                        help='latent mean of the EMA state_dict')
    parser.add_argument('--ema-std', type=float, required=True)
    parser.add_argument('--output', required=True, help='output checkpoint path')
    args = parser.parse_args()

    full_ckpt = torch.load(args.ckpt, map_location='cpu')

    save_ckpt = {
        'state_dict': renew_state_dict(full_ckpt['state_dict']),
        'state_dict_scale_factor': {  # 需要统计后获取
            'mean': args.mean,
            'std': args.std
        },

        'ema_state_dict': renew_state_dict(full_ckpt['ema_state_dict']),
        'ema_state_dict_scale_factor': {
            'mean': args.ema_mean,
            'std': args.ema_std
        }
    }

    torch.save(save_ckpt, args.output)
    print(f'saved {args.output}')


if __name__ == '__main__':
    main()
