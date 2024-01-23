import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import os
import argparse
import time


def run_program(n_epochs: int,
                batch_size: int,
                lr: float,
                n_cpu: int,
                img_size: int,
                output_dir: str=None):
    print('hi')

def create_output_dir() -> str:
    output_path = os.path.join('../../../', 'output')
    absolute_output_path = os.path.abspath(output_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f'Created folder at: {absolute_output_path}')
    else:
        print('Folder already exists')

    return output_path


if __name__ == '__main__':
    #output_path: str = create_output_dir()

    parser = argparse.ArgumentParser(
        prog="ViT training.",
        description="/",
    )
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--n_cpu", type=int, default=32, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
    #parser.add_argument("--output_dir", type=str, default=output_path, help="directory to save the generated images")

    args = vars(parser.parse_args())
    print(f'These are the args: {args}')

    run_program(**args)
