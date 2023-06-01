import argparse
import torch
import torch.nn as nn


def generate_fake_pth(output_path):
    linear = nn.Linear(1, 1)
    torch.save(linear.state_dict(), output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', type=str, default='./fake.pth')

    args = parser.parse_args()
    generate_fake_pth(args.out_path)