"""
PyTorch cuda device check
"""
import torch
import os
import sys

print("PyTorch Cuda Device Check")
print("-------------------------")
print("torch.cuda.is_available()          =", torch.cuda.is_available())
print("torch.cuda.device_count()          =", torch.cuda.device_count())
print(
    "os.environ['CUDA_VISIBLE_DEVICES'] =", os.environ.get("CUDA_VISIBLE_DEVICES", "")
)
dev = torch.device("cuda:0")
print("Trying to allocate memory on device:", dev)
linear = torch.nn.Linear(1000, 1000).to(dev)
inp = input()
print(inp)
