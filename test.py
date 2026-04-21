import torch, torchvision, torchaudio

print("=== PyTorch Packages ===")
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("torchaudio:", torchaudio.__version__)

print("\n=== CUDA Info ===")
print("CUDA available:", torch.cuda.is_available())
print("torch CUDA:", torch.version.cuda)
print("cuDNN:", torch.backends.cudnn.version())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
