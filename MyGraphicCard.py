import torch
print(torch.cuda.is_available())  # Should return True if CUDA (GPU) is available
print(torch.cuda.device_count())  # Number of GPUs available
print(torch.cuda.get_device_name(0))  # Name of the first GPU
