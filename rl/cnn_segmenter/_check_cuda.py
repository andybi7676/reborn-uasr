import torch

print("CUDA available: ", torch.cuda.is_available())
print("CUDA version: ", torch.version.cuda)
print("cuDNN version: ", torch.backends.cudnn.version())
print("CUDA device count: ", torch.cuda.device_count())
print("CUDA device name: ", torch.cuda.get_device_name(0))
print("CUDA device: ", torch.cuda.current_device())
print("CUDA capability: ", torch.cuda.get_device_capability(0))
print("CUDA memory allocated: ", torch.cuda.memory_allocated())
print("CUDA memory cached: ", torch.cuda.memory_reserved())
print("CUDA memory reserved: ", torch.cuda.memory_reserved(0))
