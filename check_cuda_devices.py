import torch

num_devices = torch.cuda.device_count()
print(f"There are {num_devices} cuda devices:")

for i in range(num_devices):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")