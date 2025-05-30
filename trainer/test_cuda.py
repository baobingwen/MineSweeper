import torch
print("PyTorch version:", torch.__version__)  # 输出 PyTorch 版本
print("CUDA version:", torch.version.cuda)  # 输出 CUDA 版本
print(torch.cuda.is_available())  # 检查 CUDA 是否可用
print(torch.version.cuda)         # 输出 PyTorch 使用的 CUDA 版本
print("CUDA Device:", torch.cuda.get_device_name(0) 
        if torch.cuda.is_available() 
        else "No CUDA device available")  # 输出 GPU 名称