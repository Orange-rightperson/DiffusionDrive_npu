# test_npu.py
import os
import torch
import torch_npu
# 确保 NPU 相关的环境变量也在这里设置，模拟 Ray Worker 的环境
os.environ["OMP_NUM_THREADS"] = "1"
#os.environ["LD_PRELOAD"] = "YOUR_LD_PRELOAD_PATH" # 替换为你的实际路径

print(f"PyTorch version: {torch.__version__}")
print(f"torch_npu version: {torch_npu.__version__}")
print(f"LD_PRELOAD: {os.environ.get('LD_PRELOAD')}")

if torch.npu.is_available():
    print(f"NPU is available. Device count: {torch.npu.device_count()}")
    try:
        # 尝试在NPU 0上创建一个张量
        x = torch.randn(2, 2).to("npu:0")
        print(f"Tensor on NPU: {x}")
        # 尝试执行一个简单计算
        y = x + x
        print(f"Simple computation on NPU successful: {y}")
    except Exception as e:
        print(f"Error during NPU tensor operation: {e}")
        import traceback
        traceback.print_exc()
else:
    print("NPU is NOT available in this environment.")

# 导入 cv2 看看是否引发 TLS 错误
try:
    import cv2
    print("cv2 imported successfully.")
except Exception as e:
    print(f"Error importing cv2: {e}")
    import traceback
    traceback.print_exc()
