import scipy.io as sio
import numpy as np
import os

# --- 配置路径 ---
# 修改为你实际的 Scene15.mat 路径
mat_path = '/home/wupeihan/Multi_View/data/Scene15.mat' 
# 输出保存路径
save_dir = './data/processed'
save_name = 'scene15.npz'

os.makedirs(save_dir, exist_ok=True)

# --- 加载数据 ---
print(f"Loading from {mat_path}...")
mat = sio.loadmat(mat_path)

# 根据你提供的参考代码提取 X1, X2 和 Y
# 转换为 float32 以匹配 PyTorch 默认类型
view_0 = mat['X1'].astype(np.float32)
view_1 = mat['X2'].astype(np.float32)
labels = np.squeeze(mat['Y']).astype(np.int64)

# --- 检查并修复标签 ---
# PyTorch 通常需要 0 到 N-1 的标签。如果 Scene15 是 1-15，需要转为 0-14
if labels.min() == 1:
    print("Detected 1-based labels, converting to 0-based...")
    labels = labels - 1

# 确保样本数一致
assert view_0.shape[0] == view_1.shape[0] == labels.shape[0], "样本数量不一致！"

# --- 打印维度信息 (用于填写配置文件) ---
print(f"View 0 shape: {view_0.shape}") # 记下这个维度 (例如: N, 20)
print(f"View 1 shape: {view_1.shape}") # 记下这个维度 (例如: N, 59)
print(f"Labels shape: {labels.shape}")
print(f"Unique classes: {len(np.unique(labels))}")

# --- 保存为 .npz ---
# 关键：键名必须是 view_0, view_1, labels
np.savez(os.path.join(save_dir, save_name), 
         view_0=view_0, 
         view_1=view_1, 
         labels=labels)

print(f"Saved to {os.path.join(save_dir, save_name)}")