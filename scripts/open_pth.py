import torch
import os

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 构建正确的权重文件路径
weight_file = os.path.join(project_root, "pretrained_weights", "breaking_bad", "predator_breaking_bad_beerbottle_best.pth")

# 检查文件是否存在
if not os.path.exists(weight_file):
    print(f"错误：文件不存在 - {weight_file}")
    print("可用的权重文件:")
    for root, dirs, files in os.walk(os.path.join(project_root, "pretrained_weights")):
        for file in files:
            print(f"  - {os.path.join(root, file)}")
else:
    # 加载文件
    data = torch.load(weight_file, map_location="cpu")
    
    # 看看里面是什么结构
    print(f"成功加载文件：{weight_file}")
    print(type(data))
    if hasattr(data, 'keys'):
        print("键名:", data.keys())  # 如果是字典，会打印键名
    
    # 打印具体内容
    print(data)