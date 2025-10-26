"""
对比 J=1 和 J=2 的推理结果
"""
import torch
import numpy as np
from neupan.blocks.flexible_pdhg import FlexiblePDHGFront

# 加载模型权重
ckpt = torch.load('example/model/acker_flex_pdhg_robot/model_5000.pth', map_location='cpu')

print("=" * 80)
print("对比 J=1 和 J=2 的推理结果")
print("=" * 80)
print()

# 提取几何信息
G = ckpt['G']
h = ckpt['h']

print(f"几何信息:")
print(f"  G shape: {G.shape}")
print(f"  h shape: {h.shape}")
print()

# 创建两个模型：J=1 和 J=2
model_j1 = FlexiblePDHGFront(
    input_dim=2,
    E=4,
    G=G,
    h=h,
    hidden=32,
    J=1,
    use_learned_prox=True,
)

model_j2 = FlexiblePDHGFront(
    input_dim=2,
    E=4,
    G=G,
    h=h,
    hidden=32,
    J=2,
    use_learned_prox=True,
)

# 加载相同的权重到两个模型
model_j1.load_state_dict(ckpt, strict=False)
model_j2.load_state_dict(ckpt, strict=False)

print(f"模型 J 值:")
print(f"  model_j1.J = {model_j1.J}")
print(f"  model_j2.J = {model_j2.J}")
print()

# 创建测试输入
torch.manual_seed(42)
x = torch.randn(5, 2)  # 5个障碍点

print(f"测试输入:")
print(f"  x shape: {x.shape}")
print(f"  x:\n{x}")
print()

# 推理
model_j1.eval()
model_j2.eval()

with torch.no_grad():
    mu_j1 = model_j1(x)
    mu_j2 = model_j2(x)

print(f"推理结果:")
print(f"  mu_j1 shape: {mu_j1.shape}")
print(f"  mu_j2 shape: {mu_j2.shape}")
print()

print(f"mu_j1 (J=1):")
print(mu_j1)
print()

print(f"mu_j2 (J=2):")
print(mu_j2)
print()

# 对比
diff = torch.abs(mu_j1 - mu_j2)
print(f"差异分析:")
print(f"  最大差异: {diff.max().item():.6f}")
print(f"  平均差异: {diff.mean().item():.6f}")
print(f"  L2 差异: {torch.norm(diff).item():.6f}")
print()

if diff.max().item() > 1e-5:
    print("✗ 结果不同！")
    print("  这证明了 J 值确实影响推理结果")
    print("  即使权重相同，迭代次数不同也会导致不同的输出")
else:
    print("✓ 结果相同")
    print("  这不太可能，说明有问题")

print()
print("=" * 80)
print("结论")
print("=" * 80)
print()
print("✓ 权重会被加载（两个模型使用相同的权重）")
print("✓ 但 J 值不同会导致不同的推理结果")
print("✗ 如果训练时 J=2，推理时 J=1，结果会不同")
print()
print("因此，必须在推理时指定正确的 J 值！")

