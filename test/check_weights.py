import torch
import numpy as np

# 加载模型权重
ckpt = torch.load('example/model/acker_flex_pdhg_robot/model_5000.pth', map_location='cpu')

print("=" * 80)
print("模型权重分析")
print("=" * 80)
print()

# 检查checkpoint结构
if isinstance(ckpt, dict):
    print("Checkpoint 是字典，包含的键:")
    for key in ckpt.keys():
        if isinstance(ckpt[key], torch.Tensor):
            print(f"  {key}: shape={ckpt[key].shape}, dtype={ckpt[key].dtype}")
        else:
            print(f"  {key}: {type(ckpt[key])}")
    print()

# 关键问题：权重是否与 J 相关？
print("=" * 80)
print("关键问题：权重是否与 J 相关？")
print("=" * 80)
print()

# 检查 encoder 权重
if 'encoder.0.weight' in ckpt:
    encoder_w = ckpt['encoder.0.weight']
    print(f"encoder.0.weight shape: {encoder_w.shape}")
    print(f"  这是第一层的权重，与 J 无关")
    print()

# 检查 init_mu 权重
if 'init_mu.0.weight' in ckpt:
    init_mu_w = ckpt['init_mu.0.weight']
    print(f"init_mu.0.weight shape: {init_mu_w.shape}")
    print(f"  这是初始化 mu 的权重，与 J 无关")
    print()

# 检查 prox_head 权重
if 'prox_head.0.weight' in ckpt:
    prox_w = ckpt['prox_head.0.weight']
    print(f"prox_head.0.weight shape: {prox_w.shape}")
    print(f"  这是 learned-prox 的权重，在每一步都会使用")
    print()

print("=" * 80)
print("关键发现")
print("=" * 80)
print()
print("✓ 所有权重都是 step-independent（与步数无关）")
print("✓ encoder, init_mu, init_y, prox_head 的权重在每一步都可以重复使用")
print("✓ 无论 J=1 还是 J=2，都会使用相同的权重")
print()
print("但是...")
print("✗ 如果 J=1，只执行 1 步迭代")
print("✗ 如果 J=2，执行 2 步迭代")
print("✗ 迭代次数不同，最终结果会不同！")
print()
print("=" * 80)
print("算法流程（forward 方法）")
print("=" * 80)
print()
print("1. 编码输入点 (encoder)")
print("   x [N,2] -> h [N,hidden]")
print()
print("2. 初始化 mu 和 y (init_mu, init_y)")
print("   h -> mu [N,E], y [N,2]")
print()
print("3. 循环 J 次迭代 (for j in range(J)):")
print("   每次迭代执行 _step():")
print("     - 更新 y")
print("     - 更新 mu")
print("     - 如果 use_learned_prox=True，使用 prox_head 调整 mu")
print("     - 投影 mu 保证可行性")
print()
print("4. 最终投影")
print("   mu = _project_mu_row(mu, G)")
print()
print("=" * 80)
print("结论")
print("=" * 80)
print()
print("✓ 是的，FlexiblePDHGFront 使用模型权重")
print("✓ 权重包括: encoder, init_mu, init_y, prox_head")
print("✓ 这些权重在每一步都会被使用")
print()
print("✗ 但是，如果 J=1 而模型训练时 J=2:")
print("  - 权重会被加载（✓）")
print("  - 但只执行 1 步，而不是 2 步（✗）")
print("  - 这会导致结果与训练时不同（✗）")
print()
print("关键区别:")
print("  J=1: encoder -> init -> 1步迭代 -> 投影 -> 输出")
print("  J=2: encoder -> init -> 2步迭代 -> 投影 -> 输出")
print()
print("权重相同，但迭代次数不同，所以结果不同！")

