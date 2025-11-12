# 对比方法开源代码汇总

本文件夹包含PDPL-Net论文中所有对比方法的开源代码和实现说明。

---

## 📊 **对比方法列表(12个)**

### ✅ **已成功下载的方法**

#### 1. Point Transformer V3 (CVPR 2024) ⭐⭐⭐
- **状态**: ✅ 已下载
- **文件夹**: `PointTransformerV3/`
- **GitHub**: https://github.com/Pointcept/PointTransformerV3
- **Star数**: 1.5k+
- **论文**: Wu et al., "Point Transformer V3: Simpler, Faster, Stronger", CVPR 2024 Oral
- **说明**: 官方PyTorch实现,需要修改输出层为对偶变量(μ, λ)
- **实现难度**: ⭐⭐ (较简单)
- **预计实现时间**: 6-8小时

---

### ⚠️ **需要手动下载的方法(网络连接失败)**

#### 2. PointNet++ (NeurIPS 2017) ⭐⭐⭐
- **状态**: ❌ 下载失败(网络问题)
- **GitHub**: https://github.com/yanx27/Pointnet_Pointnet2_pytorch
- **Star数**: 3.2k+
- **论文**: Qi et al., "PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space", NeurIPS 2017
- **说明**: 最流行的PointNet++实现,需要修改输出层
- **实现难度**: ⭐⭐ (较简单)
- **预计实现时间**: 4-6小时
- **下载命令**: 
  ```bash
  git clone https://github.com/yanx27/Pointnet_Pointnet2_pytorch.git
  ```

#### 3. CvxpyLayers (NeurIPS 2019) ⭐⭐⭐
- **状态**: ❌ 下载失败(网络问题)
- **GitHub**: https://github.com/cvxpy/cvxpylayers
- **Star数**: 1.8k+
- **论文**: Agrawal et al., "Differentiable Convex Optimization Layers", NeurIPS 2019
- **说明**: 可微分凸优化层,已在您的项目中使用(见requires.txt)
- **实现难度**: ⭐⭐⭐ (中等)
- **预计实现时间**: 6-8小时
- **下载命令**: 
  ```bash
  git clone https://github.com/cvxpy/cvxpylayers.git
  ```
- **安装命令**:
  ```bash
  pip install cvxpylayers
  ```

#### 4. DeepInverse (2024) ⭐⭐⭐
- **状态**: ❌ 下载失败(网络问题)
- **GitHub**: https://github.com/deepinv/deepinv
- **Star数**: 500+
- **论文**: Hurault et al., "DeepInverse: A Deep Learning Library for Inverse Problems", 2024
- **说明**: 包含ADMM、ISTA等多种算法展开实现
- **实现难度**: ⭐⭐ (较简单)
- **预计实现时间**: 4-6小时
- **下载命令**: 
  ```bash
  git clone https://github.com/deepinv/deepinv.git
  ```
- **安装命令**:
  ```bash
  pip install deepinv
  ```

#### 5. KKThPINN (Physics-Informed Hard Projection, 2025) ⭐⭐⭐
- **状态**: ❌ 下载失败(网络问题)
- **GitHub**: https://github.com/li-group/kkthpinn
- **论文**: Li et al., "Physics-informed neural networks with hard nonlinear equality and inequality constraints", Computers & Chemical Engineering, 2025
- **说明**: 硬约束投影方法的参考实现
- **实现难度**: ⭐⭐ (较简单)
- **预计实现时间**: 4-6小时
- **下载命令**: 
  ```bash
  git clone https://github.com/li-group/kkthpinn.git
  ```

#### 6. Voxblox (ESDF-MPC, IROS 2017) ⭐⭐
- **状态**: ⏳ 正在下载中...
- **GitHub**: https://github.com/ethz-asl/voxblox
- **Star数**: 1.2k+
- **论文**: Oleynikova et al., "Voxblox: Incremental 3D Euclidean Signed Distance Fields for On-Board MAV Planning", IROS 2017
- **说明**: C++实现,需要ROS环境,或者用Python重新实现ESDF构建
- **实现难度**: ⭐⭐⭐ (中等)
- **预计实现时间**: 1-2天
- **下载命令**: 
  ```bash
  git clone https://github.com/ethz-asl/voxblox.git
  ```

---

### 📝 **需要自己实现的方法(无现成开源代码)**

#### 7. 标准MLP ⭐⭐⭐
- **状态**: ⚠️ 需要自己实现
- **说明**: 简单的多层感知机,使用PyTorch内置nn.Linear
- **实现难度**: ⭐ (非常简单)
- **预计实现时间**: 1-2小时
- **参考代码**: 见下方"实现代码模板"

#### 8. 中心点距离-MPC ⭐⭐
- **状态**: ⚠️ 需要自己实现
- **说明**: 计算点云中心点,然后用中心点距离作为避障约束
- **实现难度**: ⭐⭐ (较简单)
- **预计实现时间**: 2-4小时
- **参考代码**: 见下方"实现代码模板"

#### 9. ISTA展开 ⭐⭐⭐
- **状态**: ✅ 可通过DeepInverse库使用
- **GitHub**: https://github.com/deepinv/deepinv
- **论文**: Gregor & LeCun, "Learning Fast Approximations of Sparse Coding", ICML 2010
- **说明**: DeepInverse库包含ISTA/FISTA展开实现,可以直接使用
- **实现难度**: ⭐⭐ (较简单)
- **预计实现时间**: 4-6小时
- **安装命令**:
  ```bash
  pip install deepinv
  ```
- **使用方式**:
  ```python
  from deepinv.unfolded import ISTA
  # 直接使用ISTA展开层
  ```

#### 10. ADMM展开 ⭐⭐⭐
- **状态**: ✅ 可通过DeepInverse库使用
- **GitHub**: https://github.com/deepinv/deepinv
- **论文**: Yang et al., "Deep ADMM-Net for Compressive Sensing MRI", CVPR 2016
- **说明**: DeepInverse库包含ADMM展开实现,可以直接使用
- **实现难度**: ⭐⭐ (较简单)
- **预计实现时间**: 4-6小时
- **安装命令**:
  ```bash
  pip install deepinv
  ```
- **使用方式**:
  ```python
  from deepinv.unfolded import ADMM
  # 直接使用ADMM展开层
  ```
- **DeepInverse包含的其他算法展开**:
  - ✅ PGD (Proximal Gradient Descent)
  - ✅ HQS (Half Quadratic Splitting)
  - ✅ RED (Regularization by Denoising)
  - ✅ FISTA (Fast ISTA)

#### 11. CVXPY求解器 ⭐⭐⭐
- **状态**: ✅ 可直接使用(已在项目中安装)
- **说明**: 使用CVXPY+CLARABEL求解器计算对偶变量真值
- **实现难度**: ⭐ (非常简单)
- **预计实现时间**: 1-2小时
- **参考代码**: 见下方"实现代码模板"

#### 12. NeuPAN ⭐⭐⭐
- **状态**: ✅ 已实现(您的baseline)
- **说明**: 您现有的代码库
- **实现难度**: ⭐ (已完成)
- **预计实现时间**: 0小时

---

## 📋 **下载状态总结**

| 方法 | 开源状态 | 下载状态 | 实现方式 |
|------|---------|---------|---------|
| Point Transformer V3 | ✅ 开源 | ✅ 已下载 | 修改官方代码 |
| PointNet++ | ✅ 开源 | ❌ 需手动下载 | 修改官方代码 |
| CvxpyLayers | ✅ 开源 | ❌ 需手动下载 | 使用库+自己实现 |
| DeepInverse | ✅ 开源 | ❌ 需手动下载 | 使用库 |
| KKThPINN | ✅ 开源 | ❌ 需手动下载 | 参考实现 |
| Voxblox | ✅ 开源 | ⏳ 下载中 | 重新实现(Python) |
| 标准MLP | ⚠️ 无需开源 | - | 自己实现 |
| 中心点距离-MPC | ⚠️ 无需开源 | - | 自己实现 |
| **ISTA展开** | ✅ **DeepInverse** | ✅ **可用** | **直接使用库** |
| **ADMM展开** | ✅ **DeepInverse** | ✅ **可用** | **直接使用库** |
| CVXPY求解器 | ✅ 已安装 | ✅ 可用 | 直接使用 |
| NeuPAN | ✅ 已实现 | ✅ 可用 | 直接使用 |

**统计**:
- ✅ 已下载/可用: **5个** (Point Transformer V3, ISTA, ADMM, CVXPY, NeuPAN)
- ⏳ 下载中: 1个 (Voxblox)
- ❌ 需手动下载: 4个 (PointNet++, CvxpyLayers, DeepInverse, KKThPINN)
- ⚠️ 需自己实现: 2个 (标准MLP, 中心点距离-MPC)

---

## 🚀 **快速开始**

### **1. 手动下载所有开源代码**

在`baseline_methods/`文件夹下执行:

```bash
# PointNet++
git clone https://github.com/yanx27/Pointnet_Pointnet2_pytorch.git

# CvxpyLayers
git clone https://github.com/cvxpy/cvxpylayers.git

# DeepInverse
git clone https://github.com/deepinv/deepinv.git

# KKThPINN
git clone https://github.com/li-group/kkthpinn.git

# Voxblox (如果未完成)
git clone https://github.com/ethz-asl/voxblox.git

# ISTA参考实现
git clone https://github.com/amzn/sparse-vqvae.git
```

### **2. 安装必要的Python包**

```bash
pip install cvxpylayers
pip install deepinv
```

### **3. 实现自定义方法**

参考下方"实现代码模板"部分。

---

## 💻 **实现代码模板**

所有自定义实现的代码将放在`baseline_methods/custom_implementations/`文件夹中。

### **文件夹结构**:

```
baseline_methods/
├── README.md (本文件)
├── PointTransformerV3/ (已下载)
├── Pointnet_Pointnet2_pytorch/ (待下载)
├── cvxpylayers/ (待下载)
├── deepinv/ (待下载)
├── kkthpinn/ (待下载)
├── voxblox/ (下载中)
├── sparse-vqvae/ (待下载,ISTA参考)
└── custom_implementations/ (自定义实现)
    ├── __init__.py
    ├── mlp_baseline.py (标准MLP)
    ├── center_distance_mpc.py (中心点距离-MPC)
    ├── ista_unrolling.py (ISTA展开)
    ├── admm_unrolling.py (ADMM展开)
    ├── cvxpy_solver.py (CVXPY求解器)
    └── README.md
```

---

## 📝 **下一步工作**

### **优先级1: 手动下载开源代码**
1. ✅ Point Transformer V3 (已完成)
2. ⏳ Voxblox (下载中)
3. ❌ PointNet++ (需手动下载)
4. ❌ CvxpyLayers (需手动下载)
5. ❌ DeepInverse (需手动下载)
6. ❌ KKThPINN (需手动下载)

### **优先级2: 实现自定义方法**
1. CVXPY求解器 (1-2小时)
2. 标准MLP (1-2小时)
3. 中心点距离-MPC (2-4小时)
4. ~~ISTA展开~~ → 使用DeepInverse库 (已包含)
5. ~~ADMM展开~~ → 使用DeepInverse库 (已包含)

### **优先级3: 适配开源代码**
1. Point Transformer V3 → 对偶变量预测 (6-8小时)
2. PointNet++ → 对偶变量预测 (4-6小时)
3. CvxpyLayers → 对偶变量求解 (6-8小时)
4. DeepInverse → ISTA/ADMM展开适配 (4-6小时) ⭐ **已包含ISTA和ADMM**
5. KKThPINN → 硬投影参考 (4-6小时)
6. Voxblox → Python ESDF实现 (1-2天)

---

## ⚠️ **注意事项**

1. **网络问题**: 如果GitHub下载失败,可以:
   - 使用代理
   - 使用GitHub镜像站(如gitee.com)
   - 手动下载ZIP文件

2. **依赖冲突**: 不同方法可能有不同的依赖版本,建议:
   - 为每个方法创建独立的conda环境
   - 或者使用Docker容器隔离

3. **C++代码**: Voxblox是C++实现,需要:
   - 安装ROS环境
   - 或者用Python重新实现ESDF构建算法

---

## 📧 **联系方式**

如有问题,请联系项目维护者。

---

**最后更新**: 2025-11-12

