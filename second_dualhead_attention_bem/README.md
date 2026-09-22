# Dual-Head Full-Graph Attention VAE + BEM

这是第二套代码的 GitHub 整理版。

核心目标保持第二套原有任务与双输出结构不变，同时把第一套代码中的
`FullGraphAttentionConv` 注意力消息传递机制迁移到第二套网络中。

## 1. 网络结构

第二套原有结构保留为：

```text
9-D node input
      │
      ▼
3 × shared FullGraphAttention encoder
      │
      ├── mu
      └── logvar
           │
           ▼
        latent z
           │
     concat [x, z]
           │
     ┌─────┴─────┐
     ▼           ▼
3 × Attention  3 × Attention
T decoder      q decoder
     │           │
     ▼           ▼
pred_T_unknown  pred_q_unknown
```

保留的第二套设计：

- 3 层共享 Encoder
- VAE `mu / logvar / reparameterization`
- 两个互相独立的 3 层 Decoder
- `T` 与 `q` 两个输出
- 只在未知边界位置计算监督误差
- 已知边界量通过 `assemble_full_fields()` 原样覆盖回来
- 原有训练、测试和 BEM 内部场恢复流程

迁移自第一套的注意力机制包括：

- multi-head Query / Key / Value
- source-target relational features
- edge-dependent attention bias
- edge-dependent message gate
- target-wise softmax
- residual branch

第二套生成的数据仍保留 `edge_weight` 字段以兼容原数据格式；新的注意力层不再使用固定的距离 GCN 权重，而由网络学习边权重。

第二套原图生成代码包含 self-loop。为了与第一套注意力实现一致，`part4_model.py`
会在 Attention 内部自动过滤 self-loop，并由 residual branch 保留节点自身信息。
因此旧的第二套数据集也可以继续读取，不需要仅为了 self-loop 重新改数据格式。

## 2. 项目结构

```text
second_dualhead_attention_bem/
├── README.md
├── requirements.txt
├── .gitignore
├── run_all.py
├── part1_bem_solver.py
├── part2_build_graph.py
├── part3_dataset.py
├── part4_model.py
├── part5_train.py
├── part6_test.py
└── extras/
    └── cloud_shape_demo.py
```

### 文件说明

- `part1_bem_solver.py`
  - 二维 Laplace 常数单元 BEM
  - 生成矩形混合 Dirichlet/Neumann 边界问题
  - 提供真实边界解与内部场计算

- `part2_build_graph.py`
  - 生成固定矩形边界图
  - 节点输入：
    `[x, y, nx, ny, length, is_T_known, is_q_known, known_T, known_q]`
  - 保存 train / val / test 三个单图样本

- `part3_dataset.py`
  - PyTorch Geometric 数据读取
  - 当前任务建议 `batch_size=1`

- `part4_model.py`
  - 第一套 `FullGraphAttentionConv`
  - 第二套 3 层共享 Encoder
  - VAE latent
  - 双 3 层 Attention Decoder
  - `T/q` 双输出
  - loss 与完整边界恢复函数

- `part5_train.py`
  - 单图监督训练
  - 未知 T / q 监督损失
  - VAE KL 正则
  - 保存 best / last checkpoint 和训练曲线

- `part6_test.py`
  - 加载最佳 checkpoint
  - 恢复完整 T / q 边界场
  - 用 BEM 计算域内场
  - 输出边界与域内误差图及 JSON 指标

- `run_all.py`
  - 一条命令串联数据生成、训练和测试

- `extras/cloud_shape_demo.py`
  - 原第二套附带的独立云形边界绘图示例
  - 不参与主训练流程

## 3. 环境

建议：

- Python 3.10+
- PyTorch
- PyTorch Geometric
- NumPy
- Matplotlib
- SciPy

创建环境后安装：

```bash
pip install -r requirements.txt
```

如果需要 CUDA，请先按照 PyTorch 官方安装方式安装与你 CUDA 版本匹配的 PyTorch，
然后再运行：

```bash
pip install torch-geometric
pip install -r requirements.txt
```

## 4. 运行方式

### 方式 A：逐步运行

生成数据：

```bash
python part2_build_graph.py
```

训练：

```bash
python part5_train.py
```

测试：

```bash
python part6_test.py
```

默认输出目录：

```text
dataset_rect_bem_gnn/
artifacts_train/
artifacts_test/
```

### 方式 B：一条命令全部运行

```bash
python run_all.py
```

例如先用较少 epoch 检查流程：

```bash
python run_all.py --epochs 20
```

正式训练：

```bash
python run_all.py --epochs 1500
```

复用已经生成的数据：

```bash
python run_all.py --skip-data
```

复用已经训练好的 `best_model.pt`，只测试：

```bash
python run_all.py --skip-data --skip-train
```

## 5. 默认训练参数

```text
hidden_channels  = 128
latent_channels  = 32
decoder_channels = 128
dropout          = 0.05
learning rate    = 1e-3
beta_kl          = 1e-5
epochs           = 1500
```

参数主要在 `part5_train.py` 或 `run_all.py` 命令行中修改。

## 6. Checkpoint 兼容性

注意力版 `part4_model.py` 的参数结构已经不同于旧 GCN 版。

因此：

> 旧的 GCN `best_model.pt` 不能直接加载到这个 Attention 模型中。

第一次使用本仓库时请重新训练：

```bash
python part5_train.py
```

之后生成的新 `best_model.pt` 可以正常由 `part6_test.py` 加载。

## 7. 当前实验含义

当前数据生成器把同一个固定 BEM 图保存到 train / val / test。

因此当前代码更接近：

> 单个固定边值问题上的图神经网络拟合实验

而不是多个独立样本上的统计泛化实验。

如果后续需要研究泛化能力，可以继续扩展 `part2_build_graph.py`，
生成不同几何、不同边界条件或不同物理参数的多个图样本。

## 8. 上传到 GitHub

在项目目录中：

```bash
git init
git add .
git commit -m "Initial dual-head full-graph attention BEM model"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

默认 `.gitignore` 已排除：

- 数据集
- checkpoint
- 训练/测试输出
- 图片
- Python cache

如果以后希望提交大型 checkpoint，建议使用 Git LFS，而不是直接提交 `.pt` 文件。

## 9. 重要说明

本项目没有自动添加开源 LICENSE。若仓库准备公开，请根据你的发布需求自行选择
MIT、Apache-2.0、GPL 等许可证。
