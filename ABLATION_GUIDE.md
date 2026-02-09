# LTGQ++ 消融实验指南 (Ablation Study Guide)

## 概述 (Overview)

`train_ablation.py` 是基于 `train` 文件创建的完整消融实验版本。该文件包含原版训练代码的所有功能，并新增了三个消融开关，用于系统性地评估 LTGQ++ Fusion 模型中各个关键模块的贡献。

## 消融模块 (Ablation Modules)

### 1. QFM (Quantum Fusion Module) - 量子融合模块
**作用**: 通过迭代融合实体、关系和时间嵌入，动态学习三者的交互权重

**消融效果** (`--disable_qfm`):
- 跳过所有 QFM 迭代融合层
- 使用均匀权重 (1/3, 1/3, 1/3) 代替学习的融合权重
- 保留原始的 e, r_emb, tau 向量不变

### 2. DTM (Dynamic Temporal Modulation) - 动态时间调制模块
**作用**: 通过旋转机制对基础融合向量进行时间感知的调制

**消融效果** (`--disable_dtm`):
- 跳过三级旋转调制
- 直接使用基础融合向量 x 代替调制后的 x_o

### 3. History Module - 历史依赖模块
**作用**: 整合历史交互信息，包括:
- 双尺度 Transformer (短期/长期历史)
- Historical CopyNet
- 双层门控机制 (关系-时间门控 + 时间门控)

**消融效果** (`--disable_history`):
- 完全跳过历史向量构建
- 跳过 CopyNet 和所有门控层
- 直接从 x_o 生成查询向量 q

**注意**: ri, ti 仍然会正常返回用于打分函数 (score) 中的 translation 和 rotation 计算

## 使用方法 (Usage)

### 基础命令

#### 对照组 (完整模型)
```bash
python train_ablation.py \
    --train_path ./data/ICEWS14/train.csv \
    --valid_path ./data/ICEWS14/valid.csv \
    --test_path ./data/ICEWS14/test.csv \
    --save_dir ./results/full \
    --epochs 100 \
    --batch_size 400
```

#### 消融实验 A: 去除 QFM
```bash
python train_ablation.py \
    --disable_qfm \
    --train_path ./data/ICEWS14/train.csv \
    --valid_path ./data/ICEWS14/valid.csv \
    --test_path ./data/ICEWS14/test.csv \
    --save_dir ./results/wo_qfm \
    --epochs 100 \
    --batch_size 400
```

#### 消融实验 B: 去除 DTM
```bash
python train_ablation.py \
    --disable_dtm \
    --train_path ./data/ICEWS14/train.csv \
    --valid_path ./data/ICEWS14/valid.csv \
    --test_path ./data/ICEWS14/test.csv \
    --save_dir ./results/wo_dtm \
    --epochs 100 \
    --batch_size 400
```

#### 消融实验 C: 去除历史依赖
```bash
python train_ablation.py \
    --disable_history \
    --train_path ./data/ICEWS14/train.csv \
    --valid_path ./data/ICEWS14/valid.csv \
    --test_path ./data/ICEWS14/test.csv \
    --save_dir ./results/wo_history \
    --epochs 100 \
    --batch_size 400
```

### 组合消融实验

#### 去除 QFM + DTM
```bash
python train_ablation.py \
    --disable_qfm \
    --disable_dtm \
    --save_dir ./results/wo_qfm_dtm
```

#### 去除所有模块 (仅保留基础嵌入)
```bash
python train_ablation.py \
    --disable_qfm \
    --disable_dtm \
    --disable_history \
    --save_dir ./results/baseline
```

## 技术细节 (Technical Details)

### 代码修改位置

1. **LTGQ.__init__** (第 400 行)
   - 新增参数: `disable_qfm=False, disable_dtm=False, disable_history=False`
   - 在初始化时存储为实例变量 (第 412-414 行)

2. **encode() 方法** (第 582-640 行)
   - 第 593-603 行: QFM 消融逻辑
   - 第 605-611 行: DTM 消融逻辑
   - 第 613-626 行: History 消融逻辑

3. **main() 函数** (第 1098-1100 行)
   - 将命令行参数传递给 LTGQ 构造函数

4. **argparse** (第 1343-1345 行)
   - 添加三个命令行参数

### 消融实现原理

**完整数据流 (无消融)**:
```
输入 → 基础嵌入 (e, r, tau) → QFM融合 (ei, ri, ti, thetas) → DTM旋转 (x_o) 
→ 历史融合 (x_hist) → 门控 (q) → 输出
```

**QFM消融**:
```
输入 → 基础嵌入 (e, r, tau) → [跳过QFM] → 均匀thetas → DTM旋转 (x_o) 
→ 历史融合 (x_hist) → 门控 (q) → 输出
```

**DTM消融**:
```
输入 → 基础嵌入 (e, r, tau) → QFM融合 (ei, ri, ti, thetas) → [跳过DTM, x_o=x] 
→ 历史融合 (x_hist) → 门控 (q) → 输出
```

**History消融**:
```
输入 → 基础嵌入 (e, r, tau) → QFM融合 (ei, ri, ti, thetas) → DTM旋转 (x_o) 
→ [跳过历史, q=MLP(x_o)] → 输出
```

## 预期效果 (Expected Results)

根据消融实验的理论预期:

1. **去除 QFM**: MRR 预计下降 2-5%
   - 失去了实体-关系-时间的动态融合能力
   - 使用固定权重降低了模型表达能力

2. **去除 DTM**: MRR 预计下降 3-6%
   - 失去了时间感知的旋转调制
   - 时间信息的利用效率下降

3. **去除 History**: MRR 预计下降 5-10%
   - 失去了历史交互模式的建模能力
   - 特别影响具有丰富历史的实体预测

## 评估指标 (Evaluation Metrics)

所有实验将输出以下指标:
- **MRR** (Mean Reciprocal Rank): 平均倒数排名
- **H@1** (Hits@1): Top-1 准确率
- **H@3** (Hits@3): Top-3 准确率
- **H@10** (Hits@10): Top-10 准确率

## 文件完整性保证

`train_ablation.py` 包含原版 `train` 文件的所有代码:
- ✅ 所有 Utils 函数 (set_seed, parse_date, load_csv, build_mappings 等)
- ✅ Dataset 和 collate_fn
- ✅ MultiHeadHistoryAttn 和 HistoricalCopyNet
- ✅ TriAff, CNN1x1, QFMBlock, RotationDTM
- ✅ EMA, FGM, NegCache
- ✅ 所有 Loss 函数 (compute_loss_adv, add_embedding_l2, rdrop_kl 等)
- ✅ score_all_entities_chunked, build_freq_tensors
- ✅ train_step, evaluate_with_scope
- ✅ build_inbatch_neg
- ✅ 所有 argparse 参数 + 3 个新增消融参数

**文件统计**:
- 原版 train: 1324 行
- train_ablation.py: 1349 行 (增加 25 行用于消融逻辑)

## 依赖要求 (Dependencies)

与原版 train 文件相同:
```
torch>=1.12.0
numpy>=1.21.0
pandas>=1.3.0
tqdm
```

## 故障排查 (Troubleshooting)

### 问题: 消融开关不生效
**解决方案**: 检查是否正确传递了命令行参数，使用 `--disable_qfm` 而非 `--disable-qfm`

### 问题: OOM (Out of Memory)
**解决方案**: 
- 减小 `--batch_size`
- 减小 `--dim`
- 使用 `--use_bf16` 启用混合精度训练

### 问题: 训练过慢
**解决方案**:
- 增加 `--num_workers`
- 减小 `--history_size`
- 使用 `--use_amp` 或 `--use_bf16`

## 引用 (Citation)

如果使用此消融实验代码，请引用原始 LTGQ++ 模型论文。

## 许可 (License)

与原版 train 文件保持相同许可。
