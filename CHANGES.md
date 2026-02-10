# train_ablation.py 修改总结

## 文件信息
- **原文件**: train (1324 行)
- **新文件**: train_ablation.py (1349 行)
- **新增行数**: 25 行
- **修改位置**: 4 处

## 详细修改

### 修改 1: LTGQ.__init__ 参数增加 (第 400 行)

**原代码:**
```python
def __init__(self, num_entities, num_relations, time_vocab_sizes, pad_ent_id, pad_rel_id,
             dim=480, qfm_layers=3, dropout=0.12,
             pretrained_ent_emb=None, pretrained_rel_emb=None,
             hist_attn_dropout=0.12, recency_decay=0.07,
             rel_dropout=0.0, time_dropout=0.0,
             use_aux_rel=False,
             history_heads=4,
             use_adapter=False, adapter_dim=64,
             time_fuse="cat", use_sin_time=True,
             trans_weight=0.25,
             hist_tf_layers=2, hist_tf_heads=8, hist_tf_dropout=0.1,
             short_hist_size=20, long_hist_size=60):
```

**新代码:**
```python
def __init__(self, num_entities, num_relations, time_vocab_sizes, pad_ent_id, pad_rel_id,
             dim=480, qfm_layers=3, dropout=0.12,
             pretrained_ent_emb=None, pretrained_rel_emb=None,
             hist_attn_dropout=0.12, recency_decay=0.07,
             rel_dropout=0.0, time_dropout=0.0,
             use_aux_rel=False,
             history_heads=4,
             use_adapter=False, adapter_dim=64,
             time_fuse="cat", use_sin_time=True,
             trans_weight=0.25,
             hist_tf_layers=2, hist_tf_heads=8, hist_tf_dropout=0.1,
             short_hist_size=20, long_hist_size=60,
             disable_qfm=False, disable_dtm=False, disable_history=False):  # ← 新增
```

### 修改 2: 存储消融标志 (第 412-414 行)

**新增代码:**
```python
self.disable_qfm = disable_qfm
self.disable_dtm = disable_dtm
self.disable_history = disable_history
```

### 修改 3: encode() 方法消融逻辑 (第 593-626 行)

**原代码 (简化):**
```python
def encode(self, s,r,y,mo,d,h_r,h_o,h_y,h_m,h_d):
    # 基础嵌入
    e = self.ent_cnn(self.get_ent_embed_h(s))
    r_emb = self.rel_cnn(self.get_rel_embed(r))
    tau = self.time_embed(y,mo,d)
    rt = self.leaky(self.Wr(torch.cat([r_emb, tau], dim=1)))
    x = torch.tanh(e + rt)

    # QFM
    Q_prev=None
    ei,ri,ti = e,r_emb,tau
    for blk in self.qfm_blocks:
        ei,ri,ti,Q_prev = blk(ei,ri,ti,Q_prev)
    thetas = torch.cat([self.We(ei), self.Wr_scalar(ri), self.Wt_scalar(ti)], dim=1)
    thetas = F.softmax(thetas, dim=1)

    # DTM
    c1,c2,c3 = self.dtm(tau, tau, tau, x)
    x_o = thetas[:,0:1]*c1 + thetas[:,1:2]*c2 + thetas[:,2:3]*c3
    x_o = torch.tanh(x_o)

    # 历史
    hist_vecs_pool, hist_seq = self.build_history_vectors(h_r,h_o,h_y,h_m,h_d)
    hist_mask = (h_o != self.pad_ent_id).long()
    x_hist = self.history_net(x_o, hist_seq, hist_mask)
    rt_gate = torch.sigmoid(self.r_time_gate(torch.cat([ri, ti], dim=1)))
    x_hist = rt_gate * x_hist + (1 - rt_gate) * ri
    t_gate = torch.sigmoid(self.time_gate(torch.cat([ri, ti], dim=1)))
    q = t_gate * x_hist + (1 - t_gate) * ri
    q = self.final_proj(self.final_ln(q))
    
    # 辅助输出...
    return q, aux_rel_logits, contrast_feats, (ri, ti), time_logits
```

**新代码 (带消融):**
```python
def encode(self, s,r,y,mo,d,h_r,h_o,h_y,h_m,h_d):
    # 基础嵌入 (不变)
    e = self.ent_cnn(self.get_ent_embed_h(s))
    r_emb = self.rel_cnn(self.get_rel_embed(r))
    tau = self.time_embed(y,mo,d)
    r_emb = self.apply_dropout_mask(r_emb, self.rel_dropout)
    tau = self.apply_dropout_mask(tau, self.time_dropout)
    rt = self.leaky(self.Wr(torch.cat([r_emb, tau], dim=1)))
    x = torch.tanh(e + rt)

    # QFM (带消融开关) ← 新增
    if self.disable_qfm:
        ei, ri, ti = e, r_emb, tau
        thetas = torch.ones(e.size(0), 3, device=e.device, dtype=e.dtype) / 3.0
    else:
        Q_prev=None
        ei,ri,ti = e,r_emb,tau
        for blk in self.qfm_blocks:
            ei,ri,ti,Q_prev = blk(ei,ri,ti,Q_prev)
        thetas = torch.cat([self.We(ei), self.Wr_scalar(ri), self.Wt_scalar(ti)], dim=1)
        thetas = F.softmax(thetas, dim=1)

    # DTM (带消融开关) ← 新增
    if self.disable_dtm:
        x_o = x
    else:
        c1,c2,c3 = self.dtm(tau, tau, tau, x)
        x_o = thetas[:,0:1]*c1 + thetas[:,1:2]*c2 + thetas[:,2:3]*c3
        x_o = torch.tanh(x_o)

    # 历史依赖 (带消融开关) ← 新增
    if self.disable_history:
        q = self.final_proj(self.final_ln(x_o))
    else:
        hist_vecs_pool, hist_seq = self.build_history_vectors(h_r,h_o,h_y,h_m,h_d)
        hist_mask = (h_o != self.pad_ent_id).long()
        x_hist = self.history_net(x_o, hist_seq, hist_mask)
        rt_gate = torch.sigmoid(self.r_time_gate(torch.cat([ri, ti], dim=1)))
        x_hist = rt_gate * x_hist + (1 - rt_gate) * ri
        t_gate = torch.sigmoid(self.time_gate(torch.cat([ri, ti], dim=1)))
        q = t_gate * x_hist + (1 - t_gate) * ri
        q = self.final_proj(self.final_ln(q))
    
    # 辅助输出 (不变)
    aux_rel_logits = self.aux_rel_head(q) if self.use_aux_rel else None
    time_logits = (...)
    contrast_feats = (...)
    return q, aux_rel_logits, contrast_feats, (ri, ti), time_logits
```

### 修改 4: main() 函数传参 (第 1098-1100 行)

**原代码:**
```python
model = LTGQ(
    num_entities=len(ent2id),
    num_relations=len(rel2id),
    time_vocab_sizes={"year": len(y2i), "month": len(m2i), "day": len(d2i)},
    pad_ent_id=pad_ent_id,
    pad_rel_id=pad_rel_id,
    dim=args.dim,
    qfm_layers=args.qfm_layers,
    dropout=args.dropout,
    pretrained_ent_emb=ent_init,
    pretrained_rel_emb=rel_init,
    hist_attn_dropout=args.hist_attn_dropout,
    recency_decay=args.recency_decay,
    rel_dropout=args.rel_dropout,
    time_dropout=args.time_dropout,
    use_aux_rel=args.use_aux_rel,
    history_heads=args.history_heads,
    use_adapter=args.use_adapter,
    adapter_dim=args.adapter_dim,
    time_fuse=args.time_fuse,
    use_sin_time=not args.no_sin_time,
    trans_weight=args.trans_weight,
    hist_tf_layers=args.hist_tf_layers,
    hist_tf_heads=args.hist_tf_heads,
    hist_tf_dropout=args.hist_tf_dropout,
    short_hist_size=args.short_hist_size,
    long_hist_size=args.long_hist_size,
).to(device)
```

**新代码:**
```python
model = LTGQ(
    num_entities=len(ent2id),
    num_relations=len(rel2id),
    time_vocab_sizes={"year": len(y2i), "month": len(m2i), "day": len(d2i)},
    pad_ent_id=pad_ent_id,
    pad_rel_id=pad_rel_id,
    dim=args.dim,
    qfm_layers=args.qfm_layers,
    dropout=args.dropout,
    pretrained_ent_emb=ent_init,
    pretrained_rel_emb=rel_init,
    hist_attn_dropout=args.hist_attn_dropout,
    recency_decay=args.recency_decay,
    rel_dropout=args.rel_dropout,
    time_dropout=args.time_dropout,
    use_aux_rel=args.use_aux_rel,
    history_heads=args.history_heads,
    use_adapter=args.use_adapter,
    adapter_dim=args.adapter_dim,
    time_fuse=args.time_fuse,
    use_sin_time=not args.no_sin_time,
    trans_weight=args.trans_weight,
    hist_tf_layers=args.hist_tf_layers,
    hist_tf_heads=args.hist_tf_heads,
    hist_tf_dropout=args.hist_tf_dropout,
    short_hist_size=args.short_hist_size,
    long_hist_size=args.long_hist_size,
    disable_qfm=args.disable_qfm,          # ← 新增
    disable_dtm=args.disable_dtm,          # ← 新增
    disable_history=args.disable_history,  # ← 新增
).to(device)
```

### 修改 5: argparse 参数 (第 1343-1345 行)

**新增代码:**
```python
# 消融开关
parser.add_argument("--disable_qfm", action="store_true", help="Disable QFM module")
parser.add_argument("--disable_dtm", action="store_true", help="Disable DTM module")
parser.add_argument("--disable_history", action="store_true", help="Disable History module")
```

## 保持不变的代码 (1299 行)

以下所有代码与原版 `train` 文件**完全一致**，一个字符都没改：

- ✅ 文件头注释和导入 (1-34 行)
- ✅ Utils 函数: set_seed, parse_date, load_csv, build_mappings 等 (36-257 行)
- ✅ Dataset 类和 collate_fn (259-274 行)
- ✅ MultiHeadHistoryAttn 类 (276-307 行)
- ✅ HistoricalCopyNet 类 (309-327 行)
- ✅ TriAff, CNN1x1 类 (329-347 行)
- ✅ QFMBlock 类 (349-363 行)
- ✅ RotationDTM 类 (365-384 行)
- ✅ LTGQ 类的其他方法:
  - reset_parameters (492-499 行)
  - _norm (501-502 行)
  - sinusoid (504-513 行)
  - time_embed (515-530 行)
  - _fuse_ent (532-537 行)
  - get_ent_embed_h/t (539-543 行)
  - get_rel_embed (545-549 行)
  - apply_dropout_mask (551-554 行)
  - build_history_vectors (556-580 行)
  - score (642-664 行)
  - forward (666-678 行)
- ✅ EMA 类 (680-699 行)
- ✅ FGM 类 (701-721 行)
- ✅ NegCache 类 (723-753 行)
- ✅ Loss 辅助函数 (755-821 行)
- ✅ Eval 辅助函数 (823-995 行)
- ✅ train_step 函数 (843-957 行)
- ✅ evaluate_with_scope 函数 (959-995 行)
- ✅ main 函数的其他部分 (997-1222 行)
- ✅ 所有原有 argparse 参数 (1224-1340 行)

## 使用示例

```bash
# 完整模型
python train_ablation.py --save_dir ./results/full

# 去 QFM
python train_ablation.py --disable_qfm --save_dir ./results/wo_qfm

# 去 DTM
python train_ablation.py --disable_dtm --save_dir ./results/wo_dtm

# 去 History
python train_ablation.py --disable_history --save_dir ./results/wo_history

# 组合消融
python train_ablation.py --disable_qfm --disable_dtm --save_dir ./results/wo_qfm_dtm
```

## 验证清单

- [x] 语法检查通过 (py_compile)
- [x] LTGQ.__init__ 包含 3 个新参数
- [x] encode() 包含 3 个 if/else 分支
- [x] main() 正确传递参数
- [x] argparse 包含 3 个新标志
- [x] 其他代码与原版完全一致
- [x] 文件行数合理 (1324 → 1349, +25)
