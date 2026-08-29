# 实验结果（PTAPE vs TSAPE）

## 实验设置

| 项目 | 内容 |
|------|------|
| 数据集 | KuaiRec（#users=7176，#items=10612，#interactions=936390） |
| 骨干 | SASRec（L=2, h=2, d=64, max_len=50, CE loss） |
| 划分 | leave-one-out（LS: valid_and_test） |
| 指标 | Recall@10 / MRR@10 / NDCG@10 / GAUC |
| 环境 | RecBole 1.2.1 + PyTorch 2.5.1 + Python 3.9（RTX 4060 8GB） |
| 随机种子 | 42（repeatable） |

## 结果（test 集）

| 模型 | Recall@10 | MRR@10 | NDCG@10 | GAUC | 参数量 |
|------|-----------|--------|---------|------|--------|
| TSAPE（LSTM） | **0.26370** | **0.12500** | **0.15768** | 0.94422 | 137.6 万 |
| PTAPE（Mamba） | 0.25784 | 0.11605 | 0.14934 | 0.94439 | **132.4 万** |

## 结论

PTAPE（用选择性状态空间模块替换 TSAPE 中的 LSTM）在 KuaiRec 上取得**与 TSAPE 接近的精度**（Recall@10 相差 2.2%、NDCG@10 相差 5.3%、GAUC 持平），同时**参数量更少**（132.4 万 vs 137.6 万），验证了「状态空间模型可替代 LSTM 做时序位置编码」的技术可行性。

**关于效率**：Mamba 的并行扫描、O(1) 推理为理论优势。当前实现是纯 PyTorch 参考实现（正确性优先），且 Windows 本机无法安装 mamba_ssm 的 CUDA 融合核，故**未做速度实测**；效率提升需在 Linux + mamba_ssm 下验证，属于后续工作。

## 复现方式

```bash
# 环境：Python 3.9 + RecBole + PyTorch（pytorch_39 环境）

# TSAPE 基线（LSTM 版）
python sasrec_test.py --dataset kuairec

# PTAPE（Mamba 版）
python ptapesasrec_test.py --dataset kuairec
```

对应模型文件：
- TSAPE：`model/sequential_recommender/tmesasrec.py`
- PTAPE：`model/sequential_recommender/ptapesasrec.py`
