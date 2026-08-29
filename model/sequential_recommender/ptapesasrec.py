# -*- coding: utf-8 -*-
"""
PTAPE: Parallel Temporal-Aware Positional Encoding (并行时序感知位置编码)

对 TSAPE 的改进实现。原论文代码见 TSAPE_Rec-main/model/sequential_recommender/tmesasrec.py，
本文件与原文件逐行对应，唯一核心改动是：用「选择性状态空间模型 Mamba」替换原
TimeAwareRNN 中的多层 LSTM（`nn.LSTM(input_size=1, hidden_size=64, num_layers=3)`）。

为什么这么改（对应报告中的两条不足）：
  1. 效率：LSTM 是串行递归，无法并行，与 Transformer 的并行性矛盾；Mamba 用并行扫描
     训练、推理时状态更新为 O(1)，消除串行瓶颈。
  2. 语义歧义：LSTM 门控是固定参数，无法区分「兴趣衰减」与「退出平台」两种成因的长间隔；
     Mamba 的选择性机制让信息保留/遗忘由当前时间间隔的具体取值决定，从机制上缓解歧义。

其余结构（1D CNN 多尺度局部建模、SE 通道注意力、残差、MLP+LayerNorm、加法式位置编码
集成策略）与 TSAPE 完全一致，因此保留了「即插即用」的特性。

运行方式：与 TSAPE 相同，通过 RecBole 的 run() 入口加载本模型的类名 PTAPESASRec 即可。
"""
import torch
from torch import nn
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 与原 TSAPE 相同的 SEBlock（Squeeze-and-Excitation 通道注意力）
# ---------------------------------------------------------------------------
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = x.mean(dim=1)
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y))
        y = y.unsqueeze(1)
        return x * y


# ---------------------------------------------------------------------------
# 新增：纯 PyTorch 的 Mamba（选择性状态空间，S6）参考实现
# ---------------------------------------------------------------------------
class MambaBlock(nn.Module):
    """最小化的 Mamba / 选择性状态空间块（Gu & Dao, COLM 2024）。

    这是「正确性优先」的参考实现：选择性扫描写成显式的序列递归，便于阅读与验证，
    但未使用 mamba_ssm 的融合 CUDA 核。在 Linux 上可将本类替换为 `mamba_ssm.Mamba`
    （接口一致）以获得硬件感知的并行扫描加速；本实现用于在任意平台验证精度。

    Args:
        d_model: 模型/嵌入维度（须与调用方 hidden_size 一致）。
        d_state: 状态空间维度。
        d_conv:  局部因果深度卷积的核大小。
        expand:  内部维度扩张倍数（d_inner = expand * d_model）。
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super(MambaBlock, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        d_inner = int(expand * d_model)
        self.d_inner = d_inner

        # 输入投影：拆分为主支路 x 与残差门控 z
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        # 因果深度卷积，混合邻近时间步
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv, padding=d_conv - 1, groups=d_inner
        )
        # 输入依赖的 SSM 参数：dt（rank-1）、B（d_state）、C（d_state）
        self.x_proj = nn.Linear(d_inner, 1 + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(1, d_inner, bias=True)

        # 对角状态转移矩阵 A（按内部通道），以 log 空间存储
        A = torch.arange(1, d_state + 1).float().view(1, d_state).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        # 跳跃连接标量 D
        self.D = nn.Parameter(torch.ones(d_inner))

        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x):
        # x: [B, L, D]
        B, L, _ = x.shape
        xz = self.in_proj(x)                    # [B, L, 2*d_inner]
        x, z = xz.chunk(2, dim=-1)              # x: [B, L, d_inner], z: 门控
        x = x.transpose(1, 2)                   # [B, d_inner, L]
        x = self.conv1d(x)[..., :L]             # 因果卷积 -> [B, d_inner, L]
        x = F.silu(x).transpose(1, 2)           # [B, L, d_inner]

        dt_bc = self.x_proj(x)                  # [B, L, 1 + 2*d_state]
        dt, bc = dt_bc.split([1, 2 * self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))       # [B, L, d_inner]，正的步长
        Bm, C = bc.split([self.d_state, self.d_state], dim=-1)  # 各 [B, L, d_state]

        A = -torch.exp(self.A_log)              # [d_inner, d_state]，负值保证稳定
        A_bar = torch.exp(dt.unsqueeze(-1) * A)          # [B, L, d_inner, d_state]
        B_bar = dt.unsqueeze(-1) * Bm.unsqueeze(2)       # [B, L, d_inner, d_state]

        # 选择性扫描（显式递归）。h: [B, d_inner, d_state]
        h = x.new_zeros(B, self.d_inner, self.d_state)
        ys = []
        for t in range(L):
            h = A_bar[:, t] * h + B_bar[:, t] * x[:, t].unsqueeze(-1)
            ys.append((C[:, t].unsqueeze(1) * h).sum(-1))  # [B, d_inner]
        y = torch.stack(ys, dim=1)              # [B, L, d_inner]

        y = y + self.D * x                      # 跳跃连接
        y = y * F.silu(z)                       # 门控
        return self.out_proj(y)                 # [B, L, D]


# ---------------------------------------------------------------------------
# 核心改动：用 Mamba 替代 LSTM 的时序感知模块（原 TimeAwareRNN 的 Mamba 版）
# ---------------------------------------------------------------------------
class TimeAwareMamba(nn.Module):
    def __init__(self, config, dataset):
        super(TimeAwareMamba, self).__init__()
        self.hidden_size = config["hidden_size"]
        self.max_time = config["max_time"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.initializer_range = config["initializer_range"]

        # Mamba 需要 d_model 维输入，先把标量时间间隔投影到隐层维度
        self.time_embedding = nn.Linear(1, self.hidden_size)
        # 用 Mamba 替代原 nn.LSTM
        self.mamba = MambaBlock(
            d_model=self.hidden_size,
            d_state=16,
            d_conv=4,
            expand=2,
        )

        # 以下与 TSAPE 完全一致：多尺度 1D CNN + 残差 + SE + MLP + LayerNorm
        self.cnn = nn.Sequential(
            nn.Conv1d(self.hidden_size, self.hidden_size, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(self.hidden_size, self.hidden_size, 5, padding=2),
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=config["layer_norm_eps"])
        self.se_block = SEBlock(self.hidden_size, reduction=16)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, time_seq, model_type="user"):
        time_diff = torch.diff(time_seq, dim=1) if model_type == "user" else time_seq
        time_diff = torch.clamp(time_diff, min=0, max=self.max_time) / self.max_time

        # 改动点：前向从 LSTM 换成「投影 + Mamba」
        x = self.time_embedding(time_diff.unsqueeze(-1))   # [B, L-1, H]
        mamba_out = self.mamba(x)                           # [B, L-1, H]

        # 后面与原 TSAPE 前向一致
        cnn_feat = self.cnn(mamba_out.permute(0, 2, 1)).permute(0, 2, 1)
        h = mamba_out + cnn_feat
        h = self.se_block(h)
        h = self.mlp(h)
        return self.LayerNorm(h)


# ---------------------------------------------------------------------------
# 新模型：PTAPE + SASRec（与原 TMESASRec 完全一致，仅把 TimeAwareRNN 换成 TimeAwareMamba）
# ---------------------------------------------------------------------------
class PTAPESASRec(SequentialRecommender):
    def __init__(self, config, dataset):
        super(PTAPESASRec, self).__init__(config, dataset)

        self.TIME_SEQ = f"{config['TIME_FIELD']}_list"
        self.ITEM_DURATION_SEQ = f"{config['ITEM_ID_FIELD']}_duration_list"
        self.n_users = dataset.num(self.USER_ID)

        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]
        self.inner_size = config["inner_size"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]

        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )
        # 唯一改动：TimeAwareRNN -> TimeAwareMamba
        self.time_aware_mamba = TimeAwareMamba(config, dataset)
        self.user_embedding = nn.Embedding(self.n_users, self.hidden_size)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.output_projection = nn.Linear(self.hidden_size, self.hidden_size)

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len, time_seq):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)  # [B L H]

        # 获取时间感知向量（Mamba 版）
        time_aware_vector = self.time_aware_mamba(time_seq)  # [B L H]

        # 加法式位置编码集成：与 TSAPE 完全一致
        concat_input = input_emb + time_aware_vector  # [B L H]

        extended_attention_mask = self.get_attention_mask(item_seq)
        trm_output = self.trm_encoder(
            concat_input, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]  # [B L H]

        output = self.output_projection(output)  # [B L H]
        output = self.gather_indexes(output, item_seq_len - 1)  # [B H]
        return output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        time_seq = interaction[self.TIME_SEQ]
        seq_output = self.forward(item_seq, item_seq_len, time_seq)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            rec_loss = self.loss_fct(pos_score, neg_score)
        else:
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            rec_loss = self.loss_fct(logits, pos_items)
        return rec_loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        time_seq = interaction[self.TIME_SEQ]
        seq_output = self.forward(item_seq, item_seq_len, time_seq)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        time_seq = interaction[self.TIME_SEQ]
        seq_output = self.forward(item_seq, item_seq_len, time_seq)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
