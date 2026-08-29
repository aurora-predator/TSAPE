import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss
import torch.nn.functional as F

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

class TimeAwareRNN(nn.Module):
    def __init__(self, config, dataset):
        super(TimeAwareRNN, self).__init__()
        self.config = config
        self.dataset = dataset
        self.initializer_range = config["initializer_range"]
        self.hidden_size = config["hidden_size"]
        self.max_time = config["max_time"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        # self.model_type = model_type
        self.rnn = nn.LSTM(input_size=1, hidden_size=self.hidden_size, num_layers=3, batch_first=True)  #正常设置为3
        self.cnn = nn.Sequential(
            nn.Conv1d(self.hidden_size, self.hidden_size, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(self.hidden_size, self.hidden_size, 5, padding=2)
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
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

    def forward(self, time_seq, model_type = 'user'):
        if model_type == "user":
            time_diff = torch.diff(time_seq, dim=1)  # [B, L-1]
        else:
            time_diff = time_seq
        time_diff = torch.clamp(time_diff, min=0, max=self.max_time)
        time_diff = time_diff / self.max_time
        rnn_output, _ = self.rnn(time_diff.unsqueeze(-1))  # [B L H]
        cnn_feat = self.cnn(rnn_output.permute(0,2,1)).permute(0,2,1)
        rnn_output = rnn_output + cnn_feat
        rnn_output = self.se_block(rnn_output)  # [B L H]
        time_aware_vector = self.mlp(rnn_output)  # [B L H]
        time_aware_vector = self.LayerNorm(time_aware_vector)  # [B L H]
        return time_aware_vector

class TMESASRec(SequentialRecommender):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, dataset):
        super(TMESASRec, self).__init__(config, dataset)

        self.TIME_SEQ = f"{config['TIME_FIELD']}_list"  # 时间序列字段名
        self.ITEM_DURATION_SEQ = f"{config['ITEM_ID_FIELD']}_duration_list"  # 物品时长序列字段名
        self.n_users = dataset.num(self.USER_ID)
        # load parameters info
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]  # same as embedding_size
        self.inner_size = config[
            "inner_size"
        ]  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]

        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]

        # define layers and loss
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
        self.time_aware_rnn = TimeAwareRNN(config, dataset)
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

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len, time_seq):
        # user_emb = self.user_embedding(user_id)  # [B H]

        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        # input_emb = item_emb
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)  # [B L H]

        # 获取时间感知向量
        time_aware_vector = self.time_aware_rnn(time_seq)  # [B L H]

        # 拼接item embedding和时间感知向量
        concat_input = input_emb + time_aware_vector  # [B L H]
        
        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(
            concat_input, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]  # [B L H]
        
        # 投影回原始维度
        output = self.output_projection(output)  # [B L H]
        output = self.gather_indexes(output, item_seq_len - 1)  # [B H]
        return output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        time_seq = interaction[self.TIME_SEQ]
        # user_id = interaction[self.USER_ID]
        # item_duration_seq = interaction[self.ITEM_DURATION_SEQ]
        # in_count = interaction['interval_count_list']
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
        # user_id = interaction[self.USER_ID]
        # item_duration_seq = interaction[self.ITEM_DURATION_SEQ]
        # in_count = interaction['interval_count_list']
        seq_output = self.forward(item_seq, item_seq_len, time_seq)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        time_seq = interaction[self.TIME_SEQ]
        # user_id = interaction[self.USER_ID]
        # item_duration_seq = interaction[self.ITEM_DURATION_SEQ]
        # in_count = interaction['interval_count_list']
        seq_output = self.forward(item_seq, item_seq_len, time_seq)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
