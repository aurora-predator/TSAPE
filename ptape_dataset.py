import pandas as pd
import torch
from torch.utils.data import Dataset


class KuaiRecDataset(Dataset):
    """
    KuaiRec Dataset

    适配 TSAPE 项目中的 RecBole .inter 数据格式。

    数据格式：

        user_id:token
        item_id:token
        timestamp:float
        label:float

    返回：

        items      : [max_len]
        delta      : [max_len]
        mask       : [max_len]
        target     : scalar
        user_id    : scalar
    """

    def __init__(
        self,
        file_path,
        max_len=50,
        num_items=None
    ):

        self.file_path = file_path
        self.max_len = max_len

        print(f"Loading: {file_path}")

        # ====================================================
        # 1. 读取 RecBole .inter 文件
        # ====================================================

        df = self._load_file(file_path)

        print("Columns:", list(df.columns))

        # ====================================================
        # 2. KuaiRec 真实字段
        # ====================================================

        self.user_col = "user_id:token"
        self.item_col = "item_id:token"
        self.time_col = "timestamp:float"
        self.label_col = "label:float"

        required_columns = [
            self.user_col,
            self.item_col,
            self.time_col,
            self.label_col
        ]

        for col in required_columns:

            if col not in df.columns:

                raise ValueError(
                    f"Cannot find required column: {col}\n"
                    f"Available columns: {list(df.columns)}"
                )

        print("User column:", self.user_col)
        print("Item column:", self.item_col)
        print("Time column:", self.time_col)
        print("Label column:", self.label_col)

        # ====================================================
        # 3. 类型转换
        # ====================================================

        df[self.user_col] = pd.to_numeric(
            df[self.user_col],
            errors="coerce"
        )

        df[self.item_col] = pd.to_numeric(
            df[self.item_col],
            errors="coerce"
        )

        df[self.time_col] = pd.to_numeric(
            df[self.time_col],
            errors="coerce"
        )

        df[self.label_col] = pd.to_numeric(
            df[self.label_col],
            errors="coerce"
        )

        # 删除非法数据
        df = df.dropna(
            subset=[
                self.user_col,
                self.item_col,
                self.time_col
            ]
        )

        # ====================================================
        # 4. 按用户 + 时间排序
        # ====================================================

        df = df.sort_values(
            by=[
                self.user_col,
                self.time_col
            ]
        ).reset_index(drop=True)

        # ====================================================
        # 5. 建立用户行为序列
        #
        # 同时保存：
        #
        # items
        # timestamps
        # ====================================================

        self.user_sequences = {}

        for user_id, group in df.groupby(self.user_col):

            items = (
                group[self.item_col]
                .astype(int)
                .tolist()
            )

            timestamps = (
                group[self.time_col]
                .astype(float)
                .tolist()
            )

            if len(items) < 2:
                continue

            self.user_sequences[int(user_id)] = {
                "items": items,
                "timestamps": timestamps
            }

        # ====================================================
        # 6. 建立训练样本
        #
        # 每个样本：
        #
        # history items
        # history timestamps
        # target item
        #
        # 例如：
        #
        # items:
        # [6812, 5274, 179]
        #
        # timestamps:
        # [t1, t2, t3]
        #
        # target:
        # 171
        #
        # delta:
        # [0, t2-t1, t3-t2]
        # ====================================================

        self.samples = []

        for user_id, data in self.user_sequences.items():

            items = data["items"]
            timestamps = data["timestamps"]

            for i in range(1, len(items)):

                start = max(
                    0,
                    i - self.max_len
                )

                sequence = items[start:i]

                time_sequence = timestamps[start:i]

                target = items[i]

                if len(sequence) == 0:
                    continue

                self.samples.append(
                    (
                        user_id,
                        sequence,
                        time_sequence,
                        target
                    )
                )

        # ====================================================
        # 7. Item 数量
        # ====================================================

        if num_items is not None:

            self.num_items = int(num_items)

        else:

            max_item_id = int(
                df[self.item_col].max()
            )

            self.num_items = max_item_id + 1

        # ====================================================
        # 8. User 数量
        # ====================================================

        self.num_users = int(
            df[self.user_col].nunique()
        )

        # ====================================================
        # 9. 输出数据集信息
        # ====================================================

        print("Users:", self.num_users)
        print("Items:", self.num_items)
        print("Sequences:", len(self.user_sequences))
        print("Training samples:", len(self.samples))

    # ========================================================
    # 读取 .inter 文件
    # ========================================================

    def _load_file(self, file_path):

        try:

            df = pd.read_csv(
                file_path,
                sep="\t"
            )

        except Exception as e:

            raise RuntimeError(
                f"Failed to read dataset: {file_path}\n"
                f"Error: {e}"
            )

        return df

    # ========================================================
    # Dataset 长度
    # ========================================================

    def __len__(self):

        return len(self.samples)

    # ========================================================
    # 获取一个训练样本
    # ========================================================

    def __getitem__(self, index):

        (
            user_id,
            sequence,
            timestamps,
            target
        ) = self.samples[index]

        # ----------------------------------------------------
        # 最近 max_len 个行为
        # ----------------------------------------------------

        sequence = sequence[-self.max_len:]

        timestamps = timestamps[-self.max_len:]

        seq_len = len(sequence)

        # ====================================================
        # 计算时间间隔 delta
        #
        # delta[0] = 0
        #
        # delta[i] =
        # timestamp[i] - timestamp[i-1]
        #
        # 单位：秒
        # ====================================================

        deltas = [0.0]

        for i in range(1, len(timestamps)):

            delta = (
                timestamps[i]
                - timestamps[i - 1]
            )

            # 防止异常数据
            if delta < 0:
                delta = 0.0

            deltas.append(delta)

        # ====================================================
        # 时间间隔归一化
        #
        # 原始 KuaiRec 时间间隔可能跨度很大。
        #
        # 使用 log(1 + delta)
        # 可以减少极端时间间隔的影响。
        # ====================================================

        deltas = [
            torch.log1p(
                torch.tensor(
                    d,
                    dtype=torch.float32
                )
            ).item()
            for d in deltas
        ]

        # ====================================================
        # Padding
        # ====================================================

        padding_len = self.max_len - seq_len

        padded_items = (
            [0] * padding_len
            + sequence
        )

        padded_deltas = (
            [0.0] * padding_len
            + deltas
        )

        # ====================================================
        # Attention Mask
        #
        # padding = 0
        # real item = 1
        # ====================================================

        mask = (
            [0] * padding_len
            + [1] * seq_len
        )

        # ====================================================
        # 转 Tensor
        # ====================================================

        items_tensor = torch.tensor(
            padded_items,
            dtype=torch.long
        )

        delta_tensor = torch.tensor(
            padded_deltas,
            dtype=torch.float32
        )

        mask_tensor = torch.tensor(
            mask,
            dtype=torch.bool
        )

        target_tensor = torch.tensor(
            target,
            dtype=torch.long
        )

        user_tensor = torch.tensor(
            user_id,
            dtype=torch.long
        )

        # ====================================================
        # 返回
        #
        # 重点：
        #
        # train.py 当前至少需要：
        #
        # batch["items"]
        # batch["delta"]
        #
        # 所以两个字段必须存在。
        # ====================================================

        return {

            # item sequence
            "items": items_tensor,

            # time interval
            "delta": delta_tensor,

            # attention mask
            "mask": mask_tensor,

            # next item
            "target": target_tensor,

            # 兼容可能使用 targets 的代码
            "targets": target_tensor,

            # user
            "user_id": user_tensor,

            # 兼容可能使用 users 的代码
            "users": user_tensor,

            # 兼容旧代码
            "sequence": items_tensor
        }