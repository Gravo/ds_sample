import json
import torch
from torch.utils.data import Dataset
from transformers import Qwen2Tokenizer


class AddressTokenDataset(Dataset):
    """地址Token分类数据集（适配Qwen2Tokenizer）"""

    def __init__(self, data_path: str, tokenizer: Qwen2Tokenizer, max_len: int = 64):
        # 加载标注数据
        self.data = [json.loads(line) for line in open(data_path, "r", encoding="utf-8")]
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label2id = {v: k for k, v in self.data[0]["id2label"].items()}  # 从数据中读取标签映射

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        input_text = sample["input_text"]

        # 1. 编码输入（用Qwen2Tokenizer，与标注时的Token一致）
        encoding = self.tokenizer(
            input_text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].flatten()
        attention_mask = encoding["attention_mask"].flatten()

        # 2. 处理标签（padding位置设为-100，模型忽略）
        token_labels = sample["token_labels"]
        if len(token_labels) < self.max_len:
            token_labels += [-100] * (self.max_len - len(token_labels))
        else:
            token_labels = token_labels[:self.max_len]
        labels = torch.tensor(token_labels, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


# 测试数据集加载
if __name__ == "__main__":
    tokenizer = Qwen2Tokenizer.from_pretrained("../qwen3_address_5w_tokenizer_final", local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = AddressTokenDataset("./annotated_brazil_address.jsonl", tokenizer)
    sample = dataset[0]
    print(f"输入ID形状：{sample['input_ids'].shape}")
    print(f"注意力掩码形状：{sample['attention_mask'].shape}")
    print(f"标签形状：{sample['labels'].shape}")
    print(f"示例标签：{sample['labels'][:10]}")