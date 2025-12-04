import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers import AutoTokenizer
import torch.nn.functional as F

# ---------------------- 1. 模型配置类（定义超参数）----------------------
class AddressModelConfig(PretrainedConfig):
    model_type = "address_analyzer"
    def __init__(
        self,
        vocab_size=50000,
        d_model=128,
        n_layers=4,
        n_heads=8,
        d_ff=512,
        dropout=0.1,
        pad_token_id=0,
        num_labels=6,  # 地址解析任务：6类核心组件（邮编、城市、州、街道、门牌号、补充信息）
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.pad_token_id = pad_token_id
        self.num_labels = num_labels

# ---------------------- 2. 注意力层+前馈层（Transformer核心模块）----------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k = d_model // n_heads  # 每头维度
        self.n_heads = n_heads
        # QKV投影层（共享权重矩阵）
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        # 输出投影层
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        # 投影并分多头：(batch_size, seq_len, d_model) → (batch_size, n_heads, seq_len, d_k)
        q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 计算注意力分数：(batch_size, n_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 注意力输出：(batch_size, n_heads, seq_len, d_k) → (batch_size, seq_len, d_model)
        output = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        output = self.w_o(output)
        return output, attn_weights

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()  # 比ReLU更适配小模型

    def forward(self, x):
        return self.linear2(self.dropout(self.gelu(self.linear1(x))))

# ---------------------- 3. 单编码器层 ----------------------
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        # 层归一化（前置，更稳定）
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 自注意力子层
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        # 前馈子层
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        return x

# ---------------------- 4. 完整地址分析模型（仅编码器架构）----------------------
class AddressAnalyzerModel(PreTrainedModel):
    config_class = AddressModelConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # 1. 词嵌入层（核心，与5万词表匹配）
        self.embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        # 位置编码（地址文本短，固定位置编码足够）
        self.position_encoding = nn.Parameter(torch.randn(1, 128, config.d_model))  # 最大序列长度128
        self.dropout = nn.Dropout(config.dropout)

        # 2. 编码器堆叠（4层）
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])

        # 3. 输出层（地址组件分类任务）
        self.classifier = nn.Linear(config.d_model, config.num_labels)

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.d_model ** -0.5)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx] = 0.0

    def create_padding_mask(self, input_ids):
        # 生成padding mask：(batch_size, 1, 1, seq_len)
        return (input_ids != self.config.pad_token_id).unsqueeze(1).unsqueeze(2)

    def forward(self, input_ids, attention_mask=None, labels=None):
        batch_size, seq_len = input_ids.size()

        # 1. 嵌入+位置编码
        x = self.embedding(input_ids)  # (batch_size, seq_len, d_model)
        x = x + self.position_encoding[:, :seq_len, :]  # 叠加位置信息
        x = self.dropout(x)

        # 2. 生成padding mask
        if attention_mask is None:
            attention_mask = self.create_padding_mask(input_ids)
        else:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # 3. 编码器前向传播
        for layer in self.encoder_layers:
            x = layer(x, attention_mask)  # (batch_size, seq_len, d_model)

        # 4. 分类输出（每个Token对应一个地址组件类别）
        logits = self.classifier(x)  # (batch_size, seq_len, num_labels)

        # 5. 计算损失（若传入labels）
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.num_labels),
                labels.reshape(-1),
                ignore_index=-100  # 忽略padding位置的损失
            )

        return {"loss": loss, "logits": logits}

# ---------------------- 5. 模型初始化+参数验证 ----------------------
if __name__ == "__main__":
    # ===================== 修复：加载剪枝后的Tokenizer =====================
    # 原始路径（假设你的Tokenizer在此文件夹）
    tokenizer_dir = "./qwen3_address_5w_tokenizer_final"

    # 关键：手动加载文件，并指定参数以避免Fast Tokenizer转换
    try:
        # 方案一：优先尝试。显式指定文件路径并禁用Fast模式
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir,
            use_fast=False,  # 强制使用慢速Tokenizer，避免protobuf和转换问题
            local_files_only=True  # 确保只从本地加载
        )
    except Exception as e:
        print(f"使用AutoTokenizer加载失败: {e}")
        print("尝试使用PreTrainedTokenizerFast直接加载...")
        # 方案二：备选方案。更底层的加载方式
        from transformers import PreTrainedTokenizerFast
        with open(f"{tokenizer_dir}/vocab.json", 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        with open(f"{tokenizer_dir}/merges.txt", 'r', encoding='utf-8') as f:
            merges = f.read()

        tokenizer = PreTrainedTokenizerFast(
            vocab_file=f"{tokenizer_dir}/vocab.json",
            merges_file=f"{tokenizer_dir}/merges.txt",
            unk_token="<unk>",
            bos_token="<|endoftext|>",
            eos_token="<|im_end|>",
            pad_token="<|endoftext|>",  # 根据你的实际情况调整
            # 注意：这种方式可能仍然会尝试构建FastTokenizer，如果失败，则必须回到方案一
        )

    # 验证加载的Tokenizer基础信息
    print(f"✅ Tokenizer加载成功。词表大小: {tokenizer.vocab_size}")
    print(f"   Pad Token ID: {tokenizer.pad_token_id}")

    # ===================== 后续模型初始化保持不变 =====================
    # 初始化模型配置
    config = AddressModelConfig(
        vocab_size=tokenizer.vocab_size,  # 确保与Tokenizer词表一致
        d_model=128,
        n_layers=4,
        n_heads=8,
        d_ff=512,
        pad_token_id=tokenizer.pad_token_id,
        num_labels=6
    )

    # 创建模型
    model = AddressAnalyzerModel(config)

    # 验证参数数量（目标：~1000万）
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数：{total_params / 1e6:.2f} 万")
    print(f"可训练参数：{trainable_params / 1e6:.2f} 万")
    print(f"嵌入层参数：{model.embedding.weight.numel() / 1e6:.2f} 万")

    # ---------------------- 6. 训练/推理示例 ----------------------
    # 示例1：推理（地址组件分类）
    test_address = "Av. Paulista, 1000 - São Paulo, SP, 01310-100"
    inputs = tokenizer(test_address, return_tensors="pt", padding=True, truncation=True, max_length=64)
    outputs = model(**inputs)
    logits = outputs["logits"]
    predictions = torch.argmax(logits, dim=-1)
    print("\n推理结果（Token级别分类）：")
    print(f"输入地址：{test_address}")
    print(f"Token序列：{tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])}")
    print(f"预测类别：{predictions[0].tolist()}")  # 0=邮编，1=城市，2=州，3=街道，4=门牌号，5=补充信息

    # 示例2：训练模板（需准备标注数据）
    # train_dataset = 你的地址标注数据集（input_ids, attention_mask, labels）
    # optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    # model.train()
    # for batch in train_dataloader:
    #     optimizer.zero_grad()
    #     outputs = model(**batch)
    #     outputs["loss"].backward()
    #     optimizer.step()