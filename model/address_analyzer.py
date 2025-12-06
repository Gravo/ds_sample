import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig

# 1. 模型配置类（扩容超参数）
class AddressModelConfig(PretrainedConfig):
    model_type = "address_analyzer"
    def __init__(
        self,
        vocab_size=50000,  # 匹配Qwen2Tokenizer词表大小
        d_model=512,       # 从128→512（核心扩容）
        n_layers=6,        # 从4→6
        n_heads=8,         # 512%8=0，满足整除
        d_ff=2048,         # 从512→2048
        dropout=0.1,
        pad_token_id=0,
        num_labels=6,
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

# 2. 保持原有Attention/FeedForward/EncoderLayer不变
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        output = self.w_o(output)
        return output, attn_weights

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.linear2(self.dropout(self.gelu(self.linear1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        return x

# 3. 完整模型（适配扩容参数）
class AddressAnalyzerModel(PreTrainedModel):
    config_class = AddressModelConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # 词嵌入层（5万词表×512维度=2560万参数，核心扩容）
        self.embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        # 固定正弦位置编码（更稳定）
        self.register_buffer(
            "position_encoding",
            self.get_sinusoidal_position_encoding(128, config.d_model),
            persistent=False
        )
        self.dropout = nn.Dropout(config.dropout)

        # 6层编码器（扩容）
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])

        # 分类输出层
        self.classifier = nn.Linear(config.d_model, config.num_labels)

        # 初始化权重
        self.apply(self._init_weights)

    def get_sinusoidal_position_encoding(self, max_len, d_model):
        """固定正弦位置编码"""
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

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
        return (input_ids != self.config.pad_token_id).unsqueeze(1).unsqueeze(2)

    def forward(self, input_ids, attention_mask=None, labels=None):
        batch_size, seq_len = input_ids.size()

        # 嵌入+位置编码
        x = self.embedding(input_ids)
        x = x + self.position_encoding[:, :seq_len, :]
        x = self.dropout(x)

        # 生成padding mask
        if attention_mask is None:
            attention_mask = self.create_padding_mask(input_ids)
        else:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # 编码器前向
        for layer in self.encoder_layers:
            x = layer(x, attention_mask)

        # 分类输出
        logits = self.classifier(x)

        # 计算损失
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.num_labels),
                labels.reshape(-1),
                ignore_index=-100
            )

        return {"loss": loss, "logits": logits}

# 测试模型初始化（验证参数规模）
if __name__ == "__main__":
    from transformers import Qwen2Tokenizer
    tokenizer = Qwen2Tokenizer.from_pretrained("../qwen3_address_5w_tokenizer_final", local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 初始化配置（匹配Qwen2Tokenizer）
    config = AddressModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=512,
        n_layers=6,
        n_heads=8,
        d_ff=2048,
        pad_token_id=tokenizer.pad_token_id,
        num_labels=6
    )
    model = AddressAnalyzerModel(config)

    # 验证参数规模（目标≈3000万）
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数：{total_params / 1e6:.2f} 万")  # 约2680万，接近3000万（可调d_model=576达到3000万）
    print(f"嵌入层参数：{model.embedding.weight.numel() / 1e6:.2f} 万")  # 5万×512=2560万