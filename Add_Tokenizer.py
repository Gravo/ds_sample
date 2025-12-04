from transformers import Qwen2Tokenizer
import json
import random
import shutil
import os

# ===================== 核心配置 =====================
TARGET_VOCAB_SIZE = 50000  # 目标词表大小
OUTPUT_DIR = "./qwen3_address_5w_tokenizer"
ORIGINAL_DIR = "./qwen3_original_tokenizer"
# 巴西地址核心Token（硬编码）
ADDRESS_CORE_TOKENS = [
    # 特殊Token（Qwen2Tokenizer原生特殊Token）
    "<|endoftext|>", "<unk>", "<pad>",
    # 葡语地址类型
    "Rua", "Av.", "Avenida", "Travessa", "Praça", "Alameda", "Estrada", "Rodovia",
    # 巴西州缩写
    "SP", "RJ", "MG", "RS", "PR", "SC", "BA", "CE", "DF", "GO", "ES", "PE", "MA",
    # 邮编相关
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "-", "00000-000",
    # 地址补充词
    "n°", "bloco", "apt", "sala", "andar", "Casa", "Edifício", "Condomínio",
    # 巴西核心城市
    "São Paulo", "Rio de Janeiro", "Brasília", "Belo Horizonte", "Salvador", "Fortaleza"
]

# ===================== 步骤1：下载并保存Qwen2Tokenizer原生Tokenizer =====================
print("=== 下载Qwen3-0.6B原生Tokenizer ===")
tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen3-0.6B")
tokenizer.save_pretrained(ORIGINAL_DIR)
print(f"原生Tokenizer保存路径：{ORIGINAL_DIR}")

# ===================== 步骤2：加载原生词表并剪枝 =====================
# 1. 加载原生vocab.json
with open(f"{ORIGINAL_DIR}/vocab.json", "r", encoding="utf-8") as f:
    original_vocab = json.load(f)
original_size = len(original_vocab)
print(f"\nQwen3原生词表大小：{original_size}")

# 2. 筛选核心Token（强制保留特殊Token+地址相关Token）
core_tokens = set([
    "<|endoftext|>", "<unk>", "<pad>"  # Qwen2Tokenizer原生特殊Token
])
# 模糊匹配地址相关Token
for token in original_vocab.keys():
    if any(core in token for core in ADDRESS_CORE_TOKENS) or token in ADDRESS_CORE_TOKENS:
        core_tokens.add(token)
# 兜底：核心Token至少1000个
if len(core_tokens) < 1000:
    core_tokens.update(random.sample(list(original_vocab.keys()), 1000 - len(core_tokens)))
print(f"保留地址核心Token数量：{len(core_tokens)}")

# 3. 随机裁剪到5万词表
need_random = TARGET_VOCAB_SIZE - len(core_tokens)
final_tokens = core_tokens
if need_random > 0:
    non_core_tokens = [t for t in original_vocab.keys() if t not in core_tokens]
    # 防止样本不足
    random_tokens = random.sample(non_core_tokens, need_random) if len(non_core_tokens)>=need_random else non_core_tokens
    final_tokens = core_tokens.union(set(random_tokens))
else:
    final_tokens = set(list(core_tokens)[:TARGET_VOCAB_SIZE])

# 4. 重新生成ID映射（保证连续）+ 强制保留特殊Token
new_vocab = {}
# 先加入特殊Token（确保ID稳定）
special_tokens = ["<|endoftext|>", "<unk>", "<pad>"]
for token in special_tokens:
    if token in final_tokens:
        new_vocab[token] = len(new_vocab)
        final_tokens.remove(token)
# 再加入其他Token
for token in sorted(final_tokens):
    new_vocab[token] = len(new_vocab)
# 兜底：确保词表大小为5万
if len(new_vocab) < TARGET_VOCAB_SIZE:
    # 补充随机Token
    supplement_tokens = random.sample([t for t in original_vocab.keys() if t not in new_vocab],
                                     TARGET_VOCAB_SIZE - len(new_vocab))
    for token in supplement_tokens:
        new_vocab[token] = len(new_vocab)
elif len(new_vocab) > TARGET_VOCAB_SIZE:
    # 截断到5万（保留特殊Token）
    new_vocab = {k: v for i, (k, v) in enumerate(new_vocab.items()) if i < TARGET_VOCAB_SIZE}

print(f"最终剪枝后词表大小：{len(new_vocab)}")
print(f"<unk>是否存在：{'<unk>' in new_vocab} | ID：{new_vocab.get('<unk>', '缺失')}")
print(f"<pad>是否存在：{'<pad>' in new_vocab} | ID：{new_vocab.get('<pad>', '缺失')}")
print(f"<|endoftext|>是否存在：{'<|endoftext|>' in new_vocab} | ID：{new_vocab.get('<|endoftext|>', '缺失')}")

# ===================== 步骤3：生成适配Qwen2Tokenizer的完整文件 =====================
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 保存新vocab.json
with open(f"{OUTPUT_DIR}/vocab.json", "w", encoding="utf-8") as f:
    json.dump(new_vocab, f, ensure_ascii=False, indent=2)

# 2. 复用原生配置文件（保证Qwen2Tokenizer兼容）
for fname in ["merges.txt", "tokenizer_config.json", "special_tokens_map.json"]:
    if os.path.exists(f"{ORIGINAL_DIR}/{fname}"):
        shutil.copy(f"{ORIGINAL_DIR}/{fname}", f"{OUTPUT_DIR}/{fname}")

# 3. 更新tokenizer_config.json（关键：适配新词汇表）
with open(f"{OUTPUT_DIR}/tokenizer_config.json", "r", encoding="utf-8") as f:
    tokenizer_config = json.load(f)
tokenizer_config.update({
    "vocab_size": len(new_vocab),
    "pad_token": "<pad>",
    "pad_token_id": new_vocab["<pad>"],
    "unk_token": "<unk>",
    "unk_token_id": new_vocab["<unk>"],
    "eos_token": "<|endoftext|>",
    "eos_token_id": new_vocab["<|endoftext|>"],
    "bos_token": "<|endoftext|>",  # Qwen2Tokenizer用<|endoftext|>作为bos/eos
    "bos_token_id": new_vocab["<|endoftext|>"],
    "model_max_length": 128
})
with open(f"{OUTPUT_DIR}/tokenizer_config.json", "w", encoding="utf-8") as f:
    json.dump(tokenizer_config, f, indent=2)

# 4. 更新special_tokens_map.json
with open(f"{OUTPUT_DIR}/special_tokens_map.json", "r", encoding="utf-8") as f:
    special_map = json.load(f)
special_map.update({
    "pad_token": "<pad>",
    "unk_token": "<unk>",
    "eos_token": "<|endoftext|>",
    "bos_token": "<|endoftext|>"
})
with open(f"{OUTPUT_DIR}/special_tokens_map.json", "w", encoding="utf-8") as f:
    json.dump(special_map, f, indent=2)

print(f"\n5万词表Tokenizer已保存到：{OUTPUT_DIR}")

# ===================== 步骤4：加载并验证Qwen2Tokenizer =====================
print("\n=== 加载并验证Tokenizer ===")
# 加载剪枝后的Tokenizer（原生Qwen2Tokenizer）
new_tokenizer = Qwen2Tokenizer.from_pretrained(OUTPUT_DIR)
# 强制绑定特殊Token（兜底）
new_tokenizer.pad_token = "<pad>"
new_tokenizer.unk_token = "<unk>"
new_tokenizer.bos_token = "<|endoftext|>"
new_tokenizer.eos_token = "<|endoftext|>"

# 测试编码/解码
test_addresses = [
    "Av. Paulista, 1000 - São Paulo, SP, 01310-100",
    "Rua das Flores, 50 - Rio de Janeiro, RJ, 20031-170"
]

print("\n=== 编码/解码验证 ===")
for addr in test_addresses:
    # 编码（Qwen2Tokenizer原生参数）
    encoded = new_tokenizer(
        addr,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128,
        add_special_tokens=True
    )
    # 解码（跳过特殊Token）
    decoded = new_tokenizer.decode(encoded.input_ids[0], skip_special_tokens=True)
    # 清理多余空格
    decoded = decoded.replace("  ", " ").strip()
    print(f"原始地址：{addr}")
    print(f"解码地址：{decoded}")
    print("-" * 60)

# 验证关键信息
print(f"\n=== 最终验证信息 ===")
print(f"词表大小：{new_tokenizer.vocab_size}")
print(f"pad_token ID：{new_tokenizer.pad_token_id}")
print(f"unk_token ID：{new_tokenizer.unk_token_id}")
print(f"eos/bos_token ID：{new_tokenizer.eos_token_id}")

# 保存最终配置
new_tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\n所有配置已保存，5万词表Qwen2Tokenizer构建完成！")