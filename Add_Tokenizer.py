from transformers import Qwen2Tokenizer
import json
import os
import shutil
from collections import defaultdict

# ===================== 核心配置 =====================
TARGET_VOCAB_SIZE = 50000
OUTPUT_DIR = "./qwen3_address_5w_tokenizer_final"
ORIGINAL_DIR = "./qwen3_original_tokenizer"

# ===================== 步骤1：下载并深度分析原生Tokenizer =====================
print("=== 1. 下载并深度分析Qwen3原生Tokenizer ===")
orig_tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen3-0.6B")
orig_tokenizer.save_pretrained(ORIGINAL_DIR)

# 获取完整词表（包括特殊Token）
with open(f"{ORIGINAL_DIR}/vocab.json", "r", encoding="utf-8") as f:
    orig_vocab = json.load(f)

print(f"原生词表总大小: {len(orig_vocab)}")

# 通过实际编码测试，找出真正的特殊Token
test_text = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant"
encoded_test = orig_tokenizer.encode(test_text, add_special_tokens=False)
tokens_test = orig_tokenizer.convert_ids_to_tokens(encoded_test)

# 找出所有的特殊Token模式
special_patterns = ["<|", "</|", "<|im_", "<|endof", "<|padding", "[UNK]", "[PAD]", "[CLS]", "[SEP]"]
actual_special_tokens = set()
for token in tokens_test:
    if any(pattern in token for pattern in special_patterns):
        actual_special_tokens.add(token)

print(f"通过编码测试发现的特殊Token: {list(actual_special_tokens)}")

# ===================== 步骤2：分析地址样本的Token频率 =====================
print("\n=== 2. 分析地址样本的Token频率 ===")

# 测试地址集
test_addresses = [
    "Av. Paulista, 1000 - São Paulo, SP, 01310-100",
    "Rua das Flores, 50 - Rio de Janeiro, RJ, 20031-170",
    "Avenida Brasil, 2000 - Belo Horizonte, MG, 30130-001",
    "Travessa da Paz, 123 - Salvador, BA, 40015-110",
    "Alameda Santos, 2000 - São Paulo, SP, 01418-200",
    "Rua Oscar Freire, 500 - São Paulo, SP, 01426-001",
    "Avenida Atlântica, 100 - Rio de Janeiro, RJ, 22010-000",
    "Praça da Sé, 1 - São Paulo, SP, 01001-001",
    "Rua 25 de Março, 1000 - São Paulo, SP, 01021-200",
    "Avenida Paulista, 1578 - São Paulo, SP, 01310-200"
]

# 扩展地址关键词（葡萄牙语/巴西地址常用词）
address_keywords = [
    "Rua", "rua", "Avenida", "avenida", "Av", "av", "Av.", "av.",
    "Travessa", "travessa", "Alameda", "alameda", "Praça", "praça",
    "São", "são", "Paulo", "Rio", "Janeiro", "Brasil", "Flores",
    "das", "dos", "das", "de", "da", "do", "e", "com", "sem",
    "SP", "RJ", "MG", "BA", "PR", "SC", "RS", "DF",
    "1000", "500", "2000", "100", "50", "123", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
    "01310", "20031", "30130", "40015", "01418", "01426", "22010", "01001", "01021"
]

# 统计地址相关Token的出现频率
address_token_freq = defaultdict(int)
all_address_tokens = set()

for addr in test_addresses:
    encoded = orig_tokenizer.encode(addr, add_special_tokens=False)
    tokens = orig_tokenizer.convert_ids_to_tokens(encoded)

    for token in tokens:
        address_token_freq[token] += 1
        all_address_tokens.add(token)

    # 同时添加原始文本中的关键词（可能被拆分成多个token）
    for keyword in address_keywords:
        if keyword.lower() in addr.lower():
            # 尝试编码单个关键词
            try:
                kw_encoded = orig_tokenizer.encode(keyword, add_special_tokens=False)
                kw_tokens = orig_tokenizer.convert_ids_to_tokens(kw_encoded)
                for kw_token in kw_tokens:
                    address_token_freq[kw_token] += 5  # 给关键词更高权重
                    all_address_tokens.add(kw_token)
            except:
                pass

print(f"从地址样本中收集到的唯一Token数: {len(all_address_tokens)}")
print(f"高频地址Token（频率≥5）:")
for token, freq in sorted(address_token_freq.items(), key=lambda x: x[1], reverse=True)[:20]:
    if freq >= 5:
        print(f"  {token:20} -> 频率: {freq}")

# ===================== 步骤3：智能构建新词表 =====================
print(f"\n=== 3. 智能构建新词表 (目标大小: {TARGET_VOCAB_SIZE}) ===")

# 策略：混合选择
# 1. 必须保留的特殊Token（通过实际编码发现的）
# 2. 高频地址Token（频率高的优先）
# 3. 原生高频Token（按使用频率排序，而非ID顺序）

# 首先，我们需要获取Token的实际使用频率信息
# 由于Qwen3没有直接提供频率信息，我们根据一些启发式规则：
# - 短Token（1-3个字符）通常是高频Token
# - 常见标点、数字、字母通常是高频Token
# - 长度适中的英文单词可能是高频Token

# 加载merges.txt来理解BPE合并顺序（越早出现的合并越可能是高频）
with open(f"{ORIGINAL_DIR}/merges.txt", "r", encoding="utf-8") as f:
    merges = f.read().splitlines()[1:]  # 跳过第一行

# 从merges中提取高频Token对
merge_pairs = []
for line in merges[:1000]:  # 只看前1000个合并（通常是最高频的）
    if ' ' in line:
        parts = line.split(' ')
        if len(parts) == 2:
            merge_pairs.extend(parts)

# 构建候选Token集（按优先级）
priority_tokens = set()

# 优先级1：实际发现的特殊Token
for token in actual_special_tokens:
    if token in orig_vocab:
        priority_tokens.add(token)

# 优先级2：高频地址Token（频率≥2）
for token, freq in address_token_freq.items():
    if freq >= 2 and token in orig_vocab:
        priority_tokens.add(token)

# 优先级3：地址关键词对应的Token
for keyword in address_keywords:
    # 尝试直接查找
    if keyword in orig_vocab:
        priority_tokens.add(keyword)
    # 尝试带空格前缀
    space_keyword = "Ġ" + keyword
    if space_keyword in orig_vocab:
        priority_tokens.add(space_keyword)

# 优先级4：从merges中提取的高频Token
for token in merge_pairs:
    if token in orig_vocab:
        priority_tokens.add(token)
    # 处理带数字的变体
    if token.isdigit() and len(token) <= 4:
        priority_tokens.add(token)

# 优先级5：常见单字符、数字、标点
common_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;:'\"()-[]{}<>/"
for char in common_chars:
    if char in orig_vocab:
        priority_tokens.add(char)

print(f"优先级Token集合大小: {len(priority_tokens)}")

# 开始构建新词表
new_vocab = {}
added_tokens = set()

# 第1步：添加特殊Token（确保它们有固定ID）
special_tokens_to_add = ["<unk>", "<|endoftext|>", "<|im_start|>", "<|im_end|>"]
for token in special_tokens_to_add:
    if token in orig_vocab and token not in added_tokens:
        new_vocab[token] = len(new_vocab)
        added_tokens.add(token)
        print(f"  添加特殊Token: {token} -> ID: {new_vocab[token]}")

# 第2步：添加高频地址Token（按频率排序）
sorted_address_tokens = sorted(
    [(t, address_token_freq.get(t, 0)) for t in priority_tokens if t in address_token_freq],
    key=lambda x: x[1],
    reverse=True
)

for token, freq in sorted_address_tokens:
    if token in orig_vocab and token not in added_tokens and len(new_vocab) < TARGET_VOCAB_SIZE:
        new_vocab[token] = len(new_vocab)
        added_tokens.add(token)

print(f"  已添加高频地址Token: {len([t for t in added_tokens if t in address_token_freq])}个")

# 第3步：添加其他优先级Token
for token in priority_tokens:
    if token in orig_vocab and token not in added_tokens and len(new_vocab) < TARGET_VOCAB_SIZE:
        new_vocab[token] = len(new_vocab)
        added_tokens.add(token)

print(f"  已添加其他优先级Token: {len(added_tokens) - len([t for t in added_tokens if t in address_token_freq])}个")

# 第4步：按原生ID顺序填充剩余位置（确保覆盖所有基础字符）
# 但跳过明显的低频Token（如非常长的Unicode字符）
sorted_orig_items = sorted(orig_vocab.items(), key=lambda x: x[1])

for token, orig_id in sorted_orig_items:
    if len(new_vocab) >= TARGET_VOCAB_SIZE:
        break

    if token not in added_tokens:
        # 过滤掉明显低频的Token（非常长的token或特殊Unicode）
        if len(token) <= 20:  # 跳过过长的token
            new_vocab[token] = len(new_vocab)
            added_tokens.add(token)

# 最终校准：如果还不够50000，添加更多基础token
if len(new_vocab) < TARGET_VOCAB_SIZE:
    for token, orig_id in sorted_orig_items:
        if token not in added_tokens:
            new_vocab[token] = len(new_vocab)
            added_tokens.add(token)
            if len(new_vocab) >= TARGET_VOCAB_SIZE:
                break

print(f"新词表构建完成。实际大小: {len(new_vocab)} (目标: {TARGET_VOCAB_SIZE})")

# 验证关键Token是否存在
key_tokens_to_check = ["Rua", "ĠRua", "Flores", "ĠFlores", "Janeiro", "ĠJaneiro", "Paulista", "ĠPaulista"]
print("\n关键Token验证:")
for token in key_tokens_to_check:
    exists = token in new_vocab
    status = "✅" if exists else "❌"
    print(f"  {status} {token:20} -> 在新词表中: {exists}")

# ===================== 步骤4：保存新的Tokenizer文件 =====================
print("\n=== 4. 保存新的Tokenizer文件 ===")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 保存新的vocab.json
with open(f"{OUTPUT_DIR}/vocab.json", "w", encoding="utf-8") as f:
    json.dump(new_vocab, f, ensure_ascii=False, indent=2)
print(f"  已保存: {OUTPUT_DIR}/vocab.json")

# 2. 完全复用原生的merges.txt
shutil.copy(f"{ORIGINAL_DIR}/merges.txt", f"{OUTPUT_DIR}/merges.txt")
print(f"  已复制: {OUTPUT_DIR}/merges.txt")

# 3. 生成tokenizer_config.json
# 使用从原始tokenizer获取的真实特殊Token
tokenizer_config = {
    "vocab_size": len(new_vocab),
    "model_max_length": 32768,
    "bos_token": orig_tokenizer.bos_token if hasattr(orig_tokenizer, 'bos_token') else "<|endoftext|>",
    "eos_token": orig_tokenizer.eos_token if hasattr(orig_tokenizer, 'eos_token') else "<|im_end|>",
    "unk_token": orig_tokenizer.unk_token if hasattr(orig_tokenizer, 'unk_token') else "<unk>",
    "pad_token": orig_tokenizer.pad_token if hasattr(orig_tokenizer, 'pad_token') else "<|endoftext|>",
    "clean_up_tokenization_spaces": False,
    "tokenizer_class": "Qwen2Tokenizer",
}

with open(f"{OUTPUT_DIR}/tokenizer_config.json", "w", encoding="utf-8") as f:
    json.dump(tokenizer_config, f, indent=2)
print(f"  已保存: {OUTPUT_DIR}/tokenizer_config.json")

# 4. 生成special_tokens_map.json
special_tokens_map = {
    "bos_token": tokenizer_config["bos_token"],
    "eos_token": tokenizer_config["eos_token"],
    "unk_token": tokenizer_config["unk_token"],
    "pad_token": tokenizer_config["pad_token"],
}
with open(f"{OUTPUT_DIR}/special_tokens_map.json", "w", encoding="utf-8") as f:
    json.dump(special_tokens_map, f, indent=2)
print(f"  已保存: {OUTPUT_DIR}/special_tokens_map.json")

# ===================== 步骤5：加载并验证新Tokenizer =====================
print("\n=== 5. 加载并验证新Tokenizer ===")

try:
    new_tokenizer = Qwen2Tokenizer.from_pretrained(OUTPUT_DIR)
    print("✅ 新Tokenizer加载成功!")
except Exception as e:
    print(f"❌ 加载失败，错误: {e}")
    # 尝试手动创建Tokenizer
    from transformers import PreTrainedTokenizerFast

    new_tokenizer = PreTrainedTokenizerFast(
        vocab_file=f"{OUTPUT_DIR}/vocab.json",
        merges_file=f"{OUTPUT_DIR}/merges.txt",
        **tokenizer_config
    )
    print("✅ 通过PreTrainedTokenizerFast创建成功!")

# 设置特殊Token ID
new_tokenizer.bos_token_id = new_vocab.get(tokenizer_config["bos_token"], 0)
new_tokenizer.eos_token_id = new_vocab.get(tokenizer_config["eos_token"], 0)
new_tokenizer.unk_token_id = new_vocab.get(tokenizer_config["unk_token"], 0)
new_tokenizer.pad_token_id = new_vocab.get(tokenizer_config["pad_token"], 0)

print(f"词表大小: {new_tokenizer.vocab_size}")
print(f"BOS: {new_tokenizer.bos_token} (ID: {new_tokenizer.bos_token_id})")
print(f"EOS: {new_tokenizer.eos_token} (ID: {new_tokenizer.eos_token_id})")
print(f"UNK: {new_tokenizer.unk_token} (ID: {new_tokenizer.unk_token_id})")
print(f"PAD: {new_tokenizer.pad_token} (ID: {new_tokenizer.pad_token_id})")

# ===================== 步骤6：核心功能测试 =====================
print("\n=== 6. 编码/解码功能测试 ===")

test_addresses = [
    "Av. Paulista, 1000 - São Paulo, SP, 01310-100",
    "Rua das Flores, 50 - Rio de Janeiro, RJ, 20031-170"
]

all_pass = True
for i, addr in enumerate(test_addresses):
    print(f"\n测试地址 {i + 1}: {addr}")

    # 编码
    encoded = new_tokenizer.encode(addr, add_special_tokens=False)
    print(f"  编码ID (前10个): {encoded[:10]}... (共{len(encoded)}个tokens)")

    # 查看Token序列
    tokens = new_tokenizer.convert_ids_to_tokens(encoded[:15])
    print(f"  Token序列: {tokens}")

    # 检查是否有<unk>
    unk_count = tokens.count('<unk>') + tokens.count('Ġ<unk>')
    if unk_count > 0:
        print(f"  警告: 发现 {unk_count} 个<unk> token")

    # 解码
    decoded = new_tokenizer.decode(encoded, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    # 清理结果
    decoded_cleaned = decoded.replace('Ġ', ' ').strip()
    import re

    decoded_cleaned = re.sub(r'\s+', ' ', decoded_cleaned)

    print(f"  解码结果: {decoded_cleaned}")

    # 对比验证
    original_clean = re.sub(r'\s+', ' ', addr.strip())
    decoded_clean = re.sub(r'\s+', ' ', decoded_cleaned.strip())

    if original_clean == decoded_clean:
        print("  ✅ 解码内容完全匹配!")
    else:
        # 尝试更宽松的匹配（忽略大小写和少量空格差异）
        if original_clean.lower() == decoded_clean.lower():
            print("  ⚠️  解码内容基本匹配（仅有大小写差异）")
        elif original_clean.replace(' ', '') == decoded_clean.replace(' ', ''):
            print("  ⚠️  解码内容基本匹配（仅有空格差异）")
        else:
            print("  ❌ 解码内容不匹配!")
            all_pass = False

# 额外测试：检查关键Token的编码
print("\n=== 7. 关键Token编码测试 ===")
key_tokens_test = ["Rua", "Flores", "Janeiro", "Paulista", "das", "de"]
for token in key_tokens_test:
    encoded = new_tokenizer.encode(token, add_special_tokens=False)
    decoded = new_tokenizer.decode(encoded, skip_special_tokens=True)

    if encoded[0] == new_tokenizer.unk_token_id:
        print(f"  ❌ {token:15} -> 被编码为<unk>")
    else:
        print(f"  ✅ {token:15} -> 编码ID: {encoded[0]}, 解码: {decoded}")

# 最终保存
new_tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\n{'=' * 60}")
if all_pass:
    print(f"✅ 所有测试通过！定制化Tokenizer已成功保存至: {OUTPUT_DIR}")
else:
    print(f"⚠️  部分测试未通过，但Tokenizer文件已保存至: {OUTPUT_DIR}")
    print("   建议检查缺失的关键Token，并考虑扩大地址样本集重新构建。")

print(f"\n下一步建议:")
print(f"1. 使用更多巴西地址样本重新运行此脚本")
print(f"2. 检查 {OUTPUT_DIR}/vocab.json 中是否包含所有关键地址Token")
print(f"3. 在实际模型训练前，使用新Tokenizer对训练数据进行预处理测试")