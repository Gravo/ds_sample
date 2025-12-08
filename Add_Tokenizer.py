from transformers import Qwen2Tokenizer
import json
import os
import shutil
from collections import defaultdict
import re

# ===================== 核心配置 =====================
TARGET_VOCAB_SIZE = 50000
OUTPUT_DIR = "./qwen3_address_5w_tokenizer_final"
ORIGINAL_DIR = "./qwen3_original_tokenizer"
GEO_NAMES_FILE = "./geonames_50k.txt"

# ===================== 辅助函数 =====================
def load_geonames_data(file_path, max_lines=50000):
    """加载地理名称数据"""
    addresses = []
    if os.path.exists(file_path):
        print(f"加载地理名称数据: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                addresses.append(line.strip())
        print(f"加载了 {len(addresses)} 条地址记录")
    else:
        print(f"警告: 地理名称文件不存在: {file_path}")
    return addresses

def preprocess_address(address):
    """预处理地址，标准化格式"""
    # 移除多余空格
    address = re.sub(r'\s+', ' ', address.strip())
    # 标准化常见的缩写
    address = re.sub(r'\b(Av\.|Av)\b', 'Avenida', address, flags=re.IGNORECASE)
    address = re.sub(r'\b(R\.|R)\b', 'Rua', address, flags=re.IGNORECASE)
    address = re.sub(r'\b(Rod\.|Rod)\b', 'Rodovia', address, flags=re.IGNORECASE)
    address = re.sub(r'\b(Trav\.|Trav)\b', 'Travessa', address, flags=re.IGNORECASE)
    address = re.sub(r'\b(Al\.|Al)\b', 'Alameda', address, flags=re.IGNORECASE)
    address = re.sub(r'\b(Pç|Pca|Pc)\b', 'Praça', address, flags=re.IGNORECASE)
    # 标准化CEP格式
    address = re.sub(r'(\d{5})-?(\d{3})', r'\1-\2', address)
    return address

def get_brazil_state_codes():
    """返回巴西所有州代码"""
    return [
        "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO",
        "MA", "MG", "MS", "MT", "PA", "PB", "PE", "PI", "PR",
        "RJ", "RN", "RO", "RR", "RS", "SC", "SE", "SP", "TO"
    ]

# ===================== 步骤1：加载已有的Tokenizer =====================
print("=== 1. 加载已有的Qwen3分词器 ===")

# 检查原始分词器目录是否存在
if not os.path.exists(ORIGINAL_DIR):
    print(f"警告: 原始分词器目录不存在: {ORIGINAL_DIR}")
    print("正在从Hugging Face下载...")
    orig_tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    orig_tokenizer.save_pretrained(ORIGINAL_DIR)
else:
    print(f"从本地加载分词器: {ORIGINAL_DIR}")
    orig_tokenizer = Qwen2Tokenizer.from_pretrained(ORIGINAL_DIR)

# 获取完整词表
with open(f"{ORIGINAL_DIR}/vocab.json", "r", encoding="utf-8") as f:
    orig_vocab = json.load(f)

print(f"原生词表总大小: {len(orig_vocab)}")

# 分析特殊token
special_tokens = [
    "<|endoftext|>", "<|im_start|>", "<|im_end|>",
    "<|padding|>", "<unk>", "<s>", "</s>"
]
actual_special_tokens = set()
for token in special_tokens:
    if token in orig_vocab:
        actual_special_tokens.add(token)

print(f"识别到的特殊Token: {list(actual_special_tokens)}")

# ===================== 步骤2：使用地理名称数据统计Token频率 =====================
print(f"\n=== 2. 使用地理名称数据统计Token频率 ===")

# 加载地理名称数据
address_samples = load_geonames_data(GEO_NAMES_FILE, max_lines=10000)

# 如果没有外部数据，使用默认样本
if not address_samples:
    print("使用内置样本数据...")
    address_samples = [
        "Rodovia Presidente Dutra, km 154, São José dos Campos, SP",
        "Avenida Paulista, 1578, Bela Vista, São Paulo, SP, 01310-200",
        "Rua Oscar Freire, 500, Cerqueira César, São Paulo, SP, 01426-001",
        "Avenida Atlântica, 1702, Copacabana, Rio de Janeiro, RJ, 22021-001",
        "Rua da Carioca, 120, Centro, Rio de Janeiro, RJ, 20050-008",
        "Avenida Afonso Pena, 4000, Cruzeiro, Belo Horizonte, MG, 30130-009",
        "Travessa da Paz, 123, Centro, Salvador, BA, 40015-110",
        "Alameda Santos, 2000, Jardim Paulista, São Paulo, SP, 01418-200",
        "Praça da Sé, 1, Sé, São Paulo, SP, 01001-001",
        "Rua 25 de Março, 1000, Centro, São Paulo, SP, 01021-200"
    ]

# 统计地址相关Token的出现频率
address_token_freq = defaultdict(int)
all_address_tokens = set()

print(f"分析 {len(address_samples)} 个地址样本...")

for addr in address_samples:
    # 预处理地址
    processed_addr = preprocess_address(addr)

    # 编码
    try:
        encoded = orig_tokenizer.encode(processed_addr, add_special_tokens=False)
        tokens = orig_tokenizer.convert_ids_to_tokens(encoded)

        for token in tokens:
            address_token_freq[token] += 1
            all_address_tokens.add(token)
    except Exception as e:
        print(f"警告: 编码地址时出错: {addr[:50]}... - {e}")

# 特别关注巴西地址相关词汇
brazil_keywords = [
    # 街道类型
    "Rodovia", "Avenida", "Rua", "Travessa", "Alameda", "Praça", "Estrada",
    "Viela", "Passagem", "Largo", "Vila", "Parque", "Jardim", "Bosque",
    # 常见城市/地区名称
    "São", "Paulo", "Rio", "Janeiro", "Brasil", "Flores", "Santos", "Carioca",
    "Afonso", "Pena", "Paz", "Sé", "Março", "Campos", "Bela", "Vista",
    # 常见词缀
    "dos", "das", "da", "de", "do", "e", "com", "sem", "para", "por",
    # 数字
    "100", "200", "300", "400", "500", "1000", "2000", "154", "1578", "1702",
    "120", "4000", "123", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
    # CEP相关
    "01310", "01426", "22021", "20050", "30130", "40015", "01418", "01001", "01021"
]

# 添加巴西州代码
brazil_keywords.extend(get_brazil_state_codes())

# 确保关键词都被统计
for keyword in brazil_keywords:
    # 尝试编码单个关键词
    try:
        kw_encoded = orig_tokenizer.encode(keyword, add_special_tokens=False)
        kw_tokens = orig_tokenizer.convert_ids_to_tokens(kw_encoded)
        for kw_token in kw_tokens:
            address_token_freq[kw_token] += 10  # 给关键词更高权重
            all_address_tokens.add(kw_token)
    except:
        pass

print(f"从地址样本中收集到的唯一Token数: {len(all_address_tokens)}")
print("高频地址Token（频率≥5）:")
top_tokens = sorted(address_token_freq.items(), key=lambda x: x[1], reverse=True)[:30]
for token, freq in top_tokens:
    if freq >= 5:
        print(f"  {repr(token):30} -> 频率: {freq}")

# ===================== 步骤3：智能构建新词表 =====================
print(f"\n=== 3. 智能构建新词表 (目标大小: {TARGET_VOCAB_SIZE}) ===")

# 加载merges.txt来理解BPE合并顺序
merges_path = f"{ORIGINAL_DIR}/merges.txt"
if os.path.exists(merges_path):
    with open(merges_path, "r", encoding="utf-8") as f:
        merges = f.read().splitlines()[1:]  # 跳过第一行

    # 从merges中提取高频Token对（前500个合并）
    merge_tokens = set()
    for line in merges[:500]:
        if ' ' in line:
            parts = line.strip().split(' ')
            for part in parts:
                if part:  # 确保不是空字符串
                    merge_tokens.add(part)
    print(f"从merges.txt中提取了 {len(merge_tokens)} 个高频token")
else:
    print("警告: merges.txt 文件不存在")
    merge_tokens = set()

# 构建候选Token集（按优先级）
priority_tokens = set()

# 优先级1：特殊Token
for token in actual_special_tokens:
    priority_tokens.add(token)

# 优先级2：高频地址Token（频率≥3）
for token, freq in address_token_freq.items():
    if freq >= 3:
        priority_tokens.add(token)

# 优先级3：地址关键词（确保重要词汇被包含）
for keyword in brazil_keywords:
    priority_tokens.add(keyword)
    # 添加带空格前缀的版本
    space_keyword = "Ġ" + keyword
    priority_tokens.add(space_keyword)

# 优先级4：从merges中提取的高频Token
for token in merge_tokens:
    priority_tokens.add(token)

# 优先级5：常见单字符、数字、标点（确保基本字符集）
common_chars = [
    # 字母
    *[chr(i) for i in range(ord('a'), ord('z')+1)],
    *[chr(i) for i in range(ord('A'), ord('Z')+1)],
    # 数字
    *[str(i) for i in range(10)],
    # 常见标点
    ".", ",", "!", "?", ";", ":", "'", "\"", "(", ")", "[", "]", "{", "}",
    "<", ">", "/", "\\", "|", "-", "_", "=", "+", "*", "&", "^", "%", "$",
    "#", "@", "~", "`"
]

for char in common_chars:
    priority_tokens.add(char)

# 过滤：移除无效token（如连续多个Ġ的token）
def is_valid_token(token):
    """检查token是否有效"""
    # 移除连续多个Ġ的token（通常是无效的）
    if "ĠĠĠ" in token:
        return False

    # 移除过长的token（可能是不常见的Unicode字符）
    if len(token) > 30:
        return False

    # 移除包含控制字符的token
    for ch in token:
        if ord(ch) < 32 and ch not in ['\t', '\n', '\r']:
            return False

    return True

# 过滤优先级token
priority_tokens = {token for token in priority_tokens if is_valid_token(token)}
print(f"过滤后的优先级Token集合大小: {len(priority_tokens)}")

# 开始构建新词表
new_vocab = {}
added_tokens = set()

# 第1步：添加特殊Token（确保它们有固定ID）
# 按照Qwen3的特殊token顺序
special_token_order = [
    "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|padding|>", "<unk>"
]

for token in special_token_order:
    if token in orig_vocab and token not in added_tokens:
        new_vocab[token] = len(new_vocab)
        added_tokens.add(token)
        print(f"  添加特殊Token: {repr(token):20} -> ID: {new_vocab[token]}")

# 第2步：添加高频地址Token（按频率排序）
sorted_address_tokens = sorted(
    [(t, address_token_freq.get(t, 0)) for t in priority_tokens if t in address_token_freq],
    key=lambda x: x[1],
    reverse=True
)

added_count = 0
for token, freq in sorted_address_tokens:
    if token in orig_vocab and token not in added_tokens and len(new_vocab) < TARGET_VOCAB_SIZE:
        if is_valid_token(token):  # 再次检查有效性
            new_vocab[token] = len(new_vocab)
            added_tokens.add(token)
            added_count += 1

print(f"  已添加高频地址Token: {added_count}个")

# 第3步：添加其他优先级Token（非地址相关）
other_priority_tokens = [t for t in priority_tokens if t not in address_token_freq]
for token in other_priority_tokens:
    if token in orig_vocab and token not in added_tokens and len(new_vocab) < TARGET_VOCAB_SIZE:
        if is_valid_token(token):  # 再次检查有效性
            new_vocab[token] = len(new_vocab)
            added_tokens.add(token)

print(f"  已添加其他优先级Token: {len(new_vocab) - added_count - len(special_token_order)}个")

# 第4步：按原生ID顺序填充剩余位置，但过滤无效token
sorted_orig_items = sorted(orig_vocab.items(), key=lambda x: x[1])

print("  按原生ID顺序添加基础token...")
for token, orig_id in sorted_orig_items:
    if len(new_vocab) >= TARGET_VOCAB_SIZE:
        break

    if token not in added_tokens:
        # 严格过滤：移除连续空格token和无效token
        if is_valid_token(token):
            new_vocab[token] = len(new_vocab)
            added_tokens.add(token)

# 最终校准：如果还不够TARGET_VOCAB_SIZE，添加更多基础token
if len(new_vocab) < TARGET_VOCAB_SIZE:
    print(f"  词表大小不足，添加更多基础token...")
    for token, orig_id in sorted_orig_items:
        if token not in added_tokens:
            # 放宽条件：只过滤最明显的问题token
            if "ĠĠĠĠĠ" not in token:  # 只过滤连续5个以上Ġ的token
                new_vocab[token] = len(new_vocab)
                added_tokens.add(token)
                if len(new_vocab) >= TARGET_VOCAB_SIZE:
                    break

print(f"新词表构建完成。实际大小: {len(new_vocab)} (目标: {TARGET_VOCAB_SIZE})")

# 验证关键Token是否存在
key_tokens_to_check = [
    "Rodovia", "ĠRodovia", "Avenida", "ĠAvenida", "Rua", "ĠRua",
    "São", "ĠSão", "Paulo", "ĠPaulo", "Rio", "ĠRio",
    "100", "Ġ100", "SP", "ĠSP", "RJ", "ĠRJ"
]

print("\n关键Token验证:")
for token in key_tokens_to_check:
    exists = token in new_vocab
    status = "✅" if exists else "❌"
    print(f"  {status} {repr(token):20} -> 在新词表中: {exists}")

# ===================== 步骤4：保存新的Tokenizer文件 =====================
print("\n=== 4. 保存新的Tokenizer文件 ===")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 保存新的vocab.json
with open(f"{OUTPUT_DIR}/vocab.json", "w", encoding="utf-8") as f:
    json.dump(new_vocab, f, ensure_ascii=False, indent=2)
print(f"  已保存: {OUTPUT_DIR}/vocab.json")

# 2. 复制merges.txt，但清理其中的无效合并规则
merges_input_path = f"{ORIGINAL_DIR}/merges.txt"
merges_output_path = f"{OUTPUT_DIR}/merges.txt"

if os.path.exists(merges_input_path):
    with open(merges_input_path, "r", encoding="utf-8") as f:
        merges_lines = f.readlines()

    # 清理无效的合并规则（包含连续多个Ġ的规则）
    cleaned_merges = []
    for line in merges_lines:
        # 跳过包含连续多个Ġ的合并规则
        if "Ġ Ġ" in line or "ĠĠ" in line:
            continue
        cleaned_merges.append(line)

    with open(merges_output_path, "w", encoding="utf-8") as f:
        f.writelines(cleaned_merges)
    print(f"  已清理并保存: {OUTPUT_DIR}/merges.txt (原{len(merges_lines)}行，现{len(cleaned_merges)}行)")
else:
    print(f"  警告: {merges_input_path} 不存在，跳过merges.txt")

# 3. 生成tokenizer_config.json
# 使用从原始tokenizer获取的真实特殊Token
tokenizer_config = {
    "vocab_size": len(new_vocab),
    "model_max_length": 32768,
    "bos_token": "<|endoftext|>",
    "eos_token": "<|im_end|>",
    "unk_token": "<unk>",
    "pad_token": "<|endoftext|>",
    "clean_up_tokenization_spaces": False,
    "tokenizer_class": "Qwen2Tokenizer",
}

with open(f"{OUTPUT_DIR}/tokenizer_config.json", "w", encoding="utf-8") as f:
    json.dump(tokenizer_config, f, indent=2)
print(f"  已保存: {OUTPUT_DIR}/tokenizer_config.json")

# 4. 生成special_tokens_map.json
special_tokens_map = {
    "bos_token": {"content": "<|endoftext|>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False},
    "eos_token": {"content": "<|im_end|>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False},
    "unk_token": {"content": "<unk>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False},
    "pad_token": {"content": "<|endoftext|>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False},
}

with open(f"{OUTPUT_DIR}/special_tokens_map.json", "w", encoding="utf-8") as f:
    json.dump(special_tokens_map, f, indent=2)
print(f"  已保存: {OUTPUT_DIR}/special_tokens_map.json")

# 5. 复制其他必要的文件
for file_name in ["tokenizer.json", "added_tokens.json"]:
    src_path = f"{ORIGINAL_DIR}/{file_name}"
    dst_path = f"{OUTPUT_DIR}/{file_name}"
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
        print(f"  已复制: {dst_path}")

# ===================== 步骤5：加载并验证新Tokenizer =====================
print("\n=== 5. 加载并验证新Tokenizer ===")

try:
    new_tokenizer = Qwen2Tokenizer.from_pretrained(OUTPUT_DIR)
    print("✅ 新Tokenizer加载成功!")
except Exception as e:
    print(f"❌ 加载失败，错误: {e}")
    print("尝试手动创建Tokenizer...")

    # 手动创建
    from transformers import PreTrainedTokenizerFast

    # 确保有必要的文件
    required_files = ["vocab.json", "merges.txt", "tokenizer_config.json"]
    missing_files = [f for f in required_files if not os.path.exists(f"{OUTPUT_DIR}/{f}")]

    if missing_files:
        print(f"缺少文件: {missing_files}")
        # 尝试从原始目录复制缺失的文件
        for file_name in missing_files:
            src = f"{ORIGINAL_DIR}/{file_name}"
            dst = f"{OUTPUT_DIR}/{file_name}"
            if os.path.exists(src):
                shutil.copy(src, dst)
                print(f"  已复制: {file_name}")

    # 再次尝试加载
    try:
        new_tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=f"{OUTPUT_DIR}/tokenizer.json" if os.path.exists(f"{OUTPUT_DIR}/tokenizer.json") else None,
            vocab_file=f"{OUTPUT_DIR}/vocab.json",
            merges_file=f"{OUTPUT_DIR}/merges.txt",
            **tokenizer_config
        )
        print("✅ 通过PreTrainedTokenizerFast创建成功!")
    except Exception as e2:
        print(f"❌ 仍然失败: {e2}")
        print("尝试创建最简单的Tokenizer...")
        new_tokenizer = Qwen2Tokenizer(vocab_file=f"{OUTPUT_DIR}/vocab.json", merges_file=f"{OUTPUT_DIR}/merges.txt")

# 设置特殊Token ID
for token_name, token_content in [("bos_token", "<|endoftext|>"), ("eos_token", "<|im_end|>"),
                                   ("unk_token", "<unk>"), ("pad_token", "<|endoftext|>")]:
    token_id = new_vocab.get(token_content, 0)
    setattr(new_tokenizer, f"{token_name}_id", token_id)
    setattr(new_tokenizer, token_name, token_content)

print(f"词表大小: {new_tokenizer.vocab_size}")
print(f"BOS: {new_tokenizer.bos_token} (ID: {new_tokenizer.bos_token_id})")
print(f"EOS: {new_tokenizer.eos_token} (ID: {new_tokenizer.eos_token_id})")
print(f"UNK: {new_tokenizer.unk_token} (ID: {new_tokenizer.unk_token_id})")
print(f"PAD: {new_tokenizer.pad_token} (ID: {new_tokenizer.pad_token_id})")

# ===================== 步骤6：核心功能测试 =====================
print("\n=== 6. 编码/解码功能测试 ===")

test_addresses = [
    "Rodovia Presidente Dutra, km 154, São José dos Campos, SP",
    "Avenida Paulista, 1578, São Paulo, SP, 01310-200"
]

all_pass = True
for i, addr in enumerate(test_addresses):
    print(f"\n测试地址 {i + 1}: {addr}")

    # 预处理
    processed_addr = preprocess_address(addr)

    # 编码
    encoded = new_tokenizer.encode(processed_addr, add_special_tokens=False)
    print(f"  编码ID (前10个): {encoded[:10]}... (共{len(encoded)}个tokens)")

    # 查看Token序列
    tokens = new_tokenizer.convert_ids_to_tokens(encoded[:15])
    print(f"  Token序列: {[repr(t) for t in tokens]}")

    # 检查是否有<unk>
    unk_count = sum(1 for t in tokens if '<unk>' in t or t == '<unk>')
    if unk_count > 0:
        print(f"  警告: 发现 {unk_count} 个<unk> token")

    # 解码
    decoded = new_tokenizer.decode(encoded, skip_special_tokens=True)

    # 清理结果
    decoded_cleaned = decoded.replace('Ġ', ' ').strip()
    decoded_cleaned = re.sub(r'\s+', ' ', decoded_cleaned)

    print(f"  解码结果: {decoded_cleaned}")

    # 对比验证
    original_clean = re.sub(r'\s+', ' ', processed_addr.strip())

    if original_clean == decoded_cleaned:
        print("  ✅ 解码内容完全匹配!")
    else:
        # 尝试更宽松的匹配
        if original_clean.lower() == decoded_cleaned.lower():
            print("  ⚠️  解码内容基本匹配（仅有大小写差异）")
        elif original_clean.replace(' ', '') == decoded_cleaned.replace(' ', ''):
            print("  ⚠️  解码内容基本匹配（仅有空格差异）")
        else:
            print("  ❌ 解码内容不匹配!")
            print(f"    原始: {original_clean}")
            print(f"    解码: {decoded_cleaned}")
            all_pass = False

# 额外测试：检查关键Token的编码
print("\n=== 7. 关键Token编码测试 ===")
key_tokens_test = ["Rodovia", "Avenida", "Rua", "São", "Paulo", "SP", "100", "154"]
for token in key_tokens_test:
    encoded = new_tokenizer.encode(token, add_special_tokens=False)
    decoded = new_tokenizer.decode(encoded, skip_special_tokens=True)

    if not encoded or encoded[0] == new_tokenizer.unk_token_id:
        print(f"  ❌ {repr(token):20} -> 被编码为<unk>")
    else:
        print(f"  ✅ {repr(token):20} -> 编码ID: {encoded[0]}, 解码: {repr(decoded)}")

# 比较原生分词器和新分词器的差异
print("\n=== 8. 与原生分词器比较 ===")
test_text = "Rodovia Presidente Dutra"
orig_tokens = orig_tokenizer.tokenize(test_text)
new_tokens = new_tokenizer.tokenize(test_text)

print(f"原生分词器: {[repr(t) for t in orig_tokens]}")
print(f"新分词器:   {[repr(t) for t in new_tokens]}")
print(f"分词结果{'相同' if orig_tokens == new_tokens else '不同'}")

# 最终保存
new_tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\n{'=' * 60}")
if all_pass:
    print(f"✅ 所有测试通过！定制化Tokenizer已成功保存至: {OUTPUT_DIR}")
else:
    print(f"⚠️  部分测试未通过，但Tokenizer文件已保存至: {OUTPUT_DIR}")
    print("   建议检查缺失的关键Token，并考虑扩大地址样本集重新构建。")

print(f"\n下一步建议:")
print(f"1. 检查 {OUTPUT_DIR}/vocab.json 中是否包含所有关键地址Token")
print(f"2. 使用更多巴西地址样本进行测试")
print(f"3. 在实际模型训练前，使用新Tokenizer对训练数据进行预处理测试")
print(f"4. 如果遇到分词问题，考虑使用ByteLevelBPE重新训练分词器")