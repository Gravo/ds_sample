#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import re
import random
from typing import List, Dict, Tuple
from transformers import Qwen2Tokenizer  # 核心替换为Qwen2专用Tokenizer

# ===================== 1. 配置项（不变） =====================
STREET_TYPES = ["Rua", "Av.", "Avenida", "Travessa", "Praça", "Alameda", "Viela", "Estrada"]
STREET_NAMES = ["das Flores", "Paulista", "Santos", "Brasília", "Rio Branco", "Amazonas"]
CITIES = [("São Paulo", "SP"), ("Rio de Janeiro", "RJ"), ("Brasília", "DF")]


def generate_zipcode() -> str:
    return f"{random.randint(10000, 99999)}-{random.randint(100, 999)}"


ID2LABEL = {0: "邮编", 1: "城市", 2: "州", 3: "街道", 4: "门牌号", 5: "补充信息"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
ZIPCODE_PATTERN = re.compile(r"\d{5}-\d{3}")
ZIPCODE_RAW_PATTERN = re.compile(r"\d{8}")
STATE_CODES = [city[1] for city in CITIES]
CITY_NAMES = [city[0].split() for city in CITIES]


# ===================== 2. Qwen2 Tokenizer加载&校验 =====================
def load_tokenizer(tokenizer_dir: str = "./qwen3_address_5w_tokenizer_final") -> Qwen2Tokenizer:
    """加载Qwen2专用Tokenizer，适配其编码规则"""
    try:
        tokenizer = Qwen2Tokenizer.from_pretrained(
            tokenizer_dir,
            local_files_only=True,
            padding_side="left",
            truncation_side="right"
        )
        tokenizer.pad_token = tokenizer.eos_token  # Qwen2无单独pad_token
        tokenizer.unk_token = "<unk>"
        return tokenizer
    except Exception as e:
        raise ValueError(f"加载Qwen2Tokenizer失败：{e}\n请确认目录下有vocab.json/merges.txt")


def validate_tokenizer(tokenizer: Qwen2Tokenizer) -> None:
    """校验Qwen2Tokenizer的基础Token（适配你的vocab.json）"""
    required_base_tokens = {
        "街道类型（ĠAV）": "ĠAV",
        "数字0": "0",
        "横杠-": "-",
        "小数点.": ".",
        "州缩写SP": "ĠSP"
    }
    missing = []
    for desc, token in required_base_tokens.items():
        if token not in tokenizer.get_vocab():
            missing.append(f"{desc} → Token「{token}」不存在")
        else:
            print(f"✅ {desc}：Token「{token}」ID = {tokenizer.get_vocab()[token]}")

    if missing:
        print(f"⚠️ 警告：缺失基础Token：{missing}")
    else:
        print(f"✅ 所有核心基础Token校验通过！")


# ===================== 3. 适配Qwen2编码的标注函数 =====================
def annotate_address_tokens(address: str, tokenizer: Qwen2Tokenizer) -> Tuple[List[str], List[int]]:
    """标注函数：基于Qwen2拆分后的基础Token匹配"""
    tokens = tokenizer.tokenize(address)
    token_labels = []

    # 临时变量：标记是否在邮编区间（处理数字+横杠的组合）
    in_zipcode = False
    zipcode_token_count = 0

    for token in tokens:
        clean_token = token.replace("Ġ", "").strip()

        # 跳过特殊Token
        if not clean_token or clean_token in [tokenizer.pad_token, tokenizer.unk_token, tokenizer.eos_token]:
            token_labels.append(-100)
            continue

        # 规则1：匹配邮编（处理拆分后的数字+横杠）
        if (clean_token.isdigit() or clean_token == "-") and zipcode_token_count < 9:
            if clean_token == "-" and zipcode_token_count == 5:  # 横杠在第5位（符合XXX-XXX格式）
                in_zipcode = True
            if in_zipcode or (clean_token.isdigit() and zipcode_token_count < 9):
                token_labels.append(LABEL2ID["邮编"])
                zipcode_token_count += 1
                continue

        # 规则2：匹配州缩写（如SP/RJ）
        if clean_token in STATE_CODES:
            token_labels.append(LABEL2ID["州"])
            continue

        # 规则3：匹配门牌号（纯数字，且不是邮编的一部分）
        if clean_token.isdigit() and not in_zipcode and len(clean_token) <= 6:
            token_labels.append(LABEL2ID["门牌号"])
            continue

        # 规则4：匹配城市名（如São/Paulo）
        if any(clean_token in city_words for city_words in CITY_NAMES):
            token_labels.append(LABEL2ID["城市"])
            continue

        # 规则5：匹配街道类型（如AV/Rua）
        if clean_token in [st.replace(".", "") for st in STREET_TYPES] or clean_token == ".":
            token_labels.append(LABEL2ID["街道"])
            continue

        # 规则6：补充信息
        token_labels.append(LABEL2ID["补充信息"])

    assert len(tokens) == len(token_labels), "Token数与标签数不匹配"
    return tokens, token_labels


# ===================== 4. 其他函数（地址生成/主函数）不变，仅调用新的Tokenizer =====================
def generate_standard_address() -> str:
    street_type = random.choice(STREET_TYPES)
    street_name = random.choice(STREET_NAMES)
    number = str(random.randint(1, 9999))
    city, state = random.choice(CITIES)
    zipcode = generate_zipcode()
    return f"{street_type} {street_name}, {number} - {city}, {state}, {zipcode}"


def generate_non_standard_address(standard_addr: str) -> str:
    ops = [lambda x: x.lower(), lambda x: x.replace(",", "").replace("-", "")]
    for _ in range(random.randint(1, 3)):
        standard_addr = random.choice(ops)(standard_addr)
    return standard_addr


def generate_address_pool(n_total: int = 10000) -> List[str]:
    standard_addrs = [generate_standard_address() for _ in range(int(n_total * 0.7))]
    non_standard_addrs = [generate_non_standard_address(addr) for addr in standard_addrs[:int(n_total * 0.3)]]
    return standard_addrs + non_standard_addrs


def generate_annotated_data(n_total: int = 10000, tokenizer_dir: str = "./qwen3_address_5w_tokenizer_final",
                            save_path: str = "./annotated_brazil_address.jsonl") -> None:
    # 加载Qwen2 Tokenizer
    tokenizer = load_tokenizer(tokenizer_dir)
    validate_tokenizer(tokenizer)

    # 生成地址池
    print(f"开始生成{n_total}条地址...")
    address_pool = generate_address_pool(n_total)

    # 批量标注
    annotated_data = []
    invalid_count = 0
    for idx, addr in enumerate(address_pool):
        if idx % 1000 == 0:
            print(f"已处理{idx}/{n_total}条，无效数据{invalid_count}条")

        try:
            tokens, token_labels = annotate_address_tokens(addr, tokenizer)
            valid_labels = [l for l in token_labels if l != -100]
            if not valid_labels:
                invalid_count += 1
                continue
            annotated_data.append({
                "input_text": addr,
                "tokens": tokens,
                "token_labels": token_labels,
                "id2label": ID2LABEL
            })
        except Exception as e:
            invalid_count += 1
            continue

    # 保存数据
    with open(save_path, "w", encoding="utf-8") as f:
        for sample in annotated_data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # 统计&示例
    print(f"\n✅ 生成完成！有效样本：{len(annotated_data)}，无效：{invalid_count}")
    if annotated_data:
        sample = annotated_data[0]
        print(f"\n示例：")
        print(f"原始地址：{sample['input_text']}")
        print(f"Token序列：{sample['tokens']}")
        print(f"标签序列：{[ID2LABEL[l] if l != -100 else '忽略' for l in sample['token_labels']]}")


# ===================== 5. 运行 =====================
if __name__ == "__main__":
    generate_annotated_data(
        n_total=10000,
        tokenizer_dir="../qwen3_address_5w_tokenizer_final",
        save_path="./annotated_brazil_address.jsonl"
    )