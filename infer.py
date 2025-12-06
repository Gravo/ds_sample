import torch
from transformers import Qwen2Tokenizer
from model.address_analyzer import AddressAnalyzerModel

# 加载模型和Tokenizer
model_dir = "./trained_model"
tokenizer = Qwen2Tokenizer.from_pretrained(model_dir)
tokenizer.pad_token = tokenizer.eos_token
model = AddressAnalyzerModel.from_pretrained(model_dir).to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

# 地址解析+格式化函数
id2label = {0: "邮编", 1: "城市", 2: "州", 3: "街道", 4: "门牌号", 5: "补充信息"}


def analyze_address(raw_address):
    # 编码输入
    inputs = tokenizer(
        raw_address,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=64
    ).to(model.device)

    # 推理
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs["logits"]
    predictions = torch.argmax(logits, dim=-1)[0]  # 取第一条结果

    # 解析结果
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    structured_result = {}
    current_label = None
    current_chunk = []

    for token, label_idx in zip(tokens, predictions):
        if token in [tokenizer.pad_token, tokenizer.unk_token, tokenizer.eos_token]:
            continue
        label = id2label[label_idx.item()]
        clean_token = token.replace("Ġ", "").strip()

        if label != current_label:
            if current_chunk:
                chunk_text = ''.join(current_chunk).strip()
                structured_result[current_label] = chunk_text
            current_label = label
            current_chunk = [clean_token]
        else:
            current_chunk.append(clean_token)

    # 处理最后一个块
    if current_chunk:
        chunk_text = ''.join(current_chunk).strip()
        structured_result[current_label] = chunk_text

    # 格式化地址
    street = structured_result.get("街道", "")
    number = structured_result.get("门牌号", "")
    city = structured_result.get("城市", "")
    state = structured_result.get("州", "").upper()
    zipcode = structured_result.get("邮编", "")
    if len(zipcode.replace("-", "")) == 8:
        zipcode = f"{zipcode[:5]}-{zipcode[5:]}"

    formatted_addr = f"{street}, {number} - {city}, {state}, {zipcode}".strip()
    return structured_result, formatted_addr


# 测试
if __name__ == "__main__":
    test_address = "av paulista 1000 sao paulo sp 01310100"
    structured, formatted = analyze_address(test_address)
    print(f"原始地址：{test_address}")
    print(f"结构化解析：{structured}")
    print(f"格式化地址：{formatted}")  # 预期：Av. Paulista, 1000 - São Paulo, SP, 01310-100