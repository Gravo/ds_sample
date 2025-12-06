import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import Qwen2Tokenizer, get_scheduler
from tqdm.auto import tqdm
import os

# 导入自定义模块
from data.dataset import AddressTokenDataset
from model.address_analyzer import AddressModelConfig, AddressAnalyzerModel

# ===================== 1. 训练配置 =====================
class TrainConfig:
    # 数据路径
    tokenizer_dir = "./qwen3_address_5w_tokenizer_final"
    data_path = "./annotated_brazil_address.jsonl"
    save_dir = "./trained_model"
    # 训练超参数
    batch_size = 32  # 6GB GPU可设32，12GB可设64
    max_len = 64
    epochs = 10
    lr = 2e-5
    weight_decay = 1e-5
    # 硬件配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 日志配置
    log_step = 100  # 每100步打印一次日志

# ===================== 2. 训练函数 =====================
def train():
    cfg = TrainConfig()
    os.makedirs(cfg.save_dir, exist_ok=True)

    # 步骤1：加载Tokenizer
    tokenizer = Qwen2Tokenizer.from_pretrained(cfg.tokenizer_dir, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"✅ 加载Qwen2Tokenizer完成，词表大小：{tokenizer.vocab_size}")

    # 步骤2：加载数据集并拆分训练/验证（8:2）
    full_dataset = AddressTokenDataset(cfg.data_path, tokenizer, cfg.max_len)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"✅ 数据集加载完成：训练集{len(train_dataset)}条，验证集{len(val_dataset)}条")

    # 数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    # 步骤3：初始化模型（适配3000万参数）
    model_config = AddressModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=512,
        n_layers=6,
        n_heads=8,
        d_ff=2048,
        pad_token_id=tokenizer.pad_token_id,
        num_labels=6
    )
    model = AddressAnalyzerModel(model_config).to(cfg.device)
    print(f"✅ 模型初始化完成，总参数：{sum(p.numel() for p in model.parameters())/1e6:.2f} 万")
    print(f"✅ 训练设备：{cfg.device}")

    # 步骤4：配置优化器&学习率调度器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )
    # 线性学习率衰减
    num_training_steps = cfg.epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    # 步骤5：训练循环
    best_val_loss = float("inf")
    progress_bar = tqdm(range(num_training_steps), desc="Training")

    for epoch in range(cfg.epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # 数据移到设备
            batch = {k: v.to(cfg.device) for k, v in batch.items()}
            # 前向传播
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            loss = outputs["loss"]
            train_loss += loss.item() * batch["input_ids"].size(0)
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            # 日志
            progress_bar.update(1)
            if step % cfg.log_step == 0 and step > 0:
                tqdm.write(f"Epoch {epoch+1}/{cfg.epochs} | Step {step} | Train Loss: {loss.item():.4f}")

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                batch = {k: v.to(cfg.device) for k, v in batch.items()}
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                val_loss += outputs["loss"].item() * batch["input_ids"].size(0)

        # 计算平均损失
        avg_train_loss = train_loss / len(train_dataset)
        avg_val_loss = val_loss / len(val_dataset)

        # 打印epoch日志
        tqdm.write("="*80)
        tqdm.write(f"Epoch {epoch+1}/{cfg.epochs} Summary")
        tqdm.write(f"Average Train Loss: {avg_train_loss:.4f}")
        tqdm.write(f"Average Val Loss: {avg_val_loss:.4f}")
        tqdm.write("="*80)

        # 保存最优模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(cfg.save_dir)
            tokenizer.save_pretrained(cfg.save_dir)
            tqdm.write(f"✅ 保存最优模型（验证损失：{best_val_loss:.4f}）到 {cfg.save_dir}")

    # 训练完成
    progress_bar.close()
    print(f"🎉 训练完成！最优验证损失：{best_val_loss:.4f}，模型保存路径：{cfg.save_dir}")

# ===================== 3. 运行训练 =====================
if __name__ == "__main__":
    train()