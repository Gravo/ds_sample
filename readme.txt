brazil_address_formatter/
├── config/                  # 配置文件（统一管理超参数/常量）
│   └── address_config.py    # 地址组件库、正则规则、标签映射
├── data/                    # 数据相关（生成/标注/预处理）
│   ├── generate_annotated_data.py  # 自动化标注数据生成脚本（核心）
│   └── dataset.py           # 训练数据集类（加载标注数据）
├── model/                   # 模型相关
│   ├── model_config.py      # 模型超参数（d_model/n_layers等）
│   └── address_analyzer.py  # 你的Transformer模型代码
├── utils/                   # 工具函数（通用能力）
│   ├── tokenizer_utils.py   # Tokenizer加载/校验
│   └── format_utils.py      # 地址格式化拼接（解析后拼接）
├── train.py                 # 训练主脚本（调用模型+数据）
└── infer.py                 # 推理主脚本（地址解析+格式化）