from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer, \
    set_seed
from datasets import load_dataset

# 一个影评数据集，就两列，一列text是评论，另一列label是该评论正面或负面
DATASET_NAME = "rotten_tomatoes"
raw_datasets = load_dataset(DATASET_NAME)
raw_train_dataset = raw_datasets["train"]
raw_valid_dataset = raw_datasets["validation"]

# 定义tokenizer
MODEL_NAME = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token_id = 0

set_seed(42)

# 标签集
named_labels = ['neg', 'pos']
# 标签转 token_id
label_ids = [
    tokenizer(label, add_special_tokens=False)["input_ids"][0]
    for label in named_labels
]

MAX_LEN = 32  # 最大序列长度（输入+输出）


# 数据处理函数，把原数据转化为模型需要的数据格式
def process_fn(input_data):
    model_inputs = {
        "input_ids": [],
        "attention_mask": [],
        "labels": []
    }

    for body, label_key in zip(input_data["text"], input_data["label"]):
        inputs = tokenizer(body, add_special_tokens=False)
        label = label_ids[label_key]
        # input_ids是每行text+eos+label
        input_ids = inputs["input_ids"] + [tokenizer.eos_token_id, label]

        raw_len = len(input_ids)
        input_len = len(inputs["input_ids"]) + 1

        # 然后就是对齐并形成矩阵，对齐方案是
        # 如果每行大于32，则input_ids截取后32个item，attention_mask都是1表示都有数据，labels的最后一个元素填充pos或neg，其他填充-100
        # 如果小于32，input_ids缺的部分用特殊的pad_token填充，attention_mask缺的部分填0，labels也填充至对应元素的位置
        if raw_len >= MAX_LEN:
            input_ids = input_ids[-MAX_LEN:]
            attention_mask = [1] * MAX_LEN
            labels = [-100] * (MAX_LEN - 1) + [label]
        else:
            input_ids = input_ids + [tokenizer.pad_token_id] * (MAX_LEN - raw_len)
            attention_mask = [1] * raw_len + [0] * (MAX_LEN - raw_len)
            labels = [-100] * input_len + [label] + [-100] * (MAX_LEN - raw_len)

        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append(attention_mask)
        model_inputs["labels"].append(labels)

    return model_inputs


tokenized_train_dataset = raw_train_dataset.map(process_fn, batched=True, remove_columns=raw_train_dataset.column_names,
                                                desc="Running tokenizer on train dataset")
tokenized_valid_dataset = raw_valid_dataset.map(process_fn, batched=True, remove_columns=raw_valid_dataset.column_names,
                                                desc="Running tokenizer on validation dataset")

# 定义数据校准器（自动生成batch）
collater = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
# 定义模型
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
# 节省显存
model.gradient_checkpointing_enable()

BATCH_SIZE = 8
INTERVAL = 100  # 每多少步打一次 log / 做一次 eval
LR = 2e-5  # 学习率
WARMUP_RATIO = 0.1  # warmup比例
# 定义训练参数
training_args = TrainingArguments(
    output_dir="./output",  # checkpoint保存路径
    evaluation_strategy="steps",  # 按步数计算eval频率
    overwrite_output_dir=True,
    num_train_epochs=1,  # 训练epoch数
    per_device_train_batch_size=BATCH_SIZE,  # 每张卡的batch大小
    gradient_accumulation_steps=1,  # 累加几个step做一次参数更新
    per_device_eval_batch_size=BATCH_SIZE,  # evaluation batch size
    eval_steps=INTERVAL,  # 每N步eval一次
    logging_steps=INTERVAL,  # 每N步log一次
    save_steps=INTERVAL,  # 每N步保存一个checkpoint
    learning_rate=LR,  # 学习率
    warmup_ratio=WARMUP_RATIO,  # warmup比例
)

# 定义训练器
trainer = Trainer(
    model=model,  # 待训练模型
    args=training_args,  # 训练参数
    data_collator=collater,  # 数据校准器
    train_dataset=tokenized_train_dataset,  # 训练集
    eval_dataset=tokenized_valid_dataset,  # 验证集
    # compute_metrics=compute_metric,  # 计算自定义指标
)

# 开始训练
trainer.train()
