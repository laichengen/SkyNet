from datasets import Dataset
from transformers import GPT2Tokenizer, TrainingArguments, Trainer

from model.rlhf.models.reward_model import GPTRewardModel



def train_rm(train_path, model_path):
    def process_func(example):
        prompt = example['question']
        cases = [prompt + example['positive']]
        negatives = [prompt + neg for neg in example['negative']]
        cases.extend(negatives)
        _input = tokenizer(cases, return_tensors='pt', padding=True)
        return _input


    tokenizer = GPT2Tokenizer.from_pretrained(model_path, pad_token="<|endoftext|>")

    model = GPTRewardModel(model_path)

    ds = Dataset.from_json(train_path)
    tokenized_id = ds.map(process_func, remove_columns=ds.column_names)

    args = TrainingArguments(
        output_dir="./logs/rm_model_train",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        logging_steps=10,
        num_train_epochs=3,
        save_steps=100,  # 为了快速演示，这里设置10，建议你设置成100
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=False
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_id,
    )
    trainer.train()  # 开始训练

if __name__ == '__main__':
    model_path = "../../checkpoints/GPT2"
    train_path = '../../data/rm_train.json'
    train_rm(train_path, model_path)