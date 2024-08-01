import torch
from torch.utils.data import DataLoader
from transformers import MarianMTModel, MarianTokenizer, Trainer, TrainingArguments
from translation_dataset import TranslationDataset

def main():
    model_name = 'Helsinki-NLP/opus-mt-en-de'  # Change to appropriate model
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    train_dataset = TranslationDataset(csv_file='translation.csv', tokenizer=tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()

if __name__ == '__main__':
    main()