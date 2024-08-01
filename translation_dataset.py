import pandas as pd
from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=128):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        original_text = self.data.iloc[idx, 0]
        translated_text = self.data.iloc[idx, 1]
        
        original_encoding = self.tokenizer(
            original_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        translated_encoding = self.tokenizer(
            translated_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        labels = translated_encoding['input_ids']
        labels[labels == self.tokenizer.pad_token_id] = -100  # Mask pad tokens

        return {
            'input_ids': original_encoding['input_ids'].flatten(),
            'attention_mask': original_encoding['attention_mask'].flatten(),
            'labels': labels.flatten()
        }
