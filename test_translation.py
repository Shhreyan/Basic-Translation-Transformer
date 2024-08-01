import torch
from torch.utils.data import DataLoader
from transformers import MarianMTModel, MarianTokenizer
from translation_dataset import TranslationDataset

def load_dataset(csv_file, tokenizer):
    return TranslationDataset(csv_file=csv_file, tokenizer=tokenizer)

def main():
    model_name = 'Helsinki-NLP/opus-mt-en-de'  # Change to appropriate model
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # Load test dataset
    test_dataset = load_dataset(csv_file='translation.csv', tokenizer=tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Switch model to evaluation mode
    model.eval()

    # Initialize variables to track performance
    total_loss = 0
    total_samples = 0

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for batch in test_dataloader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            # Move tensors to the same device as the model
            input_ids = input_ids.to(model.device)
            attention_mask = attention_mask.to(model.device)
            labels = labels.to(model.device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # Accumulate loss and sample count
            total_loss += loss.item() * input_ids.size(0)
            total_samples += input_ids.size(0)

            # Print some sample translations
            if total_samples < 20:  # Print first few samples for inspection
                decoded_preds = tokenizer.batch_decode(outputs.logits.argmax(dim=-1), skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                for pred, label in zip(decoded_preds, decoded_labels):
                    print(f"Predicted: {pred} | Ground Truth: {label}")

    average_loss = total_loss / total_samples
    print(f"Average Loss: {average_loss}")

if __name__ == '__main__':
    main()
