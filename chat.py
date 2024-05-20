import nltk
from nltk.tokenize import word_tokenize
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# Load the pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=8)

# Define a class for working with data
class IntentDataset(Dataset):
    def __init__(self, data, tokenizer):
        # Initialize data and tokenizer
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        # Return the number of elements in the data
        return len(self.data)

    def __getitem__(self, idx):
        # Get the text and label for the current element
        text = self.data.iloc[idx, 0]
        label = 0  # we temporarily assign label 0, we need to replace it with a real label

        # Tokenize text and add special tokens
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        # Return tokenized text, attention mask, and label
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Loading training and validation data
train_data = pd.read_csv('train.csv')
val_data = pd.read_csv('train1.csv')

# Create objects for working with data
train_dataset = IntentDataset(train_data, tokenizer)
val_dataset = IntentDataset(val_data, tokenizer)

# Determine the packet size and create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define the device for calculations (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define the loss function and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# Train the model for 5 epochs
for epoch in range(5):
    model.train()
    total_loss = 0
    for batch in train_loader:
        # Receive the data packet and move it to the device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        # Run the packet through the model and calculate the losses
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = criterion(outputs.logits, labels)

        # Error back propagation and model updating
        loss.backward()
        optimizer.step()

        # Summarize the losses for the current era
        total_loss += loss.item()

    print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}')

    # Estimating the model on the validation set
    model.eval()
    total_correct = 0
    with torch.no_grad():
        for batch in val_loader:
            # Receive the data packet and move it to the device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Run the packet through the model and calculate the predictions
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            _, predicted = torch.max(outputs.logits, 1)
            total_correct += (predicted == labels).sum().item()

    # Printing the accuracy for the current epoch
    accuracy = total_correct / len(val_data)
    print(f'Epoch {epoch+1}, Val Accuracy: {accuracy:.4f}')

class Chatbot:
    def __init__(self, model, tokenizer):
        # Initialize model and tokenizer
        self.model = model
        self.tokenizer = tokenizer

    def respond(self, text):
        # Tokenize text and add special tokens
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        # Move tokenized text to the device
        input_ids = encoding['input_ids'].flatten().to(device)
        attention_mask = encoding['attention_mask'].flatten().to(device)

        # Running tokenized text through the model
        outputs = self.model(input_ids, attention_mask=attention_mask)

        # Compute the predictions and select the intention with the highest probability
        _, predicted = torch.max(outputs.logits, 1)

        # Generate a response depending on the predicted intent
        if predicted.item() == 0:
            return "Hello! How can I help you today?"
        elif predicted.item() == 1:
            return "Goodbye! It was nice chatting with you."
        else:
            return "I didn't understand that. Can you please rephrase?"

chatbot = Chatbot(model, tokenizer)