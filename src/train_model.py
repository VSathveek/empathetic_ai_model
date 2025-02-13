import torch
from transformers import RobertaForSequenceClassification
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
import torch.optim as optim
import torch.nn as nn
from src.data_preprocessing import train_encodings, train_labels, test_encodings, test_labels  # Preprocessed data

# Create a custom Dataset class
class EmpatheticDialogueDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Initialize DataLoader
train_dataset = EmpatheticDialogueDataset(train_encodings, train_labels)
test_dataset = EmpatheticDialogueDataset(test_encodings, test_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Initialize the model (RoBERTa for Sequence Classification)
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=len(set(train_labels)))

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# Train the model
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = criterion(outputs.logits, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader)}')

    # Evaluate the model on the test set
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            _, predicted = torch.max(outputs.logits, dim=1)

            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    print(f'Test Accuracy: {accuracy:.4f}')

# Save the trained model to a .pt file
torch.save(model.state_dict(), 'models/trained_model.pt')

print("Training complete! Model saved as 'trained_model.pt'.")
