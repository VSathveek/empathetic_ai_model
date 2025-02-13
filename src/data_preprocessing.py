import pandas as pd
import torch
from transformers import RobertaTokenizer
from sklearn.preprocessing import LabelEncoder

# Load the raw dataset (you need to provide the correct path)
df = pd.read_csv('data/emotion_emotion_69k.csv')

# Split the data into training and testing sets
train_text, test_text = df['Situation'][:int(0.8*len(df))], df['Situation'][int(0.8*len(df)):]
train_labels, test_labels = df['emotion'][:int(0.8*len(df))], df['emotion'][int(0.8*len(df)):]

# Encode the labels (emotion categories)
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels)
test_labels = label_encoder.transform(test_labels)

# Initialize the tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Tokenize the train and test data
train_encodings = tokenizer.batch_encode_plus(
    train_text.tolist(),
    add_special_tokens=True,
    max_length=512,
    padding=True,
    truncation=True,
    return_attention_mask=True,
    return_tensors='pt'
)

test_encodings = tokenizer.batch_encode_plus(
    test_text.tolist(),
    add_special_tokens=True,
    max_length=512,
    padding=True,
    truncation=True,
    return_attention_mask=True,
    return_tensors='pt'
)

# Save the preprocessed data as .pt files
torch.save((train_encodings, train_labels), 'data/train_data.pt')
torch.save((test_encodings, test_labels), 'data/test_data.pt')

print("Preprocessed data saved as 'train_data.pt' and 'test_data.pt'.")
