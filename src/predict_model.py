import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from src.data_preprocessing import label_encoder  # Label Encoder

# Load the trained model and tokenizer
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=len(label_encoder.classes_))
model.load_state_dict(torch.load('models/trained_model.pt'))
model.eval()

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Function to predict the emotion for a given input text
def predict_emotion(text):
    encodings = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    predicted_emotion = label_encoder.inverse_transform([predicted_class])[0]
    return predicted_emotion

# Example usage
input_text = "I feel so sad today."
predicted_emotion = predict_emotion(input_text)
print(f"The predicted emotion is: {predicted_emotion}")
