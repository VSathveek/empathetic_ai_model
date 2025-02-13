
Empathetic AI Model
This repository is a work-in-progress for creating an Empathetic Conversational AI model that aims to understand the emotions of users based on their inputs and respond in a manner that shows empathy. The project is designed to detect emotions in text and, in the future, generate empathetic and emotionally-aware responses.

Current State of Development
As of now, the model is able to detect and classify emotions from text inputs. We have successfully implemented the following features:

Emotion Detection:
The model is trained to identify emotions such as anger, joy, sadness, and fear based on user input text using the RoBERTa model fine-tuned for sequence classification.
The trained model can predict emotions with acceptable accuracy, and the predictions are useful for guiding the empathetic response generation.
Model Overview
Emotion Detection (Current Focus):

We have trained a RoBERTa-based model on the emotion-emotion_69k.csv dataset (though a smaller subset for testing), which detects emotions in user input text.
The model currently detects the following emotions: anger, joy, fear, and sadness. It can classify emotions in a user's statement and is ready to be used as part of an empathetic conversational system.
Future Steps:

Dialogue Generation with Empathy: The next step in development is to integrate dialogue generation models (e.g., DialoGPT) to generate empathetic responses that align with the detected emotion.
Natural Language Processing (NLP): We are working on fine-tuning the NLP models for better context-awareness, making the conversation flow more naturally and emotionally intelligently.
Empathy Integration: Future work includes training the system to generate more contextually empathetic responses based on the user's emotions, which will help improve the overall user experience.
How to Run the Model Locally
Prerequisites
Before running the project, make sure you have the following Python libraries installed:

Transformers: For working with pre-trained models like RoBERTa and GPT.
Torch: For running the models and processing the data.
Sklearn: For data splitting and preprocessing.
You can install these dependencies using:

bash
Copy
pip install transformers torch scikit-learn
Running the Emotion Detection Model
Clone the repository to your local machine:
bash
Copy
git clone https://github.com/VSathveek/empathetic_ai_model.git
Navigate to the repository directory:
bash
Copy
cd empathetic_ai_model
Run the emotion detection model:
bash
Copy
python detect_emotion.py
Enter a text to detect the emotion. The output will be the detected emotion (e.g., anger, joy, fear, or sadness).
Generating Empathetic Responses (Under Development)
The dialogue generation component is under development. Once completed, it will generate an empathetic response based on the detected emotion from the user input.

Files Created During Execution
When running the code, the following files will be generated or expected to be available:

train_data.pt: The PyTorch serialized dataset of the training data, which is preprocessed and ready to be used for training the emotion classification model.
test_data.pt: The PyTorch serialized dataset for the test data used for model evaluation.
trained_model.pt: The final trained model that can be used to predict emotions for new inputs.
These files can be used for model evaluation or for further training if needed.

Expected Accuracy
The current emotion detection model is trained on a subset of the emotion-emotion_69k.csv dataset. Although the model has shown promising results, the accuracy for emotion detection is currently not optimal, and further improvements are expected with additional training on more data and fine-tuning.

Directory Structure
Here is the directory structure of the repository:

bash
Copy
empathetic_ai_model/
│
├── data/                       # Folder containing datasets
│   ├── emotion-emotion_69k.csv  # The dataset for emotion classification
│
├── models/                     # Folder containing pre-trained and trained models
│   ├── trained_model.pt        # The trained emotion classification model
│
├── scripts/                    # Folder containing the code files
│   ├── detect_emotion.py       # Script to run emotion detection
│   ├── generate_response.py    # Script for generating empathetic responses (Under development)
│
└── README.md                   # Project overview and instructions
Contributing
This is an open-source project, and we welcome contributions! If you'd like to contribute, please fork the repository, make changes, and create a pull request.
