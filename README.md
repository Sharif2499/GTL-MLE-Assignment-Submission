Python environment set up with the necessary libraries:

python --version # python 3.8 or higher

pip install virtualenv

virtualenv venv

python -m venv env

source env/bin/activate  # On Windows use `env\Scripts\activate`

pip install -r requirements.txt

python preprocess.py

python train.py

python evaluate.py

uvicorn app:app --host 0.0.0.0 --port 8000


To run the code Following sequence to be maintained: 

1. data_preprocessing.py for the 'Data Analysis and Preprocessing'  before training .
2. model_training.py for 'Model Development' where the model would be built, compiled, trained and saved.
3. model_evaluation.py for 'Model Evaluation' to evaluate different performance metrices on test set.
4. app.py for 'Model Deployment' to deploy the model using FastAPI. (Health Check: to http://localhost:8000/health to check the API health; Prediction Endpoint: Send a POST request to http://localhost:8000/predict with JSON data)



Model Architecture:
The BertMultiTaskModel is a neural network model designed to handle two tasks simultaneously: Named Entity Recognition (NER) and Parts of Speech (POS) tagging. The architecture is built upon the BERT (Bidirectional Encoder Representations from Transformers) model, which is a pre-trained transformer-based model known for its effectiveness in various NLP tasks. Here's a breakdown of the architecture:

BERT Base Model:
Type: TFBertModel from the transformers library.
Purpose: Extract contextual embeddings from the input text.
Input: Tokenized input sentences, including attention masks to differentiate real tokens from padding.

Dense Layer:
Type: Dense layer with ReLU activation.
Purpose: Transform the output from BERT to a hidden representation of size hidden_dim.

Task-Specific Output Layers:
NER Output: Dense layer with a softmax activation function to predict the Named Entity categories for each token.
POS Output: Dense layer with a softmax activation function to predict the Parts of Speech tags for each token.

Key Points:
Input Shape: The input to the model consists of token IDs and attention masks for sentences.
Output: The model outputs two sets of predictions: one for NER and one for POS tagging.

Dataset
The dataset used in this pipeline consists of Bangla sentences with annotated Named Entity (NER) and Parts of Speech (POS) tags. The dataset had five sentences having one tab and the rest had zero tabs which had been preprocessed during preprocessing

Data Structure:
Sentences: A collection of sentences in Bangla, each represented as a list of tokens.
NER Tags: Each token in a sentence is associated with a Named Entity tag, such as O (outside any entity) or B-ENTITY (beginning of an entity).
POS Tags: Each token in a sentence is tagged with its Part of Speech category, such as 'VNF', 'PUNCT', 'ADJ', 'VF', 'PRO', 'OTH'.

Preprocessing:
Tokenization: Sentences are tokenized using the BERT tokenizer to convert text into token IDs.
Label Encoding: NER and POS tags are encoded using LabelEncoder to convert text labels into numerical values.
Padding: Both input sequences and tag sequences are padded to ensure uniform length across the dataset.

Data Splits:
Training Set: Used to train the model.
Validation Set: Used to tune hyperparameters and avoid overfitting.
Test Set: Used to evaluate the model's performance.
