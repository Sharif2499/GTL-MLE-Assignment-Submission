import pandas as pd
file_path = 'data.tsv'

sentences = []
pos_tags = []
ner_tags = []

# Temporary storage for words, POS, and NER tags of a sentence.
temp_sentence = []
temp_pos_tags = []
temp_ner_tags = []

with open(file_path, 'r') as file:
    for line in file:
        line = line.strip()
        # Check if the line is a sentence or a word
        # In the given dataset, five sentences had one tab separation and rest had no tab separation
        tab_count = line.count('\t')
        if tab_count == 0 or tab_count == 1:

            # If it's a sentence (zero or one tabs), finalize the previous sentence
            if temp_sentence:
                sentences.append(temp_sentence)
                pos_tags.append(temp_pos_tags)
                ner_tags.append(temp_ner_tags)
                temp_sentence = []
                temp_pos_tags = []
                temp_ner_tags = []
        else:
            # If it's a word with tags, split by tabs
            parts = line.split('\t')
            temp_sentence.append(parts[0])
            if len(parts) > 1:
                temp_pos_tags.append(parts[1])
            if len(parts) > 2:
                temp_ner_tags.append(parts[2])

# Append the last sentence if the loop ends without a sentence boundary
if temp_sentence:
    sentences.append(temp_sentence)
    pos_tags.append(temp_pos_tags)
    ner_tags.append(temp_ner_tags)

# Convert sentences, pos_tags, and ner_tags to a DataFrame for better inspection
processed_data = pd.DataFrame({
    'Sentence': [" ".join(sentence) for sentence in sentences],
    'POS_Tags': pos_tags,
    'NER_Tags': ner_tags
})

# Display the processed data
print(processed_data.head())

flat_pos_labels = [label for sublist in pos_tags for label in sublist]
flat_ner_labels = [label for sublist in ner_tags for label in sublist]

unique_pos_classes = set(flat_pos_labels)
pos_classes = len(unique_pos_classes)
print(f"POS classes: {unique_pos_classes}, Number of POS classes: {pos_classes}")

# Find unique NER classes
unique_ner_classes = set(flat_ner_labels)
ner_classes = len(unique_ner_classes)
print(f"NER classes: {unique_ner_classes}, Number of NER classes: {ner_classes}")

from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Tokenize the sentences
inputs = tokenizer(sentences, padding=True, truncation=True, max_length=92, is_split_into_words=True, return_tensors='np')  # Use 'np' to return numpy arrays

max_token_id = np.max(inputs['input_ids'])
vocab_size = tokenizer.vocab_size

assert max_token_id < vocab_size, f"Token ID {max_token_id} exceeds the vocab size of {vocab_size}. Check your tokenization process."


# Convert tags to label encoding
ner_label_encoder = LabelEncoder()
pos_label_encoder = LabelEncoder()

# Flatten the list of tags, fit and transform
flat_ner_tags = [item for sublist in ner_tags for item in sublist]
flat_pos_tags = [item for sublist in pos_tags for item in sublist]

ner_label_encoder.fit(flat_ner_tags)
pos_label_encoder.fit(flat_pos_tags)

encoded_ner_tags = [ner_label_encoder.transform(tag_list) for tag_list in ner_tags]
encoded_pos_tags = [pos_label_encoder.transform(tag_list) for tag_list in pos_tags]

# Pad the encoded tags to match input length
ner_tags_padded = tf.keras.preprocessing.sequence.pad_sequences(encoded_ner_tags, maxlen=92, padding='post', truncating='post')
pos_tags_padded = tf.keras.preprocessing.sequence.pad_sequences(encoded_pos_tags, maxlen=92, padding='post', truncating='post')

# Convert to numpy arrays to ensure compatibility with scikit-learn
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

# Split the data into train, validation, and test sets
X_train, X_temp, y_train_ner, y_temp_ner, y_train_pos, y_temp_pos = train_test_split(
    input_ids, ner_tags_padded, pos_tags_padded, test_size=0.3, random_state=42)

X_val, X_test, y_val_ner, y_test_ner, y_val_pos, y_test_pos = train_test_split(
    X_temp, y_temp_ner, y_temp_pos, test_size=0.5, random_state=42)

# Split attention_mask as well to match X_train, X_val, and X_test
attention_mask_train, attention_mask_temp = train_test_split(attention_mask, test_size=0.3, random_state=42)
attention_mask_val, attention_mask_test = train_test_split(attention_mask_temp, test_size=0.5, random_state=42)

# Convert to tf.Tensor to feed into the model
X_train = tf.convert_to_tensor(X_train)
X_val = tf.convert_to_tensor(X_val)
X_test = tf.convert_to_tensor(X_test)

y_train_ner = tf.convert_to_tensor(y_train_ner)
y_val_ner = tf.convert_to_tensor(y_val_ner)
y_test_ner = tf.convert_to_tensor(y_test_ner)

y_train_pos = tf.convert_to_tensor(y_train_pos)
y_val_pos = tf.convert_to_tensor(y_val_pos)
y_test_pos = tf.convert_to_tensor(y_test_pos)

attention_mask_train = tf.convert_to_tensor(attention_mask_train)
attention_mask_val = tf.convert_to_tensor(attention_mask_val)
attention_mask_test = tf.convert_to_tensor(attention_mask_test)

def pad_sequences(sequences, maxlen):
    return tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')

# Define maximum sequence length
max_seq_length = 92

# Pad sequences and labels
X_train_padded = pad_sequences(X_train, maxlen=max_seq_length)
y_train_ner_padded = pad_sequences(y_train_ner, maxlen=max_seq_length)
y_train_pos_padded = pad_sequences(y_train_pos, maxlen=max_seq_length)

X_val_padded = pad_sequences(X_val, maxlen=max_seq_length)
y_val_ner_padded = pad_sequences(y_val_ner, maxlen=max_seq_length)
y_val_pos_padded = pad_sequences(y_val_pos, maxlen=max_seq_length)

# Convert to tensors
X_train_tensor = tf.convert_to_tensor(X_train_padded)
y_train_ner_tensor = tf.convert_to_tensor(y_train_ner_padded)
y_train_pos_tensor = tf.convert_to_tensor(y_train_pos_padded)

X_val_tensor = tf.convert_to_tensor(X_val_padded)
y_val_ner_tensor = tf.convert_to_tensor(y_val_ner_padded)
y_val_pos_tensor = tf.convert_to_tensor(y_val_pos_padded)

# Example with attention masks (all ones for simplicity; adjust as needed)
attention_mask_train = np.ones_like(X_train_tensor)
attention_mask_val = np.ones_like(X_val_tensor)


