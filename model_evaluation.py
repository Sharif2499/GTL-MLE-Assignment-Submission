from transformers import TFBertModel
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np

# Example test data preparation
X_test_padded = pad_sequences(X_test, maxlen=max_seq_length)
y_test_ner_padded = pad_sequences(y_test_ner, maxlen=max_seq_length)
y_test_pos_padded = pad_sequences(y_test_pos, maxlen=max_seq_length)

X_test_tensor = tf.convert_to_tensor(X_test_padded)
y_test_ner_tensor = tf.convert_to_tensor(y_test_ner_padded)
y_test_pos_tensor = tf.convert_to_tensor(y_test_pos_padded)

attention_mask_test = np.ones_like(X_test_tensor)

# Evaluate the model
evaluation = model.evaluate(
    {'input_ids': X_test_tensor, 'attention_mask': attention_mask_test},
    {'ner_output': y_test_ner_tensor, 'pos_output': y_test_pos_tensor}
)

print(f"Evaluation results: {evaluation}")


# Make predictions
predictions = model.predict({'input_ids': X_test_tensor, 'attention_mask': attention_mask_test})

# Get the predicted labels
pred_ner = np.argmax(predictions['ner_output'], axis=-1)
pred_pos = np.argmax(predictions['pos_output'], axis=-1)

from sklearn.metrics import confusion_matrix

# Flatten the true and predicted labels
y_true_ner = np.concatenate([y_test_ner.numpy().flatten()])
y_pred_ner = np.concatenate([pred_ner.flatten()])

# Compute confusion matrix
confusion_mat_ner = confusion_matrix(y_true_ner, y_pred_ner)
print("Confusion Matrix (NER):\n", confusion_mat_ner)

# Flatten the true and predicted labels
y_true_pos = np.concatenate([y_test_pos.numpy().flatten()])
y_pred_pos = np.concatenate([pred_pos.flatten()])

# Compute confusion matrix
confusion_mat_pos = confusion_matrix(y_true_pos, y_pred_pos)
print("Confusion Matrix (POS):\n", confusion_mat_pos)

from sklearn.metrics import classification_report

# Compute classification report
report_ner = classification_report(y_true_ner, y_pred_ner)
print("Classification Report (NER):\n", report_ner)


# Compute classification report
report_pos = classification_report(y_true_pos, y_pred_pos)
print("Classification Report (POS):\n", report_pos)

from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

# For NER
f1_ner = f1_score(y_true_ner, y_pred_ner, average='weighted', zero_division=0)
accuracy_ner = accuracy_score(y_true_ner, y_pred_ner)
recall_ner = recall_score(y_true_ner, y_pred_ner, average='weighted', zero_division=0)
precision_ner = precision_score(y_true_ner, y_pred_ner, average='weighted', zero_division=0)

print("NER Metrics:")
print("F1 Score:", f1_ner)
print("Accuracy:", accuracy_ner)
print("Recall:", recall_ner)
print("Precision:", precision_ner)

# For POS
f1_pos = f1_score(y_true_pos, y_pred_pos, average='weighted', zero_division=0)
accuracy_pos = accuracy_score(y_true_pos, y_pred_pos)
recall_pos = recall_score(y_true_pos, y_pred_pos, average='weighted', zero_division=0)
precision_pos = precision_score(y_true_pos, y_pred_pos, average='weighted', zero_division=0)

print("POS Metrics:")
print("F1 Score:", f1_pos)
print("Accuracy:", accuracy_pos)
print("Recall:", recall_pos)
print("Precision:", precision_pos)

