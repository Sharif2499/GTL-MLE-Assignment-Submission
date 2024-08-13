from transformers import TFBertModel
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np

class BertMultiTaskModel(Model):
    def __init__(self, hidden_dim, ner_classes, pos_classes, **kwargs):
        super(BertMultiTaskModel, self).__init__(**kwargs)
        self.bert = TFBertModel.from_pretrained('bert-base-multilingual-cased')
        self.hidden_dim = hidden_dim
        self.ner_classes = ner_classes
        self.pos_classes = pos_classes
        self.dense = Dense(hidden_dim, activation='relu')
        self.ner_output = Dense(ner_classes, activation='softmax', name='ner_output')
        self.pos_output = Dense(pos_classes, activation='softmax', name='pos_output')

    def call(self, inputs):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # BERT embeddings
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = bert_output.last_hidden_state

        # Apply dense layer and separate outputs
        x = self.dense(sequence_output)
        ner_logits = self.ner_output(x)
        pos_logits = self.pos_output(x)

        return {'ner_output': ner_logits, 'pos_output': pos_logits}

    def get_config(self):
        return {
            'hidden_dim': self.hidden_dim,
            'ner_classes': self.ner_classes,
            'pos_classes': self.pos_classes
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    def build_from_config(self, config):
        self.__init__(**config)
        self.build((None, 92))  # Example input shape; adjust as needed

hidden_dim = 256
class BertMultiTaskModel(Model):
    def __init__(self, hidden_dim, ner_classes, pos_classes, **kwargs):
        super(BertMultiTaskModel, self).__init__(**kwargs)
        self.bert = TFBertModel.from_pretrained('bert-base-multilingual-cased')
        self.hidden_dim = hidden_dim
        self.ner_classes = ner_classes
        self.pos_classes = pos_classes
        self.dense = Dense(hidden_dim, activation='relu')
        self.ner_output = Dense(ner_classes, activation='softmax', name='ner_output')
        self.pos_output = Dense(pos_classes, activation='softmax', name='pos_output')

    def call(self, inputs):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # BERT embeddings
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = bert_output.last_hidden_state

        # Apply dense layer and separate outputs
        x = self.dense(sequence_output)
        ner_logits = self.ner_output(x)
        pos_logits = self.pos_output(x)

        return {'ner_output': ner_logits, 'pos_output': pos_logits}

    def get_config(self):
        return {
            'hidden_dim': self.hidden_dim,
            'ner_classes': self.ner_classes,
            'pos_classes': self.pos_classes
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    def build_from_config(self, config):
        self.__init__(**config)
        self.build((None, 92))  # Example input shape; adjust as needed

hidden_dim = 256
ner_classes = 21  # Number of NER classes
pos_classes = 15  # Number of POS classes
learning_rate = 0.0001

input_ids = Input(shape=(92,), dtype=tf.int32, name='input_ids')
attention_mask = Input(shape=(92,), dtype=tf.int32, name='attention_mask')

model = BertMultiTaskModel(hidden_dim, ner_classes, pos_classes)
model({'input_ids': input_ids, 'attention_mask': attention_mask})  # This will build the model

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss={'ner_output': 'sparse_categorical_crossentropy', 'pos_output': 'sparse_categorical_crossentropy'},
    metrics={'ner_output': 'accuracy', 'pos_output': 'accuracy'}
)
model.summary()
learning_rate = 0.0001

input_ids = Input(shape=(92,), dtype=tf.int32, name='input_ids')
attention_mask = Input(shape=(92,), dtype=tf.int32, name='attention_mask')

model = BertMultiTaskModel(hidden_dim, ner_classes, pos_classes)
model({'input_ids': input_ids, 'attention_mask': attention_mask})  # This will build the model

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss={'ner_output': 'sparse_categorical_crossentropy', 'pos_output': 'sparse_categorical_crossentropy'},
    metrics={'ner_output': 'accuracy', 'pos_output': 'accuracy'}
)
model.summary()

# Train the model
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(
    monitor='val_loss',           # Metric to monitor
    patience=3,                   # Number of epochs with no improvement to wait
    restore_best_weights=True,    # Restore model weights from the epoch with the best value of the monitored quantity
    verbose=1                     # Verbosity mode
)

history = model.fit(
    {'input_ids': X_train_tensor, 'attention_mask': attention_mask_train},
    {'ner_output': y_train_ner_tensor, 'pos_output': y_train_pos_tensor},
    validation_data=(
        {'input_ids': X_val_tensor, 'attention_mask': attention_mask_val},
        {'ner_output': y_val_ner_tensor, 'pos_output': y_val_pos_tensor}
    ),
    epochs=30,
    batch_size=32,
    callbacks=[early_stopping]
)

model.save('file_path/saved_model.keras')
