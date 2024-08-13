from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
from transformers import BertTokenizer
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Load the model and tokenizer
model = tf.keras.models.load_model('path/to/your/saved/model.keras', custom_objects={'BertMultiTaskModel': BertMultiTaskModel})
tokenizer = BertTokenizer.from_pretrained('path/to/your/tokenizer')

# Define a request model for predictions
class PredictionRequest(BaseModel):
    sentences: list

# Define an endpoint for health check
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Define an endpoint for predictions
@app.post("/predict")
async def predict(request: PredictionRequest):
    sentences = request.sentences

    # Tokenize and prepare inputs
    inputs = tokenizer(sentences, padding=True, truncation=True, max_length=92, return_tensors='tf')
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Make predictions
    predictions = model({
        'input_ids': input_ids,
        'attention_mask': attention_mask
    })

    # Return predictions
    return {
        "ner_predictions": predictions['ner_output'].numpy().tolist(),
        "pos_predictions": predictions['pos_output'].numpy().tolist()
    }

# Run the application with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
