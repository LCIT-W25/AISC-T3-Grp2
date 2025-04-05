from transformers import AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify, render_template
from gensim.models import KeyedVectors
import re

app = Flask(__name__)

# Constants
MAX_SEQ_LENGTH = 128
VOCAB_SIZE = 84000  # Adjust based on your model's vocabulary size
EMBEDDING_DIM = 300  # Since you're using Google's word2vec which is 300-dimensional


class CausalTransformer(nn.Module):
    def __init__(self, d_model=300, nhead=6, num_layers=6, dim_feedforward=512, max_len=128):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation='gelu'
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, x, pos_ids):
        # Get the sequence length (number of tokens in the input sequence)
        seq_len = x.size(1)

        # Ensure that the sequence length does not exceed MAX_SEQ_LENGTH
        if seq_len > MAX_SEQ_LENGTH:
            # Truncate the input sequence if too long
            x = x[:, :MAX_SEQ_LENGTH]
            seq_len = MAX_SEQ_LENGTH   # Adjust seq_len to the max length

        # Create the position ids (indices from 0 to seq_len-1)
        pos_ids = torch.arange(seq_len).unsqueeze(0).to(x.device)

        # Add positional encoding to the input tensor
        # Use position ids instead of the input 'x'
        x = self.pos_embedding(pos_ids)

        # Continue with the rest of the model
        mask = torch.triu(torch.ones(seq_len, seq_len),
                          diagonal=1).bool().to(x.device)

        x = x.transpose(0, 1)  # (seq_len, batch, d_model)

        memory = self.encoder(x, mask=mask)
        output = self.decoder(x, memory, tgt_mask=mask, memory_mask=mask)
        output = output.transpose(0, 1)  # (batch, seq_len, d_model)

        return self.fc_out(output)


class Tokenizer:
    def __init__(self, model_name="bert-base-uncased", max_seq_length=128):
        # Load the BERT tokenizer from Hugging Face
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_seq_length = max_seq_length

    def preprocess(self, text):
        # Tokenize and encode the input text, applying padding and truncation
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s.,?!]', '', text)
        text = re.sub(r'@[\w]+', ' ', text)  # Remove mentions
        text = re.sub(r'http\S+|www\S+', ' ', text)  # Remove URLs
        text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove emojis
        # words = text.split()

        encoding = self.tokenizer.encode(
            text,
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=self.max_seq_length,
            padding='max_length',  # Pad to max length
            truncation=True,  # Truncate to max length
            return_tensors='pt'  # Return as PyTorch tensors
        )
        return encoding['input_ids']

    def detokenize(self, indices):
        # Decode the token IDs back to a string
        return self.tokenizer.decode(indices, skip_special_tokens=True)


class ChatBot:
    def __init__(self, model_path, model_name="bert-base-uncased"):
        # Initialize the tokenizer with the pre-trained BERT model
        print("Initializing tokenizer...")
        self.tokenizer = Tokenizer(model_name=model_name)

        # Initialize model
        print("Initializing model...")
        self.model = CausalTransformer()

        print("Loading model...")
        self.model = torch.load(
            model_path, map_location=torch.device('cpu'), weights_only=False)
        self.model.eval()

    def generate_response(self, input_text, max_length=50):
        input_tensor = self.tokenizer.preprocess(input_text)

        with torch.no_grad():
            for _ in range(max_length):
                seq_len = input_tensor.size(1)
                pos_ids = torch.arange(seq_len).unsqueeze(
                    0).to(input_tensor.device)

                output = self.model(input_tensor, pos_ids)

                next_token_logits = output[0, -1, :]
                next_token_probs = F.softmax(next_token_logits, dim=0)

                top_k = 5
                top_k_probs, top_k_indices = torch.topk(
                    next_token_probs, top_k)
                next_token_idx = top_k_indices[torch.multinomial(
                    top_k_probs, 1)]

                if next_token_idx == self.tokenizer.tokenizer.pad_token_id:
                    break

                input_tensor = torch.cat(
                    [input_tensor, next_token_idx.unsqueeze(0)], dim=1)

            output_indices = input_tensor[0, len(
                self.tokenizer.preprocess(input_text)[0]):].tolist()
            response = self.tokenizer.detokenize(output_indices)

        return response


# Load chatbot globally
print("Initializing chatbot...")
chatbot = ChatBot(
    model_path='causal_transformer_full.pth',
)
model_loaded = True
print("Chatbot initialized successfully!")


@app.route('/')
def home():
    return render_template('index.html', model_status=model_loaded)


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    if not user_message:
        return jsonify({'response': 'Please provide a message.'})

    response = chatbot.generate_response(user_message)
    return jsonify({'response': response})


if __name__ == '__main__':
    app.run(debug=True)
