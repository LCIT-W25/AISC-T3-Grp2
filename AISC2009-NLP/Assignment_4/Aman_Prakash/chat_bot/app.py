
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


class ChatBot:

    def __init__(self, model_path, word2vec_path):
        # Load word embeddings
        print("Loading word embeddings...")
        self.word2vec = KeyedVectors.load(word2vec_path)

        # Create vocabulary mappings
        print("Creating vocabulary mappings...")
        self.word_to_idx = {word: idx for idx, word in enumerate(
            self.word2vec.index_to_key[:VOCAB_SIZE-2])}
        self.word_to_idx['<PAD>'] = len(self.word_to_idx)
        self.word_to_idx['<UNK>'] = len(self.word_to_idx)
        self.idx_to_word = {idx: word for word,
                            idx in self.word_to_idx.items()}

        # Create pretrained embeddings tensor
        print("Creating embedding matrix...")
        embedding_matrix = torch.zeros((VOCAB_SIZE, EMBEDDING_DIM))
        for word, idx in self.word_to_idx.items():
            if word in ['<PAD>', '<UNK>']:
                embedding_matrix[idx] = torch.randn(EMBEDDING_DIM)
            else:
                embedding_matrix[idx] = torch.tensor(self.word2vec[word])

        # Initialize model
        print("Initializing model...")
        self.model = CausalTransformer()

        # Add CausalTransformer to safe globals list for PyTorch 2.6+
        print("Adding to safe globals...")
        # torch.serialization.add_safe_globals(
        #     [CausalTransformer])

        # Load pretrained weights with correct parameters
        print(f"Loading model from {model_path}...")
        try:
            # First try with weights_only=True (safer)
            # self.model.load_state_dict(torch.load(
            #     model_path, map_location=torch.device('cpu')))
            self.model = torch.load('causal_transformer_full.pth',
                                    map_location=torch.device('cpu'), weights_only=False)
        except Exception as e:
            print(f"Error loading with weights_only=True: {e}")
            print("Trying with weights_only=False (less secure but may work)...")
            # Try with weights_only=False as fallback
            checkpoint = torch.load(
                model_path, map_location=torch.device('cpu'), weights_only=False)

            # If the checkpoint contains state_dict, use that
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            # Otherwise try to load directly
            else:
                try:
                    self.model.load_state_dict(checkpoint)
                except:
                    print(
                        "Direct loading failed, trying to extract model parameters...")
                    # If the checkpoint is the model itself
                    if hasattr(checkpoint, 'state_dict'):
                        self.model.load_state_dict(checkpoint.state_dict())
                    else:
                        # Last resort: Create a new model instance from the checkpoint
                        self.model = checkpoint

        print("Model loaded successfully!")
        self.model.eval()

    def preprocess(self, text):
        # Clean and tokenize
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s.,?!]', '', text)
        text = re.sub(r'@[\w]+', ' ', text)  # Remove mentions
        text = re.sub(r'http\S+|www\S+', ' ', text)  # Remove URLs
        text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove emojis
        words = text.split()

        # Convert to indices
        indices = []
        for word in words:
            if word in self.word_to_idx:
                indices.append(self.word_to_idx[word])
            else:
                indices.append(self.word_to_idx['<UNK>'])

        if len(indices) > MAX_SEQ_LENGTH:
            indices = indices[-MAX_SEQ_LENGTH:]

        return torch.tensor([indices], dtype=torch.long)

    def generate_response(self, input_text, max_length=50):
        full_context = input_text
        input_tensor = self.preprocess(input_text)

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
                if next_token_idx == self.word_to_idx.get('<PAD>', 0):
                    break
                input_tensor = torch.cat(
                    [input_tensor, next_token_idx.unsqueeze(0)], dim=1)

            output_indices = input_tensor[0, len(
                self.preprocess(full_context)[0]):].tolist()
            response_words = [self.idx_to_word.get(
                idx, '<UNK>') for idx in output_indices]
            response = ' '.join(response_words)
            response = response.replace('<PAD>', '').replace('<UNK>', '')
            response = re.sub(r'\s+', ' ', response).strip()

        return response


# Load chatbot globally
print("Initializing chatbot...")
chatbot = ChatBot(
    model_path='causal_transformer_full.pth',
    word2vec_path='word2vec-google-news-300.model'
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
    if not model_loaded:
        return jsonify({'response': 'Sorry, the chatbot model could not be loaded. Please check the server logs.'})

    response = chatbot.generate_response(
        user_message)

    return jsonify({'response': response})


if __name__ == '__main__':
    app.run(debug=True)
