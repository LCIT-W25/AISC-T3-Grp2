import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import io
from transformers import TFAutoModel, AutoTokenizer
import numpy as np

class TimeEmbedding(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(dim)
        self.act = tf.keras.layers.Activation('swish')
        self.dense2 = tf.keras.layers.Dense(dim)

    def call(self, t):
        t = tf.cast(t, tf.float32)
        x = self.dense1(t[:, None])
        x = self.act(x)
        return self.dense2(x)


class ResBlock(tf.keras.Model):
    def __init__(self, channels, time_emb_dim, text_emb_dim):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(channels, 3, padding='same')
        self.act1 = tf.keras.layers.Activation('swish')
        self.conv2 = tf.keras.layers.Conv2D(channels, 3, padding='same')
        self.act2 = tf.keras.layers.Activation('swish')
        self.time_dense = tf.keras.layers.Dense(channels)
        self.text_dense = tf.keras.layers.Dense(channels)

        # 1x1 conv to match input shape if needed
        self.skip_conv = tf.keras.layers.Conv2D(channels, 1, padding='same')

    def call(self, x, t_emb, txt_emb):
        h = self.conv1(x)
        h += tf.reshape(self.time_dense(t_emb), [-1, 1, 1, h.shape[-1]])
        h += tf.reshape(self.text_dense(txt_emb), [-1, 1, 1, h.shape[-1]])
        h = self.act1(h)
        h = self.conv2(h)

        # Ensure skip connection matches shape
        x_proj = self.skip_conv(x) if x.shape[-1] != h.shape[-1] else x

        return self.act2(h + x_proj)

class UNet(tf.keras.Model):
    def __init__(self, base_channels=32, time_emb_dim=128, text_emb_dim=256):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_emb_dim)
        self.down1 = ResBlock(base_channels, time_emb_dim, text_emb_dim)
        self.down2 = ResBlock(base_channels*2, time_emb_dim, text_emb_dim)
        self.pool = tf.keras.layers.MaxPooling2D()

        self.middle = ResBlock(base_channels*4, time_emb_dim, text_emb_dim)

        self.up2 = ResBlock(base_channels*2, time_emb_dim, text_emb_dim)
        self.up1 = ResBlock(base_channels, time_emb_dim, text_emb_dim)
        self.upsample = tf.keras.layers.UpSampling2D()

        self.out = tf.keras.layers.Conv2D(3, 1)

    def call(self, x, t, text_emb):
        t_emb = self.time_embedding(t)

        d1 = self.down1(x, t_emb, text_emb)
        d2 = self.down2(self.pool(d1), t_emb, text_emb)

        m = self.middle(self.pool(d2), t_emb, text_emb)

        u2 = self.up2(self.upsample(m), t_emb, text_emb)
        u1 = self.up1(self.upsample(u2 + d2), t_emb, text_emb)

        out = self.out(u1 + d1)
        return out

# Load DistilBERT (or MiniLM, TinyBERT, etc.)
bert = TFAutoModel.from_pretrained("distilbert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def encode_prompt_bert(prompt_tensor):
    # Convert Tensor to list of Python strings
    prompt_list = prompt_tensor.numpy().astype(str).tolist()
    
    # Now use tokenizer
    inputs = tokenizer(prompt_list, return_tensors="tf", padding=True, truncation=True)
    outputs = bert(**inputs)
    return outputs.last_hidden_state[:, 0, :]  # CLS token

class DDPMNoiseScheduler:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        self.betas = np.linspace(beta_start, beta_end, timesteps, dtype=np.float32)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = np.cumprod(self.alphas)

    def add_noise(self, x0, t, noise):
        """
        x0: original image (B, H, W, C)
        t: timestep indices (B,)
        noise: noise to add (B, H, W, C)
        """
        alpha_bar_t = tf.gather(self.alpha_bars, t)
        alpha_bar_t = tf.reshape(alpha_bar_t, [-1, 1, 1, 1])

        return tf.sqrt(alpha_bar_t) * x0 + tf.sqrt(1 - alpha_bar_t) * noise

# Load model and scheduler (adjust to your saved path and model)
unet = UNet(base_channels=64, time_emb_dim=128, text_emb_dim=256)
unet(tf.zeros([1, 128, 128, 3]), tf.constant([0]), tf.zeros([1, 256]))  # build model
unet.load_weights("runs/20250330-110414/checkpoints/unet_epoch9.weights.h5")
scheduler = DDPMNoiseScheduler()

# BERT setup
bert_proj = tf.keras.layers.Dense(256)

def encode_prompt_bert(prompt_tensor):
    if isinstance(prompt_tensor, tf.Tensor):
        prompt_tensor = prompt_tensor.numpy()

    # If prompt_tensor is already a list of strings, keep it
    if isinstance(prompt_tensor, (list, tuple)):
        prompt_list = [s.decode("utf-8") if isinstance(s, bytes) else s for s in prompt_tensor]
    else:
        prompt_list = [prompt_tensor.decode("utf-8")] if isinstance(prompt_tensor, bytes) else [prompt_tensor]

    inputs = tokenizer(prompt_list, return_tensors="tf", padding=True, truncation=True)
    outputs = bert(**inputs)
    return outputs.last_hidden_state[:, 0, :]

def get_bert_embeddings_tf(prompts):
    output = tf.py_function(encode_prompt_bert, [prompts], tf.float32)
    output.set_shape([None, 768])
    return bert_proj(output)

def sample_from_model(prompt, steps=1000, image_size=128):
    text_emb = bert_proj(encode_prompt_bert([prompt]))
    x = tf.random.normal([1, image_size, image_size, 3])

    for t in reversed(range(steps)):
        t_tensor = tf.constant([t], dtype=tf.int32)
        pred_noise = unet(x, t_tensor, text_emb)

        alpha = tf.convert_to_tensor(scheduler.alphas[t], dtype=tf.float32)
        alpha_bar = tf.convert_to_tensor(scheduler.alpha_bars[t], dtype=tf.float32)
        alpha_bar_prev = tf.convert_to_tensor(
            scheduler.alpha_bars[t - 1] if t > 0 else 1.0, dtype=tf.float32
        )
        beta = 1 - alpha / alpha_bar_prev

        noise = tf.random.normal(shape=x.shape) if t > 0 else 0.0
        x = (1 / tf.sqrt(alpha)) * (x - (1 - alpha) / tf.sqrt(1 - alpha_bar) * pred_noise) + tf.sqrt(beta) * noise

    return tf.clip_by_value((x + 1.0) / 2.0, 0.0, 1.0)[0]

# Streamlit UI
st.set_page_config(page_title="Text-to-Image Diffusion", layout="centered")
st.title("Text-to-Image Generator")

prompt = st.text_input("Enter a prompt", value="sushi platter")
if st.button("Generate Image"):
    with st.spinner("Generating image..."):
        img = sample_from_model(prompt)
        img_np = (img.numpy() * 255).astype("uint8")
        img_pil = Image.fromarray(img_np)
        st.image(img_pil, caption=prompt, use_column_width=True)

        # Optional: download
        buf = io.BytesIO()
        img_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.download_button("Download Image", byte_im, file_name="generated.png", mime="image/png")