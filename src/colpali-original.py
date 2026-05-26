from typing import cast          # (unused here, can be removed)
import torch                     # PyTorch – tensor ops, GPU handling, etc.
from PIL import Image            # Pillow – create / load image objects

from colpali_engine.models import ColPali, ColPaliProcessor


# ------------------------------------------------------------
# Load the pretrained multimodal model and its processor
# ------------------------------------------------------------
model_name = "vidore/colpali-v1.3"

# `from_pretrained` downloads the weights from Hugging‑Face.
# * `torch_dtype=bfloat16` uses 16‑bit floats → less memory, faster on modern GPUs.
# * `device_map="cuda:0"` places the whole model on the first GPU.
model = ColPali.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
).eval()                     # switch to inference mode (no dropout, etc.)

# The processor knows how to turn raw images / text into the tensors
# the model expects (resize, normalize, tokenize, …).
processor = ColPaliProcessor.from_pretrained(model_name)


# ------------------------------------------------------------
# Your inputs – a few example images and text queries
# ------------------------------------------------------------
images = [
    Image.new("RGB", (32, 32), color="white"),   # white square
    Image.new("RGB", (16, 16), color="black"),   # black square
]

queries = [
    "Is attention really all you need?",
    "Are Benjamin, Antoine, Merve, and Jo best friends?",
]


# ------------------------------------------------------------
# Pre‑process inputs → tensors ready for the model (still on CPU)
# ------------------------------------------------------------
# Convert the list of PIL images into a dictionary of tensors (batch dimension = len(images)),
# then ('to(model.device)') move every tensor in that dictionary onto the same GPU where the model lives.
batch_images = processor.process_images(images).to(model.device)

# `process_queries` tokenizes the strings, pads them to equal length and
# returns a batch dict (keys: "input_ids", "attention_mask").
batch_queries = processor.process_queries(queries).to(model.device)


# ------------------------------------------------------------
# Forward pass – compute embeddings for images and queries
### HEAVY COMPUTE ###
# ------------------------------------------------------------
# `torch.no_grad()` disables gradient tracking → lower memory usage; just for inference
with torch.no_grad():
    image_embeddings = model(**batch_images)   # shape: (num_images, embed_dim)
    query_embeddings = model(**batch_queries)  # shape: (num_queries, embed_dim)


# ------------------------------------------------------------
# Compute similarity scores between every query‑image pair
# ------------------------------------------------------------
# The processor uses a dot‑product (or cosine) to measure how close
# each query vector is to each image vector.  The result is a matrix:
# rows = queries, columns = images.
scores = processor.score_multi_vector(query_embeddings, image_embeddings)

# `scores[i, j]` → similarity of query i to image j (higher = more related)