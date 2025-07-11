import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import pickle
from tqdm.auto import tqdm

# Set a seed for reproducibility
torch.random.manual_seed(42)

# Find intersection of norms and brain_behavior_union
norms_voc = set(
    pd.read_csv('../../data/psychNorms/psychNorms.zip', index_col=0, low_memory=False, compression='zip').index
)
with open('../../data/brain_behav_union.pkl', 'rb') as f:
    brain_behav_union = pickle.load(f)
to_extract = list(norms_voc & brain_behav_union)

# --- Device Setup ---
# Detecting device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('CUDA is available. Using GPU.')
elif torch.backends.mps.is_available(): # For Apple Silicon
    device = torch.device("mps")
    print("MPS is available. Using Apple's Metal.")
else:
    device = torch.device("cpu")
    print("No GPU or MPS available. Using CPU.")

# --- Model and Tokenizer Setup ---
model_path = '/scicore/home/matar/hussai0001/GROUP/bert-large-uncased'

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Model
# Load the model and move it to the selected device.
# For a model of this size, device_map='auto' is not necessary.
model = AutoModel.from_pretrained(model_path)
model.to(device)
model.eval()

# --- Embedding Extraction ---
# Input templates to provide context to the words
templates = [
    "{}",
    "What is the meaning of {}?",
    "Think about {}."
]

batch_size = 16
# Iterate through each template
for i, template in enumerate(templates):
    print(f"\n--- Processing Template {i + 1}/{len(templates)}: '{template}' ---")
    temp_embeds = {}
    placeholder_pos = template.find("{}")
    with torch.no_grad():
        for batch_start in tqdm(range(0, len(to_extract), batch_size), desc=f"Template {i + 1}"):
            batch_end = batch_start + batch_size
            batch_words = to_extract[batch_start:batch_end]

            formatted_batch = [template.format(word) for word in batch_words]

            # 1. Tokenize the batch. This returns a `BatchEncoding` object
            # which has the `.char_to_token` method needed below.
            inputs_encoding = tokenizer(
                formatted_batch,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            )

            # 2. Find the token indices corresponding to each target word.
            # This logic works for BERT's subword tokenization as well.
            all_token_indices = []
            for k, word in enumerate(batch_words):
                # Find the character start and end positions for the word
                start_char = placeholder_pos
                end_char = start_char + len(word) - 1

                # Convert character positions to token positions
                start_token = inputs_encoding.char_to_token(k, start_char)
                end_token = inputs_encoding.char_to_token(k, end_char)

                if start_token is not None and end_token is not None:
                    all_token_indices.append(range(start_token, end_token + 1))
                else:
                    # Handle cases where the word might be tokenized in an unexpected way
                    all_token_indices.append(None)

            # 3. Move tokenized inputs to the same device as the model
            inputs_on_device = {key: val.to(device) for key, val in inputs_encoding.items()}

            # 4. Get model outputs (hidden states)
            outputs = model(**inputs_on_device)
            last_hidden_state = outputs.last_hidden_state.cpu()

            # 5. Extract and average the embeddings for each target word
            for k, word in enumerate(batch_words):
                token_indices = all_token_indices[k]
                if token_indices is None:
                    continue  # Skip if token indices couldn't be found

                # Select the hidden states for the tokens of the current word
                word_hidden_states = last_hidden_state[k, token_indices, :]

                # Average the hidden states to get a single representation
                if word_hidden_states.shape[0] > 0:
                    averaged_representation = torch.mean(word_hidden_states, dim=0)
                    temp_embeds[word] = averaged_representation.float()

    # --- Saving the Results for the Current Template ---
    if temp_embeds:
        temp_embeds_df = pd.DataFrame(temp_embeds).T.astype(float)
        model_id = model_path.split("/")[-1]
        output_path = f'../../data/llms/{model_id}_{i}.csv'
        temp_embeds_df.to_csv(output_path)
        print(f"Saved embeddings for template {i + 1} to '{output_path}'")
        print("Example embeddings DataFrame head:")
        print(temp_embeds_df.head())
    else:
        print(f"No embeddings were extracted for template {i + 1}. No file was saved.")