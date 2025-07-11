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

# Detecting device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('CUDA is available. Using GPU.')
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS is available. Using Apple's Metal.")
else:
    device = torch.device("cpu")
    print("No GPU or MPS available. Using CPU.")

# --- Model and Tokenizer Setup ---
model_name = 'meta-llama/Llama-3.1-8B'

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Model
model = AutoModel.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # Use bfloat16 for memory efficiency
    device_map='auto'  # Automatically handle model placement on available devices
)
model.eval()

# Input templates
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
    with torch.no_grad():
        for batch_start in tqdm(range(0, len(to_extract), batch_size), desc=f"Template {i + 1}"):
            batch_end = batch_start + batch_size
            batch_words = to_extract[batch_start:batch_end]

            formatted_batch = [template.format(word) for word in batch_words]

            # 1. Tokenize the batch. This returns a `BatchEncoding` object
            # which has the `.char_to_token` method.
            inputs_encoding = tokenizer(
                formatted_batch,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            )

            # 2. FIX: Use the `inputs_encoding` object to get token spans
            # *before* converting it to a plain dictionary for the model.
            all_token_indices = []
            for k, word in enumerate(batch_words):
                current_sentence = formatted_batch[k]
                try:
                    start_char = current_sentence.index(word)
                    end_char = start_char + len(word) - 1

                    # Use the method from the BatchEncoding object
                    start_token = inputs_encoding.char_to_token(k, start_char)
                    end_token = inputs_encoding.char_to_token(k, end_char)

                    if start_token is not None and end_token is not None:
                        all_token_indices.append(range(start_token, end_token + 1))
                    else:
                        all_token_indices.append(None)  # Mark as failed

                except ValueError:
                    all_token_indices.append(None)  # Mark as failed

            # 3. Now, move the tensor data to the correct device for the model
            inputs_on_device = {key: val.to(model.device) for key, val in inputs_encoding.items()}

            # 4. Get model outputs
            outputs = model(**inputs_on_device)
            last_hidden_state = outputs.last_hidden_state.cpu()

            # 5. Extract embeddings using the pre-calculated token spans
            for k, word in enumerate(batch_words):
                token_indices = all_token_indices[k]
                if token_indices is None:
                    continue  # Skip if we failed to find the span earlier

                word_hidden_states = last_hidden_state[k, token_indices, :]

                if word_hidden_states.shape[0] > 0:
                    averaged_representation = torch.mean(word_hidden_states, dim=0)
                    temp_embeds[word] = averaged_representation.float()

    # --- Saving the Results for the Current Template ---
    if temp_embeds:
        temp_embeds_df = pd.DataFrame(temp_embeds).T.astype(float)
        model_id = model_name.split("/")[-1]
        output_path = f'./{model_id}_template_{i}.csv'
        temp_embeds_df.to_csv(output_path)
        print(f"✅ Saved embeddings for template {i + 1} to '{output_path}'")
    else:
        print(f"⚠️ No embeddings were extracted for template {i + 1}. No file was saved.")