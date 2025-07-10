import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import pickle
from tqdm.auto import tqdm
import gc  # Import garbage collector

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

mod_batch_sizes = {
    'meta-llama/Llama-3.2-1B': 16,
    'meta-llama/Llama-3.2-3B': 4,
    'meta-llama/Llama-3.1-8B': 2
}

for model_name, batch_size in mod_batch_sizes.items():
    print(f"\n--- Processing model: {model_name} ---")  # Added for clarity

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # Set padding side

    # Model
    model = AutoModel.from_pretrained(
        model_name, torch_dtype=torch.bfloat16  # Use bfloat16 for memory efficiency
    ).to(device)
    model.eval()

    mod_embeds = {}
    with torch.no_grad():
        # Loop through the data in chunks of `batch_size`
        for i in tqdm(range(0, len(to_extract), batch_size), desc="Extracting embeddings"):
            # Create a batch from the large list
            batch_words = to_extract[i:i + batch_size]

            inputs = tokenizer(batch_words, return_tensors='pt', padding=True, truncation=True)
            all_word_ids = [inputs.word_ids(j) for j in range(len(batch_words))]
            inputs = {key: val.to(device) for key, val in inputs.items()}

            outputs = model(**inputs)
            last_hidden_state = outputs.last_hidden_state.cpu()

            for k, word in enumerate(batch_words):
                word_ids = all_word_ids[k]
                word_token_indices = [j for j, wid in enumerate(word_ids) if wid is not None]

                word_hidden_states = last_hidden_state[k, word_token_indices, :]
                averaged_word_representation = torch.mean(word_hidden_states, dim=0)

                mod_embeds[word] = averaged_word_representation

    mod_embeds = pd.DataFrame(mod_embeds).T.astype(float)
    mod_embeds.to_csv(f'../../data/llms/{model_name.split("/")[-1]}.csv')
    print(f"Saved embeddings for {model_name}")

    # --- MEMORY CLEANUP ---
    # Explicitly delete model and tokenizer to free up RAM
    del model
    del tokenizer
    # Force Python's garbage collector to run
    gc.collect()
    # Clear PyTorch's cache
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()