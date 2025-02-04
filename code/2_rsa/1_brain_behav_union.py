import json
import pickle
import pandas as pd

# Getting brain and behavior names
with open('../../data/dtype_to_embed.json', 'r') as f:
    dtype_to_embed = json.load(f)
    brain_behav_names = dtype_to_embed['brain'] + dtype_to_embed['behavior']

# Getting brain and behavior vocabulary union
brain_behav_union = set()
for name in brain_behav_names:
    vocab = set(pd.read_csv('../../data/embeds/' + f'{name}.csv', index_col=0).index)
    brain_behav_union = brain_behav_union.union(vocab)

# Saving the union
with open('../../data/brain_behav_union.pkl', 'wb') as f:
    pickle.dump(brain_behav_union, f)