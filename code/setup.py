from osfclient.api import OSF
import os
import json
import pandas as pd
import pickle

###### Downloading data from OSF ######

project_id = 'nrkd7'

# Connect to OSF
osf = OSF()

# Find the project
project = osf.project(project_id)

# Get all files from the project
storage = project.storage('osfstorage')

# Loop through all files and download them
for folder in storage.folders:
    print(f'Entering folder {folder.name}...')

    output_dir = os.path.join('..', 'data', folder.name)

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in folder.files:
        print(f'Downloading {file.name}...')
        # Create full local file path
        local_path = os.path.join(output_dir, file.name)
        # Download the file
        with open(local_path, 'wb') as f:
            file.write_to(f)
        print(f'Saved {file.name} to {output_dir}')

###### Finding brain-behavior union ######

# Getting brain and behavior names
with open('../data/dtype_to_embed.json', 'r') as f:
    dtype_to_embed = json.load(f)
    brain_behav_names = dtype_to_embed['brain'] + dtype_to_embed['behavior']

# Getting brain and behavior vocabulary union
brain_behav_union = set()
for name in brain_behav_names:
    vocab = set(pd.read_csv('../data/embeds/' + f'{name}.csv', index_col=0).index)
    brain_behav_union = brain_behav_union.union(vocab)

# Saving the union
with open('../data/brain_behav_union.pkl', 'wb') as f:
    pickle.dump(brain_behav_union, f)

