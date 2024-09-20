from osfclient.api import OSF
import os

project_id = 'psychProbing'

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