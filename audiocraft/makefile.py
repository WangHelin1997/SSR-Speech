import json
import tqdm
import pandas as pd
import os
import random

# Load the JSON data
jsondata = pd.read_json(path_or_buf=os.path.join('./Gigaspeech', 'trans', 'train.json'), lines=True)

# Ensure there are at least 10,000 items in jsondata
data_length = len(jsondata)

# Randomly sample 800,000 unique indices
sample_indices = range(data_length) if data_length < 800000 else random.sample(range(data_length), 800000)

# Prepare the sampled data
data = []
for i in tqdm.tqdm(sample_indices):
    tmp = {
        "path": jsondata['wav'][i],
        "duration": jsondata['duration'][i],
        "sample_rate": 16000,
        "amplitude": None,
        "weight": None,
        "info_path": None
    }
    data.append(tmp)

# Create the output directory if it does not exist
os.makedirs('./egs/train', exist_ok=True)

# Define the output file path
output_file = './egs/train/data.jsonl'

# Write the sampled data to the JSONL file
with open(output_file, 'w') as file:
    for record in data:
        json_line = json.dumps(record)
        file.write(json_line + '\n')

print(f"Data written to {output_file}")

