import matplotlib.pyplot as plt
import numpy as np
import openai
import seaborn as sns
import torch

from embedding_ft.training.main import EmbeddingAlign


api_key= "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"


def init_device_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EmbeddingAlign(emb_dim=1536).to(device)

    model.load_state_dict(torch.load("../models/models_one_matrix_margin_0.2.pt", map_location=device))
    model.eval()
    return device, model

device, model = init_device_model()

def linear_transform(device, model, input_text):
    emb = create_openai_embedding(input=input_text)

    torch_emb = torch.tensor(emb, dtype=torch.float).to(device)
    with torch.no_grad():
        output = model(torch_emb)

    return output


sample_input = "tot"

def create_openai_embedding(input: str):
    client = openai.OpenAI(api_key=api_key)
    embeding = client.embeddings.create(
        model="text-embedding-ada-002",
        input=input,
    )
    return embeding.data[0].embedding



print("Inference output:", linear_transform(device, model, sample_input))

import os
import re


print(os.getcwd())

input_file = "product_ids.txt"
output_file = "cleaned_product_ids.txt"

pattern = re.compile(r"doc_id:\s*'([^']+)'")

doc_ids = []

with open(input_file, "r") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            doc_ids.append(match.group(1))

with open(output_file, "w") as f:
    for doc_id in doc_ids:
        f.write(doc_id + "\n")

print(f"Extracted {len(doc_ids)} doc ids to {output_file}")


cleaned_file = "cleaned_product_ids.txt"

doc_ids = []
with open(cleaned_file, "r") as file:
    for line in file:
        doc_ids.append(line.strip())


print("Loaded doc IDs:", doc_ids)
print(len(((doc_id))))

from opensearchpy import OpenSearch
from opensearchpy import RequestsHttpConnection


track = "automation"
host = "vpc-dev-main-vector-es-afa3bdqfsw247twar3lm4rmhoa.eu-central-1.es.amazonaws.com"
port = "443"
open_search_client = OpenSearch(
    hosts=[{"host": host, "port": port}],
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
    pool_maxsize=20,
)

index = "automation_11_872d78a9-3e30-4d63-a7e5-544e5638ccf0"

query_body = {
        "size": 128,
        "query": {
            "ids": {
                "values": doc_ids,
            },
        },
    }
response = open_search_client.search(index=index, body=query_body)
print((response['hits']['hits'][0]["_source"].keys()))
print((response['hits']['hits'][0]["_source"]['product_info.title.text']))


import torch.nn.functional as F


query_embedding = create_openai_embedding("cute top")

product_embeddings = []

for hit in response['hits']['hits'] :
    product_embeddings.append((hit['_source']['sanitized_knowledge_record.embedding'],hit['_source']['product_info.title.text']))


distance_scores = [
    (F.cosine_similarity(torch.tensor(query_embedding, dtype=torch.float).unsqueeze(0), torch.tensor(product_embedding, dtype=torch.float).unsqueeze(0), dim=1).item(), title)
    for product_embedding, title in product_embeddings
]
((sorted(distance_scores, key= lambda x: x[0], reverse=True)))


import matplotlib.pyplot as plt

distance_scores = [score[0] for score in distance_scores]
print(distance_scores)
scores = np.array(distance_scores)

sorted_indices = np.argsort(scores)
sorted_scores = scores[sorted_indices]

plt.figure(figsize=(10, 6))
plt.plot(sorted_scores, marker='o', linestyle='-', color='teal')
plt.xlabel("Sorted Product Index")
plt.ylabel("Cosine Similarity")
plt.title("Sorted Cosine Similarity Scores")
plt.ylim(0, 1)
plt.grid(True)
plt.show()