import logging
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import openai
import torch
import torch.nn.functional as F
from opensearchpy import OpenSearch, RequestsHttpConnection

from embed_tune.training.model import EmbeddingAlign

# Logging config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_PATH = "../models/models_title_margin_0.2.pt"
INPUT_FILE = "../results/product_ids.txt"
CLEANED_FILE = "../results/cleaned_product_ids.txt"
OPENAI_MODEL = "text-embedding-ada-002"
INDEX = os.environ["INDEX"]
HOST = os.environ["HOST"]
PORT = os.environ["PORT"]

# Initialize OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY not set in environment variables.")
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)


def init_device_model() -> tuple[torch.device, torch.nn.Module]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmbeddingAlign(emb_dim=1536).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    logger.info("Model loaded and ready on %s", device)
    return device, model


def create_openai_embedding(input_text: str) -> list[float]:
    response = openai_client.embeddings.create(model=OPENAI_MODEL, input=input_text)
    return response.data[0].embedding


def linear_transform(device: torch.device, model: torch.nn.Module, input_text: str) -> torch.Tensor:
    embedding = create_openai_embedding(input_text)
    torch_embedding = torch.tensor(embedding, dtype=torch.float).to(device)
    with torch.no_grad():
        output = model(torch_embedding)
    return output


def extract_doc_ids(input_file: str, output_file: str) -> list[str]:
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

    logger.info("Extracted %d doc ids to %s", len(doc_ids), output_file)
    return doc_ids


def load_doc_ids(file_path: str) -> list[str]:
    with open(file_path, "r") as f:
        return [line.strip() for line in f.readlines()]


def init_opensearch_client() -> OpenSearch:
    return OpenSearch(
        hosts=[{"host": HOST, "port": PORT}],
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        pool_maxsize=20,
    )


def fetch_documents(doc_ids: list[str], index: str, client: OpenSearch) -> list[dict]:
    query = {
        "size": 128,
        "query": {
            "ids": {
                "values": doc_ids,
            },
        },
    }
    response = client.search(index=index, body=query)
    return response["hits"]["hits"]


def compute_similarity(query_emb: torch.Tensor, product_embs: list[tuple[torch.Tensor, str]]) -> list[tuple[float, str]]:
    scores = [
        (F.cosine_similarity(query_emb.unsqueeze(0), emb.unsqueeze(0), dim=1).item(), title)
        for emb, title in product_embs
    ]
    return sorted(scores, key=lambda x: x[0], reverse=True)


def plot_similarity_scores(similarity_scores: list[float]) -> None:
    scores = np.array(similarity_scores)
    sorted_scores = np.sort(scores)

    plt.figure(figsize=(10, 6))
    plt.plot(sorted_scores, marker='o', linestyle='-', color='teal')
    plt.xlabel("Sorted Product Index")
    plt.ylabel("Cosine Similarity")
    plt.title("Sorted Cosine Similarity Scores")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()


def main() -> None:
    logger.info("Starting inference pipeline")

    device, model = init_device_model()
    query_text = "Cute top"
    query_vector = linear_transform(device, model, query_text)

    if not os.path.exists(CLEANED_FILE):
        extract_doc_ids(INPUT_FILE, CLEANED_FILE)

    doc_ids = load_doc_ids(CLEANED_FILE)
    client = init_opensearch_client()
    hits = fetch_documents(doc_ids, INDEX, client)

    logger.info("Fetched %d documents from OpenSearch", len(hits))

    products_transformed = []
    for hit in hits:
        raw_emb = hit['_source']['sanitized_knowledge_record.embedding']
        title = hit['_source']['product_info.title.text']
        torch_emb = torch.tensor(raw_emb, dtype=torch.float).to(device)
        with torch.no_grad():
            transformed = model(torch_emb)
        products_transformed.append((transformed, title))

    similarity = compute_similarity(query_vector, products_transformed)
    for score, title in similarity[:5]:
        logger.info("Score: %.4f | Title: %s", score, title)

    plot_similarity_scores([score for score, _ in similarity])


if __name__ == "__main__":
    main()
