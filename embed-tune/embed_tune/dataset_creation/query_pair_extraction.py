import ast
import json
import logging
import os

from typing import Any
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

import openai
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tqdm import tqdm


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables for secrets
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY not set in environment variables.")

openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

PROMPT_SYSTEM_SYNONYM = """
You need to create a query based on the product description
Your query will be in this format :
synonym of the product type and a criteria

the criteria  product description

EXAMPLES :
Manteau mi-long

this is for dataset augmentation so be creative while sticking to the product

Only generate 1 query and only one query

write in french
Tu dois utiliser des synonymes afin d'enrichir syntaxiquement mon dataset

Here is the product description :
"""

DROP_CATEGORIES = [
    "American Dream",
    "Shop by size",
    "111",
    "Achat par taille",
    "Tout voir",
    "Idées cadeaux",
]


def create_query(row: dict[str, Any]) -> str:
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.5,
        messages=[
            {"role": "system", "content": PROMPT_SYSTEM_SYNONYM},
            {"role": "user", "content": row["sanitizedKnowledgeRecord"]},
        ],
    )
    return response.choices[0].message.content if response.choices[0].message.content else ""


def create_query_embedding(row: dict[str, Any]) -> list[float]:
    embedding = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=row["query"],
    )
    return embedding.data[0].embedding


def create_sanitized_embedding(row: dict[str, Any]) -> list[float]:
    embedding = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=row["title"],
    )
    return embedding.data[0].embedding


def apply_parallel_with_progress(df: pd.DataFrame, func, num_threads: int = 10) -> list[Any]:
    results: list[Any] = [None] * len(df)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(func, row): idx for idx, (_, row) in enumerate(df.iterrows())}

        for future in tqdm(as_completed(futures), total=len(df), desc="Processing"):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.exception(f"Error processing row {idx}: {e}")
                results[idx] = None
    return results


def preprocess_csv(input_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    df["productTypes"] = df["productTypes"].apply(lambda x: ast.literal_eval(x)[0])
    df = df[~df["productTypes"].isin(DROP_CATEGORIES)]
    return df


def save_embeddings_to_parquet(df: pd.DataFrame, output_file: str) -> None:
    columns = ["query", "query_embedding", "good", "good_embedding", "bad", "bad_embedding"]
    parquet_writer = None

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Creating triplets"):
        product_type = row["productTypes"]
        filtered_df = df[df["productTypes"] != product_type].sample(frac=0.02).reset_index(drop=True)

        batch = [
            (
                row["query"],
                row["query_embedding"],
                row["title"],
                row["record_embedding"],
                f_row["title"],
                f_row["record_embedding"],
            )
            for _, f_row in filtered_df.iterrows()
        ]

        if not batch:
            continue

        batch_df = pd.DataFrame(batch, columns=columns)
        table = pa.Table.from_pandas(batch_df)

        if parquet_writer is None:
            parquet_writer = pq.ParquetWriter(output_file, table.schema, compression="snappy")

        parquet_writer.write_table(table)

    if parquet_writer:
        parquet_writer.close()


def main():
    input_csv = "../results/os_extract_catalog.csv"
    output_csv = "../results/annotated_catalog.csv"
    output_parquet = "../results/triplet_dataset.parquet"

    logger.info("Loading data...")
    df = pd.read_csv(input_csv)
    df["productTypes"] = df["productTypes"].apply(lambda x: x[2:-2].split(","))
    logger.info(f"Dataset size: {len(df)}")

    logger.info("Generating queries...")
    df["query"] = apply_parallel_with_progress(df, create_query)

    logger.info("Generating query embeddings...")
    df["query_embedding"] = apply_parallel_with_progress(df, create_query_embedding)

    logger.info("Generating record embeddings...")
    df["record_embedding"] = apply_parallel_with_progress(df, create_sanitized_embedding)

    df["query_embedding"] = df["query_embedding"].apply(json.dumps)
    df["record_embedding"] = df["record_embedding"].apply(json.dumps)

    logger.info("Saving to CSV...")
    df.to_csv(output_csv, index=False)

    logger.info("Postprocessing for Parquet creation...")
    df = preprocess_csv(output_csv)
    df["query_embedding"] = df["query_embedding"].apply(json.loads)
    df["record_embedding"] = df["record_embedding"].apply(json.loads)

    save_embeddings_to_parquet(df, output_parquet)

    logger.info("All done!")


if __name__ == "__main__":
    main()
