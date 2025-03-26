import json
import logging
import os

import click
import pandas as pd

from opensearchpy import OpenSearch
from opensearchpy import RequestsHttpConnection

from finetuning_tweaks.dataset_creation.utils import ProductParserUtils


logging.basicConfig(level=logging.INFO)


@click.command()
@click.option("-f", "--file_path", help="file path to save results", type=str, required=True)
def main(file_path: str) -> None:
    open_search_client = OpenSearch(
        hosts=[{"host": os.environ["HOST"], "port": os.environ["PORT"]}],
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        pool_maxsize=20,
    )

    index = os.environ["INDEX"]
    pagesize = 500
    scroll_time = "2m"  # Durée de vie du scroll

    # Initialiser le scroll
    result = open_search_client.search(
        index=index,
        body={"size": pagesize, "query": {"match_all": {}}},
        scroll=scroll_time,
    )

    scroll_id = result["_scroll_id"]
    hits = result["hits"]["hits"]

    columns = ["title", "description", "productTypes", "sanitizedKnowledgeRecord"]
    df = pd.DataFrame(columns=columns)
    doc_nb = 0
    while hits:
        for hit in hits:
            doc_nb += 1
            source = hit["_source"]
            raw_record = source.get("raw_record.object")

            if raw_record:
                record = json.loads(raw_record)
                sanitized_knowledge_record = ProductParserUtils.pretty_print_product_format(
                    record,
                    [
                        "availability",
                        "availabilityDate",
                        "price",
                        "salePrice",
                        "imageLink",
                        "additionalImageLink",
                        "link",
                    ],
                )

                title = record.get("title")
                description = record.get("description")
                product_type = record.get("productTypes")
                df.loc[len(df)] = [title, description, product_type, sanitized_knowledge_record]
        result = open_search_client.scroll(scroll_id=scroll_id, scroll=scroll_time)
        scroll_id = result["_scroll_id"]
        hits = result["hits"]["hits"]

        if doc_nb % 100 == 0:
            logging.info(doc_nb)

    logging.info(df.describe())
    df.to_csv(f"{file_path}.csv", index=False)
    open_search_client.clear_scroll(scroll_id=scroll_id)


if __name__ == "__main__":
    main()
