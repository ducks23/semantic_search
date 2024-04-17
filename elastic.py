from search import get_embedding
import pandas as pd
import numpy as np


from elasticsearch import Elasticsearch

password = "mw7XKZnvXKRYKDK*AG=k"
ssl_cert = "4a162c2b3a229b4c7e4deff09e2cdffea87e006dffaf57e98b1e3917868e865c"


es = Elasticsearch(
    "https://localhost:9200",
    basic_auth=("elastic", password),
    ssl_assert_fingerprint=ssl_cert,
)


index_mapping = {
    "properties": {
        "id": {"type": "long"},
        "title": {"type": "text"},
        "description": {"type": "text"},
        "label_int": {"type": "long"},
        "label": {"type": "text"},
        "embedding": {
            "type": "dense_vector",
            "dims": 1536,
            "index": True,
            "similarity": "l2_norm",
        },
    }
}

res = es.indices.create(index="article", mappings=index_mapping)

print(res)


df = pd.read_csv("word_embeddings.csv", index_col=0)
df["embedding"] = df.embedding.apply(eval).apply(np.array)

record_list = df.to_dict("records")


count = 0

for record in record_list:
    count += 1
    try:
        es.index(index="articles", document=record, id=count)
    except Exception as e:
        print(e)


search_term = "stock market crash"

search_term_vector = get_embedding(search_term)
query = {
    "field": "embedding",
    "query_vector": search_term_vector,
    "k": 20,
    "num_candidates": 10000,
}

res = es.knn_search(index="articles", knn=query, source=["title"])

response = dict(res)["hits"]["hits"]
import pprint

pprint.pp(response)
