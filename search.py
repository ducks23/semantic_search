import pandas as pd
import numpy as np
from openai import OpenAI

client = OpenAI(
    max_retries=5,
)

df = pd.read_csv("./AG_news_samples.csv")


def get_embedding(text: str, model="text-embedding-3-small", **kwargs):
    # replace newlines, which can negatively affect performance.
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model, **kwargs)
    print("embedding received")
    return response.data[0].embedding


df["embedding"] = df["title"].apply(lambda x: get_embedding(x))


df.to_csv("word_embeddings.csv")

search_term = "stock market crash"

search_term_vector = get_embedding(search_term)

df = pd.read_csv("word_embeddings.csv")

df["embedding"] = df["embedding"].apply(eval).apply(np.array)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


df["similarities"] = df["embedding"].apply(
    lambda x: cosine_similarity(x, search_term_vector)
)

df = df.sort_values("similarities", ascending=False).head(20)
df.to_csv("similarities.csv")
