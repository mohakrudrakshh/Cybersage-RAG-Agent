import pandas as pd
from sentence_transformers import SentenceTransformer, util
from typing import List, Tuple

class LocalSemanticSearch:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.model = SentenceTransformer("all-mpnet-base-v2")
        self.df["embedding"] = self.df["embedding"].apply(eval)
        self.embeddings = list(self.df["embedding"])
        self.text_chunks = list(self.df["sentence_chunk"])

    def semantic_search(self, query: str, top_k: int = 3) -> List[str]:
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, self.embeddings)[0]

        top_results = sorted(
            zip(self.text_chunks, scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        return [result[0] for result in top_results]

    def similarity_score(self, query: str, text: str) -> float:
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        text_embedding = self.model.encode(text, convert_to_tensor=True)
        score = util.cos_sim(query_embedding, text_embedding)[0][0].item()
        return score
