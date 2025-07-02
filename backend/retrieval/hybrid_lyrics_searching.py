import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from backend.retrieval.bm25 import BM25LyricsSearch
from backend.retrieval.sbert import SBERTSearcher
import numpy as np

class HybridLyricsSearch:
    def __init__(self, data_path, sbert_model='lyrics_sbert_model', alpha= 0.45):
        self.bm25 = BM25LyricsSearch(data_path)
        self.sbert = SBERTSearcher(data_path, model_name=sbert_model)
        self.alpha = alpha

    def normalize(self, scores: np.ndarray) -> np.ndarray:
            max_s = scores.max()
            min_s = scores.min()
            if max_s == min_s:
                return np.zeros_like(scores)
            return (scores - min_s) / (max_s - min_s)

    def search(self, query, top_k=5):
        bm25_scores = self.bm25.get_score(query)

        query_embedding = self.sbert.model.encode(query, convert_to_numpy=True)
        query_embedding = self.sbert._normalize(query_embedding.reshape(1, -1)).astype(np.float32)

        # Semantic scores trên toàn bộ corpus
        sbert_scores, _ = self.sbert.index.search(query_embedding, len(self.sbert.lyrics))
        sbert_scores = sbert_scores[0]  # Chỉ lấy 1 query

        bm25_norm = self.normalize(bm25_scores)
        sbert_norm_scores = self.normalize(sbert_scores)

        hybrid_scores = self.alpha * sbert_norm_scores + (1 - self.alpha) * bm25_norm

        # Sort by hybrid score
        sorted_indices = np.argsort(hybrid_scores)[::-1][:top_k]

        results = []
        for idx in sorted_indices:
            if idx < len(self.bm25.df):
                results.append({
                    "title": self.bm25.df.iloc[idx]["Title"].capitalize(),
                    "artist": self.bm25.df.iloc[idx]["Artist"],
                    "album": self.bm25.df.iloc[idx]["Album"],
                    "lyrics": self.bm25.df.iloc[idx]["Lyric"],
                    "hybrid_score": round(float(hybrid_scores[idx]), 3),
                    "bm25_score": round(float(bm25_norm[idx]), 3),
                    "sbert_score": round(float(sbert_norm_scores[idx]), 3)
                })

        return results