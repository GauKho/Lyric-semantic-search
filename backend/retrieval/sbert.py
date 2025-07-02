import faiss
import numpy as np
import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import torch
from sentence_transformers import SentenceTransformer
from backend.utils import load_and_filter_lyrics_csv


class SBERTSearcher:
    def __init__(self, data_path: str, model_name: str = 'lyrics_sbert_model', force_rebuild: bool =False):
        #Load fine-tuned SBert model 
        self.model = SentenceTransformer(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.df = load_and_filter_lyrics_csv(data_path)
        self.lyrics = self.df["Lyric"].tolist()

        self.embedding_file = "lyrics_embeddings.npy"
        self.index_file = "lyrics_faiss.index"

        self._load_or_create_index(force_rebuild=force_rebuild)


    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        faiss.normalize_L2(vectors)
        return vectors
    
    def _build_index(self):
        print("Encoding lyrics...")
        self.lyrics_embedding = self.model.encode(self.lyrics, batch_size=32, show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
        np.save(self.embedding_file, self.lyrics_embedding)

        self.lyrics_embedding = self._normalize(self.lyrics_embedding)
        self.index = faiss.IndexFlatIP(self.lyrics_embedding.shape[1])
        self.index.add(self.lyrics_embedding)
        faiss.write_index(self.index, self.index_file)

    def _load_or_create_index(self, force_rebuild: bool):
        if (
            force_rebuild
            or not os.path.exists(self.embedding_file)
            or not os.path.exists(self.index_file)
        ):
            self._build_index()
            return

        self.lyrics_embedding = np.load(self.embedding_file).astype(np.float32)
        if len(self.lyrics_embedding) != len(self.lyrics):
            print("[WARNING] Mismatch in embedding count and lyrics count. Rebuilding index...")
            self._build_index()
            return

        self.index = faiss.read_index(self.index_file)
    
    def get_score(self, query: str, top_k: int = 5):
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        query_embedding = self._normalize(query_embedding.reshape(1, -1)).astype(np.float32)
        scores, indices = self.index.search(query_embedding, top_k)


        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.df):
                results.append({
                    "title": self.df.iloc[idx]["Title"],
                    "artist": self.df.iloc[idx]["Artist"],
                    "album": self.df.iloc[idx]['Album'],
                    "lyrics": self.df.iloc[idx]["Lyric"],
                    "score": round(float(score), 3)
                })

        return results, scores[0], indices[0]
    
    def search(self, query: str, top_k: int = 5):
        result, _, _ = self.get_score(query, top_k)
        return result
