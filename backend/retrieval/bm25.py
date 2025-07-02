import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from backend.preprocessing import preprocess_and_expand
from backend.utils import load_and_filter_lyrics_csv
from rank_bm25 import BM25Okapi
from typing import List, Dict
import pandas as pd
import numpy as np

class BM25LyricsSearch: 
    def __init__(self, data_path: str):
        self.df = load_and_filter_lyrics_csv(data_path)
        self.df['processed_lyrics'] = self.df["Lyric"].apply(lambda x: preprocess_and_expand(x))
        self.tokenized_corpus = self.df['processed_lyrics'].tolist()
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def get_score(self, query: str) -> np.ndarray:
        print(f"Type of query: {type(query)}") 
        tokenized_query = preprocess_and_expand(query)
        return np.array(self.bm25.get_scores(tokenized_query))

    def search(self, query:str, top_k: int=5) -> List[Dict]:
        query_token = preprocess_and_expand(query) #Preprocess the user's query
        scores = self.bm25.get_scores(query_token) #get the relevance scores between query and all documents
        top_indices = sorted(range(len(scores)),key=lambda i:scores[i], reverse=True)[:top_k] #get the indices of the top_k documents with the high scores 

        results = []
        for idx in top_indices:
            result = {
                "title": self.df.iloc[idx]['Title'],
                "album": self.df.iloc[idx]['Album'],
                "artist": self.df.iloc[idx]["Artist"],
                "lyrics":  self.df.iloc[idx]['Lyric'][:300] + "...",
                "Score": round(scores[idx], 2)
            }
            results.append(result)

        return results