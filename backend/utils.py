import os
import glob, re
import pandas as pd
import numpy as np
from sentence_transformers import InputExample
import unicodedata

def _normalize_text(text):
    if pd.isna(text) or text is None:
        return ""
    
    text = str(text).strip()
    
    # Remove verse/chorus markers but keep the content
    text = re.sub(r'\[(Verse \d*|Chorus|Bridge|Pre-Chorus|Intro|Outro|BREAK).*?\]', '', text)
    
    # Clean up repetitive markers and annotations
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'[""''`]', '"', text)
    
    # Preserve important punctuation for meaning
    text = re.sub(r'\s+', ' ', text)
    text = unicodedata.normalize('NFKC', text)
    
    return text.strip() # text normalization


def load_and_filter_lyrics_csv(data_path: str) -> pd.DataFrame:
    csv_files = glob.glob(os.path.join(data_path, "*.csv"))
    min_word = 50

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in folder: {data_path}")

    all_data = []

    for file in csv_files:
        try:
            df = pd.read_csv(file, encoding='utf-8')

            required_cols = ["Artist", "Title", "Album", "Lyric"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"Skipping missing columns: {missing_cols}")
                continue

            df = df[required_cols].copy()
            df = df.dropna(subset=["Artist", "Title", "Lyric", "Album"])

            # Normalize text fields
            df["Album"] = df["Album"].apply(_normalize_text)
            df["Title"] = df["Title"].apply(_normalize_text)
            df["Artist"] = df["Artist"].apply(_normalize_text)
            df["Lyric"] = df["Lyric"].fillna("").str.strip()

            exclude_keywords = r"(?:remix|acoustic|live|demo|edit|version|instrumental|remaster|deluxe|extended|radio)"
            mask_variant_album = df["Album"].str.contains(exclude_keywords, case=False, na=False, regex=True)
            mask_variant_title = df["Title"].str.contains(exclude_keywords, case=False, na=False, regex=True)

             # Lọc các record chứa từ "unreleased" trong Album
            mask_unreleased = df["Album"].str.contains("unreleased", case=False, na=False)
            # Lọc placeholder lyric
            placeholder_pattern = "lyrics for this song have yet to be released please check back once the song has been released"
            mask_placeholder = df["Lyric"].str.contains(placeholder_pattern, case=False, na=False)

            df = df[~(mask_unreleased | mask_placeholder | mask_variant_album | mask_variant_title)]

            df = df[df["Lyric"].str.split().str.len() > min_word]

            # Remove empty artists/titles after normalization
            df = df[(df["Artist"] != "") & (df["Title"] != "")]

            if len(df) > 0:
                all_data.append(df)
        
        except Exception as e:
            print(f"Error reading {file}: {e}")

    if not all_data:
        raise ValueError("No valid data found after filtering.")

    all_lyrics = pd.concat(all_data, ignore_index=True)
    all_lyrics = all_lyrics.drop_duplicates(subset=["Artist", "Title", "Album", "Lyric"])
    return all_lyrics.reset_index(drop=True)


