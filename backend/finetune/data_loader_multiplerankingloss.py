import os, re, pickle, logging, glob, unicodedata
import pandas as pd
from sentence_transformers import InputExample
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
load_dotenv()
from collections import defaultdict
from tqdm import tqdm

# Configuration
CONFIG = {
    "base_model": "all-MiniLM-L12-v2",
    "epochs": 3,
    "learning_rate": 1e-4,
    "warmup_ratio": 0.2,
    "weight_decay": 0.01,

    "negative_ratio": 1.0,
    "positive_ratio": 1.0,
    "max_segments_per_song": 10,
    "segment_length": 512,
    "overlap": 30,

    "train_ratio": 0.7, 
    "val_ratio": 0.15,
    "test_ratio": 0.15,

    "min_lyric_words": 50,
    "max_lyric_length": 512,

    "batch_size_cuda": 32,
    "batch_size_cpu": 16,
    "evaluation_steps": 1500,
    "gradient_accumulation_steps": 4
}

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _normalize_text(text: str) -> str:
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
    
    return text.strip()

def load_and_prepare_data(file_path, sample_size: int = None, min_lyric_words: int = CONFIG["min_lyric_words"]):
    all_lyrics = []
    csv_files = sorted(glob.glob(os.path.join(file_path, "*.csv")))
    
    if not csv_files:
        logger.error(f"No CSV files found in {file_path}")
        return pd.DataFrame()
    logger.info(f"Found {len(csv_files)} csv files")

    for file in tqdm(csv_files, desc="Loading CSV files"):
        try:
            logger.info(f"Processing file: {file}")
            df = pd.read_csv(file, encoding="utf-8")
            
            # Check if required columns exist
            required_cols = ["Artist", "Title", "Album", "Lyric"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"Missing columns in {file}: {missing_cols}")
                continue
            
            # Clean and prepare data
            df = df[required_cols].copy()
            df = df.dropna(subset=["Artist", "Title", "Lyric", "Album"])
            
            # Normalize text fields
            df["Album"] = df["Album"].apply(_normalize_text)
            df["Title"] = df["Title"].apply(_normalize_text)
            df["Artist"] = df["Artist"].apply(_normalize_text)
            df["Lyric"] = df["Lyric"].apply(_normalize_text)

            # Fixed regex pattern - properly closed parentheses
            exclude_keywords = r"(?:remix|acoustic|live|demo|edit|version|instrumental|remaster|deluxe|extended|radio)"
            mask_variant_album = df["Album"].str.contains(exclude_keywords, case=False, na=False, regex=True)
            mask_variant_title = df["Title"].str.contains(exclude_keywords, case=False, na=False, regex=True)

            # Filter out unwanted records
            mask_unreleased = df["Album"].str.contains("unreleased", case=False, na=False)
            placeholder_pattern = "lyrics for this song have yet to be released please check back once the song has been released"
            mask_placeholder = df["Lyric"].str.contains(placeholder_pattern, case=False, na=False)
            
            df = df[~(mask_unreleased | mask_placeholder | mask_variant_album | mask_variant_title)]
            
            # Filter by lyric length
            df = df[df["Lyric"].str.split().str.len() > min_lyric_words]
            
            # Remove empty artists/titles after normalization
            df = df[(df["Artist"] != "") & (df["Title"] != "")]
            
            if len(df) > 0:
                all_lyrics.append(df)
                logger.info(f"Loaded {len(df)} records from {file}")
            else:
                logger.warning(f"No valid records found in {file}")
                
        except Exception as e:
            logger.error(f"Error reading {file}: {e}")
            continue

    if not all_lyrics:
        logger.warning("No valid data found in any files.")
        return pd.DataFrame()

    lyrics_df = pd.concat(all_lyrics, ignore_index=True)
    
    # Remove duplicates more carefully
    initial_count = len(lyrics_df)
    lyrics_df = lyrics_df.drop_duplicates(subset=["Title", "Artist", "Lyric"], keep='first')
    logger.info(f"Removed {initial_count - len(lyrics_df)} duplicates")
    
    # Log dataset statistics
    logger.info(f"Final dataset: {len(lyrics_df)} unique lyrics")
    logger.info(f"Unique artists: {lyrics_df['Artist'].nunique()}")
    logger.info(f"Average lyric length: {lyrics_df['Lyric'].str.len().mean():.1f} characters")
    logger.info(f"Average words per lyric: {lyrics_df['Lyric'].str.split().str.len().mean():.1f}")
    
    return lyrics_df

def create_data_splits(df: pd.DataFrame, train_ratio=CONFIG["train_ratio"], 
                      val_ratio=CONFIG["val_ratio"], test_ratio=CONFIG["test_ratio"], 
                      random_state=42):
    """Create proper train/validation/test splits"""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    df = df.sort_values(['Artist', 'Title']).reset_index(drop=True)

    # Count songs per artist
    artist_counts = df['Artist'].value_counts()
    
    # Create stratification groups
    stratify_groups = []
    
    for artist in df['Artist']:
        if artist_counts[artist] >= 2:
            # Multi-song artists: use artist name
            stratify_groups.append(artist)
        else:
            # Single-song artists: group them together
            stratify_groups.append('single_song_group')
    
    # Check if we can stratify
    group_counts = pd.Series(stratify_groups).value_counts()
    if group_counts.min() < 2:
        # Not enough samples per group - create bigger groups
        stratify_groups = []
        single_count = 0
        
        for artist in df['Artist']:
            if artist_counts[artist] >= 2:
                stratify_groups.append(artist)
            else:
                # Group single-song artists in pairs
                group_num = single_count // 2
                stratify_groups.append(f'single_group_{group_num}')
                single_count += 1
    
    # Split the data
    train_df, temp_df = train_test_split(
        df, 
        test_size=(val_ratio + test_ratio), 
        random_state=random_state,
        stratify=stratify_groups
    )
    
    # Split temp into val and test
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.5,
        random_state=random_state
    )
    
    logger.info(f"Data splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    logger.info(f"Train artists: {train_df['Artist'].nunique()}")
    logger.info(f"Val artists: {val_df['Artist'].nunique()}")
    logger.info(f"Test artists: {test_df['Artist'].nunique()}")
    
    return train_df, val_df, test_df

def extract_lyric_segments(lyric, artist, title, album, segment_length=CONFIG["max_lyric_length"], overlap=CONFIG["overlap"]):

    if not lyric or len(lyric.strip()) == 0:
        return []
    
    lines = [line.strip() for line in lyric.split('\n') if line.strip()]
    
    # Loại bỏ dòng metadata hoặc dòng quá ngắn
    clean_lines = [
        line for line in lines
        if (not line.startswith(('(', '[', '{')) and
            not re.match(r'^(verse|chorus|bridge)', line.strip().lower()) and
            len(line.split()) > 2)
    ]
    
    if not clean_lines:
        return [f"(Artist: {artist}. Title: {title}. Album: {album}) {lyric.strip()}"]
    
    # Rejoin cleaned lines
    clean_lyric = ' '.join(clean_lines)
    words = clean_lyric.split()
    
    if len(words) <= segment_length:
        return [f"(Artist: {artist}. Title: {title}. Album: {album}) {clean_lyric}"]
    
    segments = []
    start = 0
    
    while start < len(words):
        end = min(start + segment_length, len(words))
        segment_words = words[start:end]
        segment = ' '.join(segment_words)
        
        # Ensure segment has reasonable length
        if len(segment.split()) >= 15:  # Minimum segment size
            segments.append(segment)
        
        # Move start position with overlap
        start += segment_length - overlap
        
        # Break if we would create a very small final segment
        if end >= len(words):
            break
    
    return segments

def create_training_pairs(df: pd.DataFrame, negative_ratio=CONFIG["negative_ratio"], 
                         max_segments_per_song=CONFIG["max_segments_per_song"], 
                         min_lyric_words=CONFIG["min_lyric_words"]):
    """Create training pairs with improved negative sampling"""
    train_examples = []
    
    df = df.dropna(subset=["Lyric", "Title", "Artist"]).copy()
    
    if len(df) == 0:
        logger.warning("No valid data for creating training pairs")
        return []

    # Create artist and title mappings for better negative sampling
    artist_songs = defaultdict(list)
    title_variations = {}
    
    for idx, row in df.iterrows():
        artist = _normalize_text(row["Artist"]).lower()
        title = _normalize_text(row["Title"]).lower()
        artist_songs[artist].append(idx)
        title_variations[title] = idx

    logger.info(f"Creating training pairs from {len(df)} records")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating training pairs"):
        try:
            lyric = _normalize_text(row["Lyric"])
            title = _normalize_text(row["Title"]).lower()
            artist = _normalize_text(row["Artist"]).lower()
            album = _normalize_text(row["Album"]).lower() if pd.notna(row["Album"]) else ""
            
            if not lyric or not title or not artist or len(lyric.split()) < min_lyric_words:
                continue

            # Extract meaningful segments from lyrics
            segments = extract_lyric_segments(lyric, artist, title, album, segment_length=120, overlap=30)
            if not segments:
                continue
            
            # Limit segments per song
            segments = segments[:max_segments_per_song]

            # Create diverse positive metadata variants
            positive_queries = [
                f"the song '{title}' by {artist} from the album '{album}"
            ]

            # Remove duplicates and empty queries
            positive_queries = list(set([q.strip() for q in positive_queries if q.strip()]))

            # Create positive pairs with segments
            for segment in segments[:3]:
                for query in positive_queries[:2]:
                    train_examples.append(InputExample(texts=[segment, query], label=1))
                    train_examples.append(InputExample(texts=[query, segment], label=1))
                        
        except Exception as e:
            logger.error(f"Error processing row {idx}: {e}")
            continue

    rows = []
    for example in train_examples:
        rows.append({
            "text_1": example.texts[0],
            "text_2": example.texts[1],
            "label": example.label
        })

    all_train_example = pd.DataFrame(rows)
    all_train_example.to_csv("train_examples.csv", index=False)
    print(f"Save {len(all_train_example)} examples to train_examples.csv")
    
    logger.info(f"Created {len(train_examples)} training pairs")
    return train_examples

def save_examples(train_examples, out_file="training_data.pkl"):
    """Save training examples with error handling."""
    try:
        with open(out_file, "wb") as f:
            pickle.dump(train_examples, f)
        logger.info(f"Saved {len(train_examples)} training pairs to {out_file}")
    except Exception as e:
        logger.error(f"Error saving training examples: {e}")
        raise

def load_examples(pkl_file="training_data.pkl"):
    """Load training examples with error handling."""
    try:
        with open(pkl_file, "rb") as f:
            examples = pickle.load(f)
        logger.info(f"Loaded {len(examples)} training pairs from {pkl_file}")
        return examples
    except Exception as e:
        logger.error(f"Error loading training examples: {e}")
        raise

def save_splits(train_df, val_df, test_df, base_path="./splits"):
    """Save train/val/test splits to CSV files"""
    base_path = os.path.normpath(base_path)
    os.makedirs(base_path, exist_ok=True)
    
    train_df.to_csv(os.path.join(base_path, "train_df.csv"), index=False)
    val_df.to_csv(os.path.join(base_path, "val_df.csv"), index=False)
    test_df.to_csv(os.path.join(base_path, "test_df.csv"), index=False)
    
    logger.info(f"Saved splits to {base_path}/")
    return True

def load_splits(base_path="./splits"):
    """Load train/val/test splits from CSV files"""
    try:
        # Ensure consistent path separators
        base_path = os.path.normpath(base_path)

        train_df = pd.read_csv(os.path.join(base_path, "train_df.csv"))
        val_df = pd.read_csv(os.path.join(base_path, "val_df.csv"))
        test_df = pd.read_csv(os.path.join(base_path, "test_df.csv"))
        
        logger.info(f"Loaded splits from {base_path}/")
        logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    except Exception as e:
        logger.error(f"Error loading splits: {e}")
        return None, None, None