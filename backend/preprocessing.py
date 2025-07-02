import re
import json
import os
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from joblib import Memory
from functools import lru_cache
import nltk

cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
memory = Memory(cache_dir, verbose=0)

# Ensure NLTK resources are available
def ensure_nltk_resources():
    nltk_packages = [
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
        ('tokenizers/punkt', 'punkt'),
        ('corpora/stopwords', 'stopwords'),
        ('corpora/wordnet', 'wordnet'),
        ('corpora/omw-1.4', 'omw-1.4')
    ]

    for path, name in nltk_packages:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=False)

# Gọi kiểm tra tài nguyên khi module được import
ensure_nltk_resources()

# Load slang dictionary
current_dir = os.path.dirname(os.path.abspath(__file__))
slang_map_path = os.path.join(current_dir, "slang_map_cleaned.json")
with open(slang_map_path, "r") as f:
    slang_dict = json.load(f)

stop_word_set = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(treebank_tag):
    """Map POS tag to WordNet POS tag."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    

@memory.cache
def clean_text(text):
    """Clean, tokenize, remove stopwords, lemmatize input text."""

    # Clean and tokenize
    text = text.lower()
    text = text.encode('ascii', 'ignore').decode()              # Remove non-ASCII
    text = re.sub(r'\[(.*?)\]', '', text)                       # Remove [tags]
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)                  # Remove punctuation
    text = text.replace('\n\n', '')
    text = text.replace('\n', '. ')
    text = re.sub(r"\s+", " ", text).strip()
    # text = " ".join(text.split())

    for slang, replacement in slang_dict.items():
        # Ensure word boundaries to avoid partial replacements
        pattern = r'\b' + re.escape(slang) + r'\b'
        text = re.sub(pattern, replacement, text)

    return text

@memory.cache
def preprocess_text(text):
    if not isinstance(text, str):
        return []
    
    text = clean_text(text)
    tokens = word_tokenize(text)    

    # Replace slang
    slang_replaced = [slang_dict.get(word, word) for word in tokens]

    no_stopword = [word for word in slang_replaced if word not in stop_word_set]

    # POS tagging and lemmatization
    pos_tagged = nltk.pos_tag(no_stopword)
    lemmatized = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in pos_tagged
    ]

    return lemmatized

@memory.cache
def expand_query(tokens, max_synonyms=2):
    """Expand query using WordNet synonyms for each token."""
    expanded = set(tokens)  # Start with original tokens

    for word in tokens:

        if len(word) < 3 or word in stop_word_set:
            continue

        synsets = wordnet.synsets(word)
        synonyms = set()

        for syn in synsets:
            for lemma in syn.lemmas():
                name = lemma.name().replace('_', ' ').lower()
                if name != word:
                    synonyms.add(name.lower())
                if len(synonyms) >= max_synonyms:
                    break
            if len(synonyms) >= max_synonyms:
                break

        expanded.update(synonyms)

    return list(expanded)

def preprocess_and_expand(text, expand=True):
    tokens = preprocess_text(text)
    if expand:
        return expand_query(tokens)
    return tokens
