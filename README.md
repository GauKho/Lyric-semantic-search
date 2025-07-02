# 🎵 Lyric-Based Song Search Engine

This is a student research project for the *Searching Engine and Text Mining* course at FPT University, Ho Chi Minh City.

The goal is to develop a system that allows users to **search for songs based on lyrics**, especially in cases where users remember parts of the lyrics but not the song title.

---

## 📚 Objectives

- Explore the **Hugging Face** ecosystem, especially the **Sentence-Transformer** models
- Study the **BM25** information retrieval algorithm
- Build an end-to-end **lyric-to-song retrieval system**

---

## 🛠️ Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/lyric-song-search.git
   cd lyric-song-search
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

#🚀 Usage
1. Fine-tune Sentence-BERT:
  To fine-tune SBERT on your lyrics dataset, run the fine-tuning script:
    ```bash
    python finetune/train_sbert_model.py
2. Run the main application:
  After fine-tuning, you can run the main search engine:
     ```bash
    python main.py

# 🧠 Technologies Used
- Python 3
- Sentence-Transformers
- Hugging Face Transformers
- BM25 (via rank_bm25)
- scikit-learn, pandas, numpy, etc.

# 🧪 Example
- Input: "I'm in love with the shape of you".
- Output: 🎵 Ed Sheeran – Shape of You.
