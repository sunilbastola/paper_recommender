# paper_recommender

Recommends relevant arXiv papers using NLP, semantic embeddings, and similarity calculations on the arXiv metadata snapshot.

## 📁 Project Structure

```text
paper_recommender/
├── .venv/                          # Virtual environment (ignored)
├── .gitignore                      # Git ignore rules
├── arxiv-metadata-oai-snapshot.json # Raw Kaggle arXiv dataset
├── main.ipynb                      # Main Jupyter Notebook pipeline
├── README.md                       # Project documentation
└── requirements.txt                # Python dependencies
```

## 📊 Dataset Acquisition
This project utilizes the official [Kaggle arXiv Dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv). 
1. Download `arxiv-metadata-oai-snapshot.json` from Kaggle.
2. Place the file directly into the root folder of this repository before executing the pipeline.

## ⚙️ Features & Pipeline Architecture

### 1. Basic NLP Core
*   **Text Preprocessing:** Standardisation, tokenisation, and stop-word removal of paper abstracts.
*   **Feature Extraction:** Term frequency-inverse document frequency (TF-IDF) for keyword mining.
*   **Named Entity Recognition (NER):** Author, institution, and domain extraction.
*   **Classification & Graphing:** Structural text classification for research domains and mutual citation analysis.

### 2. Semantic Recommendation & Advanced Techniques
*   **Dense Embeddings:** Vector representation of paper texts for deep similarity calculations.
*   **LLM Synthesis:** Pipeline hooks for abstractive summarization, multi-document synthesis, and automated peer-review assistance.
*   **RAG Architecture:** Retrieval-Augmented Generation layout for conversational paper discovery and automated research gap identification.

## 🚀 Getting Started

### 1. Installation & Setup
Clone this repository, navigate into the directory, and configure your virtual environment:

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Execution
Launch the Jupyter interface and run the development notebook:
```bash
jupyter notebook main.ipynb
```
*Execute the notebook cells sequentially to progress from data parsing to generating paper recommendations.*
