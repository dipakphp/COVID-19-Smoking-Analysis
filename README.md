
# 🧠 COVID-19 & Smoking Research Assistant

A Gradio-powered dashboard for exploring the impact of smoking on COVID-19 outcomes using the CORD-19 dataset. It combines semantic search, an LLM for Q&A, sentiment analysis, and citation tracing.

## 🚀 Features
- Load & filter CORD-19 dataset for smoking-related papers.
- Vector search with MiniLM embeddings.
- Chat with TinyLlama-based Q&A system.
- Sentiment analysis on responses.
- View citations and keyword insights.
- Beautiful interactive Gradio UI.

## 🛠️ Installation
```bash
pip install gradio kagglehub llama_index.embeddings.huggingface llama-index llama-index-llms-huggingface transformers pandas tqdm
```

## 📁 Dataset
CORD-19 metadata from [KaggleHub](https://www.kaggle.com/googleai/dataset-metadata-for-cord19), filtered for smoking-related records.

## 💡 Usage
Run the main Python file in Google Colab or a Python environment. The app will launch in a browser with a public shareable link.

## 🧩 Architecture
- **Embedding**: `MiniLM-L6-v2`
- **LLM**: `TinyLlama-1.1B-Chat`
- **Q&A Framework**: `LlamaIndex`
- **UI**: `Gradio`
- **Sentiment**: `DistilBERT-sst-2`

## 📊 Output Includes:
- Answer from LLM
- Sentiment analysis
- Top keywords
- Source citations
- Response time

## 🖼️ UI Sample
![UI Screenshot](screenshot.png) *(Add a screenshot here)*

## 📄 License
This project is for educational purposes. Please cite the CORD-19 dataset and TinyLlama authors if using in publications.

---
*Developed using TinyLlama, Gradio, and LlamaIndex for exploratory research.*
