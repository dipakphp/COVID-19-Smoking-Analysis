"""
This code is suitable for the local system or PC
# Core dependencies
pip install streamlit==1.33.0 transformers==4.39.3 torch==2.2.1 sentence-transformers==2.5.1 pandas==2.2.1 kagglehub==0.1.5 tqdm==4.66.2 

# LlamaIndex (core + huggingface)
pip install llama-index==0.10.20 llama-index-core==0.10.20 llama-index-llms-huggingface==0.1.4 llama-index-embeddings-huggingface==0.1.4

# Optional utility for key term extraction (optional but useful)
pip install keybert==0.8.3 scikit-learn==1.4.1.post1

# Environment compatibility (optional)
numpy==1.26.4
"""


# CORD_APP.py

# ---------------------- import the libraries required for the Project ----------------------
import streamlit as st
import time
from transformers import pipeline
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, Document
import torch
import pandas as pd
import os
import kagglehub
from tqdm import tqdm

# ---------------------- Load Embeddings ----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device=device
)
Settings.embed_model = embed_model
Settings.llm = None

# ---------------------- Load Dataset ----------------------
def load_and_filter_data():
    path = kagglehub.dataset_download(handle="googleai/dataset-metadata-for-cord19")
    filename = path + "/" + os.listdir(path)[0]
    df = pd.read_csv(filename)
    keywords = ['smoking', 'tobacco', 'cigarette', 'nicotine', 'vaping']
    df_filtered = df[df['description'].notnull()]
    keyword_mask = df_filtered['description'].str.contains('|'.join(keywords), case=False, na=False)
    df_filtered = df_filtered[keyword_mask][['description']]
    df_filtered["word_count"] = df_filtered["description"].apply(lambda x: len(str(x).split(" ")))
    return df_filtered

def create_vector_store(dataframe):
    chunks = []
    chunk_size = 150
    for text in tqdm(dataframe["description"].values):
        if isinstance(text, str):
            words = text.split()
            for i in range(0, len(words), chunk_size):
                chunks.append(Document(text=" ".join(words[i:i+chunk_size])))
    storage_context = StorageContext.from_defaults()
    index = VectorStoreIndex.from_documents(
        chunks,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )
    index.storage_context.persist(persist_dir="covid_storage")
    return index

# ---------------------- Load LLM ----------------------
def load_models():
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    answer_llm = HuggingFaceLLM(
        tokenizer=tokenizer,
        model=model,
        context_window=2048,
        max_new_tokens=256,
        generate_kwargs={"temperature": 0.7, "do_sample": True},
    )
    return answer_llm

# ---------------------- Initialize Components ----------------------
df_processed = load_and_filter_data()
vector_index = create_vector_store(df_processed)
llm = load_models()
Settings.llm = llm

chat_agent = vector_index.as_chat_engine(
    chat_mode="context",
    memory=ChatMemoryBuffer.from_defaults(token_limit=1500),
    system_prompt=(
        "You are a medical research assistant specializing in COVID-19 and smoking-related health impacts."
        " Provide evidence-based answers using the CORD-19 dataset."
    )
)

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=0 if torch.cuda.is_available() else -1
)

# ---------------------- Dashboard Code ----------------------
st.set_page_config(page_title="COVID-19 Smoking Analysis", layout="wide")
st.title("🧬 COVID-19 Smoking Analysis")
st.caption("Analyzing the relationship between smoking and COVID-19 outcomes using CORD-19 dataset")

col1, col2 = st.columns([2, 1])

with col1:
    st.header("Research Assistant Chat")
    st.info("Ask questions about smoking and COVID-19")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    user_input = st.text_input("Your Question", key="user_input")

    if user_input:
        start = time.time()
        response = chat_agent.chat(user_input)
        answer = response.response
        elapsed = round(time.time() - start, 1)

        sentiment = sentiment_pipeline(answer[:512])[0]
        keywords = list(set([
            word.lower() for word in answer.split()
            if word.lower() in ['smoking', 'tobacco', 'risk', 'covid', 'lung', 'health']
        ]))

        sources = getattr(response, "source_nodes", [])
        citations = "\n".join([
            f"- Score: {getattr(s, 'score', 'N/A'):.2f}, Source: {s.node.text[:150]}..."
            for s in sources
        ]) if sources else "No source information available."

        st.session_state.messages.append({
            "user": user_input,
            "answer": answer,
            "sentiment": sentiment['label'],
            "confidence": sentiment['score'],
            "keywords": keywords,
            "word_count": len(answer.split()),
            "response_time": elapsed,
            "citations": citations
        })

    for msg in st.session_state.messages[::-1]:
        st.write("**You:**", msg["user"])
        st.success("**Research Assistant:** " + msg["answer"])
        with st.expander("🔍 View Citations"):
            st.markdown(msg["citations"])

with col2:
    st.header("📊 Response Analysis")

    if st.session_state.messages:
        last = st.session_state.messages[-1]

        st.subheader("📌 Sentiment Analysis")
        st.metric(label="Sentiment", value=last["sentiment"], delta=f"Confidence: {last['confidence']:.2f}")

        st.subheader("🧷 Key Terms")
        for kw in last["keywords"]:
            st.markdown(f"- {kw}")

        st.subheader("📈 Response Metrics")
        st.markdown(f"- **Word Count:** {last['word_count']}")
        st.markdown(f"- **Response Time:** {last['response_time']}s")

        st.subheader("📚 Dataset Information")
        st.markdown(f"- **Source:** CORD-19")
        st.markdown(f"- **Model:** TinyLlama-1.1B")
        st.markdown(f"- **Embeddings:** MiniLM-L6-v2")


# To run 
#streamlit run CORD_APP.py
