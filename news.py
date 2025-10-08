import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from dotenv import load_dotenv
import google.generativeai as genai
from cerebras.cloud.sdk import Cerebras


hide_github_icon = """
    <style>
        [data-testid="stToolbar"] {visibility: hidden;}
    </style>
"""
st.markdown(hide_github_icon, unsafe_allow_html=True)


# --- Load API Key ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")

# --- Inisialisasi client Gemini 2.0 ---
genai.configure(api_key=GOOGLE_API_KEY)
llm_gemini = genai  # versi lama SDK 0.8.5

# --- Inisialisasi client Qwen (Cerebras) ---
cere_client = Cerebras(api_key=CEREBRAS_API_KEY)

def narration(prompt):
    completion = cere_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="qwen-3-235b-a22b-instruct-2507",
        temperature=0  # output lebih deterministik
    )
    return completion.choices[0].message.content.strip()


def batch_sentiment(texts, selected_model="gemini"):
    """
    Batch sentiment analysis menggunakan Gemini 2.0 (SDK lama) atau Qwen
    Prompt sudah disesuaikan perspektif Kemenlu Indonesia
    """
    results = []

    for text in texts:
        prompt = f"""
        Analyze the sentiment of the following news text regarding Indonesia, from the perspective of Indonesia's Ministry of Foreign Affairs.
        Reply ONLY with Positive, Neutral, or Negative.
        - Positive: News is favorable or supportive of Indonesia's policies or international stance.
        - Neutral: News is factual or balanced without clear support or criticism.
        - Negative: News is critical, unfavorable, or challenges Indonesia's policies or international stance.
        
        Text: {text}
        """

        try:
            if "gemini" in selected_model.lower():
                response = llm_gemini.generate_content(prompt).text.strip()
            else:
                response = narration(prompt)

            sentiment = "Neutral"
            for option in ["Positive", "Neutral", "Negative"]:
                if option.lower() in response.lower():
                    sentiment = option
                    break

            results.append(sentiment)

        except Exception as e:
            st.error(f"Error {selected_model} API: {e}")
            results.append("Neutral")

    return results


# --- Fungsi analisis sentiment ---
@st.cache_data
def analyze_sentiment(df, method="roberta"):
    from transformers import pipeline
    results = []

    if method == "roberta":
        sentiment_model = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
        for i, row in df.iterrows():
            title = row["Title"]
            content = row["Content"]
            title_sent = sentiment_model(title)[0]["label"]
            content_sent = sentiment_model(content)[0]["label"]
            results.append({
                "Title": title,
                "Media": row.get("Media", ""),
                "Title Tone Sentiment": title_sent,
                "Content Tone Sentiment": content_sent
            })
    else:
        titles = df["Title"].tolist()
        contents = df["Content"].tolist()
        title_sents = batch_sentiment(titles, selected_model=method)
        content_sents = batch_sentiment(contents, selected_model=method)
        for i, row in df.iterrows():
            results.append({
                "Title": row["Title"],
                "Media": row.get("Media", ""),
                "Title Tone Sentiment": title_sents[i],
                "Content Tone Sentiment": content_sents[i]
            })

    return pd.DataFrame(results)


# --- Fungsi pewarnaan tabel ---
def color_sentiment(val):
    if val.lower() == "positive":
        return "color: #0000FF; font-weight: bold;"  # biru solid, tebal
    elif val.lower() == "negative":
        return "color: #FF0000; font-weight: bold;"  # merah solid, tebal
    else:
        return "color: #333333; font-weight: normal;"  # abu gelap / hitam



# --- Streamlit UI ---
st.set_page_config(page_title="News Sentiment Analysis", layout="wide")
st.title("📰 News Sentiment Analysis (Gemini 2.0 atau Qwen)")

uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])
#method = st.selectbox("Pilih Metode Sentiment Analysis", ["RoBERTa (BERT)", "LLM Gemini 2.0", "LLM Qwen (Cerebras)"])
method = st.selectbox("Pilih Metode Sentiment Analysis", ["LLM Gemini 2.0 Flash", "LLM Qwen (Cerebras)"])

if uploaded_file is not None:
    df_news = pd.read_excel(uploaded_file)

    if st.button("Analisis Sentiment"):
        with st.spinner("Sedang memproses..."):
            method_key = "roberta" if method.startswith("RoBERTa") else ("gemini" if "Gemini" in method else "qwen")
            df_result = analyze_sentiment(df_news, method=method_key)
            time.sleep(1)

        st.success("Selesai! Berikut hasil analisis:")
        # --- Reset index agar mulai dari 1 ---
        df_result_display = df_result.reset_index(drop=True)
        df_result_display.index += 1  # index mulai dari 1

              
        # --- Tampilkan tabel dengan index sebagai nomor ---
        #st.dataframe(
            #df_result.style.applymap(color_sentiment, subset=["Title Tone Sentiment", "Content Tone Sentiment"])
            #st.dataframe(df_result_display.style.applymap(color_sentiment, subset=["Title Tone Sentiment", "Content Tone Sentiment"]))
        #)
        st.dataframe(
            df_result_display.style.applymap(
                color_sentiment,  # <--- jangan pakai ()
                subset=["Title Tone Sentiment", "Content Tone Sentiment"]
            )
        )


        # --- Visualisasi Bar Chart pastel, compact, rata tengah ---
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)  # compact figure
            colors = ['#5DADE2', '#95A5A6', '#EC7063']  # Positive, Neutral, Negative
            
            # --- Title Tone ---
            counts_title = df_result["Title Tone Sentiment"].value_counts().reindex(["Positive","Neutral","Negative"])
            counts_title.plot(kind="bar", ax=ax[0], color=colors, rot=0)
            ax[0].set_title("Distribusi Title Tone Sentiment")
            ax[0].set_xlabel("")  # hapus label x
            ax[0].set_ylabel("Jumlah")
            ax[0].set_xticklabels(["Positive","Neutral","Negative"])
            
            # Tambahkan label angka di atas tiap bar
            for i, v in enumerate(counts_title):
                ax[0].text(i, v + 0.1, str(v), ha='center', va='bottom', fontweight='bold')
            
            # --- Content Tone ---
            counts_content = df_result["Content Tone Sentiment"].value_counts().reindex(["Positive","Neutral","Negative"])
            counts_content.plot(kind="bar", ax=ax[1], color=colors, rot=0)
            ax[1].set_title("Distribusi Content Tone Sentiment")
            ax[1].set_xlabel("")  # hapus label x
            ax[1].set_ylabel("Jumlah")
            ax[1].set_xticklabels(["Positive","Neutral","Negative"])
            
            # Tambahkan label angka di atas tiap bar
            for i, v in enumerate(counts_content):
                ax[1].text(i, v + 0.1, str(v), ha='center', va='bottom', fontweight='bold')
            
            # Tambahkan jarak antar subplot
            fig.subplots_adjust(left=0.1, right=0.9, wspace=0.6)
            
            fig.tight_layout()
            st.pyplot(fig)

