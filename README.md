#  News Article Q\&A with Groq [https://news-article-q-a-bzoucpkvtxdtbgt3pmp5mf.streamlit.app/]

This Streamlit app allows you to input news article URLs, generate vector embeddings from their content, and ask natural language questions about the articles using **Groq's LLaMA 3** models.

---

##  Features

* Input up to 3 news article URLs
* Automatically scrapes and processes content
* Chunks and embeds documents using `intfloat/e5-small-v2` via HuggingFace
* Asks questions and gets answers with Groq's `llama3-8b-8192`
* Caches embeddings in a local FAISS vectorstore
* Fully interactive Streamlit frontend

---

## Tech Stack

* [Streamlit](https://streamlit.io/)
* [LangChain](https://www.langchain.com/)
* [FAISS](https://github.com/facebookresearch/faiss)
* [HuggingFace Sentence Transformers](https://www.sbert.net/)
* [Groq LLM (via LangChain)](https://groq.com/)
* [Unstructured](https://github.com/Unstructured-IO/unstructured) for article parsing

---

## 🔧 Setup Instructions

1. **Clone the repo**

   ```bash
   git clone https://github.com/your-username/News-Article-Q-A.git
   cd News-Article-Q-A
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Add your Groq API key**
   Create a `.env` file:

   ```env
   GROQ_API_KEY=your_groq_api_key
   ```

4. **Run the app**

   ```bash
   streamlit run NewsArticleSearch.py
   ```

---


## 💡 Example Use Case

* Enter news articles like:

  * `https://motoroctane.com/news/289966-indias-best-selling-suv-to-offer-alto-like-mileage`
* Ask: *"What is the expected mileage of the new SUV?"*
* Get an AI-powered answer based on actual article content.

---

