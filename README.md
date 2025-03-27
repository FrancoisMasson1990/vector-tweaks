# vector-tweaks
Fine_tuning of embedding model linked to e-commerce


## 🔍 Project 1 : EmbeddingAlign RAG: Boosting QA Systems

**EmbeddingAlign RAG** is a lightweight enhancement to Retrieval-Augmented Generation (RAG) pipelines. It improves document retrieval performance by applying a trained **linear transformation** to both query and document embeddings—without retraining the embedding model.

---

## 📚 Background

RAG systems retrieve relevant documents using precomputed embeddings. However, as new documents are added, misalignment between query and document embeddings can reduce retrieval quality. Retraining the full embedding model is expensive and often infeasible.

---

## 🧠 Methodology

### ✅ Core Idea

Learn a **linear transformation matrix** `T`:

\[
S'(q, d) = \text{cosine\_similarity}(T \cdot E(q),\ T \cdot E(d))
\]

Where:
- `E(q)` and `E(d)` are query and document embeddings
- `T` is a learnable matrix
- Embeddings are obtained from a black-box model (e.g., OpenAI's `text-embedding-3-small`)

---

### 🧪 Synthetic Dataset

To simulate a production RAG pipeline:

- Source: Public 10-K reports from Uber and Lyft
- Chunking: Performed with **LlamaIndex**
- Query generation: Using **GPT-4o**
- Each (query, chunk) pair is augmented with distractor chunks to form triplets

---

### 🏋️ Training

- Objective: Minimize **Triplet Loss**

\[
\text{Loss}(q, c, dis) = \max(d(q, c) - d(q, dis) + \text{margin},\ 0)
\]

Where:
- `q` = query embedding  
- `c` = correct chunk  
- `dis` = distractor chunk  
- `d()` = cosine distance  
- `margin` = hyperparameter

- Training is **CPU-only**
- Only the linear transformation `T` is updated

---

## 📊 Results

| Metric       | Baseline | EmbeddingAlign |
|--------------|----------|----------------|
| Hit Rate@4   | 0.89     | **0.95**        |
| MRR          | 0.69     | **0.83**        |

- ⏱️ **+8.6% inference time** (~10ms)
- ⚡ Significant boost in retrieval accuracy

---

## 🚀 Deployment Tips

- Precompute `T ⋅ E(d)` once and store in your vector database
- Apply `T` to incoming query embeddings before retrieval
- No change needed in embedding model or vector store API

---

## ✅ Benefits

- No embedding model retraining required
- Works with any RAG pipeline (e.g., OpenAI, Cohere, etc.)
- Fast, cheap, and easy to integrate

---

## 📂 References

Original paper: [EmbeddingAlign RAG: Boosting QA Systems (Phospho, 2024)](https://research.phospho.ai/phospho_embeddingalign_rag.pdf)

---

## 💡 License

MIT
