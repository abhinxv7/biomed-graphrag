# 🧬 Biomed-GraphRAG: Multimodal Enterprise Clinical AI

**Biomed-GraphRAG** is a private, local-first multimodal AI architecture designed for secure clinical information extraction and reasoning. It fuses unstructured medical data (text, audio, and medical imaging) into a unified **Knowledge Graph**, enabling enterprise-grade retrieval that transcends standard vector-only RAG systems.

## 🚀 The Architecture

The project leverages a multi-agent orchestrated pipeline to handle clinical data:

* **Ingestion Pipeline**: A LangGraph-powered state machine that intelligently routes data:
* **Text**: Raw clinical notes processed via Llama 3 for triple extraction.
* **Audio**: Speech-to-text transcription via local OpenAI Whisper.
* **Vision**: Multimodal medical image analysis via Llava.


* **Knowledge Graph (Neo4j)**: Stores extracted entities (PATIENT, CONDITION, MEDICATION, etc.) and their relationships (HAS_CONDITION, PRESCRIBED).
* **Vector Database (Qdrant)**: Stores semantic embeddings of graph nodes to enable semantic search, ensuring the system can find clinical concepts even when query phrasing varies.
* **Hybrid Retrieval**: A dual-layer retrieval system that bridges semantic intent with structural graph traversal.

## 🛠 Tech Stack

* **Orchestration**: LangGraph, LangChain, Ollama
* **Reasoning Engine**: Llama 3 (via Ollama)
* **Databases**: Neo4j (Graph), Qdrant (Vector)
* **Multimodal**: Llava (Vision), OpenAI Whisper (Audio)
* **Containerization**: Docker Dev Containers (WSL2)

## ⚡ Quick Start

1. **Clone the Repository**:
```bash
git clone https://github.com/yourusername/biomed-graphrag.git
cd biomed-graphrag

```


2. **Spin up the infrastructure**:
Ensure Docker Desktop is running, then start the databases:
```bash
docker run -d -p 7474:7474 -p 7687:7687 --name neo4j neo4j:latest
docker run -d -p 6333:6333 -p 6334:6334 --name qdrant qdrant/qdrant

```


3. **Run Ingestion**:
Process your local data directory:
```bash
python src/ingestion.py

```


4. **Query the Graph**:
Ask complex clinical questions against your local knowledge:
```bash
python src/retriever.py

```



## 🧠 Why GraphRAG?

Standard RAG systems often suffer from "context loss" when searching through complex medical data. By building a **Knowledge Graph**, Biomed-GraphRAG allows for multi-hop reasoning. If a patient’s condition is noted in a text file and their side-effects are spoken in an audio consultation, this system can traverse the graph to connect them, providing a synthesized, accurate clinical response.

```

🔍 Query Received: 'Why was the Chest X-Ray ordered, and did the imaging results confirm those specific suspicions?'
🧠 Agent 1: Extracting clinical intent...
🎯 Intent Identified: Chest X-Ray
🧲 Agent 2: Semantic Vector Search in Qdrant...
✅ Qdrant Matched Intent 'Chest X-Ray' -> Exact Graph Node: 'X-RAY'
🕸️ Agent 3: Traversing Neo4j starting from 'X-RAY'...
✍️ Agent 4: Synthesizing clinical response...

==================================================
🏥 CLINICAL GRAPH-RAG RESPONSE:
==================================================
Based solely on the provided knowledge graph evidence, I can only retrieve information about the existence of a Chest X-Ray and its relationship to the HUMAN.

Unfortunately, the evidence does not provide any information about why the Chest X-Ray was ordered or whether the imaging results confirmed specific suspicions. Therefore, I must conclude that there is insufficient data to answer these questions accurately.

The only statement that can be made with certainty based on the provided evidence is:

"The HUMAN has a Chest X-Ray."
==================================================

```

🔍 Query Received: 'Why did Trump Bomb Iran?'

🏥 CLINICAL GRAPH-RAG RESPONSE:

I'm an expert enterprise clinical AI, and I can only provide answers based on the evidence retrieved from the knowledge graph. Since the question "Why did Trump Bomb Iran?" is unrelated to the patient's medical information, I must conclude that there is insufficient data to answer this question. There is no connection between these medical facts and the topic of Trump bombing Iran. Therefore, I cannot provide an accurate clinical response to this question. If you'd like to ask a question related to the patient's medical information, I'll be happy to help!

