import os
import json
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from database import db
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# 1. Initialize Engines
ollama_url = os.getenv(
    "OLLAMA_BASE_URL", "[http://host.docker.internal:11434](http://host.docker.internal:11434)")
llm = ChatOllama(base_url=ollama_url, model="llama3", temperature=0.1)
embedder = SentenceTransformer("all-MiniLM-L6-v2")
qdrant = QdrantClient(url="http://host.docker.internal:6333")

# 2. Prompts
entity_extraction_template = """
Extract the single most critical medical symptom, condition, or procedure from the user's question. 
Return strictly a JSON object: {{"entity": "CONCEPT"}}. No preamble.
Question: {question}
"""
entity_prompt = PromptTemplate(
    template=entity_extraction_template, input_variables=["question"])

synthesis_template = """
You are an expert enterprise clinical AI. Answer the user's question using strictly the evidence retrieved from the knowledge graph. 
Knowledge Graph Evidence (Triples):
{context}

User Question: {question}

Provide a concise, highly accurate clinical response based ONLY on the evidence above. If the evidence does not contain the answer, state that there is insufficient data.
"""
synthesis_prompt = PromptTemplate(
    template=synthesis_template, input_variables=["context", "question"])


def query_graphrag(question: str):
    print(f"\n🔍 Query Received: '{question}'")

    # Step A: Llama 3 extracts the intent
    print("🧠 Agent 1: Extracting clinical intent...")
    entity_chain = entity_prompt | llm
    entity_response = entity_chain.invoke({"question": question})

    try:
        clean_json = entity_response.content.replace(
            "```json", "").replace("```", "").strip()
        search_concept = json.loads(clean_json).get("entity", "")
        print(f"🎯 Intent Identified: {search_concept}")
    except Exception:
        print("❌ Failed to parse intent.")
        return

    # Step B: Qdrant Semantic Search (The Bridge)
    print("🧲 Agent 2: Semantic Vector Search in Qdrant...")
    concept_vector = embedder.encode(search_concept).tolist()

    # Use query_points, which is the modern standard for Qdrant search
    search_results = qdrant.query_points(
        collection_name="clinical_nodes",
        query=concept_vector,
        limit=1
    ).points  # Access the points directly from the response

    if not search_results:
        print("⚠️ Qdrant found no matching vectors.")
        return

    # Get the payload from the first hit
    exact_node_id = search_results[0].payload["node_id"]
    print(
        f"✅ Qdrant Matched Intent '{search_concept}' -> Exact Graph Node: '{exact_node_id}'")

    # Step C: Neo4j 2-Hop Traversal
    print(f"🕸️ Agent 3: Traversing Neo4j starting from '{exact_node_id}'...")
    cypher_query = """
    MATCH (anchor)
    WHERE anchor.id = $exact_node
    
    MATCH path = (anchor)-[*1..2]-(connected_node)
    UNWIND relationships(path) AS rel
    RETURN DISTINCT startNode(rel).id AS head, type(rel) AS relationship, endNode(rel).id AS tail
    """

    retrieved_triples = []
    with db.neo4j_driver.session() as session:
        results = session.run(cypher_query, exact_node=exact_node_id)
        for record in results:
            retrieved_triples.append(
                f"[{record['head']}] -({record['relationship']})-> [{record['tail']}]")

    if not retrieved_triples:
        print("⚠️ No direct relationships found in the Knowledge Graph for this entity.")
        return

    context_string = "\n".join(retrieved_triples)

    # Step D: Final Synthesis
    print("✍️ Agent 4: Synthesizing clinical response...")
    synthesis_chain = synthesis_prompt | llm
    final_answer = synthesis_chain.invoke(
        {"context": context_string, "question": question})

    print("\n==================================================")
    print("🏥 CLINICAL GRAPH-RAG RESPONSE:")
    print("==================================================")
    print(final_answer.content.strip())
    print("==================================================\n")


if __name__ == "__main__":
    test_questions = [
        "What specific medication was prescribed to the patient experiencing chest pain, and what condition is it treating?",
        "Why did Trump Bomb Iran?"
    ]

    for q in test_questions:
        query_graphrag(q)

    db.close()
