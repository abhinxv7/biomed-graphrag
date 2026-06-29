import os
import json
import re
import uuid
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from database import db

# 1. Initialize Local AI Engines
ollama_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
llm = ChatOllama(base_url=ollama_url, model="llama3", temperature=0)

print("🧠 Loading local embedding model (all-MiniLM-L6-v2)...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Initialize Qdrant Vector DB
qdrant = QdrantClient(url="http://host.docker.internal:6333")
COLLECTION_NAME = "clinical_nodes"

try:
    qdrant.get_collection(COLLECTION_NAME)
except:
    print(f"Creating new Qdrant collection: {COLLECTION_NAME}")
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

prompt_template = """
You are a strict clinical data extraction algorithm. 
Your only job is to extract Biomedical Knowledge Triples from the text.
A triple consists of a Head Entity, a Relationship, and a Tail Entity.

Entity categories allowed: PATIENT, CONDITION, MEDICATION, OBSERVATION, ANATOMY, PROCEDURE.
Relationships should be all caps and use underscores.

Output strictly in this JSON array format, and nothing else.
[
    {{"head": "Entity Name", "head_type": "CATEGORY", "relation": "RELATIONSHIP", "tail": "Entity Name", "tail_type": "CATEGORY"}}
]

Clinical Text:
{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])


def extract_triples(text: str):
    print("🧠 Llama 3 is analyzing clinical text for relationships...")
    chain = prompt | llm
    response = chain.invoke({"text": text})

    try:
        clean_text = re.sub(r'```json\n|```\n|```', '',
                            response.content).strip()
        return json.loads(clean_text)
    except json.JSONDecodeError:
        print("❌ Llama 3 failed to output valid JSON.")
        return []


def inject_into_neo4j(triples):
    if not triples:
        return

    print("🕸️ Injecting graph topology into Neo4j...")
    cypher_query = """
    UNWIND $triples AS triple
    CALL apoc.merge.node([triple.head_type], {id: toUpper(triple.head)}) YIELD node AS head
    CALL apoc.merge.node([triple.tail_type], {id: toUpper(triple.tail)}) YIELD node AS tail
    CALL apoc.merge.relationship(head, toUpper(triple.relation), {}, {}, tail) YIELD rel
    RETURN count(rel)
    """

    try:
        with db.neo4j_driver.session() as session:
            session.run(cypher_query, triples=triples)
        print("✅ Knowledge Graph Successfully Updated.")
    except Exception as e:
        print(f"❌ Neo4j Injection Error: {e}")

    # --- NEW: QDRANT VECTOR INJECTION ---
    print("🎯 Vectorizing clinical nodes for Qdrant...")
    unique_nodes = set()
    for t in triples:
        unique_nodes.add(t["head"].upper())
        unique_nodes.add(t["tail"].upper())

    points = []
    for node_name in unique_nodes:
        vector = embedder.encode(node_name).tolist()
        points.append(PointStruct(
            # Consistent ID generation
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, node_name)),
            vector=vector,
            payload={"node_id": node_name}
        ))

    if points:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"✅ {len(points)} nodes embedded and stored in Qdrant.")


if __name__ == "__main__":
    test_text = "The patient presents with severe chronic migraines and was prescribed Lisinopril."
    triples = extract_triples(test_text)
    inject_into_neo4j(triples)
    db.close()
