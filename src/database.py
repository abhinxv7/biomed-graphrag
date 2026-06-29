import os
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

load_dotenv()


class DatabaseManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance._initialize_services()
        return cls._instance

    def _initialize_services(self):
        print("Initializing Local Enterprise Architecture...")

        # Connect to Neo4j (Knowledge Graph)
        neo4j_uri = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "biomed_password123")
        self.neo4j_driver = GraphDatabase.driver(
            neo4j_uri, auth=(neo4j_user, neo4j_password))

        # Connect to Qdrant (Vector Store)
        qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")
        self.qdrant_client = QdrantClient(url=qdrant_url)

        # Initialize Free Local Embedding Model (Runs entirely on CPU/GPU locally)
        print("Loading local biomedical embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Initialize Free Local LLM Controller via Ollama
        ollama_url = os.getenv(
            "OLLAMA_BASE_URL", "http://host.docker.internal:11434")
        self.llm = OllamaLLM(base_url=ollama_url, model="llama3")

    def verify_connections(self):
        try:
            self.neo4j_driver.verify_connectivity()
            print("✅ Neo4j Connection: SUCCESS")

            self.qdrant_client.get_collections()
            print("✅ Qdrant Connection: SUCCESS")

            # Simple test embedding
            test_vector = self.embeddings.embed_query("Biomedical GraphRAG")
            print("✅ Local Embeddings Engine: SUCCESS")
            return True
        except Exception as e:
            print(f"❌ Initialization Error: {e}")
            return False

    def close(self):
        self.neo4j_driver.close()


db = DatabaseManager()
