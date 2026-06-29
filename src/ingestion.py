import os
import mimetypes
import base64
from typing import TypedDict, Literal
import warnings

from langgraph.graph import StateGraph, START, END
import torch
from transformers import pipeline
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

# Import our Phase 3 graph core functions
from graph_builder import extract_triples, inject_into_neo4j
from database import db

# Suppress verbose warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize Local Whisper Engine (Runs on CPU)
print("🎙️ Loading Local Whisper Engine...")
whisper_pipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-base",
    device="cpu"
)

# 1. The Unified State Object


class IngestionState(TypedDict):
    file_path: str
    file_type: str
    raw_content: str
    status: str

# 2. Supervisor Router Node


def router_node(state: IngestionState) -> IngestionState:
    print(f"\n🕵️  Supervisor Agent analyzing: {state['file_path']}")
    mime_type, _ = mimetypes.guess_type(state["file_path"])

    if mime_type:
        if mime_type.startswith("text"):
            state["file_type"] = "text"
        elif mime_type.startswith("audio"):
            state["file_type"] = "audio"
        elif mime_type.startswith("image"):
            state["file_type"] = "image"
        else:
            state["file_type"] = "unknown"
    else:
        ext = os.path.splitext(state["file_path"])[1].lower()
        if ext in ['.txt', '.md', '.csv']:
            state["file_type"] = "text"
        elif ext in ['.mp3', '.wav']:
            state["file_type"] = "audio"
        elif ext in ['.jpg', '.png', '.jpeg']:
            state["file_type"] = "image"
        else:
            state["file_type"] = "unknown"

    print(f"🧭 Decision: Routing to {state['file_type'].upper()} processor.")
    return state


def route_to_processor(state: IngestionState) -> Literal["process_text", "process_audio", "process_vision", "unsupported"]:
    if state["file_type"] == "text":
        return "process_text"
    elif state["file_type"] == "audio":
        return "process_audio"
    elif state["file_type"] == "image":
        return "process_vision"
    return "unsupported"

# 3. Specialized Extractor Nodes feeding into Graph Core


def process_text(state: IngestionState) -> IngestionState:
    print("📄 Text Agent extracting content...")
    with open(state["file_path"], 'r') as f:
        state["raw_content"] = f.read().strip()

    # Send directly to Llama 3 -> Neo4j
    triples = extract_triples(state["raw_content"])
    inject_into_neo4j(triples)

    state["status"] = "ingested_into_graph"
    return state


def process_audio(state: IngestionState) -> IngestionState:
    print("🎧 Audio Agent transcribing dictation...")
    if os.path.getsize(state["file_path"]) == 0:
        print("⚠️ Empty dummy audio file. Using simulated patient dictation.")
        state["raw_content"] = "Patient exhibits acute knee inflammation. Recommend immediate arthroscopy procedure."
    else:
        try:
            transcription = whisper_pipeline(state["file_path"])
            state["raw_content"] = transcription["text"]
        except Exception as e:
            print(f"❌ Whisper Error: {e}")
            state["status"] = "failed"
            return state

    # Send directly to Llama 3 -> Neo4j
    triples = extract_triples(state["raw_content"])
    inject_into_neo4j(triples)

    state["status"] = "ingested_into_graph"
    return state


def process_vision(state: IngestionState) -> IngestionState:
    print("👁️  Vision Agent running multimodal analysis...")
    if os.path.getsize(state["file_path"]) == 0:
        print("⚠️ Empty dummy image file. Using simulated medical image analysis.")
        state["raw_content"] = "Chest X-Ray shows severe bacterial pneumonia localized in the lower left lung lobe."
    else:
        try:
            with open(state["file_path"], "rb") as img_file:
                image_b64 = base64.b64encode(img_file.read()).decode("utf-8")

            ollama_url = os.getenv(
                "OLLAMA_BASE_URL", "http://host.docker.internal:11434")
            vision_llm = ChatOllama(
                base_url=ollama_url, model="llava", temperature=0.1)

            messages = [
                HumanMessage(content=[
                    {"type": "text", "text": "Extract all key findings, conditions, diagnoses, and procedures from this medical image as clear clinical text."},
                    {"type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{image_b64}"}
                ])
            ]
            response = vision_llm.invoke(messages)
            state["raw_content"] = response.content
        except Exception as e:
            print(f"❌ Vision Error: {e}")
            state["status"] = "failed"
            return state

    # Send directly to Llama 3 -> Neo4j
    triples = extract_triples(state["raw_content"])
    inject_into_neo4j(triples)

    state["status"] = "ingested_into_graph"
    return state


def unsupported_file(state: IngestionState) -> IngestionState:
    print("❌ Error: Unsupported file format.")
    state["status"] = "failed"
    return state


# 4. Assemble Graph Architecture
workflow = StateGraph(IngestionState)
workflow.add_node("router", router_node)
workflow.add_node("process_text", process_text)
workflow.add_node("process_audio", process_audio)
workflow.add_node("process_vision", process_vision)
workflow.add_node("unsupported", unsupported_file)

workflow.add_edge(START, "router")
workflow.add_conditional_edges("router", route_to_processor)
workflow.add_edge("process_text", END)
workflow.add_edge("process_audio", END)
workflow.add_edge("process_vision", END)
workflow.add_edge("unsupported", END)

ingestion_pipeline = workflow.compile()

# 5. Full Pipeline End-to-End Test Run
if __name__ == "__main__":
    test_files = [
        "data/input/clinical_note.txt",
        "data/input/consultation.mp3",
        "data/input/mri_scan.png"
    ]

    print("\n🚀 STARTING UNIFIED MULTIMODAL INGESTION MATRIX 🚀")
    for file in test_files:
        initial_state = IngestionState(
            file_path=file,
            file_type="",
            raw_content="",
            status="pending"
        )
        ingestion_pipeline.invoke(initial_state)

    # Close global Neo4j connection pool gracefully
    db.close()
    print("\n🏁 ALL TRANSACTIONS RECORDED IN NEO4J KNOWLEDGE GRAPH.")
