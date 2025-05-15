import os
from typing import List, Dict, Optional
from datetime import datetime
from dataclasses import dataclass, field
from openai import OpenAI
import numpy as np
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Document:
    question: str
    answer: str
    node_id: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    embedding: Optional[List[float]] = None

class GlobalRAGContext:
    def __init__(self, api_key=None, embedding_model="text-embedding-3-small"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.embedding_model = embedding_model
        self.client = OpenAI(api_key=self.api_key)
        self.documents: List[Document] = []
        self.global_context: List[Dict] = []  # [{question, answer, timestamp, node_id} ...]

    def add_document(self, question, answer, node_id, timestamp=None):
        doc = Document(question=question, answer=answer, node_id=node_id, timestamp=timestamp or datetime.now().isoformat())
        doc.embedding = self.get_embedding(doc.question + "\n" + doc.answer)
        self.documents.append(doc)
        return doc

    def get_embedding(self, text: str) -> List[float]:
        resp = self.client.embeddings.create(
            input=[text],
            model=self.embedding_model
        )
        return resp.data[0].embedding

    def cosine_similarity(self, v1, v2):
        v1 = np.array(v1)
        v2 = np.array(v2)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    def search(self, query: str, top_k=3) -> List[Document]:
        query_emb = self.get_embedding(query)
        scored = [
            (self.cosine_similarity(query_emb, doc.embedding), doc)
            for doc in self.documents if doc.embedding is not None
        ]
        scored.sort(reverse=True, key=lambda x: x[0])
        return [doc for score, doc in scored[:top_k]]

    def add_to_global_context_if_similar(self, query: str, threshold=0.8):
        query_emb = self.get_embedding(query)
        for doc in self.documents:
            if doc.embedding is not None:
                sim = self.cosine_similarity(query_emb, doc.embedding)
                if sim >= threshold:
                    self.global_context.append({
                        "question": doc.question,
                        "answer": doc.answer,
                        "timestamp": doc.timestamp,
                        "node_id": doc.node_id,
                        "similarity": sim
                    })

if __name__ == "__main__":
    # 테스트용 코드
    rag = GlobalRAGContext()
    rag.add_document("What is AI?", "AI stands for Artificial Intelligence.", node_id="1")
    rag.add_document("What is ML?", "ML stands for Machine Learning.", node_id="2")
    rag.add_document("What is deep learning?", "Deep learning is a subset of ML.", node_id="3")

    query = "Tell me about artificial intelligence."
    print("\n[Top Similar Documents]")
    top_docs = []
    query_emb = rag.get_embedding(query)
    for doc in rag.documents:
        if doc.embedding is not None:
            sim = rag.cosine_similarity(query_emb, doc.embedding)
            top_docs.append((sim, doc))
    top_docs.sort(reverse=True, key=lambda x: x[0])
    for sim, doc in top_docs[:3]:
        print(f"Q: {doc.question}\nA: {doc.answer}\nNode: {doc.node_id}\nTime: {doc.timestamp}\nSimilarity: {sim:.4f}\n")

    print("\n[Global Context Before]")
    print(rag.global_context)
    # top 3개를 global_context에 모두 추가 (similarity는 저장하지 않음)
    for sim, doc in top_docs[:3]:
        rag.global_context.append({
            "question": doc.question,
            "answer": doc.answer,
            "timestamp": doc.timestamp,
            "node_id": doc.node_id
        })
    print("\n[Global Context After]")
    print(rag.global_context) 