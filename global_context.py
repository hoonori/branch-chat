import os
from typing import List, Dict, Optional
from datetime import datetime
from dataclasses import dataclass, field
from openai import OpenAI
import numpy as np
from dotenv import load_dotenv
import faiss

load_dotenv()

@dataclass
class Document:
    question: str
    answer: str
    node_id: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    embedding: Optional[np.ndarray] = None

class GlobalRAGContext:
    def __init__(self, api_key=None, embedding_model="text-embedding-3-small"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.embedding_model = embedding_model
        self.client = OpenAI(api_key=self.api_key)
        self.documents: List[Document] = []
        self.global_context: List[Dict] = []  # [{question, answer, timestamp, node_id} ...]

        # FAISS specific attributes
        self.embedding_dim = 1536  # For text-embedding-3-small
        self.nlist = 10  # Number of clusters for IndexIVFFlat
        self.min_train_docs = max(self.nlist + 1, 30) # Min docs to train FAISS IVF (e.g. 30, or more robustly nlist * k)
        self.faiss_index: Optional[faiss.IndexIVFFlat] = None
        self.is_trained = False

    def _get_embedding_openai(self, text: str) -> Optional[np.ndarray]:
        try:
            resp = self.client.embeddings.create(
                input=[text],
                model=self.embedding_model
            )
            embedding_vector = np.array(resp.data[0].embedding).astype('float32')
            # OpenAI embeddings (e.g., text-embedding-3-small) are pre-normalized to length 1
            # If not, normalization would be needed: faiss.normalize_L2(embedding_vector.reshape(1, -1))
            return embedding_vector # Should be 1D array
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None

    def _train_and_populate_index(self):
        if len(self.documents) < self.min_train_docs:
            print(f"Not enough documents to train. Need {self.min_train_docs}, have {len(self.documents)}")
            return

        print(f"Training FAISS IndexIVFFlat with {len(self.documents)} documents.")
        
        # Collect embeddings, ensuring they are 2D for FAISS processing
        embeddings_for_training_list = [doc.embedding for doc in self.documents if doc.embedding is not None]
        if not embeddings_for_training_list:
            print("No valid embeddings available for training.")
            return
            
        embeddings_for_training = np.vstack(embeddings_for_training_list).astype('float32')
        
        if embeddings_for_training.shape[0] < self.nlist:
            print(f"Number of training vectors ({embeddings_for_training.shape[0]}) is less than nlist ({self.nlist}). IVF training might be unstable or fail.")
            # Consider a fallback or error, but for now, proceed with caution.
            # For very small datasets, IndexFlatIP might be more appropriate initially. User insisted on IVF.

        quantizer = faiss.IndexFlatIP(self.embedding_dim)
        self.faiss_index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, self.nlist, faiss.METRIC_INNER_PRODUCT)
        
        try:
            self.faiss_index.train(embeddings_for_training)
            print("FAISS training complete.")
            self.faiss_index.add(embeddings_for_training)
            print(f"Added {self.faiss_index.ntotal} vectors to FAISS index.")
            self.is_trained = True
        except Exception as e:
            print(f"Error during FAISS training or adding: {e}")
            self.faiss_index = None # Reset index on failure
            self.is_trained = False


    def add_document(self, question: str, answer: str, node_id: str, timestamp: Optional[str] = None):
        embedding = self._get_embedding_openai(question + "\n" + answer)
        if embedding is None:
            print(f"Failed to get embedding for document: Q: {question}")
            return None # Or handle error appropriately

        doc = Document(
            question=question, 
            answer=answer, 
            node_id=node_id, 
            timestamp=timestamp or datetime.now().isoformat(),
            embedding=embedding # Storing 1D np.ndarray
        )
        self.documents.append(doc)

        if not self.is_trained:
            if len(self.documents) >= self.min_train_docs:
                self._train_and_populate_index()
        else:
            if self.faiss_index:
                # FAISS expects a 2D array for add, so reshape the 1D embedding
                self.faiss_index.add(embedding.reshape(1, -1))
        return doc

    def cosine_similarity(self, v1: Optional[np.ndarray], v2: Optional[np.ndarray]) -> float:
        if v1 is None or v2 is None:
            return 0.0
        # Assumes v1 and v2 are 1D numpy arrays (embeddings)
        # And they are already L2 normalized as per OpenAI standard
        sim = np.dot(v1, v2)
        # Clipping to handle potential floating point inaccuracies outside [-1, 1]
        return float(np.clip(sim, -1.0, 1.0))

    def search(self, query: str, top_k=3) -> List[Document]:
        if not self.is_trained or not self.faiss_index or self.faiss_index.ntotal == 0:
            # Fallback to brute-force search if IVF not ready and documents exist
            if self.documents:
                 print("FAISS index not ready or empty. Falling back to brute-force search.")
                 return self._brute_force_search(query, top_k)
            print("FAISS index not trained or empty, and no documents for fallback. Returning no results.")
            return []

        query_emb = self._get_embedding_openai(query)
        if query_emb is None:
            return []

        # nprobe controls the speed-accuracy trade-off for IVF indexes.
        # Higher nprobe = more accurate, but slower.
        self.faiss_index.nprobe = min(self.nlist, 10) # Search up to 10 clusters or all if nlist is small

        # FAISS search expects a 2D array for queries
        similarities, indices = self.faiss_index.search(query_emb.reshape(1, -1), top_k)
        
        results = []
        if indices.size > 0:
            for i in range(len(indices[0])):
                idx = indices[0][i]
                if idx != -1 and idx < len(self.documents): # Ensure index is valid
                    doc = self.documents[idx]
                    # Optionally, you can store the similarity score with the document if needed
                    # doc.similarity_score = float(similarities[0][i]) 
                    results.append(doc)
        return results

    def _brute_force_search(self, query: str, top_k=3) -> List[Document]:
        """A simple brute-force search used as a fallback."""
        query_emb = self._get_embedding_openai(query)
        if query_emb is None or not self.documents:
            return []
        
        scored_documents = []
        for doc in self.documents:
            if doc.embedding is not None:
                sim = self.cosine_similarity(query_emb, doc.embedding)
                scored_documents.append((sim, doc))
        
        scored_documents.sort(reverse=True, key=lambda x: x[0])
        return [doc for score, doc in scored_documents[:top_k]]

    def add_to_global_context_if_similar(self, query: str, threshold=0.8):
        query_emb = self._get_embedding_openai(query)
        if query_emb is None:
            return

        for doc in self.documents:
            if doc.embedding is not None:
                sim = self.cosine_similarity(query_emb, doc.embedding)
                if sim >= threshold:
                    # Check if already in global_context by node_id to avoid duplicates
                    if not any(item['node_id'] == doc.node_id for item in self.global_context):
                        self.global_context.append({
                            "question": doc.question,
                            "answer": doc.answer,
                            "timestamp": doc.timestamp,
                            "node_id": doc.node_id,
                            "similarity": sim
                        })
        # Sort global_context by similarity descending if desired
        # self.global_context.sort(key=lambda x: x.get('similarity', 0), reverse=True)


if __name__ == "__main__":
    # Test code
    rag = GlobalRAGContext()
    print(f"FAISS IVF Index. Min training docs: {rag.min_train_docs}, nlist: {rag.nlist}")

    # Add initial documents
    rag.add_document("What is AI?", "AI stands for Artificial Intelligence.", node_id="node1")
    rag.add_document("What is ML?", "ML stands for Machine Learning.", node_id="node2")
    rag.add_document("Explain deep learning.", "Deep learning is a subset of ML using neural networks.", node_id="node3")
    rag.add_document("What are Large Language Models?", "LLMs are models trained on vast amounts of text data.", node_id="node4")

    print(f"Number of documents: {len(rag.documents)}")
    print(f"Is FAISS trained: {rag.is_trained}")

    query1 = "Tell me about artificial intelligence."
    print(f"\\nSearching for: '{query1}'")
    results1 = rag.search(query1, top_k=2)
    if results1:
        for doc in results1:
            print(f"  Q: {doc.question}, A: {doc.answer} (Node: {doc.node_id})")
    else:
        print("  No results found.")

    # Add more documents to trigger training if min_train_docs is not yet met
    # Example: if min_train_docs = 5
    if not rag.is_trained and rag.min_train_docs > 4 :
        print(f"\\nAdding more documents to meet min_train_docs ({rag.min_train_docs})...")
        for i in range(5, rag.min_train_docs + 1):
            rag.add_document(f"Sample question {i}", f"Sample answer {i}.", node_id=f"node{i}")
            if rag.is_trained:
                print(f"FAISS trained after adding document {i}.")
                break
    
    print(f"Number of documents: {len(rag.documents)}")
    print(f"Is FAISS trained: {rag.is_trained}")
    if rag.faiss_index:
        print(f"FAISS index ntotal: {rag.faiss_index.ntotal}")

    query2 = "What are LLMs?"
    print(f"\\nSearching for: '{query2}' (after potential training)")
    results2 = rag.search(query2, top_k=2)
    if results2:
        for doc in results2:
            # To see similarity, you'd need to modify search to return it or print it there
            # For now, just printing the doc info
            print(f"  Q: {doc.question}, A: {doc.answer} (Node: {doc.node_id})")
    else:
        print("  No results found.")

    # Test adding a document after training
    if rag.is_trained:
        print("\\nAdding one more document post-training...")
        rag.add_document("What is reinforcement learning?", "RL is a type of ML where agents learn by trial and error.", node_id="node_post_train")
        print(f"Number of documents: {len(rag.documents)}")
        if rag.faiss_index:
            print(f"FAISS index ntotal: {rag.faiss_index.ntotal}")
        
        query3 = "learning types in ML"
        print(f"\\nSearching for: '{query3}'")
        results3 = rag.search(query3, top_k=2)
        if results3:
            for doc in results3:
                print(f"  Q: {doc.question}, A: {doc.answer} (Node: {doc.node_id})")
        else:
            print("  No results found.")

    # Test global context
    print("\\nTesting global context feature...")
    rag.add_to_global_context_if_similar("Tell me about AI and its subsets", threshold=0.7)
    print("[Global Context Contents]")
    if rag.global_context:
        for item in rag.global_context:
            print(f"  Q: {item['question']}, Sim: {item.get('similarity', 'N/A'):.4f} (Node: {item['node_id']})")
    else:
        print("  Global context is empty.") 