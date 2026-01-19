import os
import time  # REQUIRED for the delay
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

class RAGEngine:
    def __init__(self, documents_folder="rules_documents"):
        self.documents_folder = documents_folder
        self.vectorstore = None
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        if not os.path.exists(documents_folder):
            os.makedirs(documents_folder)
            print(f"✅ Created folder: {documents_folder}")
        
        self.load_documents()
    
    def load_documents(self):
        print(f"📂 Loading documents from: {self.documents_folder}")
        all_documents = []
        file_count = 0
        loaders = {'.pdf': PyPDFLoader, '.txt': TextLoader, '.docx': Docx2txtLoader}
        
        for filename in os.listdir(self.documents_folder):
            file_path = os.path.join(self.documents_folder, filename)
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in loaders:
                try:
                    loader = loaders[file_ext](file_path)
                    documents = loader.load()
                    all_documents.extend(documents)
                    file_count += 1
                    print(f"   ✅ Loaded: {filename} ({len(documents)} pages)")
                except Exception as e:
                    print(f"   ❌ Error loading {filename}: {e}")
        
        if file_count == 0:
            print("⚠️ No documents found.")
            return
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(all_documents)
        print(f"📝 Created {len(chunks)} chunks")
        
        # --- NEW BATCHING LOGIC TO PREVENT 429 ERRORS ---
        print("🔍 Creating vector embeddings in batches (Respecting Quota)...")
        batch_size = 50  # Process 50 chunks at a time
        self.vectorstore = None

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            try:
                if self.vectorstore is None:
                    self.vectorstore = FAISS.from_documents(batch, self.embeddings)
                else:
                    self.vectorstore.add_documents(batch)
                
                print(f"   ✅ Processed chunks {i} to {min(i + batch_size, len(chunks))}")
                
                # Sleep to avoid hitting the 1,500 Requests Per Minute limit
                if i + batch_size < len(chunks):
                    time.sleep(2) 
                    
            except Exception as e:
                print(f"   ❌ API Limit Hit: {e}")
                print("   ⏳ Waiting 10 seconds before retrying batch...")
                time.sleep(10)
                # Retry logic for the failed batch
                if self.vectorstore is None:
                    self.vectorstore = FAISS.from_documents(batch, self.embeddings)
                else:
                    self.vectorstore.add_documents(batch)

        print(f"✅ RAG Engine ready! Loaded {file_count} documents.")

    def search(self, query: str, k: int = 3) -> List[Dict]:
        if not self.vectorstore: return []
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return [{'content': doc.page_content, 'source': doc.metadata.get('source', 'Unknown'),
                 'page': doc.metadata.get('page', 'N/A'), 'score': float(score)} for doc, score in results]

    def get_context_for_question(self, question: str, max_chunks: int = 3) -> str:
        if not self.vectorstore: return "No rule documents loaded."
        results = self.search(question, k=max_chunks)
        if not results: return "No relevant information found."
        return "\n".join([f"[Source {i}: {os.path.basename(r['source'])}, Page {r['page']}]\n{r['content']}\n" 
                          for i, r in enumerate(results, 1)])

if __name__ == "__main__":
    rag = RAGEngine()
    test_q = "What is the definition of a machine gun?"
    print(f"\n🔍 Test: {test_q}\n{rag.get_context_for_question(test_q)}")