# En modules/llm.py

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.schema.retriever import BaseRetriever
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.schema.document import Document
from typing import List

from sentence_transformers import CrossEncoder

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY") # Corregí el nombre de la variable (quité la 'S')

# --- NUEVA CLASE: UN RETRIEVER PERSONALIZADO CON RE-RANKING ---
class RerankingRetriever(BaseRetriever):
    vectorstore_retriever: BaseRetriever
    reranker: CrossEncoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    k: int = 3 # Número final de documentos a devolver

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # 1. Obtenemos más documentos de los necesarios del vectorstore
        initial_docs = self.vectorstore_retriever.get_relevant_documents(query)

        if not initial_docs:
            return []

        # 2. Preparamos los pares para el re-ranking
        pairs = [[query, doc.page_content] for doc in initial_docs]
        
        # 3. Calculamos las puntuaciones
        scores = self.reranker.predict(pairs)
        
        # 4. Combinamos, ordenamos y devolvemos los 'k' mejores
        reranked_docs = list(zip(scores, initial_docs))
        reranked_docs.sort(key=lambda x: x[0], reverse=True)
        
        final_docs = [doc for score, doc in reranked_docs[:self.k]]
        return final_docs

def get_llm_chain(vectorstore):
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama3-70b-8192"
    )

    # Creamos un retriever base
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    # Envolvemos el retriever base en nuestro retriever personalizado con re-ranking
    reranking_retriever = RerankingRetriever(vectorstore_retriever=base_retriever)

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=reranking_retriever, # Usamos nuestro nuevo retriever
        return_source_documents=True
    )