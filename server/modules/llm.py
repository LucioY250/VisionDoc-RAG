# En modules/llm.py

import os
from dotenv import load_dotenv
from typing import List

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from sentence_transformers import CrossEncoder

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")


# ==============================================================================
# RETRIEVER DE DOS FASES (RETRIEVAL + RE-RANKING)
# ==============================================================================
class RerankingRetriever(BaseRetriever):
    """
    Este retriever primero obtiene documentos de un vectorstore y luego usa un
    Cross-Encoder para re-ordenarlos y obtener los resultados más relevantes.
    Es ideal para la arquitectura Multi-Vector, ya que puede fusionar de forma inteligente
    los resultados de resúmenes textuales y visuales.
    """
    vectorstore_retriever: BaseRetriever
    reranker: CrossEncoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    top_k: int = 4  # El número final de documentos a enviar al LLM

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        
        # 1. Fase de Recuperación (Recall): Obtener un grupo amplio de candidatos.
        # Obtenemos más documentos (ej. 10-15) para asegurar que no nos perdemos nada importante.
        initial_docs = self.vectorstore_retriever.get_relevant_documents(query)

        if not initial_docs:
            return []

        # 2. Fase de Re-ranking (Precision): Encontrar la aguja en el pajar.
        pairs = [[query, doc.page_content] for doc in initial_docs]
        scores = self.reranker.predict(pairs)
        
        # Combinamos, ordenamos y seleccionamos los mejores 'top_k'
        reranked_docs = sorted(zip(scores, initial_docs), key=lambda x: x[0], reverse=True)
        
        final_docs = [doc for score, doc in reranked_docs[:self.top_k]]
        
        print(f"Re-ranking completado. Mejor documento (puntuación {reranked_docs[0][0]:.2f}): {reranked_docs[0][1].page_content[:100]}...")
        return final_docs


# ==============================================================================
# FUNCIÓN PRINCIPAL PARA CONSTRUIR LA CADENA DE RAG
# ==============================================================================
def get_llm_chain(vectorstore):
    """
    Crea la cadena de RAG completa, conectando el retriever de dos fases con el LLM.
    """
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama3-70b-8192",
        temperature=0
    )

    # Creamos el retriever base para la fase de recuperación
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
    
    # Envolvemos el retriever base con nuestra lógica de re-ranking
    reranking_retriever = RerankingRetriever(vectorstore_retriever=base_retriever)

    # El "Prompt Maestro" sigue siendo la mejor opción para guiar al LLM
    template = """
    Eres un asistente experto en análisis de documentos. Tu tarea es responder la pregunta del usuario basándote ÚNICA Y EXCLUSIVAMENTE en el siguiente contexto, que consiste en resúmenes de páginas de un documento.
    Sintetiza la información de todos los fragmentos para dar una respuesta completa y coherente.
    Si la pregunta es sobre un diagrama o arquitectura, prioriza la información de los resúmenes visuales si están disponibles.
    No inventes información. Si la respuesta no está en el contexto, di "Basado en el contexto proporcionado, no tengo la información para responder a esa pregunta."
    Al final de tu respuesta, cita SIEMPRE la página del documento de la que obtuviste la información, por ejemplo: [Fuente: Página 4].

    CONTEXTO:
    {context}

    PREGUNTA:
    {question}

    RESPUESTA DETALLADA:
    """
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=reranking_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )