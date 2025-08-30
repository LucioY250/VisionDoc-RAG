# En modules/llm.py

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

def get_llm_chain(vectorstore, docstore):
    """
    Crea la cadena de RAG usando el ParentDocumentRetriever para obtener un contexto de página completa.
    """
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama3-70b-8192"
    )

    ### MEJORA CRÍTICA: Implementación correcta del ParentDocumentRetriever ###
    # Este retriever buscará en el vectorstore (chunks pequeños)
    # pero devolverá los documentos del docstore (páginas completas),
    # dándole al LLM un contexto mucho más rico.
    parent_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50),
        parent_id_key="doc_id" 
    )

    template = """
    Eres un asistente experto en análisis de documentos. Tu tarea es responder la pregunta del usuario basándote ÚNICA Y EXCLUSIVAMENTE en el siguiente contexto extraído de una o varias páginas de un documento.
    Sintetiza la información para dar una respuesta completa y coherente.
    Si el contexto contiene el texto de un diagrama o una arquitectura, descríbela en detalle.
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
        retriever=parent_retriever, # <<< USAMOS EL RETRIEVER CORRECTO
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )