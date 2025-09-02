# En modules/llm.py
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

def get_rag_chain(vectorstore):
    """
    Crea la cadena de RAG de alta calidad, simplificada para un despliegue con memoria limitada.
    """
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-70b-versatile", temperature=0)
    
    # --- LA MODIFICACIÓN CRÍTICA ---
    # Eliminamos el Re-ranker y usamos un retriever simple.
    # Aumentamos 'k' para darle más contexto al LLM y mitigar la falta de re-ranking.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    # El prompt sigue siendo de alta calidad
    template = """
    You are a world-class document analysis expert. Your task is to provide a detailed and precise answer to the user's question based ONLY on the following context.
    The context may contain fused textual and visual summaries. Synthesize all information for a complete answer.
    If the question is about a diagram, focus on the visual description part of the context.
    You MUST respond in the same language as the user's QUESTION.
    At the end of your answer, ALWAYS cite the source page, for example: [Source: Page 4].

    CONTEXT:
    {context}

    QUESTION:
    {question}

    EXPERT ANSWER:
    """
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever, # Usamos el retriever simple
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )