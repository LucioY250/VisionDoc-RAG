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

class RerankingRetriever(BaseRetriever):
    vectorstore_retriever: BaseRetriever
    reranker: CrossEncoder = CrossEncoder('BAAI/bge-reranker-v2-m3')
    top_k: int = 3

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        initial_docs = self.vectorstore_retriever.get_relevant_documents(query)
        if not initial_docs: return []
        
        pairs = [[query, doc.page_content] for doc in initial_docs]
        scores = self.reranker.predict(pairs)
        
        reranked_docs = sorted(zip(scores, initial_docs), key=lambda x: x[0], reverse=True)
        final_docs = [doc for score, doc in reranked_docs[:self.top_k]]
        
        if reranked_docs:
            print(f"Retriever: Best doc (page {reranked_docs[0][1].metadata.get('page_number')}, score {reranked_docs[0][0]:.2f})")
        
        return final_docs

def get_rag_chain(vectorstore):
    """Crea la cadena de RAG de alta calidad."""
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile", temperature=0)
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    reranking_retriever = RerankingRetriever(vectorstore_retriever=base_retriever)

    template = """
    You are a world-class document analysis expert. Your task is to provide a detailed and precise answer to the user's question based ONLY on the following context.
    The context consists of fused textual and visual summaries from document pages. Synthesize all information for a complete answer.
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

    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=reranking_retriever, return_source_documents=True, chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})