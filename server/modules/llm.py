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
    reranker: CrossEncoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    top_k: int = 4

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        initial_docs = self.vectorstore_retriever.get_relevant_documents(query)
        if not initial_docs: return []
        
        pairs = [[query, doc.page_content] for doc in initial_docs]
        scores = self.reranker.predict(pairs)
        
        reranked_docs = sorted(zip(scores, initial_docs), key=lambda x: x[0], reverse=True)
        final_docs = [doc for score, doc in reranked_docs[:self.top_k]]
        
        if reranked_docs:
            print(f"Triage Retriever: Best doc (page {reranked_docs[0][1].metadata.get('page_number')}, score {reranked_docs[0][0]:.2f})")
        
        return final_docs

def get_text_rag_chain(vectorstore):
    """Creates the 'Triage' RAG chain for fast text-based questions."""
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant", temperature=0)
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
    reranking_retriever = RerankingRetriever(vectorstore_retriever=base_retriever)

    # FINAL UPGRADE: English prompt with language detection instruction
    template = """
    You are an expert document analysis assistant. Your task is to answer the user's question based ONLY on the following context extracted from one or more pages of a document.
    Synthesize the information to provide a complete and coherent answer. Do not invent any information.
    If the answer is not in the context, state "Based on the provided context, I don't have the information to answer that question."
    You MUST respond in the same language as the user's QUESTION.
    At the end of your answer, ALWAYS cite the page number of the document you got the information from, for example: [Source: Page 4].

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=reranking_retriever, return_source_documents=True, chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})