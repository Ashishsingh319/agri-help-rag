from pathlib import Path
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db_v2"

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0
)

vectorstore = Chroma(
    collection_name="agriculture_knowledge_v2",
    embedding_function=embeddings,
    persist_directory=str(CHROMA_DIR)
)

base_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm
)

conversation_history = []

rewrite_prompt = PromptTemplate.from_template(
    """
You are helping an agricultural knowledge assistant understand
follow-up questions.

Conversation history:

{history}

Current question:

{question}

Rewrite the current question so that it is completely
understandable on its own.

Resolve words such as it, this, that, they, them, these and those
using the conversation history.

If the question is already complete, keep it essentially unchanged.

Return ONLY the rewritten question.
"""
)

answer_prompt = PromptTemplate.from_template(
    """
You are an agricultural knowledge assistant.

Answer the user's question using ONLY the provided context.

Rules:

1. Do not invent agricultural facts.

3. Do not invent pesticide dosages.

5. If the answer is not available in the context, say:
"I could not find this information in the knowledge base."
6. Give the answer in a clear and farmer-friendly format.
7. Use headings, bullet points and numbered lists when useful.


Context:

{context}

Question:

{question}

Answer:
"""
)


def format_docs(docs):

    return "\n\n".join(
        f"""
Source: {doc.metadata.get('book', 'agriculture knowledge book')}
Page: {doc.metadata.get('page_number', 'Unknown')}

Content:

{doc.page_content}
"""
        for doc in docs
    )


def rewrite_question(question, history):

    if not history:
        return question

    history_text = "\n".join(
        f"User: {item['question']}\n"
        f"Assistant: {item['answer']}"
        for item in history[-5:]
    )

    rewritten = (
        rewrite_prompt
        | llm
        | StrOutputParser()
    ).invoke({
        "history": history_text,
        "question": question
    })

    return rewritten.strip()


def ask_question(question):

    standalone_question = rewrite_question(
        question,
        conversation_history
    )

    docs = multi_query_retriever.invoke(
        standalone_question
    )

    context = format_docs(docs)

    answer = (
        answer_prompt
        | llm
        | StrOutputParser()
    ).invoke({
        "context": context,
        "question": standalone_question
    })

    conversation_history.append({
        "question": question,
        "answer": answer
    })

    if len(conversation_history) > 10:
        conversation_history.pop(0)

    return answer


def clear_memory():

    conversation_history.clear()