from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_chroma import Chroma

loader = PyPDFLoader('ilovepdf_merged.pdf')
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
docs = loader.load() 
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)
chunks = text_splitter.split_documents(docs)
vectorstore = Chroma(
    collection_name="agriculture_knowledge_v2",
    embedding_function=embeddings,
    persist_directory="./chroma_db_v2"
)

batch_size = 500
for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i + batch_size]

    print(
        f"Processing {i + 1} → "
        f"{min(i + batch_size, len(chunks))} "
        f"of {len(chunks)}"
    )
    vectorstore.add_documents(batch)