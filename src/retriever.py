from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import LocalFileStore
from langchain_chroma import Chroma

from src.constants import CHROMA_PATH, DOCSTORE_PATH, ID_KEY, TOP_K_TO_RETRIEVE
from src.llm import get_embedding_model

def get_retriever() -> MultiVectorRetriever:
    """Creates and returns a MultiVectorRetriever instance.

    This function initializes a Chroma vector store and a LocalFileStore
    for document storage. It then creates a MultiVectorRetriever using
    these components, along with specified search parameters.

    Returns:
        MultiVectorRetriever: An instance configured with the vector store,
        document store, and search parameters.
    """
    vectorstore = Chroma(
        collection_name="deltagen",
        embedding_function=get_embedding_model(),
        persist_directory=CHROMA_PATH
    )

    docstore = LocalFileStore(DOCSTORE_PATH)

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key=ID_KEY,
        search_kwargs={"k": TOP_K_TO_RETRIEVE}
    )
    return retriever
