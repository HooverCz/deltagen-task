import json
import uuid

from langchain.prompts import ChatPromptTemplate
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema import StrOutputParser
from langchain.schema.document import Document
from unstructured.documents.elements import CompositeElement
from unstructured.partition.pdf import partition_pdf

from constants import ID_KEY, PDF_PATH
from llm import get_llm
from retriever import get_retriever
from loguru import logger


def process_pdf(filename: str) -> list[CompositeElement]:
    """Processes a PDF file and partitions it into chunks using specific extraction strategies.

    Args:
        filename (str): The path to the PDF file to be processed.

    Returns:
        list[CompositeElement]: A list of CompositeElement objects representing the partitioned chunks of the PDF.
    """
    logger.info(f"Processing PDF: {filename}")
    return partition_pdf(
        filename=filename,
        infer_table_structure=True,
        strategy="hi_res",
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
        chunking_strategy="by_title",
        max_characters=1024,
    )

def extract_texts(chunks: list[CompositeElement]) -> list[dict[str, any]]:
    """Extracts text content and metadata from PDF chunks.

    Args:
        chunks (list[CompositeElement]): A list of CompositeElement objects representing the PDF chunks.

    Returns:
        list[dict[str, any]]: A list of dictionaries containing text content and metadata.
    """
    logger.info("Extracting texts from PDF chunks.")
    texts: list[dict[str, any]] = []
    for chunk in chunks:
        if isinstance(chunk, CompositeElement):
            text_content: str = chunk.text
            if text_content:
                metadata_dict: dict[str, any] = chunk.metadata.to_dict()
                texts.append({
                    "content": text_content,
                    "type": "text",
                    "metadata": {
                        "filename": metadata_dict.get("filename"),
                        "page_number": metadata_dict.get("page_number"),
                    }
                })
    logger.info(f"Extracted {len(texts)} text elements.")
    return texts

def extract_images(chunks: list[CompositeElement]) -> list[dict[str, any]]:
    """Extracts image content and metadata from PDF chunks.

    Args:
        chunks (list[CompositeElement]): A list of CompositeElement objects representing the PDF chunks.

    Returns:
        list[dict[str, any]]: A list of dictionaries containing image content and metadata.
    """
    logger.info("Extracting images from PDF chunks.")
    images_data: list[dict[str, any]] = []
    for chunk in chunks:
        if isinstance(chunk, CompositeElement):
            for el in chunk.metadata.orig_elements:
                if "Image" in str(type(el)):
                    images_data.append({
                        "content": el.metadata.image_base64,
                        "type": "image",
                        "metadata": {"page_number": el.metadata.to_dict().get("page_number")}
                    })
    logger.info(f"Extracted {len(images_data)} image elements.")
    return images_data

def generate_summaries(texts: list[dict[str, any]], images: list[dict[str, any]]) -> tuple[list[str], list[str]]:
    """Generates summaries for text and image content using a language model.

    Args:
        texts (list[dict[str, any]]): A list of dictionaries containing text content and metadata.
        images (list[dict[str, any]]): A list of dictionaries containing image content and metadata.

    Returns:
        tuple[list[str], list[str]]: A tuple containing lists of text summaries and image summaries.
    """
    logger.info("Generating summaries for texts and images.")
    llm = get_llm()
    text_prompt = ChatPromptTemplate.from_template(
        """You are an assistant tasked with summarizing text.
        Give a concise summary of the text.
        Respond only with the summary, no additional comment.
        Do not start your message by saying "Here is a summary" or anything like that.
        Just give the summary as it is.
        text chunk: {element}"""
    )
    image_prompt = ChatPromptTemplate.from_messages([
        ("user", [
            {"type": "text", "text": "Be specific about graphs, such as bar plots."},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image}"}}
        ])
    ])
    summarize_text_chain = {"element": lambda x: x["content"]} | text_prompt | llm | StrOutputParser()
    summarize_image_chain = {"image": lambda x: x["content"]} | image_prompt | llm | StrOutputParser()
    text_summaries: list[str] = summarize_text_chain.batch(texts, {"max_concurrency": 3})
    image_summaries: list[str] = summarize_image_chain.batch(images, {"max_concurrency": 3})
    logger.info("Summaries generated.")
    return text_summaries, image_summaries

def ingest_documents(retriever: MultiVectorRetriever, texts: list[dict[str, any]], images: list[dict[str, any]], text_summaries: list[str], image_summaries: list[str]) -> None:
    """Ingests text and image documents into a vector store.

    Args:
        retriever (MultiVectorRetriever): The retriever object for interacting with the vector store.
        texts (list[dict[str, any]]): A list of dictionaries containing text content and metadata.
        images (list[dict[str, any]]): A list of dictionaries containing image content and metadata.
        text_summaries (list[str]): A list of text summaries.
        image_summaries (list[str]): A list of image summaries.
    """
    logger.info("Ingesting documents into the vector store.")
    doc_ids: list[str] = [str(uuid.uuid4()) for _ in texts]
    img_ids: list[str] = [str(uuid.uuid4()) for _ in images]

    summary_text_docs: list[Document] = [
        Document(
            page_content=summary,
            metadata={**texts[i]["metadata"], ID_KEY: doc_ids[i]},
        )
        for i, summary in enumerate(text_summaries)
    ]
    retriever.vectorstore.add_documents(summary_text_docs)
    retriever.docstore.mset(list(zip(doc_ids, [json.dumps(item).encode('utf-8') for item in texts])))

    summary_img_docs: list[Document] = [
        Document(
            page_content=summary,
            metadata={**images[i]["metadata"], ID_KEY: img_ids[i]},
        )
        for i, summary in enumerate(image_summaries)
    ]
    retriever.vectorstore.add_documents(summary_img_docs)
    retriever.docstore.mset(list(zip(img_ids, [json.dumps(item).encode('utf-8') for item in images])))
    logger.info("Documents ingested successfully.")

def main() -> None:
    """Main function to process a PDF, extract texts and images, generate summaries, and ingest documents."""
    logger.info("Starting main process.")
    chunks: list[CompositeElement] = process_pdf(PDF_PATH)

    texts: list[dict[str, any]] = extract_texts(chunks)
    images: list[dict[str, any]] = extract_images(chunks)

    text_summaries, image_summaries = generate_summaries(texts, images)
    retriever: MultiVectorRetriever = get_retriever()

    ingest_documents(
        retriever=retriever,
        texts=texts,
        images=images,
        text_summaries=text_summaries,
        image_summaries=image_summaries
    )
    logger.info("Main process completed.")

if __name__ == "__main__":
    main()
