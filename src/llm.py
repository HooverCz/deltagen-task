
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

def get_embedding_model() -> AzureOpenAIEmbeddings:
    return AzureOpenAIEmbeddings(model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYED_MODEL_NAME"))

def get_llm() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYED_MODEL_NAME"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )
