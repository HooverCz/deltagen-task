import asyncio
import base64
from io import BytesIO

import streamlit as st
from PIL import Image
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph
from loguru import logger

from agent import get_agents_graph, State

# Initialize session state
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []

# Display messages in chat interface
def display_messages():
    for message in st.session_state.messages:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(message.content)

# Decode base64 image and display it
def display_base64_image(base64_code):
    image_data = base64.b64decode(base64_code)
    image = Image.open(BytesIO(image_data))
    st.image(image, caption="Retrieved Image", use_container_width=True)

async def main():
    # Set up the client
    assistant: CompiledStateGraph = get_agents_graph()
    config: dict = {"configurable": {"thread_id": "abc123"}}

    # Initialize session state
    init_session_state()
    display_messages()

    # Get user input
    if prompt := st.chat_input("Write a message 👋"):

        # Display user's message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append(HumanMessage(content=prompt))

        # Assistant's response placeholder
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("Working on it 🧐")

            # Invoke the assistant
            response: State = await assistant.ainvoke(
                input={"question": prompt, "conversation": st.session_state.messages},
                config=config
            )

            # Extract the answer
            answer = response.get("answer", "Sorry something went wrong 🤯")
            placeholder.empty()
            st.markdown(answer)
            st.session_state.messages.append(AIMessage(answer))

            # Extract retrieval context (sources)
            retrieval_context = response.get("retrieval_context", {})

            # Display text sources
            if "texts" in retrieval_context:
                st.markdown("### 📄 Retrieved Documents:")
                for text_source in retrieval_context["texts"]:
                    filename = text_source["metadata"].get("filename", "Unknown File")
                    page_number = text_source["metadata"].get("page_number", "N/A")
                    content = text_source.get("content", "No content available.")

                    with st.expander(f"📑 {filename} - Page {page_number}"):
                        st.markdown(content)

            # Display images
            if "images" in retrieval_context:
                st.markdown("### 🖼️ Retrieved Images:")
                for img_data in retrieval_context["images"]:
                    display_base64_image(img_data["content"])

            logger.info(st.session_state.messages)

if __name__ == "__main__":
    st.set_page_config(page_title="DeltaGen.ai Chatbot", page_icon="🤖", layout="centered")
    asyncio.run(main())
