import os
import streamlit as st
from dotenv import load_dotenv
import datetime
import time
import uuid
from src.agent.agent import Agent
from src.utils.vector_store import VectorStore
from src.utils.search_tool import GoogleSearchTool
from src.utils.document_processor import DocumentProcessor
from src.utils.logo import get_logo_html

# Load environment variables
load_dotenv()

# Check if running on Streamlit Cloud
IS_STREAMLIT_CLOUD = "STREAMLIT_RUNTIME_PRODUCTION" in os.environ

# Application constants
APP_TITLE = "Planning Intelligence"
APP_SUBTITLE = "AI-Powered Planning Application Assistant"
APP_DESCRIPTION = "Access critical insights about planning applications in London with our AI-powered assistant."
CURRENT_YEAR = datetime.datetime.now().year

EXAMPLE_QUESTIONS = [
    "What is the current status of the planning application on New Brent Street?",
    "What details are available about the planning application on New Brent Street?",
    "What amenities are included at 99 City Road?",
    "What is the total proposed Gross Internal Area for 99 Bishopsgate?",
    "How does the proposed GIA of 99 Bishopsgate compare to that of 70 Gracechurch Street?",
    "How many planning applications have been approved in the London Borough of Barnet?",
]

# Set page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add custom CSS for styling
st.markdown(
    """
<style>
    /* Main layout styling */
    .main .block-container {
        padding-top: 1rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Header styling */
    .header-container {
        display: flex;
        align-items: center;
        padding-bottom: 1rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid #f0f0f0;
    }
    .header-logo {
        margin-right: 20px;
    }
    .header-text h1 {
        margin: 0;
        color: #1E3A8A;
        font-size: 1.8rem;
        font-weight: 600;
    }
    .header-text p {
        margin: 0;
        color: #6B7280;
    }
    
    /* Chat container styling */
    .stChatFloatingInputContainer {
        padding-bottom: 60px;  /* Add space for the footer */
    }
    .stChatMessageContent {
        border-radius: 8px !important;
    }
    .stChatMessageContent a {
        color: #1E40AF !important;
    }
    
    /* Status indicator */
    .status-indicator {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
        font-size: 0.9rem;
        color: #4B5563;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
        background-color: #10B981;
    }
    
    /* Footer styling */
    footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        text-align: center;
        padding: 0.5rem;
        background-color: rgba(255, 255, 255, 0.9);
        font-size: 0.8rem;
        color: #6B7280;
        border-top: 1px solid #f0f0f0;
        z-index: 100;
    }
    
    /* Button styling */
    .stButton button {
        border-radius: 6px;
        border: 1px solid #E5E7EB;
        padding: 0.3rem 1rem;
        font-size: 0.9rem;
        transition: all 0.2s;
    }
    .stButton button:hover {
        border-color: #1E40AF;
        color: #1E40AF;
        background-color: #EFF6FF;
    }
    
    /* Sidebar styling */
    .css-1544g2n {  /* Target sidebar */
        background-color: #F8FAFC;
    }
</style>
""",
    unsafe_allow_html=True,
)


def initialize_vector_store():
    """Initialize the vector store and build it if needed."""
    # Initialize with a data directory in "data/faiss_index"
    vector_store = VectorStore(persist_directory="data/faiss_index")

    # Check if we need to build the vector store from data files
    pdf_dir = "data/raw/pdf"
    if os.path.exists(pdf_dir) and any(f.endswith(".pdf") for f in os.listdir(pdf_dir)):
        # Check if the vector store is empty or we're on Streamlit Cloud (always rebuild)
        if IS_STREAMLIT_CLOUD or not os.path.exists("data/faiss_index/pdf_docs.faiss"):
            with st.spinner("Building knowledge base from planning documents..."):
                doc_processor = DocumentProcessor()
                documents = doc_processor.process_directory(pdf_dir)
                if documents:
                    vector_store.add_documents(documents)
                    return vector_store, True, len(documents)

    return vector_store, False, 0


def initialize_agent():
    """Initialize the agent with vector store and search tool."""
    # Initialize vector store
    vector_store, is_new_build, doc_count = initialize_vector_store()

    # Store build info in session state
    st.session_state.is_new_build = is_new_build
    st.session_state.doc_count = doc_count

    # Check if Google Search API credentials are available
    if not os.getenv("GOOGLE_API_KEY") or not os.getenv("GOOGLE_CSE_ID"):
        st.session_state.has_search = False
        search_tool = None
    else:
        st.session_state.has_search = True
        search_tool = GoogleSearchTool()

    # Initialize agent
    with st.spinner("Initializing AI Assistant..."):
        agent = Agent(vector_store=vector_store, search_tool=search_tool)

    return agent


def display_header():
    """Display a professional header."""
    header_html = f"""
    <div class="header-container">
        <div class="header-logo">
            {get_logo_html(width=50)}
        </div>
        <div class="header-text">
            <h1>{APP_TITLE}</h1>
            <p>{APP_SUBTITLE}</p>
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)


def display_status():
    """Display system status indicator."""
    status_html = """
    <div class="status-indicator">
        <div class="status-dot"></div>
        <span>AI Assistant ready</span>
    </div>
    """
    st.markdown(status_html, unsafe_allow_html=True)

    if hasattr(st.session_state, "is_new_build") and st.session_state.is_new_build:
        st.markdown(
            f"_Knowledge base built with {st.session_state.doc_count} document chunks_"
        )


def display_footer():
    """Display a professional footer."""
    footer_html = f"""
    <footer>
        © {CURRENT_YEAR} Planning Intelligence | Powered by AI | All rights reserved
    </footer>
    """
    st.markdown(footer_html, unsafe_allow_html=True)


def process_feedback(feedback_type, message_idx):
    """Process user feedback on a response."""
    if "feedback" not in st.session_state:
        st.session_state.feedback = {}

    # Store feedback
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message_content = st.session_state.messages[message_idx]["content"]

    st.session_state.feedback[message_idx] = {
        "type": feedback_type,
        "timestamp": timestamp,
        "response": (
            message_content[:100] + "..."
            if len(message_content) > 100
            else message_content
        ),
    }

    # Show a thank you message in a toast notification
    if feedback_type == "positive":
        st.toast("Thank you for your positive feedback!", icon="👍")
    else:
        st.toast(
            "Thank you for your feedback. We'll use it to improve our responses.",
            icon="👎",
        )


def setup_sidebar():
    """Set up the sidebar with example questions and reset button."""
    st.sidebar.markdown("## Planning Intelligence")
    st.sidebar.markdown("---")

    st.sidebar.markdown("### Sample Queries")
    st.sidebar.write("Click on any example to try it:")

    # Display example questions as buttons
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        # First show the example text
        st.sidebar.markdown(f"**Example {i+1}:**")
        st.sidebar.markdown(f"_{q}_")

        # Then add a button to try it
        if st.sidebar.button(
            f"Try this query", key=f"example-{i}", use_container_width=True
        ):
            # Get response from agent
            with st.chat_message("user"):
                st.markdown(q)

            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": q})

            # Force a rerun to process the message
            st.rerun()

        st.sidebar.markdown("---")

    # Add clear conversation button with better styling
    if st.sidebar.button(
        "New Conversation",
        key="clear_btn",
        help="Start a new conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.session_state.agent.reset_conversation()
        st.session_state.conversation_id = datetime.datetime.now().strftime(
            "%Y%m%d%H%M%S"
        )
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        """
        This AI-powered planning assistant helps you access and interpret planning application data 
        and documents for properties in London.
        
        It uses advanced AI to process planning documents and provide accurate, relevant information.
        """
    )


def main():
    """Main function to run the Streamlit app."""
    # Display header
    display_header()

    # Display system status
    display_status()

    # Initialize conversation ID if not exists
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = datetime.datetime.now().strftime(
            "%Y%m%d%H%M%S"
        )

    # Initialize session state for chat history if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize feedback storage
    if "feedback" not in st.session_state:
        st.session_state.feedback = {}

    # Initialize agent if it doesn't exist
    if "agent" not in st.session_state:
        st.session_state.agent = initialize_agent()

    # Setup sidebar with examples and reset button
    setup_sidebar()

    # Display welcome message if no messages
    if len(st.session_state.messages) == 0:
        st.chat_message("assistant").write(
            "👋 Hello! I'm your Planning Intelligence Assistant. Ask me anything about planning applications in London."
        )

    # Display chat messages from history
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Add feedback buttons for assistant messages using columns for layout
            if message["role"] == "assistant" and i not in st.session_state.feedback:
                cols = st.columns([7, 1, 1])
                with cols[1]:
                    if st.button(
                        "👍", key=f"fb_pos_{i}", help="This response was helpful"
                    ):
                        process_feedback("positive", i)
                with cols[2]:
                    if st.button(
                        "👎", key=f"fb_neg_{i}", help="This response was not helpful"
                    ):
                        process_feedback("negative", i)

    # Get user input
    if prompt := st.chat_input("Ask about planning applications..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Add to message history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display assistant response
        with st.chat_message("assistant"):
            # Display typing indicator
            message_placeholder = st.empty()
            full_response = ""

            # Simulate typing with a delay
            # In a real app, you could replace this with streaming from your Agent
            typing_indicator = "⏳"
            message_placeholder.markdown(typing_indicator)

            try:
                # Get response from agent
                response = st.session_state.agent.query(prompt)

                # Simulate typing effect (this could be replaced with actual streaming)
                message_placeholder.empty()

                # Optional: for a typing effect, you could uncomment this code:
                # for chunk in response.split():
                #     full_response += chunk + " "
                #     time.sleep(0.05)
                #     message_placeholder.markdown(full_response + "▌")

                # Display the final response
                message_placeholder.markdown(response)

                # Add to message history
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )

                # Add feedback buttons
                i = len(st.session_state.messages) - 1
                cols = st.columns([7, 1, 1])
                with cols[1]:
                    if st.button(
                        "👍", key=f"fb_pos_curr", help="This response was helpful"
                    ):
                        process_feedback("positive", i)
                with cols[2]:
                    if st.button(
                        "👎", key=f"fb_neg_curr", help="This response was not helpful"
                    ):
                        process_feedback("negative", i)

            except Exception as e:
                error_msg = f"I apologize, but I encountered an error: {str(e)}"
                message_placeholder.markdown(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )

    # Display footer
    display_footer()


if __name__ == "__main__":
    main()
