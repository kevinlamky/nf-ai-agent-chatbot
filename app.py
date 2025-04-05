import os
import streamlit as st
from dotenv import load_dotenv
from src.agent.agent import Agent
from src.utils.vector_store import VectorStore
from src.utils.search_tool import GoogleSearchTool

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Planning Applications Chatbot",
    page_icon="🏙️",
    layout="centered",
)

# Add custom CSS
st.markdown(
    """
<style>
    .reportview-container {
        background-color: #f5f5f5;
    }
    .chat-message {
        padding: 1.5rem; 
        border-radius: 0.5rem; 
        margin-bottom: 1rem; 
        display: flex;
        flex-direction: row;
        align-items: flex-start;
    }
    .chat-message.user {
        background-color: #DBF4FF;
    }
    .chat-message.assistant {
        background-color: #F0F0F0;
    }
    .chat-message .avatar {
        width: 40px;
        height: 40px;
        margin-right: 1rem;
        border-radius: 0.25rem;
    }
    .chat-message .content {
        width: 80%;
        padding: 0;
    }
    .css-1wrcr25 {
        margin-bottom: 6rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


def initialize_agent():
    """Initialize the agent with vector store and search tool."""
    # Initialize vector store
    vector_store = VectorStore()

    # Check if Google Search API credentials are available
    if not os.getenv("GOOGLE_API_KEY") or not os.getenv("GOOGLE_CSE_ID"):
        st.sidebar.warning(
            "Google Search API credentials not found. The agent will continue without web search capability."
        )
        search_tool = None
    else:
        search_tool = GoogleSearchTool()

    # Initialize agent
    with st.spinner("Initializing Agent..."):
        agent = Agent(vector_store=vector_store, search_tool=search_tool)

    return agent


def display_message(role, content):
    """Display a chat message with the appropriate styling."""
    if role == "user":
        st.markdown(
            f"""
        <div class="chat-message user">
            <div class="avatar">👤</div>
            <div class="content">{content}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
        <div class="chat-message assistant">
            <div class="avatar">🤖</div>
            <div class="content">{content}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )


def main():
    """Main function to run the Streamlit app."""
    # Set up the title and description
    st.title("Planning Applications AI Assistant")
    st.markdown("Ask questions about planning applications in London")

    # Initialize session state for chat history if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize agent if it doesn't exist
    if "agent" not in st.session_state:
        st.session_state.agent = initialize_agent()

    # Sidebar with examples and reset button
    with st.sidebar:
        st.header("Example Questions")
        example_questions = [
            "What is the current status of the planning application on New Brent Street?",
            "What details are available about the planning application on New Brent Street?",
            "What amenities are included at 99 City Road?",
            "What is the total proposed Gross Internal Area for 99 Bishopsgate?",
            "How does the proposed GIA of 99 Bishopsgate compare to that of 70 Gracechurch Street?",
            "How many planning applications have been approved in the London Borough of Barnet?",
        ]

        for q in example_questions:
            if st.button(q):
                st.session_state.messages.append({"role": "user", "content": q})
                # Get response from agent
                with st.spinner("Thinking..."):
                    response = st.session_state.agent.query(q)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
                # Force a rerun to display the new messages
                st.rerun()

        if st.button("Clear Conversation"):
            st.session_state.messages = []
            st.session_state.agent.reset_conversation()
            st.rerun()

    # Display chat history
    for message in st.session_state.messages:
        display_message(message["role"], message["content"])

    # Chat input
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_area("Your question:", key="user_input", height=100)
        submit_button = st.form_submit_button("Send")

        if submit_button and user_input:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_input})

            # Get response from agent
            with st.spinner("Thinking..."):
                response = st.session_state.agent.query(user_input)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

            # Force a rerun to display the new messages
            st.rerun()

    # Add some information about the app
    st.markdown("---")
    st.markdown(
        "This AI assistant uses Azure OpenAI to answer queries about planning applications in London."
    )


if __name__ == "__main__":
    main()
