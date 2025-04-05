import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.tools import Tool

from src.utils.vector_store import VectorStore
from src.utils.search_tool import GoogleSearchTool

load_dotenv()

# Fix SSL certificate verification issues
os.environ.pop("SSL_CERT_FILE", None)  # Remove problematic SSL_CERT_FILE if it exists


class Agent:
    """An AI agent that can answer queries about planning applications."""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        search_tool: Optional[GoogleSearchTool] = None,
    ):
        """
        Initialize the planning agent.

        Args:
            vector_store: Vector store for document search
            search_tool: Google search tool for web search
        """
        # Initialize vector store if not provided
        self.vector_store = vector_store or VectorStore()

        # Initialize search tool if not provided
        self.search_tool = search_tool or GoogleSearchTool()

        # Initialize chat model
        self.llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            deployment_name=os.getenv("AZURE_OPENAI_CHAT_MODEL"),
            temperature=0.0,
        )

        # Set up tools
        self.tools = self._setup_tools()

        # Set up agent with tools
        self.agent_executor = self._setup_agent()

        # Setup basic conversation chain for follow-up questions
        self.conversation_history = []
        self.conversation_chain = self._setup_conversation_chain()

    def _setup_tools(self) -> List[Tool]:
        """
        Set up the tools for the agent.

        Returns:
            List of LangChain tools
        """
        # Document search tool
        doc_search_tool = Tool(
            name="document_search",
            description="Search the planning application documents and database for information about planning applications in London.",
            func=lambda query: self._search_documents(query),
        )

        # Google search tool
        google_search_tool = self.search_tool.get_langchain_tool()

        return [doc_search_tool, google_search_tool]

    def _search_documents(self, query: str) -> str:
        """
        Search documents for information about a query.

        Args:
            query: Search query

        Returns:
            String with search results
        """
        results = self.vector_store.similarity_search(query, k=5)

        if not results:
            return "No relevant information found in the planning documents."

        documents_text = "\n\n".join(
            [
                f"Document: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
                for doc in results
            ]
        )

        return f"Here is relevant information from the planning documents:\n\n{documents_text}"

    def _setup_agent(self) -> AgentExecutor:
        """
        Set up the agent with tools.

        Returns:
            Agent executor
        """
        # Define the agent prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an AI assistant specialized in answering questions about planning applications in London.
You have access to a database of planning application documents and can search for information online if needed.

To answer questions, you should:
1. Search the planning application documents first
2. If the information isn't available in the documents, search online
3. Synthesize the information from all sources to provide a comprehensive answer

Always cite your sources, whether it's from the planning documents or online sources.
""",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        # Create the agent
        agent = create_openai_tools_agent(self.llm, self.tools, prompt)

        # Create the agent executor
        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
        )

    def _setup_conversation_chain(self):
        """
        Set up a conversation chain for follow-up questions.

        Returns:
            LangChain conversation chain
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an AI assistant specialized in answering questions about planning applications in London.
You have access to a database of planning application documents and can search for information online.

Provide helpful, accurate, and concise responses based on the conversation history and the current question.
""",
                ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )

        chain = prompt | self.llm | StrOutputParser()
        return chain

    def query(self, question: str) -> str:
        """
        Process a user query.

        Args:
            question: User's question

        Returns:
            Agent's response
        """
        # Check if it's a follow-up question (uses conversation history)
        is_followup = len(self.conversation_history) > 0 and not self._is_new_topic(
            question
        )

        if is_followup:
            # Use conversation chain for follow-up questions
            response = self.conversation_chain.invoke(
                {
                    "history": self.conversation_history,
                    "question": question,
                }
            )
        else:
            # Use agent executor for new questions
            result = self.agent_executor.invoke(
                {
                    "input": question,
                    "chat_history": self.conversation_history,
                }
            )
            response = result["output"]

        # Update conversation history
        self.conversation_history.append(HumanMessage(content=question))
        self.conversation_history.append(AIMessage(content=response))

        return response

    def _is_new_topic(self, question: str) -> bool:
        """
        Check if a question is about a new topic.

        Args:
            question: User's question

        Returns:
            True if it's a new topic, False if it's a follow-up
        """
        # Simple heuristic: if the question is short and contains pronouns like "it", "they", "them", etc.
        # it's likely a follow-up question
        followup_indicators = [
            "it",
            "they",
            "them",
            "this",
            "that",
            "these",
            "those",
            "the",
        ]

        # Split question into words
        words = question.lower().split()

        # If question is short and contains follow-up indicators, it's likely a follow-up
        if len(words) < 10 and any(word in followup_indicators for word in words):
            return False

        return True

    def reset_conversation(self):
        """Reset the conversation history."""
        self.conversation_history = []


if __name__ == "__main__":
    agent = Agent()
    # print(agent.query("Who is the winner of 2024 Olympic men single tennis?"))
    print(agent.query("What amenities are included at 99 City Road?"))
