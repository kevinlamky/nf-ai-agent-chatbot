import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents import AgentExecutor
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain_core.tools import Tool
from langchain.agents.agent_types import AgentType

from src.utils.vector_store import VectorStore
from src.utils.search_tool import GoogleSearchTool
from src.agent.csv_agent import get_csv_agent

load_dotenv()

# Fix SSL certificate verification issues
os.environ.pop("SSL_CERT_FILE", None)  # Remove problematic SSL_CERT_FILE if it exists


class Agent:
    """An AI agent that can answer queries about planning applications using PDF documents, CSV data, and web search."""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        search_tool: Optional[GoogleSearchTool] = None,
        csv_path: str = "data/raw/csv/Planning Application Details.csv",
    ):
        """
        Initialize the unified planning agent.

        Args:
            vector_store: Vector store for PDF document search
            search_tool: Google search tool for web search
            csv_path: Path to the CSV file with planning data
        """
        # Initialize vector store if not provided
        self.vector_store = vector_store or VectorStore()

        # Initialize search tool if not provided
        self.search_tool = search_tool or GoogleSearchTool()

        # Initialize CSV file paths
        self.csv_path = csv_path

        # Initialize chat model
        try:
            self.llm = AzureChatOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                deployment_name=os.getenv("AZURE_OPENAI_CHAT_MODEL"),
            )
            print("Azure OpenAI LLM initialized successfully")
        except Exception as e:
            print(f"Warning: Error initializing Azure OpenAI: {str(e)}")
            raise

        # Initialize CSV agent if CSV files are available
        self.csv_agent = None
        if os.path.exists(self.csv_path):
            try:
                self.csv_agent = get_csv_agent(self.csv_path, verbose=True)
                print(f"CSV agent initialized with file: {self.csv_path}")
            except Exception as e:
                print(f"Warning: Failed to initialize CSV agent: {str(e)}")

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
        tools = []

        # Document search tool - only if vector store is available
        if self.vector_store:
            doc_search_tool = Tool(
                name="pdf_document_search",
                description="""Search the PDF planning application documents for information about planning applications in London.
Use this tool when the question is about specific planning applications, building details, or policy information
that would be contained in official documents.""",
                func=lambda query: self._search_documents(query),
            )
            tools.append(doc_search_tool)

        # CSV data tool
        if self.csv_agent:
            csv_tool = Tool(
                name="csv_data_search",
                description="""Search the CSV database of planning applications for detailed application information.
Use this tool when the question is about planning application status, dates, locations, reference numbers,
statistics about applications, or when you need structured data about the applications.
This tool is good for queries that require analysis of planning application records.""",
                func=lambda query: self._query_csv_data(query),
            )
            tools.append(csv_tool)

        # Google search tool - only if available
        if self.search_tool:
            google_search_tool = self.search_tool.get_langchain_tool()
            tools.append(google_search_tool)

        return tools

    def _search_documents(self, query: str) -> str:
        """
        Search PDF documents for information about a query.

        Args:
            query: Search query

        Returns:
            String with search results
        """
        if not self.vector_store:
            return "PDF document search is not available."

        try:
            results = self.vector_store.similarity_search(query, k=3)

            if not results:
                return "No relevant information found in the planning documents."

            documents_text = "\n\n".join(
                [
                    f"Document: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
                    for doc in results
                ]
            )

            return f"Here is relevant information from the planning documents:\n\n{documents_text}"
        except Exception as e:
            return f"Error searching documents: {str(e)}"

    def _query_csv_data(self, query: str) -> str:
        """
        Query the CSV data using the CSV agent.

        Args:
            query: User query about planning application data

        Returns:
            String with results from the CSV agent
        """
        if not self.csv_agent:
            return "CSV data search is not available."

        try:
            # Execute the query using the CSV agent
            result = self.csv_agent.invoke({"input": query})

            # Extract and return the output
            if isinstance(result, dict) and "output" in result:
                return result["output"]
            return str(result)
        except Exception as e:
            return f"Error querying CSV data: {str(e)}"

    def _setup_agent(self) -> AgentExecutor:
        """
        Set up the agent with tools.

        Returns:
            Agent executor
        """
        # Define the agent prompt
        system_message = """You are an AI assistant specialized in answering questions about planning applications in London.
"""

        available_tools = []
        if self.vector_store:
            system_message += "You can search through PDF planning application documents for detailed information.\n"
            available_tools.append("PDF document search")

        if self.csv_agent:
            system_message += "You have access to a database of planning application records in CSV format.\n"
            available_tools.append("CSV data search")

        if self.search_tool:
            system_message += "You can search online for additional information about planning applications and related topics.\n"
            available_tools.append("web search")

        system_message += f"\nTo answer questions, you have these tools available: {', '.join(available_tools)}.\n"
        system_message += """
Follow this process when answering questions:

1. For questions about specific planning applications, statistics, or application status:
   - First try using the CSV data search tool to get accurate information from the application database.

2. For questions requiring detailed information about planning policies, building specifications, or technical details:
   - Use the PDF document search to find relevant information from official planning documents.

3. For general knowledge or current information not in the documents or database:
   - Use the web search tool to find information online.

4. Synthesize information from all sources to provide a comprehensive answer.

Always cite your sources, whether the information comes from planning documents, CSV data, or online sources.
Provide clear and accurate answers focusing on the specific question asked.
"""

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    system_message,
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        # Create the agent - using OpenAIFunctionsAgent
        agent = OpenAIFunctionsAgent(llm=self.llm, tools=self.tools, prompt=prompt)

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
You can search for information in planning documents, CSV application data, and online sources.

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
        try:
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
                # If no tools are available, fall back to just using the LLM
                if not self.tools:
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
        except Exception as e:
            # Fallback to basic response in case of errors
            print(f"Error in agent processing: {str(e)}")
            # Try a simple fallback
            try:
                fallback_response = self.llm.invoke(
                    f"Please answer this question about planning applications: {question}"
                )
                response = fallback_response.content
                return response
            except:
                return f"I'm sorry, I encountered an error while processing your question. Error details: {str(e)}"

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
    # print(
    #     agent.query(
    #         "What is the total proposed Gross Internal Area (GIA) for 99 Bishopsgate?"
    #     )
    # )
    # print(
    #     agent.query(
    #         "How does the proposed GIA of 99 Bishopsgate compare to that of 70 Gracechurch Street?"
    #     )
    # )
