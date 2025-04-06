import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import time

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
from src.utils.logger import get_logger
from src.prompts.agent_prompts import (
    TOOL_DESCRIPTIONS,
    BASE_SYSTEM_MESSAGE,
    AGENT_PROCESS_INSTRUCTIONS,
    CONVERSATION_SYSTEM_MESSAGE,
    FALLBACK_PROMPT,
)

load_dotenv()

logger = get_logger(__name__)

# Fix SSL certificate verification issues
os.environ.pop("SSL_CERT_FILE", None)


class Agent:
    """An AI agent that can answer queries about planning applications using given PDF and CSV documents, and web search."""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        search_tool: Optional[GoogleSearchTool] = None,
        csv_path: str = "data/raw/csv/Planning Application Details.csv",
    ):
        """
        Initialize the unified chatbot agent.

        Args:
            vector_store: Vector store for given PDF documents
            search_tool: Google search tool for web search
            csv_path: Path to the given CSV file with planning application data
        """
        logger.info("Initializing Planning Agent")

        # Initialize vector store if not provided
        if vector_store:
            logger.info("Using provided vector store")
            self.vector_store = vector_store
        else:
            logger.info("Creating new vector store instance")
            self.vector_store = VectorStore()

        # Initialize search tool if not provided
        if search_tool:
            logger.info("Using provided search tool")
            self.search_tool = search_tool
        else:
            logger.info("Creating new search tool instance")
            self.search_tool = GoogleSearchTool()

        # Initialize CSV file paths
        logger.info(f"Setting CSV data path to: {csv_path}")
        self.csv_path = csv_path

        # Initialize chat model
        try:
            logger.info("Initializing Azure OpenAI chat model")
            self.llm = AzureChatOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                deployment_name=os.getenv("AZURE_OPENAI_CHAT_MODEL"),
            )
            logger.info("Azure OpenAI LLM initialized successfully")
        except Exception as e:
            logger.warning(f"Warning: Error initializing Azure OpenAI: {str(e)}")
            raise

        # Initialize CSV agent
        self.csv_agent = None
        if os.path.exists(self.csv_path):
            try:
                self.csv_agent = get_csv_agent(self.csv_path, verbose=True)
                logger.info(f"CSV agent initialized with file: {self.csv_path}")
            except Exception as e:
                logger.warning(f"Warning: Failed to initialize CSV agent: {str(e)}")
        else:
            logger.warning(f"CSV file not found at {self.csv_path}")

        # Set up tools
        logger.info("Setting up agent tools")
        self.tools = self._setup_tools()
        logger.info(f"Initialized {len(self.tools)} tools for agent")

        # Set up agent with tools
        logger.info("Setting up agent executor")
        self.agent_executor = self._setup_agent()
        logger.info("Agent executor initialized")

        # Setup basic conversation chain for follow-up questions
        logger.info("Setting up conversation chain for follow-up questions")
        self.conversation_history = []
        self.conversation_chain = self._setup_conversation_chain()
        logger.info("Conversation chain initialized")

        logger.info("Planning Agent initialization complete")

    def _setup_tools(self) -> List[Tool]:
        """
        Set up the tools for the agent.

        Returns:
            List of LangChain tools
        """
        logger.info("Setting up agent tools")
        tools = []

        # Document search tool
        if self.vector_store:
            logger.info("Adding document search tool")
            doc_search_tool = Tool(
                name="pdf_document_search",
                description=TOOL_DESCRIPTIONS["pdf_document_search"],
                func=lambda query: self._search_documents(query),
            )
            tools.append(doc_search_tool)

        # CSV data tool
        if self.csv_agent:
            logger.info("Adding CSV data search tool")
            csv_tool = Tool(
                name="csv_data_search",
                description=TOOL_DESCRIPTIONS["csv_data_search"],
                func=lambda query: self._query_csv_data(query),
            )
            tools.append(csv_tool)

        # Google search tool
        if self.search_tool:
            logger.info("Adding Google search tool")
            google_search_tool = self.search_tool.get_langchain_tool()
            tools.append(google_search_tool)

        logger.info(f"Finished setting up {len(tools)} tools")
        return tools

    def _search_documents(self, query: str) -> str:
        """
        Search PDF documents for information about a query.

        Args:
            query: Search query

        Returns:
            String with search results
        """
        logger.info(f'Searching documents for: "{query}"')

        if not self.vector_store:
            logger.warning("Document search requested but vector store not available")
            return "PDF document search is not available."

        try:
            results = self.vector_store.similarity_search(
                query, k=3
            )  # reduce k=3 for token rate limit

            if not results:
                logger.info("Document search returned no results")
                return "No relevant information found in the given PDF documents."

            documents_text = "\n\n".join(
                [
                    f"[Document: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}"
                    for doc in results
                ]
            )

            sources = [doc.metadata.get("source", "Unknown") for doc in results]
            logger.info(
                f"Document search returned {len(results)} results from sources: {', '.join(sources)}"
            )

            return f"Here is relevant information from the planning documents:\n\n {documents_text}"
        except Exception as e:
            logger.error(f"Error during document search: {e}")
            return f"Error searching documents: {e}"

    def _query_csv_data(self, query: str) -> str:
        """
        Query the CSV data using the CSV agent.

        Args:
            query: User query about planning application data

        Returns:
            String with results from the CSV agent
        """
        logger.info(f'Querying CSV data for: "{query}"')

        if not self.csv_agent:
            logger.warning("CSV query requested but CSV agent not available")
            return "CSV data search is not available."

        try:
            result = self.csv_agent.invoke({"input": query})

            if isinstance(result, dict) and "output" in result:
                logger.info(
                    f"CSV agent returned structured output of {len(result['output'])} characters"
                )
                return result["output"]

            logger.info(
                f"CSV agent returned unstructured result of {len(str(result))} characters"
            )
            return str(result)
        except Exception as e:
            logger.error(f"Error during CSV query: {e}")
            return f"Error querying CSV data: {e}"

    def _setup_agent(self) -> AgentExecutor:
        """
        Set up the agent with tools.

        Returns:
            Agent executor
        """
        logger.info("Setting up agent executor")

        # Start with the base system message
        system_message = BASE_SYSTEM_MESSAGE
        logger.debug("Using base system message")

        # Add available tools to the system message
        available_tools = []
        if self.vector_store:
            logger.info("Adding PDF document search capability to system message")
            system_message += "You can search through PDF planning application documents for detailed information.\n"
            available_tools.append("PDF document search")

        if self.csv_agent:
            logger.info("Adding CSV data search capability to system message")
            system_message += "You have access to a database of planning application records in CSV format.\n"
            available_tools.append("CSV data query")

        if self.search_tool:
            logger.info("Adding web search capability to system message")
            system_message += "You can search online for additional information about planning applications and related topics.\n"
            available_tools.append("web search")

        # Add tool list and process instructions
        system_message += f"\nTo answer questions, you have these tools available: {', '.join(available_tools)}.\n"
        system_message += AGENT_PROCESS_INSTRUCTIONS

        # Create the prompt
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

        agent = OpenAIFunctionsAgent(llm=self.llm, tools=self.tools, prompt=prompt)

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
        logger.info("Setting up conversation chain")

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    CONVERSATION_SYSTEM_MESSAGE,
                ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )

        logger.debug("Creating conversation chain with LLM and prompt")
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
        logger.info(f'Processing user query: "{question}"')
        start_time = time.time()

        try:
            # Check if it's a follow-up question (uses conversation history)
            is_followup = len(self.conversation_history) > 0 and not self._is_new_topic(
                question
            )

            if is_followup:
                logger.info("Detected as follow-up question, using conversation chain")
                # Use conversation chain for follow-up questions
                response = self.conversation_chain.invoke(
                    {
                        "history": self.conversation_history,
                        "question": question,
                    }
                )
                logger.info("Generated response using conversation chain")
            else:
                logger.info("Detected as new topic question")
                # If no tools are available, fall back to just using the LLM
                if not self.tools:
                    logger.info("No tools available, using basic LLM response")
                    response = self.conversation_chain.invoke(
                        {
                            "history": self.conversation_history,
                            "question": question,
                        }
                    )
                    logger.info("Generated response using basic LLM")
                else:
                    # Use agent executor for new questions
                    logger.info(
                        f"Using agent executor with {len(self.tools)} available tools"
                    )
                    result = self.agent_executor.invoke(
                        {
                            "input": question,
                            "chat_history": self.conversation_history,
                        }
                    )
                    response = result["output"]

                    # Log tool usage if available in the result
                    if "intermediate_steps" in result:
                        tools_used = []
                        for step in result["intermediate_steps"]:
                            if len(step) >= 2 and hasattr(step[0], "tool"):
                                tools_used.append(step[0].tool)
                        if tools_used:
                            logger.info(f"Tools used: {', '.join(tools_used)}")

                    logger.info("Generated response using agent executor")

            # Update conversation history
            self.conversation_history.append(HumanMessage(content=question))
            self.conversation_history.append(AIMessage(content=response))

            # Log conversation history length
            logger.info(
                f"Conversation history updated, now contains {len(self.conversation_history) // 2} exchanges"
            )

            # Log processing time
            end_time = time.time()
            processing_time = end_time - start_time
            logger.info(f"Query processed in {processing_time:.2f} seconds")

            return response
        except Exception as e:
            # Fallback to basic response in case of errors
            end_time = time.time()
            processing_time = end_time - start_time
            logger.error(
                f"Error in agent processing after {processing_time:.2f} seconds: {str(e)}"
            )

            # Try a simple fallback
            try:
                logger.info("Attempting fallback response")
                formatted_prompt = FALLBACK_PROMPT.format(question=question)
                fallback_response = self.llm.invoke(formatted_prompt)
                response = fallback_response.content
                logger.info("Generated fallback response successfully")
                return response
            except Exception as fallback_error:
                logger.error(f"Fallback response failed: {str(fallback_error)}")
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

        # Check for follow-up indicators
        contains_indicators = any(word in followup_indicators for word in words)
        is_short = len(words) < 10

        # If question is short and contains follow-up indicators, it's likely a follow-up
        result = not (is_short and contains_indicators)

        logger.debug(
            f"Topic analysis: contains_indicators={contains_indicators}, is_short={is_short}, is_new_topic={result}"
        )

        return result

    def reset_conversation(self):
        """Reset the conversation history."""
        logger.info("Resetting conversation history")
        previous_length = len(self.conversation_history)
        self.conversation_history = []
        logger.info(f"Conversation reset (cleared {previous_length} messages)")

    def log_agent_state(self):
        """Log the current state of the agent for debugging purposes."""
        logger.info("===== Agent State =====")
        logger.info(f"Vector store available: {self.vector_store is not None}")
        if self.vector_store:
            doc_count = len(getattr(self.vector_store, "index_to_docstore_id", {}))
            logger.info(f"Vector store document count: {doc_count}")

        logger.info(f"CSV agent available: {self.csv_agent is not None}")
        logger.info(f"Web search available: {self.search_tool is not None}")
        logger.info(f"Total tools available: {len(self.tools)}")
        logger.info(
            f"Conversation history length: {len(self.conversation_history)} messages"
        )
        logger.info(f"Conversation exchanges: {len(self.conversation_history) // 2}")
        logger.info("=======================")


if __name__ == "__main__":
    agent = Agent()
    # logger.info(agent.query("Who is the winner of 2024 Olympic men single tennis?"))
    # logger.info(agent.query("What amenities are included at 99 City Road from PDF?"))
    logger.info(
        agent.query(
            "What is the total proposed Gross Internal Area (GIA) for 99 Bishopsgate?"
        )
    )
    # logger.info(
    #     agent.query(
    #         "How does the proposed GIA of 99 Bishopsgate compare to that of 70 Gracechurch Street?"
    #     )
    # )
