import os
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from googleapiclient.discovery import build
from langchain_core.tools import Tool

load_dotenv()


class GoogleSearchTool:
    """A tool for searching the web using Google Search API."""

    def __init__(self, api_key: Optional[str] = None, cse_id: Optional[str] = None):
        """
        Initialize the Google Search tool.

        Args:
            api_key: Google API key (if None, loads from environment)
            cse_id: Google Custom Search Engine ID (if None, loads from environment)
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.cse_id = cse_id or os.getenv("GOOGLE_CSE_ID")

        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        if not self.cse_id:
            raise ValueError("GOOGLE_CSE_ID environment variable not set")

    def search(self, query: str, num_results: int = 2) -> List[Dict[str, Any]]:
        """
        Search Google for results.

        Args:
            query: Search query
            num_results: Number of results to return

        Returns:
            List of search results
        """
        service = build("customsearch", "v1", developerKey=self.api_key)
        results = service.cse().list(q=query, cx=self.cse_id, num=num_results).execute()

        if "items" not in results:
            return []

        search_results = []
        for item in results["items"]:
            search_results.append(
                {
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "source": "google_search",
                }
            )

        return search_results

    def get_langchain_tool(self) -> Tool:
        """
        Get a LangChain tool for use with agents.

        Returns:
            LangChain Tool
        """
        return Tool(
            name="google_search",
            description="Search Google for information about a query. Useful for finding up-to-date information that might not be in the document database.",
            func=self.search,
        )
