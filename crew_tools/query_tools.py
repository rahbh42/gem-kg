# crew_tools/query_tools.py
from typing import Any
from langchain_community.llms import CTransformers
from langchain_community.graphs import Neo4jGraph
from crewai_tools import BaseTool

class CypherQueryTool(BaseTool):
    name: str = "Cypher Query Tool"
    description: str = "Generates and executes a Cypher query to answer a user's question from the knowledge graph."
    llm: Any = None
    graph: Any = None

    # The tool now ACCEPTS an LLM during initialization
    def __init__(self, llm: CTransformers):
        super().__init__()
        self.llm = llm # Use the provided, correctly configured LLM
        self.graph = Neo4jGraph()

    def _run(self, user_query: str) -> str:
        cypher_prompt = f"""You are a Neo4j expert. Given a question, generate a Cypher query to answer it from a graph with nodes labeled 'Entity' and relationships labeled 'RELATION'.
        Schema: {self.graph.schema}
        Question: {user_query}
        Cypher Query:"""
        
        cypher_query = self.llm.invoke(cypher_prompt)
        
        try:
            results = self.graph.query(cypher_query)
        except Exception:
            # If the Cypher query is invalid, try to recover
            results = "The generated Cypher query was invalid. Please try rephrasing the question."

        response_prompt = f"""Based on the following information, provide a natural language answer to the user's question. If the results indicate an error or are empty, say that you could not find an answer.
        Question: {user_query}
        Query Results: {results}
        Answer:"""

        final_answer = self.llm.invoke(response_prompt)
        return final_answer