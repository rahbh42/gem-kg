# crew_logic.py
from crewai import Agent, Task, Crew
from langchain_community.llms import CTransformers
from crew_tools.chunking_tools import DocumentChunkingTool
from crew_tools.ingestion_tools import KnowledgeExtractorTool
from crew_tools.query_tools import CypherQueryTool

# --- Create ONE central, correctly configured LLM ---
shared_llm = CTransformers(
    model="/app/models/mistral-7b-instruct.gguf",
    model_type="mistral",
    config={'max_new_tokens': 1024, 'temperature': 0.1, 'context_length': 4096}
)

# --- Instantiate all tools ---
chunking_tool = DocumentChunkingTool()
knowledge_extractor_tool = KnowledgeExtractorTool(llm=shared_llm)
cypher_query_tool = CypherQueryTool(llm=shared_llm)

# --- CREW 1: Document Chunking ---
def create_chunking_crew():
    chunker_agent = Agent(
        role='Document Processing Specialist',
        goal='Read a document and split it into a list of semantic text chunks.',
        backstory='An expert in using the Unstructured library to partition and chunk complex documents.',
        tools=[chunking_tool],
        llm=shared_llm,
        verbose=True
    )
    chunking_task = Task(
        description="Process the document specified by the 'file_path' input to generate text chunks.",
        expected_output="A Python list of strings, where each string is a semantic chunk of the document.",
        agent=chunker_agent,
    )
    return Crew(agents=[chunker_agent], tasks=[chunking_task], verbose=2)

# --- CREW 2: Single Chunk Ingestion ---
def create_ingestion_crew():
    kg_builder = Agent(
        role='Knowledge Graph Engineer',
        goal="Extract, refine, and structure information from a single text chunk into a knowledge graph.",
        backstory="A specialist in identifying entities and relationships.",
        tools=[knowledge_extractor_tool],
        llm=shared_llm,
        verbose=True
    )
    build_task = Task(
        description="Process a single 'document_chunk' to build the knowledge graph. The 'overwrite_graph' setting is also provided.",
        expected_output="A confirmation message indicating the number of triples added from the chunk.",
        agent=kg_builder,
    )
    return Crew(agents=[kg_builder], tasks=[build_task], verbose=2)

# --- CREW 3: Querying ---
def create_query_crew():
    # (This crew remains the same)
    query_agent = Agent(
        role='Knowledge Graph Query Specialist',
        goal='Answer user questions by querying the Neo4j knowledge graph.',
        backstory='An expert in natural language processing and Cypher query generation.',
        tools=[cypher_query_tool],
        llm=shared_llm,
        verbose=True
    )
    query_task = Task(
        description="Answer the user's question, which will be provided in the inputs.",
        expected_output="A clear, concise, and natural language answer.",
        agent=query_agent
    )
    return Crew(agents=[query_agent], tasks=[query_task], verbose=2)