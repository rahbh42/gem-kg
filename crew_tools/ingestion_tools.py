# crew_tools/ingestion_tools.py
import logging
import ast
import re
from typing import Any
from langchain_community.llms import CTransformers
from langchain_community.graphs import Neo4jGraph
from crewai_tools import BaseTool
from transformers import pipeline

class KnowledgeExtractorTool(BaseTool):
    name: str = "Knowledge Extractor and Refiner Tool"
    description: str = "Extracts, refines, and saves KG triples from a single chunk of text."
    
    rebel_model: Any = None
    llm: Any = None
    
    def __init__(self, llm: CTransformers):
        super().__init__()
        self.rebel_model = pipeline("text2text-generation", model="/app/models/rebel-large") 
        self.llm = llm

    def _extract_triples(self, text: str):
        """Extracts raw triples from a single text chunk."""
        triples = []
        # The input `text` is already a single, manageable chunk.
        # We split it by newlines just in case the chunk itself contains multiple paragraphs.
        text_lines = [line for line in text.split("\n") if line.strip()]
        if not text_lines:
            return []
        
        try:
            # Process lines in batches for efficiency
            extracted_text = self.rebel_model(text_lines, return_tensors=True, return_text=False)
            decoded_text = self.rebel_model.tokenizer.batch_decode([gen_ids[0] for gen_ids in extracted_text])

            for sentence in decoded_text:
                for triple_str in sentence.split("<triplet>"):
                    if "<subj>" not in triple_str:
                        continue
                    try:
                        head = triple_str.split("<subj>")[1].split("<obj>")[0].strip()
                        obj = triple_str.split("<obj>")[1].split("<pred>")[0].strip()
                        rel = triple_str.split("<pred>")[1].replace("</triplet>", "").strip()
                        if head and obj and rel:
                            triples.append({"head": head, "type": rel, "tail": obj})
                    except IndexError:
                        continue
        except Exception as e:
            logging.error(f"Error during triple extraction: {e}")
        return triples

    def _refine_triples(self, triples: list):
        """Refines a list of raw triples using the LLM."""
        if not triples:
            return []
        triples_str = "\n".join([f"- {t['head']} | {t['type']} | {t['tail']}" for t in triples])
        prompt = f"""You are a Knowledge Graph expert. Refine this list of triples: normalize entities, canonicalize relations to SNAKE_CASE, and remove duplicates. Return ONLY a Python list of dictionaries.
        Raw Triples:\n{triples_str}"""
        try:
            refined_str = self.llm.invoke(prompt)
            # Use regex to robustly find the list within the LLM's output
            match = re.search(r"\[.*\]", refined_str, re.DOTALL)
            return ast.literal_eval(match.group(0)) if match else triples
        except Exception as e:
            logging.error(f"Error during triple refinement, returning raw triples: {e}")
            return triples

    def _run(self, document_chunk: str, overwrite_graph: bool = False) -> str:
        """The main execution method for processing one chunk."""
        graph = Neo4jGraph()
        if overwrite_graph:
            logging.info("Overwriting existing graph...")
            graph.query("MATCH (n) DETACH DELETE n")

        raw_triples = self._extract_triples(document_chunk)
        if not raw_triples:
            return "No knowledge was extracted from this chunk."
        
        refined_triples = self._refine_triples(raw_triples)
        
        for triple in refined_triples:
            graph.query(
                "MERGE (h:Entity {name: $head}) MERGE (t:Entity {name: $tail}) MERGE (h)-[r:RELATION {type: $type}]->(t)",
                triple,
            )
        return f"Successfully added {len(refined_triples)} triples from this chunk."