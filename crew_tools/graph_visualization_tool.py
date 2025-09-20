# crew_tools/graph_visualization_tool.py
import os
from pyvis.network import Network
from langchain_community.graphs import Neo4jGraph
from crewai_tools import BaseTool

class GraphVisualizerTool(BaseTool):
    name: str = "Graph Visualizer Tool"
    description: str = "Generates an interactive HTML visualization of the knowledge graph."

    def _run(self) -> str:
        graph = Neo4jGraph()
        query_result = graph.query("MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 50")

        net = Network(height="750px", width="100%", notebook=True, cdn_resources="in_line")
        added_nodes = set()
        for record in query_result:
            source_node, rel, target_node = record['n'], record['r'], record['m']
            source_id, target_id, rel_type = source_node.get('name', ''), target_node.get('name', ''), rel.type
            if source_id not in added_nodes:
                net.add_node(source_id, label=source_id)
                added_nodes.add(source_id)
            if target_id not in added_nodes:
                net.add_node(target_id, label=target_id)
                added_nodes.add(target_id)
            net.add_edge(source_id, target_id, label=rel_type)

        file_path = os.path.join("data", "knowledge_graph.html")
        net.save_graph(file_path)
        return file_path