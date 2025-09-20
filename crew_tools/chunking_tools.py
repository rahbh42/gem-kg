# crew_tools/chunking_tools.py
from crewai_tools import BaseTool
import requests
import os

class DocumentChunkingTool(BaseTool):
    name: str = "Document Chunking Tool"
    description: str = "Uploads a document to the chunking service and returns a list of text chunks."

    def _run(self, file_path: str) -> list[str]:
        """Uploads a file to the unstructured service and gets the chunks."""

        # The service is available at this hostname inside the Docker network
        unstructured_service_url = "http://unstructured_service:8008/chunk"

        try:
            with open(file_path, "rb") as f:
                files = {"file": (os.path.basename(file_path), f)}
                response = requests.post(unstructured_service_url, files=files)

            response.raise_for_status() # Raise an exception for bad status codes

            return response.json().get("chunks", [])
        except requests.exceptions.RequestException as e:
            return [f"Error calling the chunking service: {e}"]