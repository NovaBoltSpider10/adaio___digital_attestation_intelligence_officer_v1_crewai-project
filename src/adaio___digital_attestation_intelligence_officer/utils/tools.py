from crewai.tools import tool
from crewai_tools import FileReadTool, SerperDevTool
import requests

# Shared in-memory case store
case_store = {}

# Global tools
file_reader = FileReadTool()
search_tool = SerperDevTool()


@tool("Case Log Search")
def search_case_logs(query: str, query_type: str = "case_id"):
    """
    Look up cases. 
    query_type: 'case_id' or 'id_number'.
    """
    if query_type == "case_id":
        return case_store.get(query, "Case ID not found.")

    if query_type == "id_number":
        # Search all cases for matching applicant id_number
        matches = [
            cid for cid, data in case_store.items()
            if data.get("request", {}).get("applicant", {}).get("id_number") == query
        ]
        return matches if matches else "No duplicate cases found."

@tool("Verify Document Existence")
def verify_document_existence(doc_id: str) -> str:
    """
    Calls the mock API endpoint to check if a document reference exists.
    Input should be the exact doc_id (e.g., 'attestation_digital.pdf' or 'MOHE-2026-00084721').
    """
    # Replace with your actual base URL / port
    url = f"http://localhost:8000/mock/documents/{doc_id}"
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return f"SUCCESS: Document '{doc_id}' exists and is valid."
        elif response.status_code == 404:
            return f"ERROR: Document '{doc_id}' was not found on the server."
        else:
            return f"ERROR: Received status code {response.status_code} when verifying '{doc_id}'."
    except Exception as e:
        return f"ERROR: Failed to connect to document server: {str(e)}"