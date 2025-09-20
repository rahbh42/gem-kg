# app.py
import os
import streamlit as st
import logging
from dotenv import load_dotenv

# Import the new, separated crews
from crew_logic import create_chunking_crew, create_ingestion_crew, create_query_crew
from crew_tools.graph_visualization_tool import GraphVisualizerTool

load_dotenv()
UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Configure the page
st.set_page_config(page_title="Agentic KG Chatbot", layout="wide")
st.title("🤖 Agentic KG Chatbot (CrewAI & Unstructured)")

# --- Sidebar for Controls ---
with st.sidebar:
    st.header("Controls")

    uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt"])
    overwrite_kg = st.toggle("Overwrite existing KG", value=False)

    if st.button("📄 Ingest Document"):
        if uploaded_file:
            # --- THIS IS THE NEW BACKEND CHUNKING WORKFLOW ---
            file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.status("Step 1: Chunking document...", expanded=True) as status:
                try:
                    chunking_crew = create_chunking_crew()
                    chunks = chunking_crew.kickoff(inputs={"file_path": file_path})
                    status.update(label=f"Document split into {len(chunks)} chunks.", state="complete")
                except Exception as e:
                    st.error(f"Failed to chunk the document. Error: {e}")
                    st.stop()

            # 3. Process each chunk sequentially
            with st.spinner("Step 2: Ingesting chunks..."):
                ingestion_crew = create_ingestion_crew()
                progress_bar = st.progress(0, text="Initializing ingestion...")

                for i, chunk in enumerate(chunks):
                    try:
                        # Only overwrite the graph on the very first chunk
                        should_overwrite = overwrite_kg if i == 0 else False

                        inputs = {
                            "document_chunk": chunk,
                            "overwrite_graph": should_overwrite
                        }
                        result = ingestion_crew.kickoff(inputs=inputs)

                        # Update progress bar and status
                        progress_percentage = (i + 1) / len(chunks)
                        progress_bar.progress(progress_percentage, text=f"Processing chunk {i+1}/{len(chunks)}...")

                    except Exception as e:
                        st.error(f"Failed to process chunk {i+1}. Error: {e}")
                        logging.error(f"Failed on chunk {i+1}: {chunk}")
                        break # Stop the process if a chunk fails

                progress_bar.empty() # Clear the progress bar
                st.success("Ingestion complete! The Knowledge Graph has been updated.")
        else:
            st.warning("Please upload a document first.")

    if st.button("📊 Visualize Graph"):
        # (This section remains the same)
        # ...
        with st.spinner("Generating visualization..."):
            try:
                visualizer = GraphVisualizerTool()
                file_path = visualizer.run()
                with open(file_path, 'r', encoding='utf-8') as f:
                    st.session_state.graph_html = f.read()
                st.success("Visualization created!")
            except Exception as e:
                st.error(f"Failed to create visualization: {e}")

# --- Main Content Area ---
if "graph_html" in st.session_state:
    with st.expander("Knowledge Graph Visualization", expanded=True):
        st.components.v1.html(st.session_state.graph_html, height=750, scrolling=True)

st.divider()

# Chat Interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Crew at work: Answering question..."):
            query_crew = create_query_crew()
            try:
                result = query_crew.kickoff(inputs={"question": prompt})
                st.markdown(result)
                st.session_state.messages.append({"role": "assistant", "content": result})
            except Exception as e:
                error_message = f"Query failed. The agent may be stuck. Error: {e}"
                st.error(error_message)
                logging.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})