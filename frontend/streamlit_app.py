import streamlit as st
import requests

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="Autonomous QA Agent", layout="wide")

st.title("🤖 Autonomous QA Agent – Test Case & Selenium Script Generator")

# ---------------- Phase 1: Build KB ----------------
st.sidebar.header("Phase 1: Build Knowledge Base")
uploaded_files = st.sidebar.file_uploader(
    "Upload support documents + checkout.html",
    accept_multiple_files=True
)

if st.sidebar.button("Build Knowledge Base"):
    if not uploaded_files:
        st.sidebar.error("Please upload at least one file.")
    else:
        files = [
            ("files", (f.name, f.getvalue(), f"type" or "application/octet-stream"))
            for f in uploaded_files
        ]
        with st.spinner("Building knowledge base..."):
            resp = requests.post(f"{API_BASE}/api/build_kb", files=files)
        if resp.status_code == 200:
            data = resp.json()
            st.sidebar.success(data.get("message", "Done"))
            st.sidebar.write(
                f"Documents: {data['num_documents']}, Chunks: {data['num_chunks']}"
            )
        else:
            st.sidebar.error("Failed to build knowledge base.")

# ---------------- Phase 2: Test Case Generation ----------------
st.header("Phase 2: Test Case Generation")
default_query = "Generate all positive and negative test cases for the discount code feature."
query = st.text_area("Ask the agent for test cases", default_query, height=120)

if "test_cases" not in st.session_state:
    st.session_state["test_cases"] = []

if st.button("Generate Test Cases"):
    with st.spinner("Generating test cases..."):
        resp = requests.post(
            f"{API_BASE}/api/generate_test_cases",
            json={"query": query, "output_format": "json"}
        )
    if resp.status_code == 200:
        data = resp.json()
        st.session_state["test_cases"] = data.get("test_cases", [])
        st.success(f"Generated {len(st.session_state['test_cases'])} test cases.")
        st.json(data)
    else:
        st.error("Failed to generate test cases.")

# ---------------- Phase 3: Script Generation ----------------
test_cases = st.session_state.get("test_cases", [])
if test_cases:
    st.header("Phase 3: Selenium Script Generation")
    options = {tc["test_id"]: tc for tc in test_cases}
    selected_id = st.selectbox("Select a test case", list(options.keys()))

    if st.button("Generate Selenium Script"):
        payload = {
            "test_case_id": selected_id,
            "test_case": options[selected_id],
        }
        with st.spinner("Generating Selenium script..."):
            resp = requests.post(
                f"{API_BASE}/api/generate_selenium_script",
                json=payload
            )
        if resp.status_code == 200:
            script = resp.json().get("script", "")
            st.success("Selenium script generated.")
            st.code(script, language="python")

            # ✅ Download button
            st.download_button(
                label="Download script as generated_test_case.py",
                data=script,
                file_name=f"{selected_id.lower()}_selenium_test.py",
                mime="text/x-python",
            )
        else:
            st.error("Failed to generate script.")
else:
    st.info("Generate test cases first to enable script generation.")
