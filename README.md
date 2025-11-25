🤖 Autonomous QA Agent – Test Case & Selenium Script Generator
Project Overview

I built an Autonomous QA Agent that generates test cases and Selenium automation scripts automatically from project documentation and the checkout.html page.
The system ingests documents, builds a knowledge base, generates test cases using RAG + Ollama LLM, and converts selected test cases into runnable Selenium Python scripts.

Tech Stack
Component	Technology
Backend	FastAPI
Frontend	Streamlit
AI Model	Ollama – llama3.2:1b
Embeddings	SentenceTransformer
Vector Store	FAISS-style in-memory index
Automation	Selenium + ChromeDriver
Language	Python 3.10
Key Features

✔ Upload project docs + checkout.html
✔ Build vector-based knowledge base
✔ Generate positive & negative test cases grounded only in uploaded docs
✔ Select a test case & generate Selenium automation script
✔ Output is clean JSON for test cases and runnable Python code for Selenium
✔ No hallucinated UI elements or random selectors — script is based strictly on HTML structure

Workflow

Upload support docs (product_specs.md, ui_ux_guide.txt, api_endpoints.json) + checkout.html

Click Build Knowledge Base

Ask the agent:
“Generate all positive and negative test cases for the discount code feature”

Test cases appear in structured JSON format

Select one test case

Click Generate Selenium Script

Download runnable .py script

How to Run
Backend
uvicorn backend.app.main:app --reload

Frontend
streamlit run frontend/streamlit_app.py

Model
ollama pull llama3.2:1b

Deliverables Included

Source Code Repository

Streamlit UI + FastAPI backend

Support Documents (3–4 files)

checkout.html

Demo video covering ingestion → test case generation → script generation

Status

All assignment requirements met:
Ingestion ✔ | Test Case Agent ✔ | Script Generator ✔ | Knowledge Grounding ✔ | Selenium Output ✔ | UI & UX ✔