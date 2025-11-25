from pathlib import Path
from sentence_transformers import SentenceTransformer

from .config import settings
from .services.vector_store_service import VectorStoreService
from .services.ingestion_service import IngestionService
from .services.rag_service import RAGService
from .services.test_generation_service import TestGenerationService
from .services.script_generation_service import ScriptGenerationService
from .utils.logging_utils import get_logger

logger = get_logger(__name__)


class StubLLMClient:
    """
    Simple stub LLM client so the app can run without a real LLM.
    Used for the hosted version (Render + Streamlit Cloud) where Ollama
    is not available.
    """

    def generate(self, prompt: str) -> str:
        # If we are asking for test cases, return a fixed JSON array
        if "Return ONLY a JSON array of test cases" in prompt or "Return ONLY a JSON array" in prompt:
            return """
[
  {
    "test_id": "TC-001",
    "feature": "Discount Code",
    "scenario": "Apply valid discount code 'SAVE15'",
    "preconditions": ["User has items in cart"],
    "steps": [
      "Open checkout page",
      "Enter discount code 'SAVE15'",
      "Click 'Apply' button"
    ],
    "expected_result": "Total price is reduced by 15%",
    "grounded_in": ["product_specs.md"]
  },
  {
    "test_id": "TC-002",
    "feature": "Discount Code",
    "scenario": "Apply invalid discount code",
    "preconditions": ["User has items in cart"],
    "steps": [
      "Open checkout page",
      "Enter discount code 'INVALID'",
      "Click 'Apply' button"
    ],
    "expected_result": "Invalid discount code. Total price is unchanged",
    "grounded_in": ["product_specs.md", "ui_ux_guide.txt"]
  }
]
"""
        # If we are asking for Selenium script, return a simple example
        if "You are an expert in Selenium (Python)." in prompt:
            return """from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def test_discount_code():
    driver = webdriver.Chrome()
    driver.maximize_window()
    wait = WebDriverWait(driver, 10)

    try:
        driver.get("http://example.com/checkout")

        code_input = wait.until(EC.visibility_of_element_located((By.ID, "discount-code")))
        code_input.clear()
        code_input.send_keys("SAVE15")

        apply_btn = wait.until(EC.element_to_be_clickable((By.ID, "apply-discount")))
        apply_btn.click()

        success_msg = wait.until(EC.visibility_of_element_located((By.ID, "discount-success")))
        assert "15%" in success_msg.text or "Payment Successful" in success_msg.text

    finally:
        driver.quit()

if __name__ == "__main__":
    test_discount_code()
"""

        # Fallback
        return "[]"


# ---------- Singletons ----------

logger.info("Loading SentenceTransformer embedder...")
embedder = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)

vector_store = VectorStoreService()
llm_client = StubLLMClient()

ingestion_service = IngestionService(vector_store, embedder)
rag_service = RAGService(vector_store, embedder, llm_client)
test_generation_service = TestGenerationService(rag_service)
script_generation_service = ScriptGenerationService(
    rag_service, checkout_html_path=Path(settings.CHECKOUT_HTML_PATH)
)
