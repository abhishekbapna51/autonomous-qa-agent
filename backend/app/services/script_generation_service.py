from pathlib import Path
import json

from .rag_service import RAGService
from ..models import TestCase
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class ScriptGenerationService:
    def __init__(self, rag_service: RAGService, checkout_html_path: Path):
        self.rag_service = rag_service
        self.checkout_html_path = checkout_html_path

        if self.checkout_html_path.exists():
            self.checkout_html = self.checkout_html_path.read_text(encoding="utf-8")
        else:
            logger.warning(
                f"checkout.html not found at {self.checkout_html_path}, HTML context will be empty."
            )
            self.checkout_html = ""

    def generate_script(self, test_case: TestCase) -> str:
        # ✅ Pydantic v2: use model_dump / model_dump_json
        tc_json_compact = test_case.model_dump_json()
        tc_json_pretty = json.dumps(test_case.model_dump(), indent=2)

        query = f"Generate Selenium Python script for this test case: {tc_json_compact}"
        retrieved = self.rag_service.retrieve_context(query)
        docs_context = self.rag_service.build_context_block(retrieved)

        prompt = f"""
You are an expert in Selenium (Python).

Use ONLY the following HTML and documentation context to write the test script.
You MUST NOT invent any HTML elements or IDs that do not exist in the provided HTML.

HTML (checkout.html):
{self.checkout_html}

Additional context from docs:
{docs_context}

Test case (JSON):
{tc_json_pretty}

Known important element IDs and names (use them if present in the HTML):
- discount-code, apply-discount, discount-success, discount-error
- qty-item-a, qty-item-b
- name, email, address
- pay-now, payment-message

Requirements:
- Use Selenium WebDriver with Chrome (webdriver.Chrome()).
- Navigate to a placeholder URL: "http://localhost:8000/checkout".
- Implement the steps from the test case in order.
- Use clear, robust locators (By.ID, By.NAME, By.CSS_SELECTOR) that exist in the HTML.
- Add at least one assertion that verifies the 'expected_result' from the test case.
- Include proper setup and teardown: create driver, run test, then driver.quit().
- Code must be fully executable.
- Return ONLY Python code, with no backticks, no markdown, and no explanation text.
"""
        script = self.rag_service.generate_from_context(prompt)
        return script
