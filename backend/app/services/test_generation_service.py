import json
from typing import List, Any

from .rag_service import RAGService
from ..models import TestCase
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class TestGenerationService:
    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service

    def _clean_llm_json(self, raw_output: str) -> str:
        """
        Handles cases where the LLM wraps JSON in ``` or ```json code fences.
        Returns just the inner JSON string.
        """
        text = raw_output.strip()

        # Strip leading ```json or ``` if present
        if text.startswith("```"):
            # Remove the first ``` (and possible "json")
            parts = text.split("```", 1)
            if len(parts) > 1:
                text = parts[1]
            text = text.lstrip()  # remove newline / spaces
            if text.lower().startswith("json"):
                text = text[4:].lstrip()  # remove 'json' + spaces/newline

        # Strip trailing ``` if present
        if "```" in text:
            text = text.rsplit("```", 1)[0].strip()

        return text

    def _normalize_data_to_list(self, data: Any) -> List[dict]:
        """
        Accepts either:
        - a list of test case dicts
        - a dict like {"TC-001": {...}, "TC-002": {...}}
        and always returns a list of dicts where each has a test_id field.
        """
        # If already a list: ensure each has test_id
        if isinstance(data, list):
            normalized = []
            for idx, tc in enumerate(data, start=1):
                if not isinstance(tc, dict):
                    continue
                if "test_id" not in tc:
                    tc["test_id"] = f"TC-{idx:03d}"
                normalized.append(tc)
            return normalized

        # If dict keyed by test_id
        if isinstance(data, dict):
            normalized = []
            for key, value in data.items():
                if not isinstance(value, dict):
                    continue
                if "test_id" not in value:
                    value["test_id"] = str(key)
                normalized.append(value)
            return normalized

        logger.error(f"Unexpected JSON shape from LLM: {type(data)}")
        return []

    def generate_test_cases(self, query: str) -> List[TestCase]:
        retrieved = self.rag_service.retrieve_context(query)
        context_str = self.rag_service.build_context_block(retrieved)

        prompt = f"""
You are a QA test design assistant.

You MUST base all reasoning ONLY on the context below.
If something is not present in the context, you MUST say you don't know
and you MUST NOT invent features, fields, or UI elements.

Context:
{context_str}

User request:
{query}

Return ONLY a JSON array of test cases.
Do NOT include any explanation, comments, or markdown.

Each test case must have the following fields:
- test_id (string, like "TC-001")
- feature (string)
- scenario (string)
- preconditions (array of strings)
- steps (array of strings)
- expected_result (string)
- grounded_in (array of source document names as strings,
               taken strictly from the "Source: ..." lines in the Context)
"""

        raw_output = self.rag_service.generate_from_context(prompt)
        cleaned = self._clean_llm_json(raw_output)

        # Debug logs (helpful while developing)
        logger.debug(f"Raw LLM output: {raw_output}")
        logger.debug(f"Cleaned JSON string: {cleaned}")

        # 1) Parse JSON
        try:
            data = json.loads(cleaned)
        except Exception as e:
            logger.error(f"Failed to json.loads LLM output: {e}")
            logger.error(f"Raw output: {raw_output}")
            return []

        # 2) Normalize to a list of dicts and convert to TestCase models
        try:
            normalized_list = self._normalize_data_to_list(data)
            test_cases = [TestCase(**tc) for tc in normalized_list]
            return test_cases
        except Exception as e:
            logger.error(f"Failed to convert JSON to TestCase models: {e}")
            logger.error(f"Normalized JSON: {normalized_list}")
            return []
