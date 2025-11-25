from pydantic import BaseModel
from typing import List


class BuildKBResponse(BaseModel):
    status: str
    num_documents: int
    num_chunks: int
    message: str


class TestCase(BaseModel):
    test_id: str
    feature: str
    scenario: str
    preconditions: List[str]
    steps: List[str]
    expected_result: str
    grounded_in: List[str]


class GenerateTestCasesRequest(BaseModel):
    query: str
    output_format: str = "json"  # or "markdown"


class GenerateTestCasesResponse(BaseModel):
    test_cases: List[TestCase]


class GenerateScriptRequest(BaseModel):
    test_case_id: str
    test_case: TestCase


class GenerateScriptResponse(BaseModel):
    script: str
