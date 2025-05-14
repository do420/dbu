import os
from typing import Dict, Any
from .base import BaseAgent

import PyPDF2
import docx

class DocumentParserAgent(BaseAgent):
    """Agent that parses PDF and Word files and outputs their text."""

    async def process(self, input_data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Args:
            input_data: Path to the document file (PDF or DOCX)
        Returns:
            Dict with extracted text
        """
        context = context or {}
        file_path = input_data if isinstance(input_data, str) else input_data.get("file_path")
        if not file_path or not os.path.exists(file_path):
            return {
                "output": f"File not found: {file_path}",
                "error": "File not found",
                "agent_type": "document_parser"
            }

        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == ".pdf":
                text = self._parse_pdf(file_path)
            elif ext in [".docx", ".doc"]:
                text = self._parse_docx(file_path)
            else:
                return {
                    "output": f"Unsupported file type: {ext}",
                    "error": "Unsupported file type",
                    "agent_type": "document_parser"
                }
        except Exception as e:
            return {
                "output": f"Error parsing document: {str(e)}",
                "error": str(e),
                "agent_type": "document_parser"
            }

        return {
            "output": text,
            "agent_type": "document_parser",
            "file_path": file_path
        }

    def _parse_pdf(self, file_path: str) -> str:
        text = ""
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text.strip()

    def _parse_docx(self, file_path: str) -> str:
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs]).strip()