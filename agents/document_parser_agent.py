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
        
        if not file_path:
            return {
                "error": "No file path provided",
                "output": "Error: No file path provided",
                "agent_type": "document_parser"
            }
            
        # Check if the file exists at the provided path
        if not os.path.exists(file_path):
            # Try alternative paths
            input_path = os.path.join("_INPUT", os.path.basename(file_path))
            output_path = os.path.join("_OUTPUT", os.path.basename(file_path))
            
            if os.path.exists(input_path):
                file_path = input_path
            elif os.path.exists(output_path):
                file_path = output_path

        if not file_path or not os.path.exists(file_path):
            # Log all possible paths we tried
            possible_input = os.path.join("_INPUT", os.path.basename(str(file_path)))
            possible_output = os.path.join("_OUTPUT", os.path.basename(str(file_path)))
            
            return {
                "error": f"Input document file not found: {file_path}",
                "output": f"Error: Input document file not found. Checked paths: {file_path}, {possible_input}, {possible_output}",
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