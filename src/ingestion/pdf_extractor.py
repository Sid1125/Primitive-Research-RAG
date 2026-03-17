"""
PDF Text Extraction Module
Extracts text from PDF files with page-level metadata using PyMuPDF.
"""

import os
import fitz  # PyMuPDF
from typing import List, Dict


class PDFExtractor:
    """Extract text content from PDF documents with page metadata."""

    def extract(self, pdf_path: str) -> List[Dict]:
        """
        Extract text from a single PDF file.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of dicts: [{"text": str, "page_num": int, "source": str}, ...]
        """
        pages = []
        doc = fitz.open(pdf_path)
        filename = os.path.basename(pdf_path)

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            if text.strip():
                pages.append({
                    "text": text.strip(),
                    "page_num": page_num + 1,  # 1-indexed
                    "source": filename,
                })

        doc.close()
        return pages

    def extract_all(self, directory: str) -> List[Dict]:
        """
        Extract text from all PDFs in a directory.

        Args:
            directory: Path to directory containing PDF files.

        Returns:
            Flat list of page-level dicts across all PDFs.
        """
        all_pages = []
        pdf_files = [f for f in os.listdir(directory) if f.lower().endswith(".pdf")]

        if not pdf_files:
            print(f"[WARN] No PDF files found in {directory}")
            return all_pages

        for pdf_file in sorted(pdf_files):
            pdf_path = os.path.join(directory, pdf_file)
            print(f"  Extracting: {pdf_file}")
            pages = self.extract(pdf_path)
            all_pages.extend(pages)

        return all_pages
