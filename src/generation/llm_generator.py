"""
LLM-based answer generation for the RAG pipeline.
Uses retrieved chunks as context for an LLM.
"""

import os
import json
import urllib.error
import urllib.request
from typing import Dict, List

from src.generation.context_optimizer import ContextOptimizer


class LLMGenerator:
    """
    Hybrid answer generation using an external LLM.

    Retrieval stays local and document-grounded. The LLM is used to
    synthesize a natural-language answer from retrieved context.
    """

    def __init__(self, config: dict):
        self.provider = config.get("llm_provider", "openai")
        self.api_key = (
            config.get("llm_api_key")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("GEMINI_API_KEY")
        )
        self.max_context = config.get("max_context_tokens", 4000)
        self.openai_model = config.get("openai_model", "gpt-4o-mini")
        self.google_model = config.get("google_model", "gemini-1.5-flash")
        self.ollama_model = config.get("ollama_model", "phi3:mini")
        self.ollama_base_url = config.get("ollama_base_url", "http://localhost:11434")

    def generate(self, query: str, chunks: List[Dict]) -> Dict:
        """Generate an answer from retrieved chunks using an LLM."""
        if not chunks:
            return {
                "text": "Answer not found - no relevant passages above confidence threshold.",
                "sources": [],
                "confidence": 0.0,
                "used_llm": False,
                "fallback_reason": "no_context",
            }

        if self.provider in {"openai", "google"} and not self.api_key:
            return {
                "text": "LLM API key missing.",
                "sources": [],
                "confidence": 0.0,
                "used_llm": False,
                "fallback_reason": "missing_api_key",
            }

        optimized_chunks = ContextOptimizer(self.max_context).optimize(chunks)
        context_parts = []
        sources = []
        for i, chunk in enumerate(optimized_chunks, 1):
            src = f"{chunk.get('source', 'Unknown')} (Page {chunk.get('page', '?')})"
            context_parts.append(f"[{i}] {src}:\n{chunk.get('text', '')}")
            if src not in sources:
                sources.append(src)

        system_prompt = (
            "You are a document question-answering assistant. "
            "Answer only from the provided context. "
            "If the answer is not supported by the context, say that the answer is not available in the documents. "
            "Start with a direct answer, then add a short explanation only if useful. "
            "Do not invent facts and do not rely on outside knowledge."
        )
        user_prompt = (
            "Use the context below to answer the question.\n\n"
            "Context:\n"
            "---------------------\n"
            f"{chr(10).join(context_parts)}\n"
            "---------------------\n"
            f"Question: {query}\n\n"
            "Return a concise answer grounded in the context."
        )

        try:
            answer_text = self._run_chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
        except Exception as exc:
            return {
                "text": f"LLM generation failed: {str(exc)}",
                "sources": sources,
                "confidence": optimized_chunks[0].get("score", 0.0),
                "used_llm": False,
                "fallback_reason": "llm_error",
            }

        if not answer_text:
            return {
                "text": "LLM returned an empty answer.",
                "sources": sources,
                "confidence": optimized_chunks[0].get("score", 0.0),
                "used_llm": False,
                "fallback_reason": "empty_llm_response",
            }

        return {
            "text": answer_text,
            "sources": sources,
            "confidence": optimized_chunks[0].get("score", 0.0),
            "used_llm": True,
        }

    def chat(self, prompt: str, system_prompt: str = None) -> Dict:
        """Run a general-purpose chat request without document retrieval."""
        system_prompt = system_prompt or (
            "You are a sharp AI research assistant. "
            "Give concise, clear, and useful answers. "
            "If asked about the uploaded documents, explain that no relevant document context was provided."
        )

        if self.provider in {"openai", "google"} and not self.api_key:
            return {
                "text": "LLM API key missing.",
                "sources": [],
                "confidence": 0.0,
                "used_llm": False,
                "fallback_reason": "missing_api_key",
            }

        try:
            answer_text = self._run_chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
            )
        except Exception as exc:
            return {
                "text": f"LLM generation failed: {str(exc)}",
                "sources": [],
                "confidence": 0.0,
                "used_llm": False,
                "fallback_reason": "llm_error",
            }

        return {
            "text": answer_text,
            "sources": [],
            "confidence": 0.0,
            "used_llm": True,
        }

    def _run_chat(self, messages: List[Dict[str, str]]) -> str:
        """Dispatch a chat request to the configured provider."""
        if self.provider == "google":
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.google_model)
            response = model.generate_content([message["content"] for message in messages])
            answer_text = (response.text or "").strip()
        elif self.provider == "openai":
            import openai

            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.openai_model,
                messages=messages,
            )
            answer_text = (response.choices[0].message.content or "").strip()
        elif self.provider == "ollama":
            answer_text = self._generate_with_ollama(messages)
        else:
            raise RuntimeError(f"Unsupported LLM provider: {self.provider}")

        if not answer_text:
            raise RuntimeError("The configured LLM returned an empty response.")
        return answer_text

    def _generate_with_ollama(self, messages: List[Dict[str, str]]) -> str:
        """Generate an answer using a local Ollama model."""
        payload = {
            "model": self.ollama_model,
            "messages": messages,
            "stream": False,
        }

        request = urllib.request.Request(
            f"{self.ollama_base_url.rstrip('/')}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Could not reach Ollama. Make sure the Ollama server is running."
            ) from exc

        message = body.get("message", {})
        answer_text = (message.get("content") or "").strip()
        if not answer_text:
            raise RuntimeError("Ollama returned an empty response.")
        return answer_text
