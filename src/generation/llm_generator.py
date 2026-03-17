"""
LLM-based Answer Generator (Optional)
Uses retrieved chunks as context for an external LLM.
LLM is used ONLY for natural language generation — not retrieval or learning.
"""

from typing import List, Dict


class LLMGenerator:
    """
    Optional hybrid answer generation using an external LLM.

    Note: The LLM is used ONLY for generating human-readable answers
    from retrieved context. All retrieval and learning is done by
    the NLTK + Keras pipeline.
    """

    def __init__(self, config: dict):
        self.provider = config.get("llm_provider")  # "google" or "openai"
        self.api_key = config.get("llm_api_key")
        self.max_context = config.get("max_context_tokens", 4000)

    def generate(self, query: str, chunks: List[Dict]) -> Dict:
        """
        Generate an answer using LLM with retrieved chunks as context.

        Args:
            query: User's question.
            chunks: Retrieved and ranked chunks.

        Returns:
            Answer dict with text, sources, confidence.
        """
        if not self.api_key:
            return {
                "text": "API key missing. Enable hybrid mode by setting llm_api_key in config.",
                "sources": [],
                "confidence": 0.0
            }

        context_parts = []
        sources = []
        for i, chunk in enumerate(chunks, 1):
            src = f"{chunk.get('source', 'Unknown')} (Page {chunk.get('page', '?')})"
            context_parts.append(f"[{i}] {src}:\n{chunk.get('text', '')}")
            if src not in sources:
                sources.append(src)

        context_str = "\n\n".join(context_parts)
        prompt = f"Context information is below.\n---------------------\n{context_str}\n---------------------\nGiven the context information and not prior knowledge, answer the query.\nQuery: {query}\nAnswer:"

        answer_text = ""
        try:
            if self.provider == "google":
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(prompt)
                answer_text = response.text
            elif self.provider == "openai":
                import openai
                client = openai.OpenAI(api_key=self.api_key)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer questions."},
                        {"role": "user", "content": prompt}
                    ]
                )
                answer_text = response.choices[0].message.content
            else:
                answer_text = f"Unsupported provider: {self.provider}"
        except Exception as e:
            answer_text = f"Error calling LLM API: {str(e)}"

        return {
            "text": answer_text,
            "sources": sources,
            "confidence": chunks[0].get("score", 0.0) if chunks else 0.0
        }
