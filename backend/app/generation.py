from google import genai
from typing import Dict, Optional
import os

class AnswerGenerator:
    def __init__(self, model_name: str = "gemini-3-flash-preview"):
        self.client = None
        self.model_name = model_name

    def _get_client(self):
        if self.client is None:
            self.client = genai.Client()
        return self.client

    def generate_answer(self, query: str, context: str) -> str:
        """
        Generates a grounded answer using the provided context.
        """
        if not context:
            return "I'm sorry, but I couldn't find any relevant information in the uploaded document to answer that question."

        prompt = f"""Answer ONLY from context. If the context is insufficient, say you don't know.
Be precise, grounded, and concise.
When useful, cite page markers like [Page X] that already appear in the context.
Do not invent facts, sources, or interpretations beyond the context.

Context:
{context}

Q: {query}
A:"""

        interaction = self.client.interactions.create(
            model=self.model_name,
            input=prompt
        )
        
        # Access the last output text
        return interaction.outputs[-1].text

if __name__ == "__main__":
    # Smoke test (requires API key)
    # generator = AnswerGenerator()
    # print(generator.generate_answer("What is RAG?", "RAG is Retrieval-Augmented Generation."))
    pass
