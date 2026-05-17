"""Mock example for Gemma 4 Reasoner demonstrating thinking mode without a real backend."""

from __future__ import annotations
from typing import Optional
from PIL import Image
from gemma4_reasoner import Gemma4Reasoner, ReasoningConfig, BackendType
from gemma4_reasoner.backends.base import BaseBackend
from gemma4_reasoner.models import ReasoningResult


class MockBackend(BaseBackend):
    """A mock backend that simulates Gemma 4's multimodal reasoning."""

    def generate(
        self,
        prompt: str,
        images: Optional[list[Image.Image]] = None,
        system_prompt: Optional[str] = None,
    ) -> ReasoningResult:
        # Simulate thinking process
        thinking = (
            "1. User provided an image and asked a question.\n"
            "2. I see a solid red square with the text 'TEST' in white.\n"
            "3. The red color is very vibrant.\n"
            "4. The text is located at the top-left corner."
        )
        
        # Simulate final answer
        response = "The image shows a red square with the word 'TEST' written in white text near the top-left corner."
        
        # In a real Gemma 4 model, these would be interleaved with <|channel|> tags
        # and parsed by the base class if needed, but here we return directly.
        return ReasoningResult(
            response=response,
            thinking=thinking if self.config.thinking else None,
            model="mock-gemma4-e4b"
        )


def main():
    print("Initializing Gemma 4 Reasoner with Mock Backend...")
    
    # Configure for thinking mode
    config = ReasoningConfig(
        backend=BackendType.OLLAMA,  # Placeholder, we will override
        thinking=True
    )
    
    reasoner = Gemma4Reasoner(config)
    
    # Inject our mock backend
    reasoner.backend = MockBackend(config)
    
    # Analyze our test image
    print("\n--- Analyzing Image ---")
    image_path = "test_image.jpg"
    result = reasoner.analyze_image(image_path, question="What is in this image?")
    
    if result.has_thinking:
        print("\n[Thinking Process]")
        print(result.thinking)
    
    print("\n[Final Response]")
    print(result.response)
    
    # Document analysis example
    print("\n--- Document Analysis (Chart) ---")
    result = reasoner.analyze_document(image_path, doc_type="chart")
    print(f"Response: {result.response}")


if __name__ == "__main__":
    main()
