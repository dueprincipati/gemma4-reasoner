"""Basic usage examples for the Gemma 4 Reasoner."""

from gemma4_reasoner import Gemma4Reasoner, ReasoningConfig, ModelSize, BackendType


def main():
    # Initialize
    config = ReasoningConfig(
        model_size=ModelSize.E4B,
        backend=BackendType.OLLAMA,
        thinking=True,
    )
    reasoner = Gemma4Reasoner(config)

    # 1. Simple image analysis
    print("=== Image Analysis ===")
    result = reasoner.analyze_image(
        "photo.jpg",
        question="What is the main subject of this image?",
    )
    print(f"Response: {result.response}")
    if result.has_thinking:
        print(f"Thinking: {result.thinking[:200]}...")

    # 2. Document analysis
    print("\n=== Chart Analysis ===")
    result = reasoner.analyze_document("chart.png", doc_type="chart")
    print(f"Response: {result.response}")

    # 3. Math problem
    print("\n=== Math Problem ===")
    result = reasoner.solve_math("math_problem.jpg")
    print(f"Answer: {result.response}")
    print(f"Work: {result.thinking}")

    # 4. Multi-turn chat
    print("\n=== Chat ===")
    r1 = reasoner.chat("What's in this image?", images=["scene.jpg"])
    print(f"Turn 1: {r1.response}")
    r2 = reasoner.chat("What colors are most prominent?")
    print(f"Turn 2: {r2.response}")


if __name__ == "__main__":
    main()
