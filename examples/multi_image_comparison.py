"""Multi-image comparison example."""

from gemma4_reasoner import Gemma4Reasoner, ReasoningConfig, ModelSize, BackendType


def main():
    config = ReasoningConfig(
        model_size=ModelSize.E4B,
        backend=BackendType.OLLAMA,
    )
    reasoner = Gemma4Reasoner(config)

    # Compare before/after
    result = reasoner.compare_images(
        images=["before.png", "after.png"],
        question="What changed between these two images? Be specific.",
    )
    print(f"Comparison:\n{result.response}")
    print(f"\nVision tokens used: {result.total_vision_tokens}")

    # Compare multiple products
    result = reasoner.compare_images(
        images=["product_a.png", "product_b.png", "product_c.png"],
        question="Compare these products. What are the key differences in design, color, and features?",
    )
    print(f"\nProduct comparison:\n{result.response}")


if __name__ == "__main__":
    main()
