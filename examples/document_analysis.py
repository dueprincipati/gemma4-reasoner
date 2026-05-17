"""Document analysis examples."""

from gemma4_reasoner import Gemma4Reasoner, ReasoningConfig, ModelSize, BackendType
from gemma4_reasoner.tools import ChartAnalyzer, DocumentParser


def main():
    config = ReasoningConfig(
        model_size=ModelSize.E4B,
        backend=BackendType.OLLAMA,
    )
    reasoner = Gemma4Reasoner(config)

    # Using specialized tools
    charts = ChartAnalyzer(reasoner)
    docs = DocumentParser(reasoner)

    # Chart analysis
    print("=== Chart ===")
    print(charts.analyze("sales_chart.png").response)
    print("\nData extraction:")
    print(charts.extract_data("sales_chart.png"))

    # Table parsing
    print("\n=== Table ===")
    print(docs.parse_table("data_table.png"))

    # Receipt parsing
    print("\n=== Receipt ===")
    print(docs.parse_receipt("receipt.jpg"))

    # Handwriting
    print("\n=== Handwriting ===")
    print(docs.transcribe_handwriting("note.jpg"))


if __name__ == "__main__":
    main()
