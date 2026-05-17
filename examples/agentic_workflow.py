"""Agentic workflow example — screen understanding + function calling pattern."""

from gemma4_reasoner import Gemma4Reasoner, ReasoningConfig, ModelSize, BackendType
from gemma4_reasoner.tools import ScreenAnalyzer


def navigate_app(reasoner: Gemma4Reasoner, screenshot_path: str, goal: str):
    """Simple agentic navigation loop.

    Analyzes a screenshot, determines the next action, and repeats.
    In a real system, you'd execute the action and take a new screenshot.
    """
    analyzer = ScreenAnalyzer(reasoner)

    print(f"Goal: {goal}")
    print(f"Analyzing: {screenshot_path}\n")

    # Step 1: Describe the current screen
    description = analyzer.describe(screenshot_path)
    print(f"Screen: {description}\n")

    # Step 2: Get interactable elements
    elements = analyzer.get_interactable_elements(screenshot_path)
    print(f"Interactable: {elements}\n")

    # Step 3: Ask the model what to do next
    result = reasoner.chat(
        f"Given the goal '{goal}', the current screen shows: {description}\n"
        f"Available elements: {elements}\n"
        f"What is the single best next action? Respond with just the action."
    )
    print(f"Next action: {result.response}\n")

    # Step 4: Check accessibility
    a11y = analyzer.check_accessibility(screenshot_path)
    print(f"Accessibility: {a11y}")

    return result.response


def main():
    config = ReasoningConfig(
        model_size=ModelSize.E4B,
        backend=BackendType.OLLAMA,
        thinking=True,
    )
    reasoner = Gemma4Reasoner(config)

    navigate_app(
        reasoner,
        screenshot_path="app_screenshot.png",
        goal="Find and click the settings button",
    )


if __name__ == "__main__":
    main()
