from dataclasses import asdict
from pprint import pprint
from typing import List

from utils import (
    ConfusionMatrix,
    render_confusion_matrix,
    generate_data,
    Label,
    LabelMetrics,
)


class Evaluator:
    def __init__(self, actuals: List[Label], predictions: List[Label]):
        """Initialize an Evaluator using two lists representing actual labels and predictions for some samples."""
        self.actuals = actuals
        self.predictions = predictions
        self.matrix = ConfusionMatrix()

    @staticmethod
    def build_matrix(actuals: List[Label], predictions: List[Label]) -> ConfusionMatrix:
        """Builds and return the ConfusionMatrix."""
        raise NotImplementedError

    def calculate_metrics(self) -> List[LabelMetrics]:
        """Calculates and returns metrics."""
        raise NotImplementedError


def main():
    # Generate some fake data.
    actuals, predictions = generate_data()

    # Test Evaluator.
    evaluator = Evaluator(actuals, predictions)
    print(render_confusion_matrix(evaluator.matrix))
    for metric in evaluator.calculate_metrics():
        pprint(asdict(metric))


if __name__ == "__main__":
    main()
