from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union


class Metric(ABC):
    """Abstract base class for metrics."""

    @abstractmethod
    def __init__(self, name: str, **kwargs):
        """Initialize the metric.

        Args:
            name: Name of the metric.
        """
        self.name = name

    @abstractmethod
    def __call__(
        self,
        predictions: List[str],
        observations: Optional[List[Dict[str, Union[str, int]]]] = None,
    ) -> List[Union[int, float]]:
        """Compute the metric.

        Args:
            predictions: List of responses to check.
            observations: List of observations. This list contains the original queries
                used to generate the responses.

        Returns:
            List of scores.
        """
        pass
