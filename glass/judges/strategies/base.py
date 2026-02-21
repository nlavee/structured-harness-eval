from abc import ABC, abstractmethod

from glass.judges.base import Judge


class JudgeStrategy(ABC):
    @abstractmethod
    def assign_judge(self, system_name: str) -> Judge:
        """Assign a judge for the given system."""
        pass
