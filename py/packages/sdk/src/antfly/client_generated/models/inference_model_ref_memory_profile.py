from enum import Enum


class InferenceModelRefMemoryProfile(str, Enum):
    COMPACT_2GBS = "compact_2gbs"

    def __str__(self) -> str:
        return str(self.value)
