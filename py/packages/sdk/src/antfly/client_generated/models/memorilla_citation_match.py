from enum import StrEnum


class MemorillaCitationMatch(StrEnum):
    CASE_INSENSITIVE = "case_insensitive"
    EXACT = "exact"
    NONE = "none"

    def __str__(self) -> str:
        return str(self.value)
