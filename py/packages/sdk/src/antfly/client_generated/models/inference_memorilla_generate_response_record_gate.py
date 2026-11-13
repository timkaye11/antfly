from enum import StrEnum


class InferenceMemorillaGenerateResponseRecordGate(StrEnum):
    CLOSED = "closed"
    OPEN = "open"

    def __str__(self) -> str:
        return str(self.value)
