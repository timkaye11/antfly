from enum import StrEnum


class MemorillaGenerateResponseRecordGate(StrEnum):
    CLOSED = "closed"
    OPEN = "open"

    def __str__(self) -> str:
        return str(self.value)
