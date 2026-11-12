from enum import StrEnum


class InferenceMemorillaGenerateResponseAnswerGuard(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"

    def __str__(self) -> str:
        return str(self.value)
