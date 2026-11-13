from enum import IntEnum


class MemorillaGenerateResponseMemoryTokens(IntEnum):
    VALUE_0 = 0
    VALUE_16 = 16
    VALUE_32 = 32

    def __str__(self) -> str:
        return str(self.value)
