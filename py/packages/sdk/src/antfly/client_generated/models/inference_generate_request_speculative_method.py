from enum import Enum


class InferenceGenerateRequestSpeculativeMethod(str, Enum):
    AR = "ar"
    DFLASH = "dflash"

    def __str__(self) -> str:
        return str(self.value)
