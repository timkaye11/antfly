from enum import StrEnum


class MemorillaProfileConfigReaderKind(StrEnum):
    ENCODER_LORA_RESEARCH = "encoder_lora_research"
    HYBRID_RESEARCH = "hybrid_research"
    RELEASED = "released"

    def __str__(self) -> str:
        return str(self.value)
