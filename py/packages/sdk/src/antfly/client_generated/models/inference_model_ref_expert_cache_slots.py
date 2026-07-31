from enum import IntEnum


class InferenceModelRefExpertCacheSlots(IntEnum):
    VALUE_8 = 8
    VALUE_12 = 12
    VALUE_16 = 16

    def __str__(self) -> str:
        return str(self.value)
