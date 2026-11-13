from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define

from ..models.memorilla_citation_match import MemorillaCitationMatch

T = TypeVar("T", bound="MemorillaCitation")


@_attrs_define
class MemorillaCitation:
    """A region of one caller document that the copy head copied into the memory record (encoder_lora_research profiles).

    Attributes:
        document_index (int): Index into the request's document_texts.
        start (int): Inclusive UTF-8 byte offset into document_texts[document_index]. Offsets count bytes of the UTF-8
            encoding, not characters or UTF-16 code units, and always fall on character boundaries.
        end (int): Exclusive UTF-8 byte offset into document_texts[document_index]; end > start.
        text (str): Exactly the bytes document_texts[document_index][start:end]. The region covers the copied tokens
            widened to whole characters, with surrounding whitespace trimmed.
        supports_answer (bool): True when the published answer occurs in text (match is exact or case_insensitive).
            Always false when the published answer is UNKNOWN, including a guard abstention.
        match (MemorillaCitationMatch): How the published answer occurs in text; case_insensitive ignores ASCII case
            only.
    """

    document_index: int
    start: int
    end: int
    text: str
    supports_answer: bool
    match: MemorillaCitationMatch

    def to_dict(self) -> dict[str, Any]:
        document_index = self.document_index

        start = self.start

        end = self.end

        text = self.text

        supports_answer = self.supports_answer

        match = self.match.value

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "document_index": document_index,
                "start": start,
                "end": end,
                "text": text,
                "supports_answer": supports_answer,
                "match": match,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        document_index = d.pop("document_index")

        start = d.pop("start")

        end = d.pop("end")

        text = d.pop("text")

        supports_answer = d.pop("supports_answer")

        match = MemorillaCitationMatch(d.pop("match"))

        memorilla_citation = cls(
            document_index=document_index,
            start=start,
            end=end,
            text=text,
            supports_answer=supports_answer,
            match=match,
        )

        return memorilla_citation
