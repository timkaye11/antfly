from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define

from ..models.memorilla_profile_config_reader_kind import MemorillaProfileConfigReaderKind
from ..types import UNSET, Unset

T = TypeVar("T", bound="MemorillaProfileConfig")


@_attrs_define
class MemorillaProfileConfig:
    """An administrator-configured, immutable Memorilla model composition. Only
    the qualified Qwen3 F32 CUDA profiles are supported for the released
    reader kind. The encoder_lora_research kind serves the experimental
    encoder-LoRA copy candidate from its own locked artifacts; it is not
    qualified and must not be promoted or presented as such.

        Attributes:
            name (str):
            encoder (str): Managed encoder reference. For reader_kind=released this uses the
                qwen3_encoder_f32_v1 execution profile. For
                reader_kind=encoder_lora_research it is the merged research encoder
                directory exported by encoder_lora_export.py (manifest profile
                qwen3_encoder_lora_research_f32_v1 with its memorilla_lock.json).
            decoder (str): Managed decoder reference using the qwen3_decoder_f32_kv_f32_v1 execution profile.
            memory_module_path (str): Local reader bundle directory. For reader_kind=released this is an
                exported released-checkpoint bundle; for
                reader_kind=encoder_lora_research it is the research reader bundle
                (reader, copy head, candidate.json and protocol.json). This path is
                configured by the administrator and cannot be supplied by an
                inference request.
            reader_kind (MemorillaProfileConfigReaderKind | Unset): Which reader the composition serves. released consumes
                caller-supplied document embeddings and supports source_verified
                mode. encoder_lora_research encodes caller-supplied document_texts
                with the merged research encoder, rejects document_embeddings, does
                not support source_verified, and reports candidate_sha256.
                 Default: MemorillaProfileConfigReaderKind.RELEASED.
            max_documents (int | Unset):  Default: 64.
            max_new_tokens (int | Unset):  Default: 128.
    """

    name: str
    encoder: str
    decoder: str
    memory_module_path: str
    reader_kind: MemorillaProfileConfigReaderKind | Unset = MemorillaProfileConfigReaderKind.RELEASED
    max_documents: int | Unset = 64
    max_new_tokens: int | Unset = 128

    def to_dict(self) -> dict[str, Any]:
        name = self.name

        encoder = self.encoder

        decoder = self.decoder

        memory_module_path = self.memory_module_path

        reader_kind: str | Unset = UNSET
        if not isinstance(self.reader_kind, Unset):
            reader_kind = self.reader_kind.value

        max_documents = self.max_documents

        max_new_tokens = self.max_new_tokens

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "name": name,
                "encoder": encoder,
                "decoder": decoder,
                "memory_module_path": memory_module_path,
            }
        )
        if reader_kind is not UNSET:
            field_dict["reader_kind"] = reader_kind
        if max_documents is not UNSET:
            field_dict["max_documents"] = max_documents
        if max_new_tokens is not UNSET:
            field_dict["max_new_tokens"] = max_new_tokens

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        name = d.pop("name")

        encoder = d.pop("encoder")

        decoder = d.pop("decoder")

        memory_module_path = d.pop("memory_module_path")

        _reader_kind = d.pop("reader_kind", UNSET)
        reader_kind: MemorillaProfileConfigReaderKind | Unset
        if isinstance(_reader_kind, Unset):
            reader_kind = UNSET
        else:
            reader_kind = MemorillaProfileConfigReaderKind(_reader_kind)

        max_documents = d.pop("max_documents", UNSET)

        max_new_tokens = d.pop("max_new_tokens", UNSET)

        memorilla_profile_config = cls(
            name=name,
            encoder=encoder,
            decoder=decoder,
            memory_module_path=memory_module_path,
            reader_kind=reader_kind,
            max_documents=max_documents,
            max_new_tokens=max_new_tokens,
        )

        return memorilla_profile_config
