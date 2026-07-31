from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.inference_model_backend import InferenceModelBackend
from ..models.inference_model_format import InferenceModelFormat
from ..models.inference_model_kind import InferenceModelKind
from ..models.inference_model_quantization import InferenceModelQuantization
from ..models.inference_model_ref_memory_profile import InferenceModelRefMemoryProfile
from ..types import UNSET, Unset

T = TypeVar("T", bound="InferenceModelRef")


@_attrs_define
class InferenceModelRef:
    """Model reference used by startup preload and model-loading configuration.

    Attributes:
        kind (InferenceModelKind): Model registry kind.
        name (str): Model name to resolve within the registry for the selected kind, usually in `<owner>/<repo>` format.
            Example: antflydb/gemma-e2b.
        backend (InferenceModelBackend | Unset): Optional backend preference for model loading or request execution.
            `auto` keeps the node default behavior.
            `xla` selects the PJRT/XLA backend and may require a PJRT plugin path via
            `ANTFLY_INFERENCE_XLA_PLUGIN`, `ANTFLY_INFERENCE_PJRT_PLUGIN`,
            `PJRT_PLUGIN_PATH`, or `PJRT_PLUGIN`.
            `webgpu` selects the Wasm/WebGPU backend in Wasm builds; pair it with
            `mode: "compiled"` on generation requests to request WebGPU graph partition execution.
        format_ (InferenceModelFormat | Unset): Optional artifact format preference for loading a model.
        quantization (InferenceModelQuantization | Unset): Optional quantization preference for loading a model.
        memory_profile (InferenceModelRefMemoryProfile | Unset): Load-time memory profile for this model instance.
            `compact_2gbs`
            selects the bounded streaming routed-expert runtime with an
            enforced resident footprint ceiling sized by `memory_budget_mb`
            (default 2048 MiB). Only qualified model geometries load under a
            compact profile, the load fails closed otherwise, and
            request-time APIs cannot change residency policy.
        memory_budget_mb (int | Unset): Total resident memory budget in MiB for the compact memory
            profile. The enforced footprint ceiling, the KV-cache share, and
            the automatic expert-cache slot count all derive from it.
            Omission keeps the 2048 MiB floor; values below the floor are
            rejected. Setting a budget without `memory_profile` implies the
            compact profile.
        expert_cache_slots (int | Unset): Resident routed-expert cache slots per MoE layer under a compact
            memory profile. Omission derives the count from
            `memory_budget_mb`, downshifting under residency-ceiling
            pressure; explicit values pin it within [4, 128].
    """

    kind: InferenceModelKind
    name: str
    backend: InferenceModelBackend | Unset = UNSET
    format_: InferenceModelFormat | Unset = UNSET
    quantization: InferenceModelQuantization | Unset = UNSET
    memory_profile: InferenceModelRefMemoryProfile | Unset = UNSET
    memory_budget_mb: int | Unset = UNSET
    expert_cache_slots: int | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        kind = self.kind.value

        name = self.name

        backend: str | Unset = UNSET
        if not isinstance(self.backend, Unset):
            backend = self.backend.value

        format_: str | Unset = UNSET
        if not isinstance(self.format_, Unset):
            format_ = self.format_.value

        quantization: str | Unset = UNSET
        if not isinstance(self.quantization, Unset):
            quantization = self.quantization.value

        memory_profile: str | Unset = UNSET
        if not isinstance(self.memory_profile, Unset):
            memory_profile = self.memory_profile.value

        memory_budget_mb = self.memory_budget_mb

        expert_cache_slots = self.expert_cache_slots

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "kind": kind,
                "name": name,
            }
        )
        if backend is not UNSET:
            field_dict["backend"] = backend
        if format_ is not UNSET:
            field_dict["format"] = format_
        if quantization is not UNSET:
            field_dict["quantization"] = quantization
        if memory_profile is not UNSET:
            field_dict["memory_profile"] = memory_profile
        if memory_budget_mb is not UNSET:
            field_dict["memory_budget_mb"] = memory_budget_mb
        if expert_cache_slots is not UNSET:
            field_dict["expert_cache_slots"] = expert_cache_slots

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        kind = InferenceModelKind(d.pop("kind"))

        name = d.pop("name")

        _backend = d.pop("backend", UNSET)
        backend: InferenceModelBackend | Unset
        if isinstance(_backend, Unset):
            backend = UNSET
        else:
            backend = InferenceModelBackend(_backend)

        _format_ = d.pop("format", UNSET)
        format_: InferenceModelFormat | Unset
        if isinstance(_format_, Unset):
            format_ = UNSET
        else:
            format_ = InferenceModelFormat(_format_)

        _quantization = d.pop("quantization", UNSET)
        quantization: InferenceModelQuantization | Unset
        if isinstance(_quantization, Unset):
            quantization = UNSET
        else:
            quantization = InferenceModelQuantization(_quantization)

        _memory_profile = d.pop("memory_profile", UNSET)
        memory_profile: InferenceModelRefMemoryProfile | Unset
        if isinstance(_memory_profile, Unset):
            memory_profile = UNSET
        else:
            memory_profile = InferenceModelRefMemoryProfile(_memory_profile)

        memory_budget_mb = d.pop("memory_budget_mb", UNSET)

        expert_cache_slots = d.pop("expert_cache_slots", UNSET)

        inference_model_ref = cls(
            kind=kind,
            name=name,
            backend=backend,
            format_=format_,
            quantization=quantization,
            memory_profile=memory_profile,
            memory_budget_mb=memory_budget_mb,
            expert_cache_slots=expert_cache_slots,
        )

        inference_model_ref.additional_properties = d
        return inference_model_ref

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
