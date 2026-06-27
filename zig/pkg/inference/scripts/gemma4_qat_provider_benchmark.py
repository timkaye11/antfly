#!/usr/bin/env python3
"""Collect OpenAI-compatible provider baselines for the Gemma4 QAT gate."""

import argparse
import csv
import json
import os
import pathlib
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_PROMPT = (
    "Write a detailed technical explanation of how database indexes improve read "
    "queries while slowing down writes. Include examples, tradeoffs, tuning advice, "
    "and operational caveats."
)

PROVIDER_METRICS = {
    "target_decode_tok_s",
    "long_decode_tok_s",
    "resident_e2e_tok_s",
    "soak_aggregate_tok_s",
    "backpressure_accepted_e2e_tok_s",
}


def to_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def average(values: Iterable[float]) -> Optional[float]:
    items = list(values)
    if not items:
        return None
    return sum(items) / len(items)


def percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[index]


def parse_header(raw: str) -> Tuple[str, str]:
    if ":" in raw:
        name, value = raw.split(":", 1)
    elif "=" in raw:
        name, value = raw.split("=", 1)
    else:
        raise ValueError(f"header must use Name:Value or Name=Value: {raw!r}")
    name = name.strip()
    value = value.strip()
    if not name:
        raise ValueError(f"header name is empty: {raw!r}")
    return name, value


def completion_url(base_url: str, endpoint: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions") or base.endswith("/completions"):
        return base
    if endpoint == "chat":
        return base + "/chat/completions"
    return base + "/completions"


def build_headers(args: argparse.Namespace) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if not args.no_auth:
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise RuntimeError(f"${args.api_key_env} is required unless --no-auth is set")
        if args.auth_scheme:
            headers[args.auth_header] = f"{args.auth_scheme} {api_key}"
        else:
            headers[args.auth_header] = api_key
    for raw in args.header:
        name, value = parse_header(raw)
        headers[name] = value
    return headers


def build_payload(args: argparse.Namespace) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": args.model,
        "temperature": args.temperature,
        "stream": bool(args.stream),
        args.max_token_field: args.tokens,
    }
    if args.stream and args.stream_include_usage:
        payload["stream_options"] = {"include_usage": True}
    if args.top_p is not None:
        payload["top_p"] = args.top_p
    if args.seed is not None:
        payload["seed"] = args.seed
    if args.endpoint == "chat":
        payload["messages"] = [{"role": "user", "content": args.prompt}]
    else:
        payload["prompt"] = args.prompt
    return payload


def http_post_json(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: float) -> Tuple[int, Dict[str, Any]]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = int(response.status)
            response_body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {response_body[:1000]}") from exc
    try:
        return status, json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"provider returned non-JSON response: {response_body[:1000]}") from exc


def http_post_stream(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout: float,
) -> Tuple[int, List[Tuple[float, Dict[str, Any]]]]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    events: List[Tuple[float, Dict[str, Any]]] = []
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = int(response.status)
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data_text = line[len("data:") :].strip()
                if data_text == "[DONE]":
                    break
                try:
                    events.append((time.perf_counter(), json.loads(data_text)))
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"provider returned malformed SSE JSON: {data_text[:1000]}") from exc
    except urllib.error.HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {response_body[:1000]}") from exc
    return status, events


def completion_tokens_from_response(data: Dict[str, Any]) -> Optional[int]:
    usage = data.get("usage") or {}
    for key in ("completion_tokens", "output_tokens", "generated_tokens"):
        value = to_int(usage.get(key))
        if value is not None:
            return value
    return None


def completion_tokens_from_stream_events(events: List[Tuple[float, Dict[str, Any]]]) -> Optional[int]:
    for _, data in reversed(events):
        tokens = completion_tokens_from_response(data)
        if tokens is not None:
            return tokens
    return None


def finish_reason_from_response(data: Dict[str, Any]) -> Optional[str]:
    choices = data.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return None
    finish_reason = choices[0].get("finish_reason")
    return str(finish_reason) if finish_reason is not None else None


def finish_reason_from_stream_events(events: List[Tuple[float, Dict[str, Any]]]) -> Optional[str]:
    for _, data in reversed(events):
        finish_reason = finish_reason_from_response(data)
        if finish_reason is not None:
            return finish_reason
    return None


def response_id(data: Dict[str, Any]) -> Optional[str]:
    value = data.get("id")
    return str(value) if value is not None else None


def response_id_from_stream_events(events: List[Tuple[float, Dict[str, Any]]]) -> Optional[str]:
    for _, data in events:
        value = response_id(data)
        if value is not None:
            return value
    return None


def stream_event_has_content(data: Dict[str, Any]) -> bool:
    choices = data.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return False
    choice = choices[0]
    delta = choice.get("delta")
    if isinstance(delta, dict) and delta.get("content"):
        return True
    if choice.get("text"):
        return True
    message = choice.get("message")
    return isinstance(message, dict) and bool(message.get("content"))


def run_one(args: argparse.Namespace, url: str, headers: Dict[str, str], case: str) -> Dict[str, Any]:
    payload = build_payload(args)
    started = time.perf_counter()
    ttft_ms: Optional[float] = None
    stream_ms: Optional[float] = None
    stream_decode_tok_s: Optional[float] = None
    if args.stream:
        status, events = http_post_stream(url, headers, payload, args.timeout)
        ended = time.perf_counter()
        content_times = [event_time for event_time, data in events if stream_event_has_content(data)]
        if content_times:
            ttft_ms = (content_times[0] - started) * 1000.0
            stream_ms = max(0.0, (content_times[-1] - content_times[0]) * 1000.0)
        completion_tokens = completion_tokens_from_stream_events(events)
        response_id_value = response_id_from_stream_events(events)
        finish_reason_value = finish_reason_from_stream_events(events)
    else:
        status, data = http_post_json(url, headers, payload, args.timeout)
        ended = time.perf_counter()
        completion_tokens = completion_tokens_from_response(data)
        response_id_value = response_id(data)
        finish_reason_value = finish_reason_from_response(data)
    elapsed_ms = (ended - started) * 1000.0
    if completion_tokens is None and args.allow_token_fallback:
        completion_tokens = args.tokens
    if completion_tokens is None:
        raise RuntimeError("provider response did not include usage.completion_tokens or usage.output_tokens")
    e2e_tok_s = completion_tokens / (elapsed_ms / 1000.0) if elapsed_ms > 0 else 0.0
    if args.stream and stream_ms is not None and stream_ms > 0:
        stream_decode_tokens = max(1, completion_tokens - 1)
        stream_decode_tok_s = stream_decode_tokens / (stream_ms / 1000.0)
    return {
        "case": case,
        "status": status,
        "e2e_ms": elapsed_ms,
        "ttft_ms": ttft_ms,
        "stream_ms": stream_ms,
        "completion_tokens": completion_tokens,
        "e2e_tok_s": e2e_tok_s,
        "stream_decode_tok_s": stream_decode_tok_s,
        "response_id": response_id_value,
        "finish_reason": finish_reason_value,
    }


def aggregate_rows(rows: List[Dict[str, Any]], measured_prefix: str) -> Dict[str, Any]:
    measured = [row for row in rows if str(row.get("case", "")).startswith(measured_prefix)]
    rates = [float(row["e2e_tok_s"]) for row in measured]
    stream_rates = [
        float(row["stream_decode_tok_s"])
        for row in measured
        if row.get("stream_decode_tok_s") is not None
    ]
    latencies = [float(row["e2e_ms"]) for row in measured]
    ttfts = [float(row["ttft_ms"]) for row in measured if row.get("ttft_ms") is not None]
    tokens = [int(row["completion_tokens"]) for row in measured]
    return {
        "measured_repeats": len(measured),
        "min_e2e_tok_s": min(rates) if rates else None,
        "avg_e2e_tok_s": average(rates),
        "median_e2e_tok_s": statistics.median(rates) if rates else None,
        "max_e2e_tok_s": max(rates) if rates else None,
        "min_stream_decode_tok_s": min(stream_rates) if stream_rates else None,
        "avg_stream_decode_tok_s": average(stream_rates),
        "median_stream_decode_tok_s": statistics.median(stream_rates) if stream_rates else None,
        "max_stream_decode_tok_s": max(stream_rates) if stream_rates else None,
        "p50_e2e_ms": percentile(latencies, 50),
        "p95_e2e_ms": percentile(latencies, 95),
        "p50_ttft_ms": percentile(ttfts, 50),
        "p95_ttft_ms": percentile(ttfts, 95),
        "min_completion_tokens": min(tokens) if tokens else None,
        "total_completion_tokens": sum(tokens) if tokens else None,
    }


def baseline_tok_s(aggregate: Dict[str, Any], stat: str, rate_source: str) -> Optional[float]:
    if rate_source == "stream_decode":
        return aggregate.get(f"{stat}_stream_decode_tok_s")
    if stat == "avg":
        return aggregate.get("avg_e2e_tok_s")
    if stat == "min":
        return aggregate.get("min_e2e_tok_s")
    if stat == "median":
        return aggregate.get("median_e2e_tok_s")
    raise ValueError(f"unsupported baseline stat: {stat}")


def parse_baseline_stats(raw_stats: str, fallback_stat: str) -> List[str]:
    raw = raw_stats.strip() if raw_stats else fallback_stat
    if raw.lower() == "all":
        return ["avg", "median", "min"]
    stats = []
    for item in raw.split(","):
        stat = item.strip().lower()
        if not stat:
            continue
        if stat not in {"avg", "min", "median"}:
            raise ValueError(f"unsupported baseline stat: {stat}")
        if stat not in stats:
            stats.append(stat)
    if not stats:
        raise ValueError("at least one baseline stat is required")
    return stats


def make_baseline(
    args: argparse.Namespace,
    aggregate: Dict[str, Any],
    measured_at: str,
    stat: str,
    all_stats: List[str],
) -> Dict[str, Any]:
    return {
        "provider": args.provider,
        "label": f"{args.provider}:{args.metric}:{args.rate_source}:{stat}",
        "metric": args.metric,
        "stat": stat,
        "rate_source": args.rate_source,
        "tok_s": baseline_tok_s(aggregate, stat, args.rate_source),
        "min_ratio": args.min_ratio,
        "model": args.model,
        "hardware": args.hardware,
        "tokens": args.tokens,
        "workload": args.workload,
        "measured_at": measured_at,
        "source_url": args.source_url,
        "context": {
            "api": "openai-compatible",
            "endpoint": args.endpoint,
            "base_url": args.base_url.rstrip("/"),
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "stream": bool(args.stream),
            "stream_include_usage": bool(args.stream_include_usage),
            "rate_source": args.rate_source,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "baseline_stat": stat,
            "baseline_stats": all_stats,
            "max_token_field": args.max_token_field,
            "min_completion_tokens": args.min_completion_tokens,
        },
        "notes": args.notes,
    }


def write_rows_tsv(path: pathlib.Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case",
                "status",
                "e2e_ms",
                "ttft_ms",
                "stream_ms",
                "completion_tokens",
                "e2e_tok_s",
                "stream_decode_tok_s",
                "response_id",
                "finish_reason",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "case": row.get("case"),
                    "status": row.get("status"),
                    "e2e_ms": f"{float(row.get('e2e_ms') or 0.0):.3f}",
                    "ttft_ms": "" if row.get("ttft_ms") is None else f"{float(row.get('ttft_ms') or 0.0):.3f}",
                    "stream_ms": "" if row.get("stream_ms") is None else f"{float(row.get('stream_ms') or 0.0):.3f}",
                    "completion_tokens": row.get("completion_tokens"),
                    "e2e_tok_s": f"{float(row.get('e2e_tok_s') or 0.0):.3f}",
                    "stream_decode_tok_s": "" if row.get("stream_decode_tok_s") is None else f"{float(row.get('stream_decode_tok_s') or 0.0):.3f}",
                    "response_id": row.get("response_id") or "",
                    "finish_reason": row.get("finish_reason") or "",
                }
            )


def benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    url = completion_url(args.base_url, args.endpoint)
    headers = build_headers(args)
    rows: List[Dict[str, Any]] = []
    errors: List[str] = []
    for i in range(args.warmup):
        rows.append(run_one(args, url, headers, f"warmup_{i + 1}"))
        if args.sleep_s > 0:
            time.sleep(args.sleep_s)
    for i in range(args.repeats):
        rows.append(run_one(args, url, headers, f"run_{i + 1}"))
        if args.sleep_s > 0 and i + 1 < args.repeats:
            time.sleep(args.sleep_s)

    aggregate = aggregate_rows(rows, "run_")
    if int(aggregate.get("measured_repeats") or 0) != args.repeats:
        errors.append(f"measured_repeats={aggregate.get('measured_repeats')} expected={args.repeats}")
    min_completion = aggregate.get("min_completion_tokens")
    if min_completion is None or int(min_completion) < args.min_completion_tokens:
        errors.append(f"min_completion_tokens={min_completion} floor={args.min_completion_tokens}")
    try:
        baseline_stats = parse_baseline_stats(args.baseline_stats, args.baseline_stat)
    except ValueError as exc:
        errors.append(str(exc))
        baseline_stats = [args.baseline_stat]
    for stat in baseline_stats:
        tok_s = baseline_tok_s(aggregate, stat, args.rate_source)
        if tok_s is None or tok_s <= 0:
            errors.append(f"baseline_{args.rate_source}_{stat}_tok_s={tok_s}")

    measured_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    baselines = [make_baseline(args, aggregate, measured_at, stat, baseline_stats) for stat in baseline_stats]
    ok = not errors
    return {
        "baselines": baselines if ok else [],
        "provider_benchmark": {
            "ok": ok,
            "errors": errors,
            "provider": args.provider,
            "metric": args.metric,
            "rate_source": args.rate_source,
            "baseline_stats": baseline_stats,
            "url": url,
            "aggregate": aggregate,
            "rows": rows,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible base URL, for example https://api.example.com/v1")
    parser.add_argument("--endpoint", choices=("chat", "completions"), default="chat")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--auth-header", default="Authorization")
    parser.add_argument("--auth-scheme", default="Bearer")
    parser.add_argument("--no-auth", action="store_true")
    parser.add_argument("--header", action="append", default=[], help="extra HTTP header as Name:Value or Name=Value")
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--metric", choices=sorted(PROVIDER_METRICS), default="resident_e2e_tok_s")
    parser.add_argument("--workload", default="antfly-resident-index-explanation-512")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--min-completion-tokens", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--sleep-s", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-token-field", choices=("max_completion_tokens", "max_tokens"), default="max_completion_tokens")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--stream-include-usage", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rate-source", choices=("e2e", "stream_decode"), default="e2e")
    parser.add_argument("--baseline-stat", choices=("avg", "min", "median"), default="avg")
    parser.add_argument("--baseline-stats", default="", help="comma-separated stats to emit, or all; overrides --baseline-stat")
    parser.add_argument("--min-ratio", type=float, default=1.0)
    parser.add_argument("--notes", default="")
    parser.add_argument("--allow-token-fallback", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows-tsv")
    args = parser.parse_args()

    if args.tokens <= 0:
        parser.error("--tokens must be positive")
    if args.min_completion_tokens < 0:
        parser.error("--min-completion-tokens must be non-negative")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")

    try:
        result = benchmark(args)
    except Exception as exc:
        print(f"provider_benchmark_error: {exc}", file=sys.stderr)
        return 1

    output = pathlib.Path(args.output)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.rows_tsv:
        write_rows_tsv(pathlib.Path(args.rows_tsv), result["provider_benchmark"]["rows"])

    aggregate = result["provider_benchmark"]["aggregate"]
    print(
        f"provider_benchmark={output} ok={result['provider_benchmark']['ok']} "
        f"provider={args.provider} metric={args.metric} "
        f"avg_e2e_tok_s={aggregate.get('avg_e2e_tok_s')} "
        f"min_completion_tokens={aggregate.get('min_completion_tokens')}"
    )
    if not result["provider_benchmark"]["ok"]:
        for error in result["provider_benchmark"]["errors"]:
            print(f"provider_benchmark_failure: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
