"""Aggregate and render anomaly benchmark results."""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from towervision.utils.io import ensure_dir, read_json, write_json, write_text


MAIN_METRICS = (
    "roi_auroc",
    "roi_auprc",
    "f1",
    "precision",
    "recall",
    "accuracy",
)


def _metric_mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(mean(values))


def _metric_std(values: Sequence[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return float(pstdev(values))


def aggregate_seed_results(seed_results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate completed seed results by anomaly model."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for result in seed_results:
        grouped.setdefault(str(result["model_name"]), []).append(result)

    aggregated: list[dict[str, Any]] = []
    for model_name, model_results in sorted(grouped.items()):
        completed = [item for item in model_results if item.get("status") == "completed"]
        display_name = str(model_results[0].get("display_name", model_name))
        payload: dict[str, Any] = {
            "model_name": model_name,
            "display_name": display_name,
            "num_completed_seeds": len(completed),
            "num_total_seeds": len(model_results),
        }
        for split_name in ("val", "test"):
            for metric_name in MAIN_METRICS:
                values = [
                    float(result[f"{split_name}_metrics"][metric_name])
                    for result in completed
                    if result.get(f"{split_name}_metrics", {}).get(metric_name) is not None
                ]
                payload[f"{split_name}_{metric_name}_mean"] = _metric_mean(values)
                payload[f"{split_name}_{metric_name}_std"] = _metric_std(values)
        thresholds = [
            float(result.get("val_metrics", {}).get("threshold"))
            for result in completed
            if result.get("val_metrics", {}).get("threshold") is not None
        ]
        payload["threshold_mean"] = _metric_mean(thresholds)
        payload["threshold_std"] = _metric_std(thresholds)
        aggregated.append(payload)
    return aggregated


def _read_breakdown_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            normalized: dict[str, Any] = {}
            for key, value in row.items():
                if key in {"group_field", "group_value"}:
                    normalized[key] = value
                elif value in ("", None):
                    normalized[key] = None
                else:
                    normalized[key] = float(value)
            rows.append(normalized)
        return rows


def aggregate_breakdown_results(
    seed_results: Sequence[Mapping[str, Any]],
    *,
    breakdown_key: str,
) -> list[dict[str, Any]]:
    """Aggregate breakdown CSVs across seeds by model and group value."""

    grouped_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    display_names: dict[str, str] = {}
    for result in seed_results:
        if result.get("status") != "completed":
            continue
        model_name = str(result["model_name"])
        display_names[model_name] = str(result.get("display_name", model_name))
        breakdown_path = Path(str(result.get(breakdown_key, "")))
        for row in _read_breakdown_csv(breakdown_path):
            grouped_rows[(model_name, str(row["group_value"]))].append(row)

    aggregated: list[dict[str, Any]] = []
    for (model_name, group_value), rows in sorted(grouped_rows.items()):
        payload: dict[str, Any] = {
            "model_name": model_name,
            "display_name": display_names[model_name],
            "group_field": str(rows[0]["group_field"]),
            "group_value": group_value,
        }
        for metric_name in ("roi_auroc", "roi_auprc", "f1", "precision", "recall"):
            values = [float(row[metric_name]) for row in rows if row.get(metric_name) is not None]
            payload[f"{metric_name}_mean"] = _metric_mean(values)
            payload[f"{metric_name}_std"] = _metric_std(values)
        aggregated.append(payload)
    return aggregated


def write_aggregated_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write one aggregated CSV table."""

    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _format_metric(value: float | None, std: float | None) -> str:
    if value is None:
        return "n/a"
    if std is None:
        return f"{value:.4f}"
    return f"{value:.4f} ± {std:.4f}"


def render_benchmark_report_markdown(
    *,
    benchmark_name: str,
    dataset_name: str,
    dataset_version: str,
    training_source: str,
    synthetic_pack: str,
    aggregated_results: Sequence[Mapping[str, Any]],
    generator_breakdown: Sequence[Mapping[str, Any]],
    anomaly_type_breakdown: Sequence[Mapping[str, Any]],
    severity_breakdown: Sequence[Mapping[str, Any]],
    operating_recall_floor: float,
    proxy_backend_present: bool,
) -> str:
    """Render the consolidated anomaly benchmark report."""

    lines = [
        f"# {benchmark_name}",
        "",
        "## Protocolo",
        "",
        f"- dataset: `{dataset_name}/{dataset_version}`",
        f"- trilha oficial: `{training_source}`",
        f"- pack sintético: `{synthetic_pack}`",
        "- treino: apenas ROIs normais de `isoladores`",
        "- calibração de threshold: somente em `val`",
        "- ranking principal: `test_roi_auroc`",
        "- ranking secundário: `test_roi_auprc`",
        f"- piso operacional de recall em validação: `{operating_recall_floor:.2f}`",
        "",
    ]
    if proxy_backend_present:
        lines.extend(
            [
                "## Aviso de Implementação",
                "",
                "- os resultados atuais foram gerados com o backend proxy local do repositório;",
                "- eles validam protocolo, manifests, métricas e outputs, mas não substituem as implementações de referência de `PatchCore`, `PaDiM` e `CutPaste`.",
                "",
            ]
        )
    lines.extend(
        [
            "## Resultados Consolidados",
            "",
            "| Modelo | Seeds | Val ROI AUROC | Val ROI AUPRC | Val F1 | Val Recall | Test ROI AUROC | Test ROI AUPRC | Test F1 | Test Recall | Threshold |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for result in sorted(
        aggregated_results,
        key=lambda item: (-(item.get("test_roi_auroc_mean") or float("-inf")), item["model_name"]),
    ):
        lines.append(
            (
                f"| {result['display_name']} | {result['num_completed_seeds']}/{result['num_total_seeds']} | "
                f"{_format_metric(result.get('val_roi_auroc_mean'), result.get('val_roi_auroc_std'))} | "
                f"{_format_metric(result.get('val_roi_auprc_mean'), result.get('val_roi_auprc_std'))} | "
                f"{_format_metric(result.get('val_f1_mean'), result.get('val_f1_std'))} | "
                f"{_format_metric(result.get('val_recall_mean'), result.get('val_recall_std'))} | "
                f"{_format_metric(result.get('test_roi_auroc_mean'), result.get('test_roi_auroc_std'))} | "
                f"{_format_metric(result.get('test_roi_auprc_mean'), result.get('test_roi_auprc_std'))} | "
                f"{_format_metric(result.get('test_f1_mean'), result.get('test_f1_std'))} | "
                f"{_format_metric(result.get('test_recall_mean'), result.get('test_recall_std'))} | "
                f"{_format_metric(result.get('threshold_mean'), result.get('threshold_std'))} |"
            )
        )

    lines.extend(_render_breakdown_section("generator_family", generator_breakdown))
    lines.extend(_render_breakdown_section("anomaly_type", anomaly_type_breakdown))
    lines.extend(_render_breakdown_section("severity", severity_breakdown))
    return "\n".join(lines)


def _render_breakdown_section(
    title: str,
    rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    pretty_title = {
        "generator_family": "Generator Family",
        "anomaly_type": "Anomaly Type",
        "severity": "Severity",
    }.get(title, title)
    lines = [
        "",
        f"## Breakdown por {pretty_title}",
        "",
        "| Modelo | Grupo | ROI AUROC | ROI AUPRC | F1 | Recall |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    filtered_rows = [row for row in rows if row.get("group_field") == title]
    if not filtered_rows:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a |")
        return lines
    for row in sorted(filtered_rows, key=lambda item: (item["display_name"], item["group_value"])):
        lines.append(
            (
                f"| {row['display_name']} | {row['group_value']} | "
                f"{_format_metric(row.get('roi_auroc_mean'), row.get('roi_auroc_std'))} | "
                f"{_format_metric(row.get('roi_auprc_mean'), row.get('roi_auprc_std'))} | "
                f"{_format_metric(row.get('f1_mean'), row.get('f1_std'))} | "
                f"{_format_metric(row.get('recall_mean'), row.get('recall_std'))} |"
            )
        )
    return lines


def persist_benchmark_report(
    *,
    root_dir: Path,
    benchmark_name: str,
    dataset_name: str,
    dataset_version: str,
    training_source: str,
    synthetic_pack: str,
    operating_recall_floor: float,
    seed_results: Sequence[Mapping[str, Any]],
) -> dict[str, Path]:
    """Persist consolidated anomaly benchmark artifacts."""

    aggregated_results = aggregate_seed_results(seed_results)
    generator_breakdown = aggregate_breakdown_results(
        seed_results,
        breakdown_key="generator_breakdown_path",
    )
    anomaly_type_breakdown = aggregate_breakdown_results(
        seed_results,
        breakdown_key="anomaly_type_breakdown_path",
    )
    severity_breakdown = aggregate_breakdown_results(
        seed_results,
        breakdown_key="severity_breakdown_path",
    )
    proxy_backend_present = any(
        result.get("status") == "completed" and str(result.get("backend")) == "placeholder"
        for result in seed_results
    )

    json_path = root_dir / "benchmark_results.json"
    csv_path = root_dir / "benchmark_results.csv"
    markdown_path = root_dir / "benchmark_report.md"
    generator_csv_path = root_dir / "generator_breakdown.csv"
    anomaly_type_csv_path = root_dir / "anomaly_type_breakdown.csv"
    severity_csv_path = root_dir / "severity_breakdown.csv"

    write_json(
        json_path,
        {
            "benchmark_name": benchmark_name,
            "dataset": {"name": dataset_name, "version": dataset_version},
            "training_source": training_source,
            "synthetic_pack": synthetic_pack,
            "seed_results": list(seed_results),
            "aggregated_results": aggregated_results,
            "breakdowns": {
                "generator_family": generator_breakdown,
                "anomaly_type": anomaly_type_breakdown,
                "severity": severity_breakdown,
            },
        },
    )
    write_aggregated_csv(csv_path, aggregated_results)
    write_aggregated_csv(generator_csv_path, generator_breakdown)
    write_aggregated_csv(anomaly_type_csv_path, anomaly_type_breakdown)
    write_aggregated_csv(severity_csv_path, severity_breakdown)
    write_text(
        markdown_path,
        render_benchmark_report_markdown(
            benchmark_name=benchmark_name,
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            training_source=training_source,
            synthetic_pack=synthetic_pack,
            aggregated_results=aggregated_results,
            generator_breakdown=generator_breakdown,
            anomaly_type_breakdown=anomaly_type_breakdown,
            severity_breakdown=severity_breakdown,
            operating_recall_floor=operating_recall_floor,
            proxy_backend_present=proxy_backend_present,
        ),
    )
    return {
        "json": json_path,
        "csv": csv_path,
        "markdown": markdown_path,
        "generator_breakdown_csv": generator_csv_path,
        "anomaly_type_breakdown_csv": anomaly_type_csv_path,
        "severity_breakdown_csv": severity_csv_path,
    }
