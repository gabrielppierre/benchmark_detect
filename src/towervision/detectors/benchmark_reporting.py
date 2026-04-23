"""Aggregate and render fair benchmark results."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from towervision.utils.io import ensure_dir, write_json, write_text


def _metric_mean(values: Sequence[float]) -> float | None:
    """Return the mean or None when the sequence is empty."""

    if not values:
        return None
    return float(mean(values))


def _metric_std(values: Sequence[float]) -> float | None:
    """Return the population std or None when the sequence is empty."""

    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return float(pstdev(values))


def aggregate_seed_results(seed_results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate completed seed results by model."""

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for result in seed_results:
        grouped.setdefault(str(result["model_name"]), []).append(result)

    aggregated: list[dict[str, Any]] = []
    for model_name, model_results in sorted(grouped.items()):
        completed = [result for result in model_results if result.get("status") == "completed"]
        display_name = str(model_results[0].get("display_name", model_name))

        val_map50_95 = [
            float(
                result["val_best_metrics"].get("val_map50_95", result["val_best_metrics"].get("mAP50_95"))
            )
            for result in completed
            if result.get("val_best_metrics", {}).get("val_map50_95") is not None
            or result.get("val_best_metrics", {}).get("mAP50_95") is not None
        ]
        val_ap_isoladores = [
            float(result["val_best_metrics"]["AP50_95_isoladores"])
            for result in completed
            if result.get("val_best_metrics", {}).get("AP50_95_isoladores") is not None
        ]
        val_recall_isoladores = [
            float(result["val_best_metrics"]["Recall_isoladores"])
            for result in completed
            if result.get("val_best_metrics", {}).get("Recall_isoladores") is not None
        ]
        test_map50_95 = [
            float(result["test_metrics"]["mAP50_95"])
            for result in completed
            if result.get("test_metrics", {}).get("mAP50_95") is not None
        ]
        test_ap_isoladores = [
            float(result["test_metrics"]["AP50_95_isoladores"])
            for result in completed
            if result.get("test_metrics", {}).get("AP50_95_isoladores") is not None
        ]
        test_recall_isoladores = [
            float(result["test_metrics"]["Recall_isoladores"])
            for result in completed
            if result.get("test_metrics", {}).get("Recall_isoladores") is not None
        ]

        aggregated.append(
            {
                "model_name": model_name,
                "display_name": display_name,
                "num_completed_seeds": len(completed),
                "num_total_seeds": len(model_results),
                "val_map50_95_mean": _metric_mean(val_map50_95),
                "val_map50_95_std": _metric_std(val_map50_95),
                "val_AP50_95_isoladores_mean": _metric_mean(val_ap_isoladores),
                "val_AP50_95_isoladores_std": _metric_std(val_ap_isoladores),
                "val_Recall_isoladores_mean": _metric_mean(val_recall_isoladores),
                "val_Recall_isoladores_std": _metric_std(val_recall_isoladores),
                "test_map50_95_mean": _metric_mean(test_map50_95),
                "test_map50_95_std": _metric_std(test_map50_95),
                "test_AP50_95_isoladores_mean": _metric_mean(test_ap_isoladores),
                "test_AP50_95_isoladores_std": _metric_std(test_ap_isoladores),
                "test_Recall_isoladores_mean": _metric_mean(test_recall_isoladores),
                "test_Recall_isoladores_std": _metric_std(test_recall_isoladores),
            }
        )
    return aggregated


def select_detector_for_anomaly(
    aggregated_results: Sequence[Mapping[str, Any]],
    *,
    recall_floor_isoladores: float,
) -> dict[str, Any] | None:
    """Select the operational detector for anomaly crops using validation metrics."""

    eligible = [
        result
        for result in aggregated_results
        if (result.get("val_Recall_isoladores_mean") or 0.0) >= recall_floor_isoladores
    ]
    if not eligible:
        return None

    ranked = sorted(
        eligible,
        key=lambda result: (
            -(result.get("val_AP50_95_isoladores_mean") or float("-inf")),
            -(result.get("val_map50_95_mean") or float("-inf")),
            result["model_name"],
        ),
    )
    return dict(ranked[0])


def write_aggregated_csv(path: Path, aggregated_results: Sequence[Mapping[str, Any]]) -> None:
    """Write the aggregated benchmark table to CSV."""

    ensure_dir(path.parent)
    fieldnames = [
        "model_name",
        "display_name",
        "num_completed_seeds",
        "num_total_seeds",
        "val_map50_95_mean",
        "val_map50_95_std",
        "val_AP50_95_isoladores_mean",
        "val_AP50_95_isoladores_std",
        "val_Recall_isoladores_mean",
        "val_Recall_isoladores_std",
        "test_map50_95_mean",
        "test_map50_95_std",
        "test_AP50_95_isoladores_mean",
        "test_AP50_95_isoladores_std",
        "test_Recall_isoladores_mean",
        "test_Recall_isoladores_std",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in aggregated_results:
            writer.writerow({field: result.get(field) for field in fieldnames})


def render_benchmark_report_markdown(
    *,
    benchmark_name: str,
    dataset_name: str,
    dataset_version: str,
    split_name: str,
    classes: Sequence[str],
    critical_class: str,
    aggregated_results: Sequence[Mapping[str, Any]],
    selected_detector: Mapping[str, Any] | None,
    recall_floor_isoladores: float,
) -> str:
    """Render the final benchmark report in Markdown."""

    lines = [
        f"# {benchmark_name}",
        "",
        "## Protocolo",
        "",
        f"- dataset: `{dataset_name}/{dataset_version}`",
        f"- split oficial: `{split_name}`",
        f"- classes: `{', '.join(classes)}`",
        f"- classe crítica: `{critical_class}`",
        "- early stopping: simétrico por `val_map50_95`",
        "- melhor checkpoint: maior `val_map50_95`",
        "- ranking geral: `test_map50_95`",
        "- ranking orientado à classe crítica: `test_AP50_95_isoladores`",
        "",
        "## Resultados Consolidados",
        "",
        "| Modelo | Seeds | Val mAP50-95 | Val AP50-95 Isoladores | Val Recall Isoladores | Test mAP50-95 | Test AP50-95 Isoladores | Test Recall Isoladores |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for result in sorted(
        aggregated_results,
        key=lambda item: (-(item.get("test_map50_95_mean") or float("-inf")), item["model_name"]),
    ):
        lines.append(
            (
                f"| {result['display_name']} | {result['num_completed_seeds']}/{result['num_total_seeds']} | "
                f"{_format_metric(result.get('val_map50_95_mean'), result.get('val_map50_95_std'))} | "
                f"{_format_metric(result.get('val_AP50_95_isoladores_mean'), result.get('val_AP50_95_isoladores_std'))} | "
                f"{_format_metric(result.get('val_Recall_isoladores_mean'), result.get('val_Recall_isoladores_std'))} | "
                f"{_format_metric(result.get('test_map50_95_mean'), result.get('test_map50_95_std'))} | "
                f"{_format_metric(result.get('test_AP50_95_isoladores_mean'), result.get('test_AP50_95_isoladores_std'))} | "
                f"{_format_metric(result.get('test_Recall_isoladores_mean'), result.get('test_Recall_isoladores_std'))} |"
            )
        )

    lines.extend(
        [
            "",
            "## Seleção Para Anomalia",
            "",
            f"- piso de `Recall_isoladores` em validação: `{recall_floor_isoladores:.3f}`",
        ]
    )
    if selected_detector is None:
        lines.append("- nenhum modelo completado atingiu o piso configurado")
    else:
        lines.extend(
            [
                f"- detector selecionado: `{selected_detector['display_name']}`",
                (
                    f"- critério: maior `val_AP50_95_isoladores` entre os modelos com "
                    f"`val_Recall_isoladores >= {recall_floor_isoladores:.3f}`; "
                    "desempate por `val_map50_95`"
                ),
            ]
        )

    return "\n".join(lines)


def persist_benchmark_report(
    *,
    root_dir: Path,
    benchmark_name: str,
    dataset_name: str,
    dataset_version: str,
    split_name: str,
    classes: Sequence[str],
    critical_class: str,
    recall_floor_isoladores: float,
    seed_results: Sequence[Mapping[str, Any]],
) -> dict[str, Path]:
    """Persist consolidated benchmark JSON, CSV and Markdown artifacts."""

    aggregated_results = aggregate_seed_results(seed_results)
    selected_detector = select_detector_for_anomaly(
        aggregated_results,
        recall_floor_isoladores=recall_floor_isoladores,
    )
    json_path = root_dir / "benchmark_results.json"
    csv_path = root_dir / "benchmark_results.csv"
    markdown_path = root_dir / "benchmark_report.md"

    write_json(
        json_path,
        {
            "benchmark_name": benchmark_name,
            "dataset": {"name": dataset_name, "version": dataset_version},
            "split_name": split_name,
            "seed_results": list(seed_results),
            "aggregated_results": aggregated_results,
            "selected_detector_for_anomaly": selected_detector,
        },
    )
    write_aggregated_csv(csv_path, aggregated_results)
    write_text(
        markdown_path,
        render_benchmark_report_markdown(
            benchmark_name=benchmark_name,
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            split_name=split_name,
            classes=classes,
            critical_class=critical_class,
            aggregated_results=aggregated_results,
            selected_detector=selected_detector,
            recall_floor_isoladores=recall_floor_isoladores,
        ),
    )
    return {
        "json": json_path,
        "csv": csv_path,
        "markdown": markdown_path,
    }


def _format_metric(value: float | None, std: float | None) -> str:
    """Format a mean ± std metric for Markdown tables."""

    if value is None:
        return "n/a"
    if std is None:
        return f"{value:.4f}"
    return f"{value:.4f} ± {std:.4f}"
