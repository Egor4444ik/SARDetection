from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class ModelMetrics:
    name: str
    total: int
    top1: float
    top3: float
    top5: float
    macro_p: float
    macro_r: float
    macro_f1: float
    weighted_p: float
    weighted_r: float
    weighted_f1: float
    mean_conf: float
    per_class: list[dict]


class MetricsLoader:
    def __init__(self, metrics_path: Path) -> None:
        self._path = metrics_path

    def load(self) -> list[ModelMetrics]:
        raw = json.loads(self._path.read_text())
        out: list[ModelMetrics] = []
        for m in raw:
            out.append(ModelMetrics(
                name=m["model"],
                total=m["total"],
                top1=m["top1"],
                top3=m["top3"],
                top5=m["top5"],
                macro_p=m["macro_p"],
                macro_r=m["macro_r"],
                macro_f1=m["macro_f1"],
                weighted_p=m["weighted_p"],
                weighted_r=m["weighted_r"],
                weighted_f1=m["weighted_f1"],
                mean_conf=m["mean_conf"],
                per_class=m["per_class"],
            ))
        return out


class ConfusionLoader:
    def __init__(self, out_dir: Path) -> None:
        self._out_dir = out_dir

    def load(self, model_name: str) -> np.ndarray | None:
        path = self._out_dir / f"confusion_{model_name}.csv"
        if not path.exists():
            return None
        return np.loadtxt(path, dtype=np.int64, delimiter=",")


class ClassNamesLoader:
    def __init__(self, coco_json: Path) -> None:
        self._coco_json = coco_json

    def load(self) -> list[str]:
        d = json.loads(self._coco_json.read_text())
        used_ids = {a["category_id"] for a in d["annotations"]}
        cats = sorted(
            [c for c in d["categories"] if c["id"] in used_ids],
            key=lambda c: c["id"],
        )
        return [c["name"] for c in cats]


class GlobalBarPlotter:
    def __init__(self, out_dir: Path) -> None:
        self._out_dir = out_dir

    def plot(self, models: list[ModelMetrics]) -> None:
        if not models:
            return
        names = [m.name for m in models]
        metrics = {
            "top1 (OA)": [m.top1 for m in models],
            "top3": [m.top3 for m in models],
            "top5": [m.top5 for m in models],
            "macro F1": [m.macro_f1 for m in models],
            "weighted F1": [m.weighted_f1 for m in models],
        }

        x = np.arange(len(names))
        width = 0.15
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, (label, vals) in enumerate(metrics.items()):
            offset = (i - len(metrics) / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=label)
            for bar, v in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("score")
        ax.set_title("Model comparison — all metrics")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(self._out_dir / "01_model_comparison.png", dpi=140)
        plt.close(fig)
        print("saved 01_model_comparison.png")


class PerClassF1Plotter:
    def __init__(self, out_dir: Path) -> None:
        self._out_dir = out_dir

    def plot(self, models: list[ModelMetrics], classes: list[str]) -> None:
        if not models:
            return
        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(classes))
        width = 0.8 / max(len(models), 1)
        for i, m in enumerate(models):
            f1 = [m.per_class[j]["f1"] for j in range(len(classes))]
            ax.bar(x + i * width, f1, width, label=m.name)

        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(classes, rotation=90, fontsize=7)
        ax.set_ylabel("F1")
        ax.set_ylim(0, 1.05)
        ax.set_title("Per-class F1 — все модели")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(self._out_dir / "02_per_class_f1.png", dpi=140)
        plt.close(fig)
        print("saved 02_per_class_f1.png")


class PerClassRecallPlotter:
    def __init__(self, out_dir: Path) -> None:
        self._out_dir = out_dir

    def plot(self, models: list[ModelMetrics], classes: list[str]) -> None:
        if not models:
            return
        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(classes))
        width = 0.8 / max(len(models), 1)
        for i, m in enumerate(models):
            r = [m.per_class[j]["recall"] for j in range(len(classes))]
            ax.bar(x + i * width, r, width, label=m.name)

        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(classes, rotation=90, fontsize=7)
        ax.set_ylabel("recall")
        ax.set_ylim(0, 1.05)
        ax.set_title("Per-class recall")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(self._out_dir / "03_per_class_recall.png", dpi=140)
        plt.close(fig)
        print("saved 03_per_class_recall.png")


class ConfusionPlotter:
    def __init__(self, out_dir: Path) -> None:
        self._out_dir = out_dir

    def plot(self, model_name: str, cm: np.ndarray, classes: list[str]) -> None:
        fig, ax = plt.subplots(figsize=(14, 12))
        im = ax.imshow(cm, cmap="Blues", aspect="auto")
        ax.set_xticks(np.arange(len(classes)))
        ax.set_yticks(np.arange(len(classes)))
        ax.set_xticklabels(classes, rotation=90, fontsize=6)
        ax.set_yticklabels(classes, fontsize=6)
        ax.set_xlabel("predicted")
        ax.set_ylabel("ground truth")
        ax.set_title(f"Confusion matrix — {model_name}")
        fig.colorbar(im, ax=ax, fraction=0.03)
        fig.tight_layout()
        fig.savefig(self._out_dir / f"04_confusion_{model_name}.png", dpi=140)
        plt.close(fig)
        print(f"saved 04_confusion_{model_name}.png")


class ConfusionNormalizedPlotter:
    def __init__(self, out_dir: Path) -> None:
        self._out_dir = out_dir

    def plot(self, model_name: str, cm: np.ndarray, classes: list[str]) -> None:
        row_sum = cm.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1
        norm = cm / row_sum

        fig, ax = plt.subplots(figsize=(14, 12))
        im = ax.imshow(norm, cmap="viridis", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(classes)))
        ax.set_yticks(np.arange(len(classes)))
        ax.set_xticklabels(classes, rotation=90, fontsize=6)
        ax.set_yticklabels(classes, fontsize=6)
        ax.set_xlabel("predicted")
        ax.set_ylabel("ground truth")
        ax.set_title(f"Normalized confusion — {model_name}")
        fig.colorbar(im, ax=ax, fraction=0.03)
        fig.tight_layout()
        fig.savefig(self._out_dir / f"05_confusion_norm_{model_name}.png", dpi=140)
        plt.close(fig)
        print(f"saved 05_confusion_norm_{model_name}.png")


class TopErrorsPlotter:
    def __init__(self, out_dir: Path) -> None:
        self._out_dir = out_dir

    def plot(self, model_name: str, cm: np.ndarray, classes: list[str], top_k: int = 15) -> None:
        errors = []
        for i in range(len(classes)):
            for j in range(len(classes)):
                if i != j and cm[i, j] > 0:
                    errors.append((int(cm[i, j]), classes[i], classes[j]))
        errors.sort(reverse=True)
        errors = errors[:top_k]
        if not errors:
            return

        labels = [f"{gt} → {pred}" for _, gt, pred in errors]
        values = [c for c, _, _ in errors]

        fig, ax = plt.subplots(figsize=(10, max(4, len(errors) * 0.4)))
        y = np.arange(len(labels))
        ax.barh(y, values, color="#e53935")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("count")
        ax.set_title(f"Top-{top_k} ошибок — {model_name}")
        for i, v in enumerate(values):
            ax.text(v, i, f" {v}", va="center", fontsize=7)
        fig.tight_layout()
        fig.savefig(self._out_dir / f"06_top_errors_{model_name}.png", dpi=140)
        plt.close(fig)
        print(f"saved 06_top_errors_{model_name}.png")

class SupportPlotter:
    def __init__(self, out_dir: Path) -> None:
        self._out_dir = out_dir

    def plot(self, models: list[ModelMetrics], classes: list[str]) -> None:
        if not models:
            return
        ref = models[0]
        support = [ref.per_class[j]["support"] for j in range(len(classes))]
        fig, ax = plt.subplots(figsize=(14, 5))
        x = np.arange(len(classes))
        ax.bar(x, support, color="#3949ab")
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=90, fontsize=7)
        ax.set_ylabel("samples")
        ax.set_title("Distribution of test samples per class")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(self._out_dir / "07_support.png", dpi=140)
        plt.close(fig)
        print("saved 07_support.png")


class RadarPlotter:
    def __init__(self, out_dir: Path) -> None:
        self._out_dir = out_dir

    def plot(self, models: list[ModelMetrics]) -> None:
        if not models:
            return
        labels = ["top1", "top3", "top5", "macro P", "macro R", "macro F1",
                  "weighted F1", "confidence"]
        n = len(labels)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        for m in models:
            vals = [
                m.top1, m.top3, m.top5,
                m.macro_p, m.macro_r, m.macro_f1,
                m.weighted_f1, m.mean_conf,
            ]
            vals += vals[:1]
            ax.plot(angles, vals, linewidth=2, label=m.name)
            ax.fill(angles, vals, alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title("Radar — общее сравнение моделей")
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=8)
        fig.tight_layout()
        fig.savefig(self._out_dir / "08_radar.png", dpi=140)
        plt.close(fig)
        print("saved 08_radar.png")


class PerClassMetricDump:
    def __init__(self, out_dir: Path) -> None:
        self._out_dir = out_dir

    def dump(self, models: list[ModelMetrics], classes: list[str]) -> None:
        for m in models:
            lines = ["idx,class,precision,recall,f1,support"]
            for j, c in enumerate(classes):
                pc = m.per_class[j]
                lines.append(
                    f"{j},{c},{pc['precision']:.6f},{pc['recall']:.6f},"
                    f"{pc['f1']:.6f},{pc['support']}"
                )
            (self._out_dir / f"per_class_{m.name}.csv").write_text(
                "\n".join(lines)
            )
        print("saved per_class_*.csv")


class VisualizeApp:
    def __init__(
        self,
        metrics_path: Path,
        coco_json: Path,
        out_dir: Path,
    ) -> None:
        self._metrics_path = metrics_path
        self._coco_json = coco_json
        self._out_dir = out_dir
        self._out_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        models = MetricsLoader(self._metrics_path).load()
        classes = ClassNamesLoader(self._coco_json).load()
        print(f"models: {[m.name for m in models]}")
        print(f"classes: {len(classes)}")

        GlobalBarPlotter(self._out_dir).plot(models)
        PerClassF1Plotter(self._out_dir).plot(models, classes)
        PerClassRecallPlotter(self._out_dir).plot(models, classes)
        SupportPlotter(self._out_dir).plot(models, classes)
        RadarPlotter(self._out_dir).plot(models)
        PerClassMetricDump(self._out_dir).dump(models, classes)

        loader = ConfusionLoader(self._metrics_path.parent)
        for m in models:
            cm = loader.load(m.name)
            if cm is None:
                print(f"skip confusion {m.name}: csv not found")
                continue
            ConfusionPlotter(self._out_dir).plot(m.name, cm, classes)
            ConfusionNormalizedPlotter(self._out_dir).plot(m.name, cm, classes)
            TopErrorsPlotter(self._out_dir).plot(m.name, cm, classes)


def main() -> None:
    VisualizeApp(
        metrics_path=Path("runs/soc40_eval/metrics.json"),
        coco_json=Path(
            "data/_downloads/Ground_Range/Annotation_COCO/"
            "annotation_coco/SOC_40classes/annotations/train.json"
        ),
        out_dir=Path("runs/soc40_eval/figures"),
    ).run()


if __name__ == "__main__":
    main()