"""CIFAR-10 KNN experiments implemented with NumPy. See README.md for usage."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import time
from pathlib import Path

import numpy as np


METRICS = ("euclidean", "manhattan", "cosine")
SCORES = ("accuracy", "precision", "recall", "f1")
ROOT = Path(__file__).resolve().parent


def load_cifar10(data_dir: Path):
    """Read the official local CIFAR-10 Python batches (trusted files only)."""
    def read(name):
        path = data_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing CIFAR-10 file: {path}")
        with path.open("rb") as handle:
            return pickle.load(handle, encoding="bytes")

    batches = [read(f"data_batch_{i}") for i in range(1, 6)]
    test = read("test_batch")
    names = [name.decode("utf-8") for name in read("batches.meta")[b"label_names"]]
    train_x = np.concatenate([batch[b"data"] for batch in batches])
    train_y = np.concatenate([np.asarray(batch[b"labels"]) for batch in batches])
    return train_x, train_y, test[b"data"], np.asarray(test[b"labels"]), names


def stratified_sample(labels, size, seed):
    """Sample without replacement, retaining class proportions and exact size."""
    if not 1 <= size <= len(labels):
        raise ValueError(f"Sample size must be in [1, {len(labels)}], got {size}")
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(labels, return_counts=True)
    quotas = counts * (size / len(labels))
    allocations = np.floor(quotas).astype(int)
    remainder = size - int(allocations.sum())
    allocations[np.argsort(-(quotas - allocations), kind="stable")[:remainder]] += 1
    selected = np.concatenate([
        rng.permutation(np.flatnonzero(labels == label))[:count]
        for label, count in zip(classes, allocations)
    ])
    return rng.permutation(selected)


def make_features(pixels, image_size=8):
    """Average-pool each RGB channel, flatten, and scale fixed pixels to [0, 1]."""
    if image_size not in (4, 8, 16, 32):
        raise ValueError("image_size must be one of 4, 8, 16, 32")
    images = pixels.reshape(-1, 3, 32, 32).astype(np.float32)
    block = 32 // image_size
    images = images.reshape(-1, 3, image_size, block, image_size, block)
    pooled = images.mean(axis=(3, 5))
    return np.ascontiguousarray(pooled.reshape(len(pixels), -1) / 255.0)


def stratified_folds(labels, n_splits=5, seed=42):
    """Each sample is validation data exactly once; classes are shuffled separately."""
    rng = np.random.default_rng(seed)
    buckets = [[] for _ in range(n_splits)]
    for label in np.unique(labels):
        indices = rng.permutation(np.flatnonzero(labels == label))
        if len(indices) < n_splits:
            raise ValueError(f"Class {label} needs at least {n_splits} samples")
        for fold, part in enumerate(np.array_split(indices, n_splits)):
            buckets[fold].append(part)
    all_indices = np.arange(len(labels))
    result = []
    for bucket in buckets:
        validation = np.sort(np.concatenate(bucket))
        keep = np.ones(len(labels), dtype=bool)
        keep[validation] = False
        result.append((all_indices[keep], validation))
    return result


def classification_scores(y_true, y_pred, n_classes=10):
    """Accuracy plus macro-averaged precision, recall, F1; zero divisions become 0."""
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(confusion, (y_true, y_pred), 1)
    tp = confusion.diagonal().astype(float)
    predicted = confusion.sum(axis=0)
    support = confusion.sum(axis=1)
    precision = np.divide(tp, predicted, out=np.zeros(n_classes), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros(n_classes), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(n_classes), where=(precision + recall) != 0)
    scores = dict(accuracy=float(tp.sum() / len(y_true)),
                  precision=float(precision.mean()), recall=float(recall.mean()),
                  f1=float(f1.mean()))
    per_class = [dict(label=i, precision=float(precision[i]), recall=float(recall[i]),
                      f1=float(f1[i]), support=int(support[i])) for i in range(n_classes)]
    return scores, confusion, per_class


class KNNClassifier:
    """Brute-force KNN, with bounded distance blocks and uniform majority voting."""

    def __init__(self, metric="euclidean", query_batch_size=64, train_batch_size=256):
        if metric not in METRICS:
            raise ValueError(f"Unknown metric: {metric}")
        if query_batch_size < 1 or train_batch_size < 1:
            raise ValueError("Batch sizes must be positive")
        self.metric = metric
        self.query_batch_size = query_batch_size
        self.train_batch_size = train_batch_size

    def fit(self, x, y):
        self.x = np.ascontiguousarray(x, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
        if self.x.ndim != 2 or len(self.x) != len(self.y) or len(self.y) == 0:
            raise ValueError("Training data must be a nonempty 2D array matching labels")
        if np.any(self.y < 0):
            raise ValueError("Labels must be nonnegative integers")
        self.n_classes = int(self.y.max()) + 1
        self.norm_squared = np.einsum("ij,ij->i", self.x, self.x)
        self.norms = np.sqrt(self.norm_squared)
        return self

    def _distances(self, queries):
        # Euclidean ordering is identical for d and d^2; no square root is needed.
        if self.metric == "euclidean":
            q_squared = np.einsum("ij,ij->i", queries, queries)
            return np.maximum(q_squared[:, None] + self.norm_squared[None, :]
                              - 2 * (queries @ self.x.T), 0)
        if self.metric == "cosine":
            q_norms = np.linalg.norm(queries, axis=1)
            denominator = q_norms[:, None] * self.norms[None, :]
            similarity = np.divide(queries @ self.x.T, denominator,
                                   out=np.zeros_like(denominator), where=denominator != 0)
            distance = 1 - np.clip(similarity, -1, 1)
            # Explicit convention: two zero vectors have distance 0; one zero has 1.
            distance[(q_norms == 0)[:, None] & (self.norms == 0)[None, :]] = 0
            return distance
        distances = np.zeros((len(queries), len(self.x)), dtype=np.float32)
        # Chunk features as well, so the temporary array stays small at 32x32.
        for start in range(0, len(self.x), self.train_batch_size):
            train = self.x[start:start + self.train_batch_size]
            block = distances[:, start:start + len(train)]
            for feature in range(0, self.x.shape[1], 128):
                delta = queries[:, None, feature:feature + 128] - train[None, :, feature:feature + 128]
                np.abs(delta, out=delta)
                block += delta.sum(axis=2)
        return distances

    def predict_many_k(self, queries, k_values):
        """Reuse one neighbor ordering for all requested k values."""
        queries = np.asarray(queries, dtype=np.float32)
        ks = sorted(set(k_values))
        if not ks or ks[0] < 1 or ks[-1] > len(self.x):
            raise ValueError(f"k must be in [1, {len(self.x)}]")
        if queries.ndim != 2 or queries.shape[1] != self.x.shape[1]:
            raise ValueError("Query feature dimensions must match training data")
        predictions = {k: np.empty(len(queries), dtype=np.int64) for k in ks}
        for start in range(0, len(queries), self.query_batch_size):
            batch = queries[start:start + self.query_batch_size]
            distances = self._distances(batch)
            # Equal distances retain training order. Equal vote counts choose lower label.
            neighbors = np.argsort(distances, axis=1, kind="stable")[:, :ks[-1]]
            labels = self.y[neighbors]
            votes = np.zeros((len(batch), self.n_classes), dtype=np.int64)
            rows = np.arange(len(batch))
            for rank in range(1, ks[-1] + 1):
                votes[rows, labels[:, rank - 1]] += 1
                if rank in predictions:
                    predictions[rank][start:start + len(batch)] = votes.argmax(axis=1)
        return predictions


def cross_validate(x, y, k_values, metrics, seed, query_batch_size, train_batch_size):
    folds = stratified_folds(y, n_splits=5, seed=seed)
    if max(k_values) > min(len(train) for train, _ in folds):
        raise ValueError("Largest k exceeds the training size of a CV fold")
    rows = []
    for metric in metrics:
        for fold, (train, validation) in enumerate(folds, start=1):
            started = time.perf_counter()
            model = KNNClassifier(metric, query_batch_size, train_batch_size).fit(x[train], y[train])
            predictions = model.predict_many_k(x[validation], k_values)
            elapsed = time.perf_counter() - started
            for k, prediction in predictions.items():
                scores, _, _ = classification_scores(y[validation], prediction)
                rows.append(dict(metric=metric, k=k, fold=fold, **scores,
                                 shared_predict_seconds=elapsed))
            print(f"  {metric:10s} fold {fold}/5: {elapsed:.2f}s", flush=True)
    summary = []
    for metric in metrics:
        for k in k_values:
            selected = [row for row in rows if row["metric"] == metric and row["k"] == k]
            result = dict(metric=metric, k=k)
            for score in SCORES:
                values = [row[score] for row in selected]
                result[f"{score}_mean"] = float(np.mean(values))
                result[f"{score}_std"] = float(np.std(values, ddof=1))
            summary.append(result)
    # Primary criterion: mean accuracy; ties use F1, then smaller k, then metric name.
    summary.sort(key=lambda row: (-row["accuracy_mean"], -row["f1_mean"], row["k"], row["metric"]))
    return rows, summary


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plots(output_dir, summary, confusion, names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, score in zip(axes.flat, SCORES):
        for metric in dict.fromkeys(row["metric"] for row in summary):
            selected = sorted([row for row in summary if row["metric"] == metric], key=lambda r: r["k"])
            ax.errorbar([row["k"] for row in selected],
                        [row[f"{score}_mean"] for row in selected],
                        yerr=[row[f"{score}_std"] for row in selected], marker="o", capsize=3, label=metric)
        ax.set(title=f"5-fold CV: {score} (mean +/- sample SD)", xlabel="k", ylabel=score)
        ax.set_xticks(sorted({row["k"] for row in summary}))
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(output_dir / "cv_comparison.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    im = ax.imshow(confusion, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(xticks=range(len(names)), yticks=range(len(names)), xticklabels=names,
           yticklabels=names, xlabel="Predicted class", ylabel="True class",
           title="Held-out test confusion matrix (CV-selected model)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, str(confusion[i, j]), ha="center", va="center",
                    color="white" if confusion[i, j] > confusion.max() / 2 else "black")
    fig.savefig(output_dir / "test_confusion_matrix.png", dpi=160)
    plt.close(fig)


def write_report(output_dir, config, summary, test_scores):
    best = summary[0]
    lines = ["# CIFAR-10 KNN 실험 결과", "",
             f"- 학습 표본: {config['train_size']:,}개 / 별도 테스트 표본: {config['test_size']:,}개",
             f"- 이미지: {config['image_size']}×{config['image_size']} RGB, 특성 {config['features']}개",
             f"- 표본 추출·분할 seed: {config['seed']}",
             "- 검증: 동일한 Stratified 5-fold 분할을 모든 조합에 사용",
             "- Precision, Recall, F1: 클래스별 값을 동일 가중치로 평균한 macro 방식",
             "- 표준편차: 5개 fold 점수의 표본 표준편차(ddof=1); 신뢰구간을 뜻하지 않음", "",
             "## 교차검증 비교", "",
             "| 거리 | k | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in summary:
        values = [f"{row[f'{score}_mean']:.4f} ± {row[f'{score}_std']:.4f}" for score in SCORES]
        lines.append(f"| {row['metric']} | {row['k']} | " + " | ".join(values) + " |")
    lines += ["", "## 선택된 모델의 별도 테스트 결과", "",
              f"교차검증 평균 Accuracy가 가장 높은 **{best['metric']}, k={best['k']}**를 선택했다. "
              "동률이면 평균 macro F1, 작은 k, 거리 이름 순으로 선택했다. "
              "테스트 데이터는 모델 선택에 사용하지 않고, 선택 후 한 번 평가했다.", "",
              "| Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |",
              "|---:|---:|---:|---:|",
              "| " + " | ".join(f"{test_scores[score]:.4f}" for score in SCORES) + " |", "",
              "## 간단한 비교 및 설명", ""]
    for metric in config["metrics"]:
        candidates = [row for row in summary if row["metric"] == metric]
        winner = max(candidates, key=lambda row: row["accuracy_mean"])
        minimum = min(row["accuracy_mean"] for row in candidates)
        maximum = max(row["accuracy_mean"] for row in candidates)
        lines.append(f"- {metric}: 최상위 k={winner['k']}, 평균 Accuracy={winner['accuracy_mean']:.4f}. "
                     f"실험한 k에 따른 Accuracy 범위는 {minimum:.4f}~{maximum:.4f}이다.")
    lines += [f"- 선택 조합의 CV Accuracy는 {best['accuracy_mean']:.4f}, "
              f"테스트 Accuracy는 {test_scores['accuracy']:.4f}이다. 두 값은 서로 다른 표본에서 계산되므로 차이가 날 수 있다.",
              "- k가 작으면 가까운 소수 표본의 영향을 크게 받고, 커지면 다수결이 안정되는 대신 다른 클래스가 섞일 수 있다. "
              "따라서 성능이 k에 따라 반드시 단조롭게 증가하지는 않는다.",
              "- 유클리드는 큰 픽셀 차이를 제곱하여 반영하고, 맨해튼은 절댓값 차이를 합한다. "
              "코사인은 벡터의 크기보다 방향을 비교한다. 각 거리의 우열은 위 실험 범위에서 판단해야 한다.",
              "- 픽셀 기반 KNN은 물체의 위치·배경·조명 변화에 민감하며, 물체의 의미를 학습하는 특징 추출기는 사용하지 않았다."]
    if config["image_size"] != 32:
        lines.append(f"- 이번 실험은 평균 풀링으로 이미지를 {config['image_size']}×{config['image_size']}로 축소했다. "
                     "계산량을 줄이지만 세부 정보도 사라지므로 원본 32×32 결과와 구분해야 한다.")
    if config["train_size"] < 50000 or config["test_size"] < 10000:
        lines.append("- 전체 CIFAR-10이 아닌 층화 추출한 부분집합 결과이다. 전체 데이터의 성능으로 일반화해서 보고하지 않는다.")
    lines += ["", "## 생성 파일", "",
              "- `cv_folds.csv`: 각 조합의 fold별 점수 (시간은 같은 fold/거리의 모든 k에 공유된 예측 시간)",
              "- `cv_summary.csv`: 교차검증 평균과 표준편차",
              "- `test_metrics.csv`, `test_per_class.csv`: 최종 테스트 전체/클래스별 평가",
              "- `test_confusion_matrix.csv`, `test_predictions.csv`: 혼동행렬과 예측",
              "- `cv_comparison.png`, `test_confusion_matrix.png`: 결과 그래프",
              "- `config.json`: 실행 조건 및 선택 조합", ""]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "cifar-10-batches-py")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--image-size", type=int, choices=(4, 8, 16, 32), default=8)
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 3, 5, 7, 9])
    parser.add_argument("--metrics", nargs="+", choices=METRICS, default=list(METRICS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--train-batch-size", type=int, default=256)
    args = parser.parse_args()
    args.k_values = sorted(set(args.k_values))
    args.metrics = list(dict.fromkeys(args.metrics))
    if min(args.k_values) < 1 or args.query_batch_size < 1 or args.train_batch_size < 1:
        parser.error("k values and batch sizes must be positive")
    return args


def main():
    args = parse_args()
    started = time.perf_counter()
    raw_train, labels, raw_test, test_labels, names = load_cifar10(args.data_dir)
    train_indices = stratified_sample(labels, args.train_size, args.seed)
    test_indices = stratified_sample(test_labels, args.test_size, args.seed + 1)
    x = make_features(raw_train[train_indices], args.image_size)
    y = labels[train_indices]
    test_x = make_features(raw_test[test_indices], args.image_size)
    test_y = test_labels[test_indices]
    del raw_train, raw_test
    print(f"Train={len(x)}, test={len(test_x)}, features={x.shape[1]}, seed={args.seed}", flush=True)
    folds, summary = cross_validate(x, y, args.k_values, args.metrics, args.seed,
                                    args.query_batch_size, args.train_batch_size)
    best = summary[0]
    print(f"Selected by CV: {best['metric']}, k={best['k']}, accuracy={best['accuracy_mean']:.4f}", flush=True)
    model = KNNClassifier(best["metric"], args.query_batch_size, args.train_batch_size).fit(x, y)
    prediction = model.predict_many_k(test_x, [best["k"]])[best["k"]]
    scores, confusion, per_class = classification_scores(test_y, prediction)
    config = dict(train_size=len(x), test_size=len(test_x), image_size=args.image_size,
                  features=x.shape[1], seed=args.seed, k_values=args.k_values,
                  metrics=args.metrics, folds=5, averaging="macro", class_names=names,
                  data_dir=str(args.data_dir.resolve()), query_batch_size=args.query_batch_size,
                  train_batch_size=args.train_batch_size,
                  preprocessing="per-channel average pooling, flatten, fixed divide by 255",
                  numpy_version=np.__version__, selected_metric=best["metric"], selected_k=best["k"],
                  selection_rule="CV accuracy descending, CV macro F1 descending, k ascending, metric name ascending")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "cv_folds.csv", folds)
    write_csv(output_dir / "cv_summary.csv", summary)
    write_csv(output_dir / "test_metrics.csv", [dict(metric=best["metric"], k=best["k"], **scores)])
    write_csv(output_dir / "test_per_class.csv", [dict(class_name=names[row["label"]], **row) for row in per_class])
    write_csv(output_dir / "test_confusion_matrix.csv", [dict(true_class=name, **dict(zip(names, row.tolist())))
                                                        for name, row in zip(names, confusion)])
    write_csv(output_dir / "test_predictions.csv", [dict(test_batch_index=int(idx), true_label=int(actual),
                                                        predicted_label=int(pred))
                                                   for idx, actual, pred in zip(test_indices, test_y, prediction)])
    make_plots(output_dir, summary, confusion, names)
    config["elapsed_seconds"] = time.perf_counter() - started
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output_dir, config, summary, scores)
    print("Test: " + ", ".join(f"{key}={value:.4f}" for key, value in scores.items()), flush=True)
    print(f"Results: {output_dir.resolve()} ({config['elapsed_seconds']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
