import argparse
import csv
import json
import math
import os
import random
import shutil
import socket
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

if __name__ == "__main__":
    print("[python] Démarrage ; import de NumPy et PyTorch...", flush=True)

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

if __name__ == "__main__":
    print("[python] NumPy et PyTorch importés.", flush=True)


MODEL_ID = "openai-community/openai-gpt"
TEXT_COLUMNS = {
    "sst2": ("sentence",),
    "cola": ("sentence",),
    "mrpc": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "mnli": ("premise", "hypothesis"),
}
TASK_NAMES = list(TEXT_COLUMNS)
SCORE_NAMES = {name: "accuracy" for name in TASK_NAMES}
SCORE_NAMES.update(cola="mcc", mrpc="f1", qqp="f1")
N_LAYERS = 12
N_EXPERIMENTS = len(TASK_NAMES) * (N_LAYERS + 1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_dataset(data, columns, tokenizer, max_length, batch_size=1024):
    # Tokeniser par petits morceaux évite de garder toutes les listes en mémoire.
    input_ids = torch.empty((len(data), max_length), dtype=torch.long)
    attention_mask = torch.empty((len(data), max_length), dtype=torch.bool)
    separator = tokenizer.encode(" ; ", add_special_tokens=False)
    first_limit = (max_length - len(separator)) // 2
    second_limit = max_length - len(separator) - first_limit
    for start in range(0, len(data), batch_size):
        batch = data[start:start + batch_size]
        if len(columns) == 1:
            sequences = tokenizer(
                batch[columns[0]], add_special_tokens=False,
                truncation=True, max_length=max_length,
            )["input_ids"]
        else:
            first = tokenizer(
                batch[columns[0]], add_special_tokens=False,
                truncation=True, max_length=first_limit,
            )["input_ids"]
            second = tokenizer(
                batch[columns[1]], add_special_tokens=False,
                truncation=True, max_length=second_limit,
            )["input_ids"]
            sequences = [a + separator + b for a, b in zip(first, second)]
        encoded = tokenizer.pad(
            {"input_ids": sequences}, padding="max_length", max_length=max_length,
            return_attention_mask=True, return_tensors="pt",
        )
        if not (encoded["attention_mask"].sum(dim=1) > 0).all():
            raise ValueError("Une entrée ne contient aucun token.")
        end = start + len(sequences)
        input_ids[start:end] = encoded["input_ids"]
        attention_mask[start:end] = encoded["attention_mask"].bool()
        if (start // batch_size) % 100 == 0 or end == len(data):
            print(f"  Tokenisation : {end}/{len(data)}", flush=True)
    labels = torch.tensor(list(data["label"]), dtype=torch.long)
    return [input_ids, attention_mask, labels]


def prepare(args):
    # Une seule préparation, sur CPU. Les caches existants peuvent être réutilisés.
    from transformers import AutoTokenizer, OpenAIGPTConfig
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    args.data_dir.mkdir(parents=True, exist_ok=True)
    metadata_file = args.data_dir / "metadata.json"
    if metadata_file.exists():
        raise ValueError("Ce dossier est déjà préparé. Choisir un nouveau --data-dir.")
    model_dir = args.data_dir / "model"
    if args.model_dir is not None:
        if args.model_dir.resolve() == model_dir.resolve():
            raise ValueError("--model-dir doit désigner les anciens poids, dans un autre dossier.")
        shutil.copytree(args.model_dir, model_dir, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns(".cache"))
    else:
        snapshot_download(
            MODEL_ID, local_dir=str(model_dir),
            allow_patterns=[
                "config.json", "model.safetensors", "tokenizer.json",
                "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt",
            ],
        )
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.unk_token
    config = OpenAIGPTConfig.from_pretrained(model_dir, local_files_only=True)
    if not 4 <= args.max_length <= config.n_positions or config.n_layer != N_LAYERS:
        raise ValueError("Longueur invalide ou checkpoint différent de GPT-1 à 12 blocs.")

    for name, columns in TEXT_COLUMNS.items():
        raw = load_dataset("nyu-mll/glue", name)
        split = "validation_matched" if name == "mnli" else "validation"
        train = raw["train"].shuffle(seed=args.seed)
        validation = raw[split].shuffle(seed=args.seed)
        if args.max_train_size is not None:
            train = train.select(range(min(args.max_train_size, len(train))))
        if args.max_eval_size is not None:
            validation = validation.select(range(min(args.max_eval_size, len(validation))))
        print(f"Préparation de {name.upper()} : {len(train)} train, {len(validation)} validation", flush=True)
        torch.save({
            "train": encode_dataset(train, columns, tokenizer, args.max_length),
            "eval": encode_dataset(validation, columns, tokenizer, args.max_length),
            "num_labels": raw["train"].features["label"].num_classes,
        }, args.data_dir / f"{name}.pt")
        print(f"{name.upper()} : {len(train)} train, {len(validation)} validation", flush=True)

    metadata = {
        "model_id": MODEL_ID, "seed": args.seed, "max_length": args.max_length,
        "max_train_size": args.max_train_size, "max_eval_size": args.max_eval_size,
    }
    # Ce fichier est écrit à la fin : sa présence indique une préparation terminée.
    metadata_file.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("Préparation terminée :", args.data_dir.resolve())


class GPT1Classifier(nn.Module):
    def __init__(self, model_dir, num_labels, k):
        super().__init__()
        self.k = k
        print("Import de Transformers...", flush=True)
        from transformers import OpenAIGPTModel
        print("Chargement des poids GPT-1 sur CPU...", flush=True)
        self.backbone = OpenAIGPTModel.from_pretrained(model_dir, local_files_only=True)
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(self.backbone.config.n_embd, num_labels)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        # k=0 : tout est entraîné. k=12 : seule la tête est entraînée.
        if k > 0:
            self.backbone.tokens_embed.requires_grad_(False)
            self.backbone.positions_embed.requires_grad_(False)
        for block in self.backbone.h[:k]:
            block.requires_grad_(False)

    def forward(self, input_ids, attention_mask, compute_lm_loss=False):
        hidden = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask,
        ).last_hidden_state
        positions = attention_mask.sum(dim=1) - 1
        rows = torch.arange(input_ids.size(0), device=input_ids.device)
        logits = self.head(self.dropout(hidden[rows, positions]))
        if not compute_lm_loss:
            return logits
        # Même matrice pour lire les tokens et prédire le suivant : aucun nouveau poids.
        lm_logits = nn.functional.linear(hidden[:, :-1], self.backbone.tokens_embed.weight)
        targets = input_ids[:, 1:].masked_fill(~attention_mask[:, 1:].bool(), -100)
        token_losses = nn.functional.cross_entropy(
            lm_logits.reshape(-1, lm_logits.size(-1)), targets.reshape(-1),
            ignore_index=-100, reduction="none",
        ).reshape_as(targets)
        # Moyenne des pertes par phrase, en excluant le padding (comme dans le code du papier).
        counts = attention_mask[:, 1:].sum(dim=1).clamp_min(1)
        lm_loss = (token_losses.sum(dim=1) / counts).mean()
        return logits, lm_loss


def compute_scores(y_true, y_pred, num_labels):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    scores = {"accuracy": float(np.mean(y_true == y_pred))}
    if num_labels == 2:
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        scores["f1"] = 2 * tp / max(2 * tp + fp + fn, 1)
        denominator = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        scores["mcc"] = (tp * tn - fp * fn) / denominator if denominator else 0.0
    return scores


def make_optimizer(model, learning_rate):
    # Pas de weight decay sur les biais et les gains des normalisations.
    parameters = [p for p in model.parameters() if p.requires_grad]
    groups = [
        {"params": [p for p in parameters if p.ndim > 1], "weight_decay": 0.01},
        {"params": [p for p in parameters if p.ndim == 1], "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)


def make_scheduler(optimizer, total_steps, warmup_ratio):
    def multiplier(step):
        progress = min(step / total_steps, 1.0)
        warmup = min(progress / warmup_ratio, 1.0) if warmup_ratio else 1.0
        # Formule warmup_linear du code OpenAI de 2018.
        return warmup * (1.0 - progress)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def train_one_epoch(model, loader, optimizer, scheduler, device, lm_weight):
    model.train()
    if model.k > 0:
        model.backbone.drop.eval()
    for block in model.backbone.h[:model.k]:
        block.eval()
    loss_sums = {"train_loss": 0.0, "classification_loss": 0.0, "lm_loss": 0.0}
    # k=12 : la perte LM est constante, elle ne dépend pas de la tête de classification.
    lm_active = lm_weight > 0 and model.k < N_LAYERS
    for step, (ids, mask, labels) in enumerate(loader, start=1):
        ids, mask, labels = [x.to(device) for x in (ids, mask, labels)]
        optimizer.zero_grad(set_to_none=True)
        if lm_active:
            logits, lm_loss = model(ids, mask, compute_lm_loss=True)
        else:
            logits, lm_loss = model(ids, mask), torch.zeros((), device=device)
        classification_loss = nn.functional.cross_entropy(logits, labels)
        loss = classification_loss + lm_weight * lm_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        for key, value in zip(loss_sums, (loss, classification_loss, lm_loss)):
            loss_sums[key] += value.item() * labels.size(0)
        if step % 500 == 0 or step == len(loader):
            print(f"  Batch {step}/{len(loader)} | loss={loss.item():.3f}", flush=True)
    return {key: value / len(loader.dataset) for key, value in loss_sums.items()}


@torch.no_grad()
def evaluate(model, loader, device, num_labels):
    model.eval()
    y_true, y_pred = [], []
    for ids, mask, labels in loader:
        logits = model(ids.to(device), mask.to(device))
        y_true.extend(labels.tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())
    return compute_scores(y_true, y_pred, num_labels)


def check_cuda(args):
    """Diagnostic à lancer dans une allocation Slurm GPU, sans charger GPT-1."""
    print("Nœud :", socket.gethostname(), flush=True)
    print("PyTorch :", torch.__version__, "| CUDA PyTorch :", torch.version.cuda, flush=True)
    for key in (
        "SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS",
        "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "GPU_DEVICE_ORDINAL",
    ):
        print(f"{key}={os.environ.get(key, '(non défini)')}", flush=True)
    for query in (
        ["--query-gpu=index,uuid,name,driver_version,compute_mode,memory.used", "--format=csv"],
        ["--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv"],
    ):
        try:
            subprocess.run(["nvidia-smi", *query], check=False, timeout=15)
        except (OSError, subprocess.TimeoutExpired) as exc:
            print("nvidia-smi indisponible :", exc, flush=True)
    print("CUDA disponible :", torch.cuda.is_available(), flush=True)
    print("GPU visibles par PyTorch :", torch.cuda.device_count(), flush=True)
    print("Test : allocation d'un seul nombre sur cuda:0...", flush=True)
    probe = torch.ones(1, device="cuda:0")
    torch.cuda.synchronize()
    print("Test CUDA réussi :", probe.item(), "sur", torch.cuda.get_device_name(0), flush=True)


def train(args):
    task_index, k = divmod(args.experiment_id, N_LAYERS + 1)
    name = TASK_NAMES[task_index]
    data_config = json.loads((args.data_dir / "metadata.json").read_text())
    train_config = {
        "epochs": args.epochs, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "weight_decay": 0.01,
        "optimizer": "AdamW", "betas": [0.9, 0.999], "epsilon": 1e-8,
        "decay_bias_and_norm": False, "max_grad_norm": 1.0,
        "schedule": "openai_warmup_linear", "warmup_ratio": args.warmup_ratio,
        "classifier_dropout": 0.1, "classifier_init_std": 0.02,
        "lm_weight": args.lm_weight,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_file = args.output_dir / f"{name}_k{k:02d}.json"
    if result_file.exists() and not args.overwrite:
        previous = json.loads(result_file.read_text())
        if previous["data_config"] != data_config or previous["train_config"] != train_config:
            raise ValueError("Résultat avec d'autres réglages : utiliser un nouveau --output-dir.")
        print("Déjà terminé, essai ignoré :", result_file)
        return

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA indisponible. Vérifier le job GPU et l'installation de PyTorch.")
    if device.type == "cuda":
        print("Vérification de l'accès au GPU avant de charger GPT-1...", flush=True)
        try:
            probe = torch.empty(1, device=device)
            torch.cuda.synchronize()
            del probe
        except RuntimeError as exc:
            raise RuntimeError(
                "L'accès CUDA échoue avant le chargement du modèle. "
                "Lancer 'python supervised_fine_tuning.py check-cuda' dans un job GPU "
                "pour vérifier l'allocation Slurm, les GPU visibles et leur état."
            ) from exc
    set_seed(data_config["seed"])
    payload = torch.load(args.data_dir / f"{name}.pt", map_location="cpu", weights_only=True)
    train_data = TensorDataset(*payload["train"])
    eval_data = TensorDataset(*payload["eval"])
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(data_config["seed"]),
        pin_memory=device.type == "cuda",
    )
    eval_loader = DataLoader(eval_data, batch_size=args.batch_size, pin_memory=device.type == "cuda")
    model = GPT1Classifier(args.data_dir / "model", payload["num_labels"], k)
    print(f"Transfert du modèle sur {device}...", flush=True)
    model = model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = make_optimizer(model, args.learning_rate)
    total_steps = args.epochs * len(train_loader)
    scheduler = make_scheduler(optimizer, total_steps, args.warmup_ratio)
    lm_active = args.lm_weight > 0 and k < N_LAYERS
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    metric = SCORE_NAMES[name]
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else str(device)
    print(f"Essai {args.experiment_id} : {name}, k={k}, appareil={gpu}, paramètres={n_trainable:,}", flush=True)
    print(f"{len(train_data)} train, {len(eval_data)} validation, {total_steps} mises à jour, batch={args.batch_size}", flush=True)
    if args.lm_weight > 0 and not lm_active:
        print("k=12 : perte LM constante, calcul ignoré ; seule la tête est entraînée.", flush=True)
    start = time.perf_counter()
    history = []
    for epoch in range(args.epochs):
        losses = train_one_epoch(model, train_loader, optimizer, scheduler, device, args.lm_weight)
        scores = evaluate(model, eval_loader, device, payload["num_labels"])
        learning_rate = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch + 1, **losses, "learning_rate": learning_rate, **scores})
        print(f"Époque {epoch + 1}/{args.epochs} | loss={losses['train_loss']:.3f} "
              f"| classification={losses['classification_loss']:.3f} "
              f"| LM={losses['lm_loss']:.3f} | {metric}={scores[metric]:.3f}", flush=True)

    majority = int(np.bincount(payload["train"][2].numpy()).argmax())
    baseline = compute_scores(
        payload["eval"][2].numpy(), np.full(len(eval_data), majority), payload["num_labels"],
    )
    result = {
        "experiment_id": args.experiment_id, "task": name, "k": k,
        "trainable_parameters": n_trainable, "total_parameters": n_total,
        "trainable_fraction": n_trainable / n_total,
        "metric": metric, "score": scores[metric], "majority_score": baseline[metric],
        "accuracy": scores["accuracy"], "f1": scores.get("f1", ""), "mcc": scores.get("mcc", ""),
        **losses, "lm_active": lm_active, "optimizer_steps": total_steps,
        "seconds": time.perf_counter() - start,
        "n_train": len(train_data), "n_eval": len(eval_data), "device": gpu,
        "data_config": data_config, "train_config": train_config, "history": history,
        "versions": {package: version(package) for package in ("torch", "transformers", "datasets", "numpy")},
    }
    # Chaque essai a son fichier : pas d'écritures concurrentes dans un même CSV.
    temporary = result_file.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(result_file)
    print(f"Terminé en {result['seconds'] / 60:.1f} min : {result_file}", flush=True)


def collect(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = [json.loads(p.read_text()) for p in args.output_dir.glob("*_k[0-9][0-9].json")]
    if not results:
        raise ValueError("Aucun résultat dans ce dossier.")
    results.sort(key=lambda row: row["experiment_id"])
    signatures = {json.dumps([r["data_config"], r["train_config"]], sort_keys=True) for r in results}
    if len(signatures) != 1:
        raise ValueError("Réglages différents dans ce dossier : séparer les campagnes d'expériences.")
    fields = [key for key in results[0] if key not in ("data_config", "train_config", "history", "versions")]
    with (args.output_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    for name in TASK_NAMES:
        rows = [row for row in results if row["task"] == name]
        if not rows:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
        axes[0].plot([r["k"] for r in rows], [r["score"] for r in rows], "o-")
        axes[0].set_xlabel("Nombre de blocs gelés (k)")
        axes[0].set_xticks([r["k"] for r in rows])
        by_parameters = sorted(rows, key=lambda r: r["trainable_fraction"])
        axes[1].plot([100 * r["trainable_fraction"] for r in by_parameters], [r["score"] for r in by_parameters], "o-")
        axes[1].set_xlabel("Paramètres entraînés (% du modèle avec tête)")
        for ax in axes:
            ax.axhline(rows[0]["majority_score"], color="gray", linestyle="--", label="Majorité")
            ax.set_ylabel(rows[0]["metric"])
            ax.set_ylim(-1.05 if rows[0]["metric"] == "mcc" else 0, 1.05)
            ax.grid(alpha=0.3)
            ax.legend()
        fig.suptitle(name.upper())
        fig.tight_layout()
        fig.savefig(args.output_dir / f"{name}.png", dpi=150)
        plt.close(fig)
    missing = sorted(set(range(N_EXPERIMENTS)) - {r["experiment_id"] for r in results})
    print(f"{len(results)}/{N_EXPERIMENTS} résultats collectés dans {args.output_dir}")
    if missing:
        print("Indices manquants :", ",".join(map(str, missing)))


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("La valeur doit être strictement positive.")
    return value


def nonnegative_float(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("La valeur doit être positive ou nulle et finie.")
    return value


def positive_float(value):
    value = nonnegative_float(value)
    if value == 0:
        raise argparse.ArgumentTypeError("La valeur doit être strictement positive.")
    return value


def ratio(value):
    value = nonnegative_float(value)
    if value >= 1:
        raise argparse.ArgumentTypeError("La proportion doit être comprise entre 0 inclus et 1 exclu.")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("check-cuda", help="Tester l'accès au GPU sans charger le modèle.")
    p.set_defaults(function=check_cuda)
    p = commands.add_parser("prepare", help="Télécharger les poids et préparer les 7 jeux de données.")
    p.add_argument("--data-dir", type=Path, default=Path("data_full"))
    p.add_argument("--model-dir", type=Path, help="Réutiliser des poids locaux, par exemple data/model.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train-size", type=positive_int, help="Limiter pour un test ; sinon tout utiliser.")
    p.add_argument("--max-eval-size", type=positive_int, help="Limiter pour un test ; sinon tout utiliser.")
    p.add_argument("--max-length", type=positive_int, default=128)
    p.set_defaults(function=prepare)
    p = commands.add_parser("train", help="Lancer un essai indépendant sur un GPU.")
    p.add_argument("--experiment-id", type=int, choices=range(N_EXPERIMENTS), required=True)
    p.add_argument("--data-dir", type=Path, default=Path("data_full"))
    p.add_argument("--output-dir", type=Path, default=Path("results_full"))
    p.add_argument("--epochs", type=positive_int, default=3)
    p.add_argument("--batch-size", type=positive_int, default=32)
    p.add_argument("--learning-rate", type=positive_float, default=6.25e-5)
    p.add_argument("--warmup-ratio", type=ratio, default=0.002)
    p.add_argument("--lm-weight", type=nonnegative_float, default=0.0,
                   help="Poids de la perte LM auxiliaire : 0 pour désactiver, 0.5 dans le papier.")
    p.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
    p.add_argument("--overwrite", action="store_true", help="Refaire un essai déjà terminé.")
    p.set_defaults(function=train)
    p = commands.add_parser("collect", help="Créer le CSV récapitulatif et les graphiques, sans GPU.")
    p.add_argument("--output-dir", type=Path, default=Path("results_full"))
    p.set_defaults(function=collect)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
