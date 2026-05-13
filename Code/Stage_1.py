import csv
import os
import time
import random
import inspect
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    roc_auc_score,
)
from sklearn.decomposition import PCA
from scipy.special import softmax

from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    TrainerCallback,
)

# =========================================================
# GLOBAL CONFIG
# =========================================================
PCA_COMPONENTS = 256
SEED = 42
MAX_LENGTH = 1022

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== PATHS ======
model_checkpoint = "/home/adeel/Transformer/esm2_t30_150M_UR50D/"
poz_path = "/home/adeel/pos_train.csv"
non_path = "/home/adeel/non_train.csv"
val_poz_path = "/home/adeel/pos_test.csv"
val_non_path = "/home/adeel/non_test.csv"
aaindex_file = "/home/adeel/aaindex_feature.csv"
output_dir = "/home/adeel/Transformer/Output/"


# =========================================================
# AAINDEX FEATURE EXTRACTION
# =========================================================
def aaindex(sequences, feature_file=aaindex_file):
    try:
        with open(feature_file, "r") as file:
            csv_reader = csv.reader(file)
            _ = next(csv_reader)
            data_dict = {}

            first_row = next(csv_reader)
            feature_length = len(first_row[1:])
            data_dict[first_row[0]] = [
                float(value) if value != "" else 0.0 for value in first_row[1:]
            ]

            for row in csv_reader:
                name = row[0]
                values = [float(value) if value != "" else 0.0 for value in row[1:]]
                if len(values) != feature_length:
                    raise ValueError(f"Feature length mismatch for {name}")
                data_dict[name] = values

    except FileNotFoundError:
        raise FileNotFoundError(f"AAIndex feature file not found: {feature_file}")

    encodings = []
    for sequence in sequences:
        code = []
        for aa in str(sequence).upper():
            if aa in data_dict:
                code.append(data_dict[aa])
            else:
                code.append([0.0] * feature_length)
        encodings.append(code)

    return encodings


# =========================================================
# BLOSUM62 FEATURE EXTRACTION
# =========================================================
def BLOSUM62(sequences):
    blosum62 = {
        "A": [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],
        "R": [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],
        "N": [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],
        "D": [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],
        "C": [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
        "Q": [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],
        "E": [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],
        "G": [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],
        "H": [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],
        "I": [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],
        "L": [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],
        "K": [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],
        "M": [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],
        "F": [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],
        "P": [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],
        "S": [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],
        "T": [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],
        "W": [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],
        "Y": [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],
        "V": [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4],
        "*": [0] * 20,
        "_": [0] * 20,
        "X": [0] * 20,
    }

    encodings = []
    for sequence in sequences:
        code = []
        for aa in str(sequence).upper():
            code.append(blosum62.get(aa, [0] * 20))
        encodings.append(code)

    return encodings


# =========================================================
# HANDCRAFTED FEATURES
# =========================================================
def extract_handcrafted_features(sequence):
    blosum62_seq = BLOSUM62([sequence])[0]
    aaindex_seq = aaindex([sequence], feature_file=aaindex_file)[0]

    blosum62_avg = np.mean(np.array(blosum62_seq), axis=0)
    aaindex_avg = np.mean(np.array(aaindex_seq), axis=0)

    return np.concatenate([blosum62_avg, aaindex_avg])  # 20 + 531 = 551


# =========================================================
# DATA LOADING AND TOKENIZATION
# =========================================================
def load_data(file_path):
    df = pd.read_csv(file_path)

    if "data" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{file_path} must contain columns named 'data' and 'label'.")

    df["data"] = df["data"].astype(str).str.upper()
    df["label"] = df["label"].astype(int)
    return df


def tokenize_data(tokenizer, data):
    return tokenizer(
        list(map(str, data)),
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )


# =========================================================
# CAPSULE NETWORK MODULES
# =========================================================
class PrimaryCapsule(nn.Module):
    def __init__(self, in_dim, out_capsules, out_dim):
        super().__init__()
        self.out_capsules = out_capsules
        self.capsules = nn.Linear(in_dim, out_capsules * out_dim)

    def forward(self, x):
        batch_size = x.size(0)
        outputs = self.capsules(x)
        outputs = outputs.view(batch_size, self.out_capsules, -1)
        return outputs


class DigitCapsule(nn.Module):
    def __init__(self, in_capsules, in_dim, out_capsules, out_dim, num_iterations=3):
        super().__init__()
        self.num_iterations = num_iterations
        self.W = nn.Parameter(torch.randn(1, in_capsules, out_capsules, in_dim, out_dim))

    def squash(self, inputs, dim=-1):
        norm = torch.norm(inputs, p=2, dim=dim, keepdim=True)
        scale = norm ** 2 / (1 + norm ** 2) / (norm + 1e-8)
        return scale * inputs

    def forward(self, u):
        batch_size = u.size(0)

        u = u.unsqueeze(2).unsqueeze(3)
        W = self.W.expand(batch_size, -1, -1, -1, -1)

        u_hat = torch.matmul(u, W).squeeze(3)

        b = torch.zeros(
            batch_size,
            u_hat.size(1),
            u_hat.size(2),
            1,
            device=u.device,
        )

        for i in range(self.num_iterations):
            c = F.softmax(b, dim=2)
            s = (c * u_hat).sum(dim=1)
            v = self.squash(s)

            if i < self.num_iterations - 1:
                agreement = (u_hat * v.unsqueeze(1)).sum(dim=-1, keepdim=True)
                b = b + agreement

        return v


# =========================================================
# HYBRID MODEL
# =========================================================
class HybridModel(nn.Module):
    def __init__(self, transformer_checkpoint, dropout=0.3):
        super().__init__()

        self.transformer = AutoModel.from_pretrained(transformer_checkpoint)
        transformer_dim = self.transformer.config.hidden_size

        self.feature_compressor = nn.Sequential(
            nn.Linear(PCA_COMPONENTS, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.primary_caps = PrimaryCapsule(
            in_dim=transformer_dim + 128,
            out_capsules=16,
            out_dim=8,
        )

        self.digit_caps = DigitCapsule(
            in_capsules=16,
            in_dim=8,
            out_capsules=2,
            out_dim=16,
        )

        self.classifier = nn.Sequential(
            nn.Linear(2 * 16, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None, labels=None, hc_features=None):
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        transformer_features = transformer_outputs.last_hidden_state[:, 0]

        if hc_features is not None:
            compressed_hc = self.feature_compressor(hc_features)
        else:
            compressed_hc = torch.zeros(
                transformer_features.size(0),
                128,
                device=transformer_features.device,
            )

        combined = torch.cat([transformer_features, compressed_hc], dim=1)

        primary_caps = self.primary_caps(combined)
        digit_caps = self.digit_caps(primary_caps)

        flattened = digit_caps.view(digit_caps.size(0), -1)
        logits = self.classifier(flattened)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)

        return {"loss": loss, "logits": logits} if loss is not None else logits


# =========================================================
# DATASET WITH PCA-COMPRESSED HANDCRAFTED FEATURES
# =========================================================
class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None, sequences=None, pca=None):
        self.encodings = encodings
        self.labels = labels
        self.sequences = sequences
        self.pca = pca

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx]) if not isinstance(val[idx], torch.Tensor)
            else val[idx].clone().detach()
            for key, val in self.encodings.items()
        }

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.sequences is not None:
            seq = self.sequences[idx]
            hc_features = extract_handcrafted_features(seq)

            if self.pca is not None:
                hc_features = self.pca.transform(
                    np.array(hc_features).reshape(1, -1)
                )[0]

            item["hc_features"] = torch.tensor(hc_features, dtype=torch.float32)

        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


# =========================================================
# METRICS
# =========================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred

    probabilities = softmax(logits, axis=1)[:, 1]
    preds = np.argmax(logits, axis=1)

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    mcc = matthews_corrcoef(labels, preds)

    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    auc_score = roc_auc_score(labels, probabilities)

    return {
        "eval_accuracy": accuracy,
        "eval_precision": precision,
        "eval_recall": recall,
        "eval_f1": f1,
        "eval_mcc": mcc,
        "eval_sensitivity": sensitivity,
        "eval_specificity": specificity,
        "eval_fpr": fpr,
        "eval_auc": auc_score,
    }


# =========================================================
# EARLY STOPPING
# =========================================================
class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, early_stopping_patience=8):
        self.early_stopping_patience = early_stopping_patience
        self.best_metric = None
        self.patience_counter = 0

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        metric = metrics.get("eval_accuracy", None)

        if metric is None:
            return

        if self.best_metric is None:
            self.best_metric = metric
            self.patience_counter = 0
            return

        if metric > self.best_metric:
            self.best_metric = metric
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.early_stopping_patience:
                control.should_training_stop = True
                print(f"Early stopping triggered at epoch {state.epoch}")


# =========================================================
# TRAINING
# =========================================================
def build_training_args():
    ta_params = set(inspect.signature(TrainingArguments).parameters.keys())

    args_dict = dict(
        output_dir=output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_dir="./logs",
        logging_steps=2,
        gradient_accumulation_steps=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        max_grad_norm=1.0,
        fp16=torch.cuda.is_available(),
        optim="adamw_torch",
        report_to="tensorboard",
        dataloader_num_workers=4,
        save_total_limit=2,
    )

    if "evaluation_strategy" in ta_params:
        args_dict["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in ta_params:
        args_dict["eval_strategy"] = "epoch"

    if "save_strategy" in ta_params:
        args_dict["save_strategy"] = "epoch"

    return TrainingArguments(**args_dict)


def train_model(model, tokenizer, train_dataset, val_dataset):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    optimizer_grouped_parameters = [
        {"params": model.transformer.parameters(), "lr": 2e-5},
        {"params": model.feature_compressor.parameters(), "lr": 1e-4},
        {"params": model.primary_caps.parameters(), "lr": 1e-4},
        {"params": model.digit_caps.parameters(), "lr": 1e-4},
        {"params": model.classifier.parameters(), "lr": 5e-4},
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    args = build_training_args()

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        optimizers=(optimizer, None),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=8)],
    )

    start_time = time.time()
    trainer.train()
    trainer.save_model()
    trainer.save_state()
    end_time = time.time()

    print(f"Training completed in {end_time - start_time:.2f} seconds")
    return trainer


# =========================================================
# PREDICTION
# =========================================================
def predict_model(trainer, dataset):
    predictions = trainer.predict(dataset)

    probabilities = torch.nn.functional.softmax(
        torch.from_numpy(predictions.predictions),
        dim=1,
    )[:, 1].numpy()

    predicted_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids

    return true_labels, predicted_labels, probabilities


# =========================================================
# PLOTTING
# =========================================================
def plot_roc_curves(fold_results, overall_true_labels, overall_probabilities, save_path=None):
    plt.figure(figsize=(10, 8))

    fold_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    average_color = "#2a52be"
    overall_color = "#e377c2"

    all_fpr = np.linspace(0, 1, 100)
    interp_tprs = []

    for i, (fold_num, true_labels, probabilities) in enumerate(fold_results):
        fpr, tpr, _ = roc_curve(true_labels, probabilities)
        fold_auc = auc(fpr, tpr)

        interp_tpr = np.interp(all_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tprs.append(interp_tpr)

        plt.plot(
            fpr,
            tpr,
            lw=1.5,
            alpha=0.7,
            color=fold_colors[i % len(fold_colors)],
            label=f"Fold {fold_num} (AUC = {fold_auc:.4f})",
        )

    mean_tpr = np.mean(interp_tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(all_fpr, mean_tpr)

    plt.plot(
        all_fpr,
        mean_tpr,
        color=average_color,
        linestyle=":",
        lw=3,
        label=f"Average (AUC = {mean_auc:.4f})",
        alpha=0.9,
    )

    overall_fpr, overall_tpr, _ = roc_curve(overall_true_labels, overall_probabilities)
    overall_auc = auc(overall_fpr, overall_tpr)

    plt.plot(
        overall_fpr,
        overall_tpr,
        color=overall_color,
        linestyle="-",
        lw=3,
        label=f"Overall (AUC = {overall_auc:.4f})",
        alpha=0.9,
    )

    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)

    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curves from Cross-Validation", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_independent_test_roc(true_labels, probabilities, save_path=None):
    fpr, tpr, _ = roc_curve(true_labels, probabilities)
    test_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 7))
    plt.plot(fpr, tpr, lw=3, label=f"Independent Test AUC = {test_auc:.4f}")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)

    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("Independent Test ROC Curve", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_radar_chart_with_mean_values(metrics_dict, avg_metrics, save_path=None):
    def convert_to_pct(val):
        return val * 100 if val <= 1.0 else val

    categories = ["precision", "recall", "f1", "accuracy", "specificity", "mcc"]
    category_labels = ["Precision", "Recall", "F1-Score", "Accuracy", "Specificity", "MCC"]

    labels = list(metrics_dict.keys())
    values = []

    for label in labels:
        fold_values = [convert_to_pct(metrics_dict[label][cat]) for cat in categories]
        values.append(fold_values)

    mean_values = [convert_to_pct(avg_metrics[f"eval_{cat}"]) for cat in categories]

    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    mean_wrapped = mean_values + [mean_values[0]]

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)

    fold_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    mean_color = "#e377c2"

    for idx, vals in enumerate(values[:-1]):
        vals_wrapped = vals + [vals[0]]
        ax.plot(
            angles,
            vals_wrapped,
            linewidth=2.0,
            alpha=0.7,
            color=fold_colors[idx % len(fold_colors)],
            label=f"Fold {idx + 1}",
        )

    ax.plot(
        angles,
        mean_wrapped,
        linewidth=4,
        color=mean_color,
        label="Mean",
        marker="o",
        markersize=10,
        zorder=10,
    )

    ax.fill(angles, mean_wrapped, color=mean_color, alpha=0.15)

    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.set_yticklabels([])

    for angle, val, cat_label in zip(angles[:-1], mean_values, category_labels):
        ax.text(
            angle,
            105,
            f"{cat_label}\n{val:.2f}%",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=mean_color,
            bbox=dict(
                facecolor="white",
                alpha=0.95,
                boxstyle="round,pad=0.3",
                edgecolor=mean_color,
                linewidth=1,
            ),
        )

    ax.set_xticklabels([])
    ax.set_title("Cross-Validation Performance Metrics", pad=30, fontsize=16, fontweight="bold")

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=True,
        fontsize=11,
    )

    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.grid(axis="x", linestyle="-", alpha=0.8)
    ax.spines["polar"].set_visible(False)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)

    plt.show()


# =========================================================
# MAIN FUNCTION
# CV ON TRAINING SET + FINAL INDEPENDENT TEST
# =========================================================
def fold():
    # -----------------------------------------------------
    # Load original train and independent test sets
    # -----------------------------------------------------
    pos_train = load_data(poz_path)
    neg_train = load_data(non_path)
    pos_test = load_data(val_poz_path)
    neg_test = load_data(val_non_path)

    train_all_data = pd.concat([pos_train, neg_train], ignore_index=True)
    train_all_data = shuffle(train_all_data, random_state=SEED).reset_index(drop=True)

    independent_test_data = pd.concat([pos_test, neg_test], ignore_index=True)
    independent_test_data = shuffle(independent_test_data, random_state=SEED).reset_index(drop=True)

    train_all_labels = train_all_data["label"].values

    print("\n========== DATA SUMMARY ==========")
    print(
        f"Training data: {len(train_all_data)} "
        f"(ACP: {sum(train_all_data['label'] == 1)}, "
        f"Non-ACP: {sum(train_all_data['label'] == 0)})"
    )
    print(
        f"Independent test data: {len(independent_test_data)} "
        f"(ACP: {sum(independent_test_data['label'] == 1)}, "
        f"Non-ACP: {sum(independent_test_data['label'] == 0)})"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = getattr(tokenizer, "eos_token", "[PAD]")

    tokenizer.model_max_length = MAX_LENGTH

    # -----------------------------------------------------
    # 5-fold cross-validation on original training set only
    # -----------------------------------------------------
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    metrics_list = []
    cv_true_labels = []
    cv_probabilities = []
    fold_results = []

    for fold_idx, (train_index, val_index) in enumerate(
        skf.split(train_all_data, train_all_labels), start=1
    ):
        print(f"\n========== Processing CV Fold {fold_idx}/{n_folds} ==========")

        train_data_fold = train_all_data.iloc[train_index].reset_index(drop=True)
        val_data_fold = train_all_data.iloc[val_index].reset_index(drop=True)

        print(
            f"Fold {fold_idx} training size: {len(train_data_fold)} "
            f"(ACP: {sum(train_data_fold['label'] == 1)}, "
            f"Non-ACP: {sum(train_data_fold['label'] == 0)})"
        )
        print(
            f"Fold {fold_idx} validation size: {len(val_data_fold)} "
            f"(ACP: {sum(val_data_fold['label'] == 1)}, "
            f"Non-ACP: {sum(val_data_fold['label'] == 0)})"
        )

        train_sequences = train_data_fold["data"].tolist()
        train_labels = train_data_fold["label"].tolist()

        val_sequences = val_data_fold["data"].tolist()
        val_labels = val_data_fold["label"].tolist()

        # PCA fitted only on the current training fold
        print("  Extracting handcrafted features for fold-specific PCA...")
        train_hc_matrix = np.array([extract_handcrafted_features(seq) for seq in train_sequences])

        print("  Fitting PCA on training fold only...")
        pca = PCA(n_components=PCA_COMPONENTS, random_state=SEED)
        pca.fit(train_hc_matrix)

        X_train_tokenized = tokenize_data(tokenizer, train_sequences)
        X_val_tokenized = tokenize_data(tokenizer, val_sequences)

        train_dataset = SeqDataset(
            X_train_tokenized,
            train_labels,
            train_sequences,
            pca=pca,
        )

        val_dataset = SeqDataset(
            X_val_tokenized,
            val_labels,
            val_sequences,
            pca=pca,
        )

        model = HybridModel(model_checkpoint).to(DEVICE)

        trainer = train_model(model, tokenizer, train_dataset, val_dataset)

        eval_results = trainer.evaluate(val_dataset)
        metrics_list.append(eval_results)

        true_labels, predicted_labels, probabilities = predict_model(trainer, val_dataset)

        fold_results.append((fold_idx, true_labels, probabilities))
        cv_true_labels.extend(true_labels)
        cv_probabilities.extend(probabilities)

    # -----------------------------------------------------
    # Cross-validation summary
    # -----------------------------------------------------
    avg_metrics = {
        metric: np.mean([m[metric] for m in metrics_list])
        for metric in metrics_list[0]
        if metric.startswith("eval_")
    }

    print("\n========== CROSS-VALIDATION RESULTS ON TRAINING SET ==========")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.4f}")

    cv_overall_auc = roc_auc_score(cv_true_labels, cv_probabilities)
    fold_aucs = [m["eval_auc"] for m in metrics_list]
    avg_fold_auc = np.mean(fold_aucs)

    print("\nCV AUC Metrics:")
    for i, auc_val in enumerate(fold_aucs):
        print(f"Fold {i + 1}: {auc_val:.4f}")

    print(f"Average Fold AUC: {avg_fold_auc:.4f}")
    print(f"Overall CV AUC: {cv_overall_auc:.4f}")

    metrics_dict = {
        f"Fold {i + 1}": {
            "precision": m["eval_precision"],
            "recall": m["eval_recall"],
            "f1": m["eval_f1"],
            "accuracy": m["eval_accuracy"],
            "specificity": m["eval_specificity"],
            "mcc": m["eval_mcc"],
        }
        for i, m in enumerate(metrics_list)
    }

    metrics_dict["Average"] = {
        "precision": avg_metrics["eval_precision"],
        "recall": avg_metrics["eval_recall"],
        "f1": avg_metrics["eval_f1"],
        "accuracy": avg_metrics["eval_accuracy"],
        "specificity": avg_metrics["eval_specificity"],
        "mcc": avg_metrics["eval_mcc"],
    }

    plot_radar_chart_with_mean_values(
        metrics_dict,
        avg_metrics,
        save_path="radar_cv_training_set.png",
    )

    plot_roc_curves(
        fold_results,
        cv_true_labels,
        cv_probabilities,
        save_path="roc_cv_training_set.png",
    )

    # =====================================================
    # FINAL MODEL TRAINING
    # Training set is split internally into train/validation.
    # Independent test set is evaluated only after training.
    # =====================================================
    print("\n========== FINAL MODEL TRAINING FOR INDEPENDENT TEST ==========")

    final_train_df, final_val_df = train_test_split(
        train_all_data,
        test_size=0.15,
        random_state=SEED,
        stratify=train_all_data["label"].values,
    )

    final_train_df = final_train_df.reset_index(drop=True)
    final_val_df = final_val_df.reset_index(drop=True)

    print(
        f"Final training split: {len(final_train_df)} "
        f"(ACP: {sum(final_train_df['label'] == 1)}, "
        f"Non-ACP: {sum(final_train_df['label'] == 0)})"
    )
    print(
        f"Final validation split: {len(final_val_df)} "
        f"(ACP: {sum(final_val_df['label'] == 1)}, "
        f"Non-ACP: {sum(final_val_df['label'] == 0)})"
    )
    print(
        f"Final independent test set: {len(independent_test_data)} "
        f"(ACP: {sum(independent_test_data['label'] == 1)}, "
        f"Non-ACP: {sum(independent_test_data['label'] == 0)})"
    )

    final_train_sequences = final_train_df["data"].tolist()
    final_train_labels = final_train_df["label"].tolist()

    final_val_sequences = final_val_df["data"].tolist()
    final_val_labels = final_val_df["label"].tolist()

    test_sequences = independent_test_data["data"].tolist()
    test_labels = independent_test_data["label"].tolist()

    # PCA fitted only on final training split
    print("  Extracting handcrafted features for final PCA...")
    final_train_hc_matrix = np.array(
        [extract_handcrafted_features(seq) for seq in final_train_sequences]
    )

    print("  Fitting final PCA on final training split only...")
    final_pca = PCA(n_components=PCA_COMPONENTS, random_state=SEED)
    final_pca.fit(final_train_hc_matrix)

    X_final_train_tokenized = tokenize_data(tokenizer, final_train_sequences)
    X_final_val_tokenized = tokenize_data(tokenizer, final_val_sequences)
    X_test_tokenized = tokenize_data(tokenizer, test_sequences)

    final_train_dataset = SeqDataset(
        X_final_train_tokenized,
        final_train_labels,
        final_train_sequences,
        pca=final_pca,
    )

    final_val_dataset = SeqDataset(
        X_final_val_tokenized,
        final_val_labels,
        final_val_sequences,
        pca=final_pca,
    )

    independent_test_dataset = SeqDataset(
        X_test_tokenized,
        test_labels,
        test_sequences,
        pca=final_pca,
    )

    final_model = HybridModel(model_checkpoint).to(DEVICE)

    # The validation split from training data is used for checkpoint selection.
    # The independent test set is not used during training.
    final_trainer = train_model(
        final_model,
        tokenizer,
        final_train_dataset,
        final_val_dataset,
    )

    # -----------------------------------------------------
    # Independent test evaluation
    # -----------------------------------------------------
    print("\n========== INDEPENDENT TEST SET PERFORMANCE ==========")

    test_eval_results = final_trainer.evaluate(independent_test_dataset)

    for metric, value in test_eval_results.items():
        print(f"{metric}: {value:.4f}")

    test_true_labels, test_predicted_labels, test_probabilities = predict_model(
        final_trainer,
        independent_test_dataset,
    )

    test_auc = roc_auc_score(test_true_labels, test_probabilities)

    print("\nIndependent Test AUC:")
    print(f"Test AUC: {test_auc:.4f}")

    print("\nIndependent Test Classification Report:")
    print(
        classification_report(
            test_true_labels,
            test_predicted_labels,
            target_names=["Non-ACP", "ACP"],
            digits=4,
            zero_division=0,
        )
    )

    plot_independent_test_roc(
        test_true_labels,
        test_probabilities,
        save_path="roc_independent_test_set.png",
    )

    # Save results
    results = {
        "cv_metrics_list": metrics_list,
        "cv_avg_metrics": avg_metrics,
        "cv_true": np.array(cv_true_labels),
        "cv_probs": np.array(cv_probabilities),
        "cv_fold_results": fold_results,
        "independent_test_eval": test_eval_results,
        "independent_test_true": np.array(test_true_labels),
        "independent_test_pred": np.array(test_predicted_labels),
        "independent_test_probs": np.array(test_probabilities),
        "independent_test_auc": test_auc,
    }

    np.save("results_cv_and_independent_test.npy", results)
    print("\nSaved results to results_cv_and_independent_test.npy")


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    fold()