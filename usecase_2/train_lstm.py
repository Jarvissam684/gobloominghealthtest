#!/usr/bin/env python3
"""
LSTM classifier for 2500 partial call sequences.
Event-based input: [event_type_idx, duration_ms, words] per timestep; pad to 50.
Fixed architecture: Embedding(8,16) + LSTM(32, dropout=0.3) + Dense(64, relu) + Dense(4, softmax).
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

# Event type mapping: 8 types for embedding input_dim=8
EVENT_TYPE_MAP = {
    "call_start": 0,
    "agent_speech": 1,
    "user_speech": 2,
    "silence": 3,
    "tool_call": 4,
    "call_end": 5,
    "padding": 6,
    "unknown": 7,
}
OUTCOME_CLASSES = ["completed", "abandoned", "transferred", "error"]
MAX_SEQ_LEN = 50
NUM_EVENT_TYPES = 8
EMBEDDING_DIM = 16
LSTM_UNITS = 32
DROPOUT = 0.3
DENSE_UNITS = 64
NUM_CLASSES = 4


def load_and_encode(data_path: Path):
    """Load partial_sequences.json; build X_type (batch, max_len), X_cont (batch, max_len, 2), and y."""
    with open(data_path) as f:
        data = json.load(f)
    sequences = data.get("partial_sequences", [])
    X_type_list = []
    X_cont_list = []
    y_list = []
    completion_percents = []
    for rec in sequences:
        events = rec.get("sequence", [])
        type_seq = []
        cont_seq = []
        for ev in events:
            t = ev.get("type", "")
            type_idx = EVENT_TYPE_MAP.get(t, EVENT_TYPE_MAP["unknown"])
            dur = ev.get("duration_ms", 0) or 0
            words = ev.get("words", 0) or 0
            type_seq.append(type_idx)
            cont_seq.append([float(dur), float(words)])
        # Pad
        while len(type_seq) < MAX_SEQ_LEN:
            type_seq.append(EVENT_TYPE_MAP["padding"])
            cont_seq.append([0.0, 0.0])
        type_seq = type_seq[:MAX_SEQ_LEN]
        cont_seq = cont_seq[:MAX_SEQ_LEN]
        X_type_list.append(type_seq)
        X_cont_list.append(cont_seq)
        outcome = rec.get("outcome", "")
        class_idx = OUTCOME_CLASSES.index(outcome) if outcome in OUTCOME_CLASSES else 0
        y_list.append(class_idx)
        completion_percents.append(rec.get("completion_percent", 0))
    X_type = np.array(X_type_list, dtype=np.int32)
    X_cont = np.array(X_cont_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    completion_percents = np.array(completion_percents)
    return X_type, X_cont, y, completion_percents, sequences


def build_model():
    """Fixed architecture: Embedding(8,16) for type; concat with duration/words; LSTM(32); Dense(64); Dense(4, softmax)."""
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError:
        raise SystemExit("Install tensorflow: pip install tensorflow")

    # Two inputs: type indices (batch, max_len), continuous (batch, max_len, 2)
    inp_type = keras.Input(shape=(MAX_SEQ_LEN,), dtype="int32", name="event_type")
    inp_cont = keras.Input(shape=(MAX_SEQ_LEN, 2), dtype="float32", name="duration_words")
    duration_words = layers.LayerNormalization(axis=-1)(inp_cont)

    # Embedding: input_dim=8, output_dim=16
    emb = layers.Embedding(input_dim=NUM_EVENT_TYPES, output_dim=EMBEDDING_DIM, mask_zero=False)(inp_type)
    # Concat (batch, max_len, 16) with (batch, max_len, 2) -> (batch, max_len, 18)
    combined = layers.Concatenate(axis=-1)([emb, duration_words])

    # LSTM: units=32, dropout=0.3, return_sequences=False
    lstm_out = layers.LSTM(LSTM_UNITS, dropout=DROPOUT, return_sequences=False)(combined)
    dense1 = layers.Dense(DENSE_UNITS, activation="relu")(lstm_out)
    out = layers.Dense(NUM_CLASSES, activation="softmax")(dense1)

    model = keras.Model(inputs=[inp_type, inp_cont], outputs=out)
    return model, keras


def main():
    base = Path(__file__).parent
    data_path = base / "partial_sequences.json"
    out_path = base / "lstm_training_results.json"

    X_type, X_cont, y, completion_percents, raw_sequences = load_and_encode(data_path)
    n = len(y)

    # Stratified train/test split 80/20
    from sklearn.model_selection import train_test_split
    (X_type_train, X_type_test, X_cont_train, X_cont_test,
     y_train, y_test, pct_train, pct_test) = train_test_split(
        X_type, X_cont, y, completion_percents, test_size=0.2, stratify=y, random_state=42, shuffle=True
    )
    # Validation: 10% of train = 200
    (X_type_train, X_type_val, X_cont_train, X_cont_val,
     y_train, y_val) = train_test_split(
        X_type_train, X_cont_train, y_train, test_size=0.1, stratify=y_train, random_state=42, shuffle=True
    )

    n_train, n_val, n_test = len(X_type_train), len(X_type_val), len(X_type_test)
    from collections import Counter
    class_dist = Counter(int(c) for c in y)
    class_dist_named = {OUTCOME_CLASSES[i]: int(class_dist.get(i, 0)) for i in range(4)}

    model, keras = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
    )

    history = model.fit(
        [X_type_train, X_cont_train], y_train,
        validation_data=([X_type_val, X_cont_val], y_val),
        batch_size=32,
        epochs=50,
        callbacks=[early_stop],
        verbose=1,
    )

    epochs_trained = len(history.history["loss"])
    final_train_loss = float(history.history["loss"][-1])
    final_val_loss = float(history.history["val_loss"][-1])

    # Save model for ensemble pipeline
    model_path = base / "lstm_model.keras"
    model.save(str(model_path))

    # Evaluate on test
    test_loss, test_acc = model.evaluate([X_type_test, X_cont_test], y_test, verbose=0)
    test_pred = np.argmax(model.predict([X_type_test, X_cont_test], verbose=0), axis=1)

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
    )
    test_accuracy = float(accuracy_score(y_test, test_pred))
    test_precision_macro = float(precision_score(y_test, test_pred, average="macro", zero_division=0))
    test_recall_macro = float(recall_score(y_test, test_pred, average="macro", zero_division=0))
    test_f1_macro = float(f1_score(y_test, test_pred, average="macro", zero_division=0))

    # Convergence: stable if val_loss plateaued in last 5 epochs
    if epochs_trained >= 5:
        last_5_val = history.history["val_loss"][-5:]
        stable = max(last_5_val) - min(last_5_val) < 0.05
        convergence = "stable" if stable else ("unstable" if epochs_trained >= 50 else "not_converged")
    else:
        convergence = "not_converged"

    # Confusion matrix (4x4)
    cm = confusion_matrix(y_test, test_pred, labels=[0, 1, 2, 3])
    confusion_dict = {}
    for i, true_label in enumerate(OUTCOME_CLASSES):
        confusion_dict[true_label] = {OUTCOME_CLASSES[j]: int(cm[i, j]) for j in range(4)}

    # Per-class metrics
    per_class_metrics = {}
    for i, c in enumerate(OUTCOME_CLASSES):
        mask = y_test == i
        pred_mask = test_pred == i
        tp = np.sum(mask & pred_mask)
        p = tp / np.sum(pred_mask) if np.sum(pred_mask) > 0 else 0.0
        r = tp / np.sum(mask) if np.sum(mask) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_class_metrics[c] = {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f), 4),
        }

    # Completion percent analysis on test set
    acc_at_pct = {}
    for p in [20, 40, 60, 80, 100]:
        mask = pct_test == p
        if np.sum(mask) == 0:
            acc_at_pct[f"accuracy_at_{p}_percent"] = 0.0
            continue
        acc = float(accuracy_score(y_test[mask], test_pred[mask]))
        acc_at_pct[f"accuracy_at_{p}_percent"] = round(acc, 4)
    accs = [acc_at_pct.get(f"accuracy_at_{p}_percent", 0) for p in [20, 40, 60, 80, 100]]
    if all(accs[i] <= accs[i + 1] for i in range(len(accs) - 1)):
        trend = "monotonic_improvement"
    elif accs[-1] > accs[0]:
        trend = "mixed"
    else:
        trend = "declining"
    completion_percent_analysis = {
        "accuracy_at_20_percent": acc_at_pct.get("accuracy_at_20_percent", 0),
        "accuracy_at_40_percent": acc_at_pct.get("accuracy_at_40_percent", 0),
        "accuracy_at_60_percent": acc_at_pct.get("accuracy_at_60_percent", 0),
        "accuracy_at_80_percent": acc_at_pct.get("accuracy_at_80_percent", 0),
        "accuracy_at_100_percent": acc_at_pct.get("accuracy_at_100_percent", 0),
        "trend": trend,
    }

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_id = f"lstm_v1_{ts}"

    report = {
        "model_id": model_id,
        "architecture": {
            "embedding_dim": EMBEDDING_DIM,
            "lstm_units": LSTM_UNITS,
            "lstm_layers": 1,
            "dropout": DROPOUT,
            "dense_layer": DENSE_UNITS,
            "output_classes": NUM_CLASSES,
            "max_sequence_length": MAX_SEQ_LEN,
            "num_event_types": NUM_EVENT_TYPES,
        },
        "training_metadata": {
            "total_sequences": n,
            "train_sequences": n_train,
            "val_sequences": n_val,
            "test_sequences": n_test,
            "class_distribution": class_dist_named,
        },
        "confusion_matrix": confusion_dict,
        "training_results": {
            "epochs_trained": epochs_trained,
            "final_train_loss": round(final_train_loss, 4),
            "final_val_loss": round(final_val_loss, 4),
            "test_accuracy": round(test_accuracy, 4),
            "test_precision_macro": round(test_precision_macro, 4),
            "test_recall_macro": round(test_recall_macro, 4),
            "test_f1_macro": round(test_f1_macro, 4),
            "convergence": convergence,
        },
        "per_class_metrics": per_class_metrics,
        "completion_percent_analysis": completion_percent_analysis,
    }

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"LSTM training complete. Results: {out_path}")
    print(f"Test accuracy: {test_accuracy:.4f}, F1 macro: {test_f1_macro:.4f}")
    print(f"Completion trend: {completion_percent_analysis}")


if __name__ == "__main__":
    main()
