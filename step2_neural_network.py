"""
STEP 2: Neural Network Model
=============================
- Defines and builds the feedforward neural network
- Same architecture used for both GA and Random initialized models
- Only the initial weights differ between the two models

"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


def build_model(input_dim, num_classes):
    """

    Architecture:
        Input → Dense(64) → Dense(32) → Output(num_classes)

    """
    model = Sequential([
        Dense(64,  activation='relu', input_shape=(input_dim,), name='hidden_1'),
        Dropout(0.2),
        Dense(32,  activation='relu', name='hidden_2'),
        Dropout(0.1),
        Dense(num_classes, activation='softmax', name='output')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def get_total_weights(model):
    """Get total number of trainable weights (= GA chromosome length)."""
    total = sum([np.prod(w.shape) for w in model.trainable_weights])
    print(f"[INFO] Total trainable weights (chromosome length): {total}")
    return total


def set_weights_from_chromosome(model, chromosome):
    """Set model weights from a flat GA chromosome array."""
    idx         = 0
    new_weights = []

    for w in model.trainable_weights:
        shape = w.shape
        size  = np.prod(shape)
        new_weights.append(
            np.array(chromosome[idx:idx + size]).reshape(shape)
        )
        idx += size

    model.set_weights(new_weights)
    return model


def train_model(model, X_train, y_train, epochs=100, batch_size=32, label="Model"):
    """
    Train the neural network with early stopping on validation loss.

    """
    print(f"\n[INFO] Training {label}...")

    # Evaluate BEFORE training starts ─────────────────────────────────
    # Manually split matching Keras validation_split=0.1 (last 10% = val)
    val_split_idx = int(len(X_train) * 0.9)
    X_tr_part     = X_train[:val_split_idx]
    y_tr_part     = y_train[:val_split_idx]
    X_val_part    = X_train[val_split_idx:]
    y_val_part    = y_train[val_split_idx:]

    print(f"[INFO] Evaluating {label} BEFORE training starts (epoch 0)...")
    initial_train_loss, initial_train_acc = model.evaluate(X_tr_part,  y_tr_part,  verbose=0)
    initial_val_loss,   initial_val_acc   = model.evaluate(X_val_part, y_val_part, verbose=0)

    print(f"[INFO] {label} | Epoch 0 (before training) | "
          f"train_acc: {initial_train_acc:.4f} | val_acc: {initial_val_acc:.4f} | "
          f"train_loss: {initial_train_loss:.4f} | val_loss: {initial_val_loss:.4f}")

    # ── Early Stopping ────────────────────────────────────────────────────────
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )

    # Prepend epoch 0 to history ──────────────────────────────────────
    history.history['accuracy'].insert(0,     float(initial_train_acc))
    history.history['val_accuracy'].insert(0, float(initial_val_acc))
    history.history['loss'].insert(0,         float(initial_train_loss))
    history.history['val_loss'].insert(0,     float(initial_val_loss))

    print(f"[INFO] {label} training complete.")
    print(f"[INFO] {label} | Final train_acc: {history.history['accuracy'][-1]:.4f} | "
          f"Final val_acc: {history.history['val_accuracy'][-1]:.4f}")

    return history


def plot_training_curves(history_ga, history_random):
    """Plot and compare training + validation accuracy/loss curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(history_ga.history['accuracy'],     color='green', label='GA Model')
    axes[0, 0].plot(history_random.history['accuracy'], color='red',   label='Random Model')
    axes[0, 0].set_title("Training Accuracy")
    axes[0, 0].set_xlabel("Epochs")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(history_ga.history['val_accuracy'],     color='green', label='GA Model')
    axes[0, 1].plot(history_random.history['val_accuracy'], color='red',   label='Random Model')
    axes[0, 1].set_title("Validation Accuracy")
    axes[0, 1].set_xlabel("Epochs")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(history_ga.history['loss'],     color='green', label='GA Model')
    axes[1, 0].plot(history_random.history['loss'], color='red',   label='Random Model')
    axes[1, 0].set_title("Training Loss")
    axes[1, 0].set_xlabel("Epochs")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(history_ga.history['val_loss'],     color='green', label='GA Model')
    axes[1, 1].plot(history_random.history['val_loss'], color='red',   label='Random Model')
    axes[1, 1].set_title("Validation Loss")
    axes[1, 1].set_xlabel("Epochs")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("GA vs Random Initialized NN — Training Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    plt.show()
    print("[INFO] Training curves saved as 'training_curves.png'")


if __name__ == "__main__":
    INPUT_DIM   = 30   
    NUM_CLASSES = 16

    model = build_model(INPUT_DIM, NUM_CLASSES)
    model.summary()

    total = get_total_weights(model)
    print(f"\n[TEST] Chromosome length for GA: {total}")
    print(f"[TEST] Old chromosome length was ~36,592 — now {total} — much better for GA!")