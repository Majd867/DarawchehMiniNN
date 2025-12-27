import numpy as np

from Darawcheh_mini_nn import (
    NeuralNetwork, Trainer,
    Dense, ReLU, Sigmoid, Dropout, BatchNorm1D,
    SoftmaxCrossEntropy,
    Adam
)

def main():
    """
    Full integration test for DarawchehMiniNN.
    Uses all major components required by the homework.
    """

    # -------------------------------------------------
    # 1) Load dataset (sklearn only for data loading)
    # -------------------------------------------------
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    data = load_iris()
    X = data["data"].astype(float)
    y = data["target"].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # -------------------------------------------------
    # 2) Build network (explicitly matching homework)
    # Dense → Dense → Sigmoid → BatchNorm → Dense → ReLU → Dense → Softmax
    # -------------------------------------------------
    model = NeuralNetwork(
        layers=[
            Dense(4, 32, weight_init="he", seed=1),
            Dense(32, 32, weight_init="he", seed=2),
            Sigmoid(),
            BatchNorm1D(32),
            Dropout(p=0.1, seed=1),
            Dense(32, 16, weight_init="he", seed=3),
            ReLU(),
            Dense(16, 3, weight_init="xavier", seed=4),
        ],
        loss=SoftmaxCrossEntropy()
    )

    # -------------------------------------------------
    # 3) Optimizer
    # -------------------------------------------------
    optimizer = Adam(
        lr=1e-2,
        weight_decay=1e-4
    )

    # -------------------------------------------------
    # 4) Trainer
    # -------------------------------------------------
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        batch_size=16,
        epochs=50,
        shuffle=True
    )

    # -------------------------------------------------
    # 5) Sanity check: single gradient computation
    # -------------------------------------------------
    print("\n[Gradient check on one batch]")
    grads = model.gradient(X_train[:8], y_train[:8])
    print(f"Number of trainable parameters: {len(grads)}")

    # -------------------------------------------------
    # 6) Sanity check: one manual train step
    # -------------------------------------------------
    print("\n[Single train_step test]")
    loss_before = model.loss(X_train[:16], y_train[:16])
    trainer.train_step(X_train[:16], y_train[:16])
    loss_after = model.loss(X_train[:16], y_train[:16])

    print(f"Loss before step: {loss_before:.4f}")
    print(f"Loss after  step: {loss_after:.4f}")

    # -------------------------------------------------
    # 7) Full training
    # -------------------------------------------------
    print("\n[Full training]")
    trainer.fit(
        X_train, y_train,
        X_val, y_val,
        verbose=True
    )

    # -------------------------------------------------
    # 8) Final evaluation
    # -------------------------------------------------
    train_acc = model.accuracy(X_train, y_train)
    val_acc = model.accuracy(X_val, y_val)

    print("\n[Final results]")
    print(f"Train accuracy: {train_acc:.3f}")
    print(f"Validation accuracy: {val_acc:.3f}")

    # -------------------------------------------------
    # 9) Prediction API test
    # -------------------------------------------------
    preds = model.predict(X_val[:5])
    print("\n[Prediction shape check]")
    print("Logits shape:", preds.shape)
    print("Predicted classes:", np.argmax(preds, axis=1))
    print("True classes     :", y_val[:5])


if __name__ == "__main__":
    main()
