import numpy as np


from Darawcheh_mini_nn import (
    NeuralNetwork, Trainer,
    Dense, ReLU, Sigmoid, Dropout, BatchNorm1D,
    SoftmaxCrossEntropy,
    Adam
)


def main():


    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler



    data=load_iris()
    X=data["data"].astype(float )
    y=data["target"].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(

        X , y , test_size=0.25, random_state=42 , stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    model = NeuralNetwork(
        layers=[
            Dense(4, 32, weight_init="he", seed=1),
            BatchNorm1D(32),
            ReLU(),
            Dropout(p=0.1, seed=1),
            Dense(32, 16, weight_init="he", seed=2),
            ReLU(),
            Dense(16, 3, weight_init="xavier", seed=3),
        ],
        loss=SoftmaxCrossEntropy()
    )
    

    opt = Adam(lr=1e-2, weight_decay=1e-4)
    trainer = Trainer(model, opt, batch_size=16, epochs=60, shuffle=True)
    trainer.fit(X_train, y_train, X_val, y_val, verbose=True)


    print("Final validation accuracy:", model.accuracy(X_val, y_val))

if __name__ == "__main__":
    main()
