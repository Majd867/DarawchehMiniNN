from __future__ import annotations
from typing import List
import numpy as np


from .layer import Layer
from .parameter import Parameter

class Loss:

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray ) ->  float:
        raise NotImplementedError
    

    def backward(self) -> np.ndarray:
        raise NotImplementedError
  



class NeuralNetwork:

    def __init__(self, layers: List[Layer] , loss: Loss,):
        self.layers = layers
        self.loss_fn = loss


    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        out = x
        for layer in self.layers:
            out = layer.forward(out, training=training)
        return out


    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x, training=False)
    
    

    def loss(self, x: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.forward(x, training=True)
        return float(self.loss_fn.forward(y_pred, y))
    

    def backward(self) -> None:
        grad = self.loss_fn.backward()
        for layer in reversed(self.layers):
            grad = layer.backward(grad)



    def parameters(self) -> List[Parameter]:
        params: List[Parameter] = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
    


    
    def gradient(self, x: np.ndarray, y: np.ndarray):
        logits = self.forward(x, training=True)
        _ = self.loss_fn.forward(logits, y)
        self.zero_grad()
        self.backward()
        return self.parameters()



    def zero_grad(self) -> None:
        for layer in self.layers:
            layer.zero_grad()




    def accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        logits = self.predict(x)
        pred = np.argmax(logits, axis=1)
        true = y if y.ndim == 1 else np.argmax(y, axis=1)
        return float(np.mean(pred == true))
