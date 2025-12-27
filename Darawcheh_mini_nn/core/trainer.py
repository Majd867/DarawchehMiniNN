from __future__ import annotations
import numpy as np
from typing import Optional, Dict, List, Any

from .network import NeuralNetwork
from ..optimizers.base import Optimizer
from ..utils.metrics import batch_iter

class Trainer:


    def __init__(
        self,
        model:NeuralNetwork,
        optimizer: Optimizer,
        batch_size: int = 32,
        epochs: int= 20,
        shuffle:bool = True
    ):
        self.model= model
        self.opt =optimizer
        self.batch_size = batch_size
        self.epochs= epochs
        self.shuffle=shuffle

        self.history: Dict[str, List[float]]={
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": []
        }
    
    def train_step(self, xb: np.ndarray, yb: np.ndarray) -> float:
    
        logits = self.model.forward(xb, training=True)
        loss = float(self.model.loss_fn.forward(logits, yb))

        self.model.zero_grad()
        self.model.backward()
        self.opt.step(self.model.parameters())

        return loss

    step_train = train_step




    def fit(
        self,
        x_train: np.ndarray, y_train: np.ndarray,
        x_val: Optional[np.ndarray ] = None, y_val: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> Dict[str, List[float]]:

        for ep in range(1, self.epochs + 1):
            losses = []

            for xb, yb in batch_iter(x_train, y_train, self.batch_size, shuffle=self.shuffle):
                # Forward + loss
                logits = self.model.forward(xb, training=True)
                loss = self.model.loss_fn.forward(logits, yb)
                losses.append(loss)

                # Backward
                self.model.zero_grad()
                self.model.backward()

                # Update
                self.opt.step(self.model.parameters())

            train_loss = float(np.mean(losses) )
            train_acc = self.model.accuracy(x_train, y_train )
            self.history["train_loss"].append(train_loss )
            self.history["train_acc"].append(train_acc)


            if x_val is not None and y_val is not None:
                val_logits = self.model.forward(x_val, training=False)
                val_loss = float(self.model.loss_fn.forward(val_logits, y_val))
                val_acc = self.model.accuracy(x_val, y_val)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)

                if verbose:
                    print(f"Epoch {ep:03d}: train_loss={train_loss:.4f}, train_acc={train_acc:.3f}, "
                          f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}")
            else:
                if verbose:
                    print(f"Epoch {ep:03d}: train_loss={train_loss:.4f}, train_acc={train_acc:.3f}")

        return self.history



def train_step(self, xb, yb) -> float:
    logits = self.model.forward(xb, training=True)
    loss = self.model.loss_fn.forward(logits, yb)
    self.model.zero_grad()
    self.model.backward()
    self.opt.step(self.model.parameters())
    return float(loss)

step_train = train_step



def n_stepTrain(self, x_train, y_train, n_steps: int = 100):
    steps = 0
    losses = []
    for xb, yb in batch_iter(x_train, y_train, self.batch_size, shuffle=True):
        losses.append(self.train_step(xb, yb))
        steps += 1
        if steps >= n_steps:
            break
    return float(np.mean(losses)) if losses else 0.0
