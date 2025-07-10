import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight

from tqdm.auto import tqdm
from typing import Union


class CSAModel:
    def __init__(
            self, 
            input_size: int = 768, 
            hidden_size: int = 700,
            nonlinear: nn.Module = nn.ReLU(),
            optimizer: torch.optim.Optimizer = torch.optim.Adam,
            lr: float = 1e-4
        ):
        self._model = CosineSimModel(
            input_size=input_size, hidden_size=hidden_size, nonlinear=nonlinear
        )
        self._criterion = BCELoss_weighted()
        self._optimizer = optimizer(self._model.parameters(), lr)


    def fit(self, max_epoch: int, dataloader: DataLoader) -> None:
        for epoch in tqdm(range(max_epoch), desc="Epoch"):
            self._fit_one_epoch(dataloader)

    
    def predict(self, dataloader: DataLoader) -> np.array:
        self._model.eval()
        predictions = np.array([])
        with torch.no_grad():
            for (fembs, tembs) in tqdm(dataloader, "Predicting", leave=False):
                fembs, tembs = fembs[0], tembs[0]
                scores = self._model(tembs, fembs)
                predictions = np.hstack((predictions.numpy() > 0.5, scores.numpy().flatten()))
        return predictions.flatten()


    def predict_proba(self, dataloader: DataLoader) -> np.array:
        self._model.eval()
        predictions = np.array([])
        with torch.no_grad():
            for (fembs, tembs) in tqdm(dataloader, "Predicting", leave=False):
                fembs, tembs = fembs[0], tembs[0]
                scores = self._model(tembs, fembs)
                predictions = np.hstack((predictions, scores.numpy().flatten()))
        return predictions.flatten()


    def evaluate(self, dataloader: DataLoader) -> tuple[float, float]:
        self._model.eval()
        loss, correct, all = 0, 0, 0
        with torch.no_grad():
             for (fembs, tembs, targets) in tqdm(dataloader, desc="Evaluating", leave=False):
                fembs, tembs, targets =  fembs[0], tembs[0], targets[0]
                if len(np.unique(targets)) < 2:
                    continue
                weights = self._get_batch_cweights(targets)
                scores = self._model(tembs, fembs)
                
                loss += self._criterion(weights, scores, targets).item()
                correct += ((scores > 0.5).type(torch.long) == targets).sum().item()
                all += tembs.shape[0]
        loss /= len(dataloader)
        accuracy = correct / all
        return loss, accuracy
    

    def save_model_optimizer(self, path: str) -> None:
        """Save model and optimizer parameters in specific path"""
        torch.save({
            "model_state_dict": self._model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict()
        }, path)

    
    def load_model_optimizer(self, path: str) -> None:
        """Load model and optimizer parameters from specific path"""
        checkpoint = torch.load(path, weights_only=True)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


    def _fit_one_epoch(self, train_dataloader : DataLoader) -> None:
        self._model.train()
        for (fembs, tembs, targets) in tqdm(train_dataloader, desc="Training", leave=False):
            fembs, tembs, targets =  fembs[0], tembs[0], targets[0]

            if len(np.unique(targets)) < 2:
                continue
            weights = self._get_batch_cweights(targets) 

            proba = self._model(tembs, fembs)
            loss = self._criterion(weights, proba, targets)
            loss.backward()
            self._optimizer.step()
            self._optimizer.zero_grad()


    def _get_batch_cweights(self, target: torch.Tensor) -> torch.Tensor:
        weights = compute_class_weight(
            class_weight="balanced", classes=np.array([0., 1.]), 
            y=target.numpy().flatten()
        )
        return torch.Tensor(weights)

class CosineSimModel(nn.Module):
    """
    A PyTorch module that fine-tunes embeddings to predict the similarity between 
    test and file embeddings, indicating the likelihood of a test failure 
    if the files and tests are similar.
    """
    def __init__(
            self, 
            input_size: int = 768, 
            hidden_size: int = 700, 
            nonlinear: nn.Module = nn.ReLU(),
        ):
        super().__init__()
        self.test_linear = nn.Linear(input_size, hidden_size)
        self.file_delta_linear = nn.Linear(2 * input_size, hidden_size)
        self.nonlinear = nonlinear
        self.cossim = CosineSimProba()

    def forward(
            self, 
            tests_embs: torch.Tensor, 
            file_delta_embs: torch.Tensor
        ):
        tests_embs = self.nonlinear(self.test_linear(tests_embs))
        file_delta_embs = self.nonlinear(self.file_delta_linear(file_delta_embs))

        cossim = self.cossim(tests_embs, file_delta_embs)
        cossim, inds = cossim.max(dim=1)
        return cossim


class CosineSimProba(nn.Module):
    """Сomputes the cosine similarity between two tensors and converts it to a probability."""
    def __init__(self):
        super().__init__()
    
    def forward(self, t1: torch.Tensor, t2: torch.Tensor):
        t1_norm = F.normalize(t1, dim=1)
        t2_norm = F.normalize(t2, dim=1)
        cossim = torch.matmul(t1_norm, t2_norm.t())
        proba = (1 + cossim) / 2
        return proba
    

class BCELoss_weighted(nn.Module):
    def __init__(self, error: float = 1e-10):
        super().__init__()
        self.error = error
        
    def forward(
            self, 
            weights: Union[torch.Tensor, np.array, list], 
            scores: torch.Tensor, 
            targets: torch.Tensor
        ):
        loss = -weights[1] * targets * torch.log(scores + self.error) - (1 - targets) * weights[0] * torch.log(1 - scores + self.error)
        return torch.mean(loss)