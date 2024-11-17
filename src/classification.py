import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from src.DataSet import DataSet
from src.settings import settings, logger
from sklearn.model_selection import train_test_split
from typing import Optional
import ghostml
from typing import Dict


class SimpleDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class MultiLabelModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MultiLabelModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))  # Sigmoid for multi-label outputs
        return x


class ClassificationDataSet(DataSet):
    def __init__(self, embedding_path: Optional[str], annotation_path: Optional[str]):
        super().__init__(embedding_path, annotation_path)
        features = []
        raw_labels = []  # Store raw string labels for processing
        for index, vector in self.sorted_index_embedding.items():
            features.append(vector)
            nct_id = self.index_name_dict[index]
            raw_labels.append(self.id_conditions_dict[nct_id])  # Assuming this is a list of strings

        unique_labels = set(label for sublist in raw_labels for label in sublist)
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        index_to_label = {idx: label for label, idx in label_to_index.items()}
        num_classes = len(unique_labels)
        encoded_labels = []
        for label_list in raw_labels:
            one_hot = torch.zeros(num_classes, dtype=torch.float32)
            for label in label_list:
                one_hot[label_to_index[label]] = 1
            encoded_labels.append(one_hot)

        # Convert features and labels to tensors
        features = torch.tensor(features, dtype=torch.float32)
        labels = torch.stack(encoded_labels)

        self.features = features
        self.labels = labels
        self.num_classes = num_classes
        self.label_to_index = label_to_index
        self.index_to_label = index_to_label


def training_loop(model: nn.Module,
                  train_dataloader: DataLoader,
                  validation_dataloader: DataLoader,
                  num_epochs=50,
                  patience=3,
                  optimizer=None,
                  criterion=None) -> None:
    """
    Basic training loop with early stopping

    :param model: Untrained model
    :param train_dataloader: train dataloader
    :param validation_dataloader: Validation dataloader for early stopping
    :param num_epochs: maximum number of epochs
    :param patience: Patience for early stopping
    :param optimizer: Optimiser to use, defaults to Adam with a learning rate of 0.001
    :param criterion: Criterion to use, defaults to binary cross entropy
    :return:
    """
    if not optimizer:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    if not criterion:
        criterion = nn.BCELoss()
    best_loss = float('inf')  # Best validation loss
    counter = 0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for inputs, targets in train_dataloader:
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_dataloader)
        logger.info(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in validation_dataloader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(validation_dataloader)
        logger.info(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")

        # Check early stopping condition
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            counter = 0  # Reset patience counter
            torch.save(model.state_dict(), f'{settings.data_dir}/best_model.pth')  # Save the best model
        else:
            counter += 1
            logger.info(f"No improvement in validation loss for {counter} epochs.")

        if counter >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break


def test(model: nn.Module, test_dataloader: DataLoader, criterion=None) -> None:
    """
    Outputs the loss over the testing dataloader

    :param model: Trained Model
    :param test_dataloader: Testing Dataloader
    :param criterion: Criterion to use, defaults to binary cross entropy
    :return:
    """
    if not criterion:
        criterion = nn.BCELoss()
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
    logger.info(f"Test Loss: {test_loss / len(test_dataloader):.4f}")


# def get_all_predicted_labels(model: nn.Module, train_dataloader: DataLoader, threshold=0,
#                              index_to_label: Optional[dict] = None):
#     model.eval()
#     all_predictions = []
#
#     with torch.no_grad():
#         for inputs, _ in train_dataloader:
#             outputs = model(inputs)  # Get predictions
#             batch_predicted_indices = (outputs >= threshold).nonzero(
#                 as_tuple=False)  # Get indices of predictions above threshold
#
#             # Group predictions by sample
#             for i in range(inputs.size(0)):
#                 sample_indices = batch_predicted_indices[batch_predicted_indices[:, 0] == i, 1]
#                 if index_to_label:
#                     predicted_labels = [index_to_label[idx.item()] for idx in sample_indices]
#                     all_predictions.append(predicted_labels)
#                 else:
#                     predicted_index = [idx.item() for idx in sample_indices]
#                     all_predictions.append(predicted_index)
#     return all_predictions
def get_all_probabilities(model: nn.Module, dataloader: DataLoader) -> torch.Tensor:
    """
    Returns all probabilities

    :param model: Trained model
    :param dataloader: Data loader
    :return:
    """
    model.eval()
    all_probabilities = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            outputs = model(inputs)
            all_probabilities.append(outputs)
    all_probabilities = torch.cat(all_probabilities, dim=0)
    return all_probabilities


def optimise_thresholds(train_labels: torch.Tensor,
                        train_probabilities: torch.Tensor,
                        thresholds: list[float],
                        subset: Optional[list[int]] = None,
                        min_positive_count: int = 3) -> Dict[int:float]:
    """
    Uses GhostML to optimise the threshold on a per class basis.

    :param train_labels: Ground truth Tensor
    :param train_probabilities: Predictions Tensor
    :param thresholds: Threshold to try
    :param subset: If only a subset of classes are of interest
    :param min_positive_count: Minimum positive count in the dataset to bother finding a threshold
    :return: Dictionary of class label to threshold
    """
    num_classes = train_labels.shape[1]
    optimal_thresholds = {}
    for class_index in range(num_classes):
        if subset:
            if class_index not in subset:
                continue
        class_labels = train_labels[:, class_index]
        class_probabilities = train_probabilities[:, class_index]
        positive_count = torch.sum(class_labels > 0).item()
        # Skip items if not enough examples
        if positive_count < min_positive_count:
            logger.warning(
                f"Class {class_index} skipped: only {positive_count} positive labels (min required: {min_positive_count})")
            optimal_thresholds[class_index] = None
            continue
        optimal_threshold = ghostml.optimize_threshold_from_predictions(
            class_labels, class_probabilities, thresholds, ThOpt_metrics='Kappa'
        )

        optimal_thresholds[class_index] = optimal_threshold
        logger.debug(f"{class_index} has optimal threshold of {optimal_threshold}")
    return optimal_thresholds
