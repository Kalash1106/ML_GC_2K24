import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score

import torch
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import CyclicLR, ReduceLROnPlateau

from PIL import Image
from tqdm import tqdm

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_clean_df(mappings, gt_file):
    name_series = pd.read_csv(gt_file, index_col="image").idxmax(axis=1)
    id_series = name_series.map(mappings)

    new_df = pd.DataFrame({"image_id": id_series, "image_name": name_series})
    new_df = new_df[new_df["image_id"] != -1].reset_index()
    return new_df


class Classify:
    def __init__(self, mappings, inv_mappings, gt_file):
        self.mappings = mappings
        self.inv_mappings = inv_mappings
        self.gt_file = gt_file  # clean gt file from data folder
        self.clean_df = get_clean_df(mappings, gt_file)
        self.num_classes = self.clean_df.image_id.nunique()

    def get_final_df(self, trained_model, test_dl, device="cuda"):
        # Validation Step
        trained_model.to(device).eval()
        all_predictions = []
        all_filenames = []

        with torch.no_grad():
            for test_inputs, test_filenames in tqdm(test_dl, desc="Predicting"):
                test_inputs = test_inputs.to(device)
                test_outputs = trained_model(test_inputs)

                _, test_predicted = torch.max(test_outputs, 1)

                all_predictions.extend(test_predicted.detach().cpu().numpy())
                all_filenames.extend(test_filenames)

        print(f"Inverse mapping used here is: {self.inv_mappings}")
        all_predictions_name = [self.inv_mappings[label] for label in all_predictions]

        pred_df = pd.DataFrame({"image": all_filenames, "Class": all_predictions_name})
        return pred_df


def get_splits(df, VAL_SIZE, stratify=True):
    # Splitting based on stratification if 'stratify' is True
    if stratify:
        train_df, val_df = train_test_split(
            df, test_size=VAL_SIZE, stratify=df["image_id"], random_state=42
        )
    else:
        train_df, val_df = train_test_split(df, test_size=VAL_SIZE, random_state=42)
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    return train_df, val_df


class CustomDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None, is_test=False):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if self.is_test:
            img_path = os.path.join(self.root_dir, self.dataframe.image[idx]) + ".jpg"
            label = self.dataframe.image[
                idx
            ]  # assigning a random value - because its test dataset
        else:
            img_path = os.path.join(self.root_dir, self.dataframe.image[idx]) + ".jpg"
            label = self.dataframe.image_id[idx]

        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)

        return image, label


def train_model(
    model,
    train_dataloader,
    val_dataloader,
    criterion,
    optimizer,
    num_epochs,
    device,
    scheduler=None,
):
    model.to(device)
    best_accuracy = 0.0
    best_model_wts = model.state_dict()
    last_model_wts = model.state_dict()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for inputs, labels in tqdm(
            train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"
        ):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_dataloader.dataset)
        epoch_accuracy = correct_predictions / total_predictions
        print(
            f"Epoch {epoch+1}/{num_epochs} - Training, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}"
        )

        # Validation Step
        model.eval()
        val_running_loss = 0.0
        val_correct_predictions = 0
        val_total_predictions = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for val_inputs, val_labels in tqdm(
                val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"
            ):
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_labels)

                _, val_predicted = torch.max(val_outputs, 1)
                val_total_predictions += val_labels.size(0)
                val_correct_predictions += (val_predicted == val_labels).sum().item()
                val_running_loss += val_loss.item() * val_inputs.size(0)

                all_predictions.extend(val_predicted.cpu().numpy())
                all_labels.extend(val_labels.cpu().numpy())

        val_epoch_loss = val_running_loss / len(val_dataloader.dataset)
        val_epoch_accuracy = val_correct_predictions / val_total_predictions
        # Calculate average recall
        recall = recall_score(all_labels, all_predictions, average="macro")
        print(
            f"Epoch {epoch+1}/{num_epochs} - Validation, Loss: {val_epoch_loss:.4f}, Accuracy: {val_epoch_accuracy:.4f}, Average Recall: {recall:.4f}"
        )

        if scheduler is not None:
            # If scheduler requires a validation metric, pass it
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_epoch_accuracy)
            elif isinstance(scheduler, CyclicLR):
                scheduler.step(val_epoch_loss)
            else:
                scheduler.step()

        # Check if this is the best model so far
        if val_epoch_accuracy > best_accuracy:
            best_accuracy = val_epoch_accuracy
            best_model_wts = model.state_dict()
        # saving the current model state also
        last_model_wts = model.state_dict()

        print(f"---- Epoch {epoch+1}/{num_epochs} completed ---")
        print()

    model.load_state_dict(best_model_wts)
    return model, {"best_weights": best_model_wts, "last_weights": last_model_wts}
