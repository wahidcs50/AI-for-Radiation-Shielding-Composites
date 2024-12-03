import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import random
import matplotlib.pyplot as plt

class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
    def load_data(self):
        self.df = pd.read_excel(self.file_path)
        return self.df
    def preprocess_data(self):
        X = self.df[['Density:', 'AMW:', 'C2H6OSi_fraction', 'CdO_fraction', 'B4C_fraction', 'Gd2O3_fraction']]
        y = self.df['FNRCS:']
        return train_test_split(X, y, test_size=0.2, random_state=42)
class Model(nn.Module):

    def __init__(self, input_size):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class Trainer:

    def __init__(self, model, train_loader, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.0001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5 
        )

    def train(self, epochs, early_stopping):
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            self.model.train()
            running_train_loss = 0.0

            for X_batch, y_batch in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                running_train_loss += loss.item()

            avg_train_loss = running_train_loss / len(self.train_loader)
            train_losses.append(avg_train_loss)

            avg_val_loss = self._validate()
            val_losses.append(avg_val_loss)

            print(
                f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}"
            )

            self.scheduler.step(avg_val_loss)

            if early_stopping(avg_val_loss):
                break

        return train_losses, val_losses

    def _validate(self):
        self.model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for X_val_batch, y_val_batch in self.test_loader:
                val_outputs = self.model(X_val_batch)
                val_loss = self.criterion(val_outputs, y_val_batch)
                running_val_loss += val_loss.item()

        return running_val_loss / len(self.test_loader)

    def evaluate(self):
        self.model.eval()
        all_predictions = []

        with torch.no_grad():
            for X_test_batch in self.test_loader:
                X_batch = X_test_batch[0]
                predictions = self.model(X_batch)
                all_predictions.append(predictions)

        return torch.cat(all_predictions).cpu().numpy()
    def scores(self,y_test_actual):
        r2 = r2_score(y_test_actual, all_predictions)
        mse = mean_squared_error(y_test_actual, all_predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_actual, all_predictions)
        print(f'R² Score: {r2:.4f}')
        print(f'Mean Squared Error (MSE): {mse:.4f}')
        print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
        print(f'Mean Absolute Error (MAE): {mae:.4f}')
        return r2


class Visualizer:

    @staticmethod
    def plot_loss(train_losses, val_losses):
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label="Training Loss", color="blue", lw=3)
        plt.plot(val_losses, label="Validation Loss", color="orange", lw=3)
        plt.title("Training and Validation Loss Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Loss (MSE)")
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_predictions(y_true, all_predictions, train_losses, val_losses):

        exclude_epochs = 50
        smoothed_train_losses = train_losses[exclude_epochs:]
        smoothed_val_losses = val_losses[exclude_epochs:]
        
        plt.figure(figsize=(10, 6))
        plt.plot(smoothed_train_losses, label='Training Loss', color='blue', lw=3)
        plt.plot(smoothed_val_losses, label='Validation Loss', color='orange', lw=3)
        plt.title(f'Training and Validation Loss Curve', fontsize=16, fontweight='bold')
        plt.xlabel('Epochs', fontsize=12, fontweight='bold')
        plt.ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
        plt.legend(loc='upper right', prop={'size': 12, 'weight': 'bold'})
        plt.xticks(fontweight='bold')
        plt.yticks(fontweight='bold')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'bestnn_loss_curve_excluding_first.png', dpi=600) 
        plt.show()
        

        plt.figure(figsize=(8, 6))
        plt.scatter(y_test_actual, all_predictions, color='blue', label='Predictions')
        min_value = min(min(y_test_actual), min(all_predictions))
        max_value = max(max(y_test_actual), max(all_predictions))
        plt.plot([min_value, max_value], [min_value, max_value], color='red', linestyle='--', label='Perfect Prediction') 
        plt.title('Predicted vs Actual FNRC Values', fontsize=16, fontweight='bold')
        plt.xlabel('Actual FNRC', fontsize=12, fontweight='bold')
        plt.ylabel('Predicted FNRC', fontsize=12, fontweight='bold')
        plt.legend(loc='upper left', prop={'size': 12, 'weight': 'bold'})
        plt.xticks(fontweight='bold')
        plt.yticks(fontweight='bold')
        plt.grid(True)
        plt.tight_layout()
        # plt.savefig('nnpredicted_vs_actual_FNRC.png', dpi=600) 
        plt.show()

class EarlyStopping:

    def __init__(self, patience=20, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            print("Early stopping triggered")
            return True
        return False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)
torch.manual_seed(0)
np.random.seed(0)

if __name__ == "__main__":
    data_handler = DataHandler('/kaggle/input/second-clean-data.xlsx')
    df = data_handler.load_data()
    X_train, X_test, y_train, y_test = data_handler.preprocess_data()
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, worker_init_fn=seed_worker, generator=g)

    model = Model(input_size=X_train.shape[1])

    trainer = Trainer(model, train_loader, test_loader)
    early_stopping = EarlyStopping(patience=20, min_delta=0.001) 
    train_losses, val_losses = trainer.train(epochs=500, early_stopping=early_stopping)
    y_test_actual = y_test_tensor.cpu().numpy()
    all_predictions= trainer.evaluate()
    scores= trainer.scores(y_test_actual)
    
    Visualizer.plot_loss(train_losses, val_losses)
    Visualizer.plot_predictions(y_test_actual, all_predictions, train_losses, val_losses)
    
