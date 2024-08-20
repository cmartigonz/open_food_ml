import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss

"""
Trainer class for training the model.
"""
class Trainer:
    def __init__(self, model, data_loader, config):
        """
        Initialize the trainer with model, optimizer, loss function, and data loader.
        """
        self.model = model
        self.data_loader = data_loader
        self.config = config
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.criterion = CrossEntropyLoss()

    def train_one_epoch(self):
        """
        Train the model for one epoch.
        """
        self.model.train()
        for batch_idx, (images, texts, numericals, labels) in enumerate(self.data_loader):
            self.optimizer.zero_grad()
            outputs = self.model(images, texts, numericals)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item()}")

    def validate(self, val_loader):
        """
        Evaluate the model on the validation set.
        """
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for images, texts, numericals, labels in val_loader:
                outputs = self.model(images, texts, numericals)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
        avg_loss = total_loss / len(val_loader)
        print(f"Validation Loss: {avg_loss}")
        return avg_loss

    def fit(self, train_loader, val_loader):
        """
        Full training loop with validation at each epoch.
        """
        best_val_loss = float('inf')
        for epoch in range(self.config['epochs']):
            print(f"Epoch {epoch+1}/{self.config['epochs']}")
            self.train_one_epoch(train_loader)
            val_loss = self.validate(val_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print("Saving best model...")
                torch.save(self.model.state_dict(), self.config['model_save_path'])
