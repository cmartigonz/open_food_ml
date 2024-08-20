import torch

class InferencePipeline:
    def __init__(self, model, data_loader, config):
        """
        Initialize the inference pipeline with the trained model and data loader.
        """
        self.model = model
        self.data_loader = data_loader
        self.config = config
        self.model.load_state_dict(torch.load(self.config['model_save_path']))
        self.model.eval()

    def predict(self):
        """
        Run inference on the data and return predictions.
        """
        predictions = []
        with torch.no_grad():
            for images, texts, numericals, _ in self.data_loader:
                outputs = self.model(images, texts, numericals)
                _, preds = torch.max(outputs, 1)
                predictions.extend(preds.tolist())
        return predictions
