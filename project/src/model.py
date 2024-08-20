import torch.nn as nn
"""
Multimodal model for nutrition classification.
"""
class MultimodalNutritionModel(nn.Module):
    def __init__(self, config):
        """
        Initialize the multimodal model with the specified configuration.
        """
        super(MultimodalNutritionModel, self).__init__()
        
        # Image processing layers
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Text processing layers
        self.embedding = nn.Embedding(config['vocab_size'], config['embedding_dim'])
        self.rnn = nn.GRU(config['embedding_dim'], config['hidden_dim'], batch_first=True)
        
        # Numerical data processing layer
        self.fc_numerical = nn.Linear(len(config['numerical_features']), 64)
        
        # Combined processing layers
        self.fc1 = nn.Linear(64*config['image_size']//4*config['image_size']//4 + config['hidden_dim'] + 64, 128)
        self.fc2 = nn.Linear(128, config['num_classes'])
        
    def forward(self, image, text, numerical):
        """
        Define the forward pass for the model.
        """
        # Image forward pass
        image_features = self.cnn_layers(image)
        image_features = image_features.view(image_features.size(0), -1)
        
        # Text forward pass
        text_features, _ = self.rnn(self.embedding(text))
        text_features = text_features[:, -1, :]
        
        # Numerical data forward pass
        numerical_features = self.fc_numerical(numerical)
        
        # Combine image, text, and numerical features
        combined_features = torch.cat((image_features, text_features, numerical_features), dim=1)
        
        # Classification output
        output = self.fc2(self.fc1(combined_features))
        return output
