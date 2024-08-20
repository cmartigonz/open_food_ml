import pytest
import torch
from src.data_loader import DataLoader, get_data_loader
from src.model import MultimodalNutritionModel
from src.inference import InferencePipeline
from src.train import Trainer

@pytest.fixture
def sample_data():
    """
    Fixture to provide sample data for testing.
    """
    config = {
        'image_size': 64,
        'image_mean': [0.485, 0.456, 0.406],
        'image_std': [0.229, 0.224, 0.225],
        'vocab_size': 5000,
        'embedding_dim': 128,
        'hidden_dim': 128,
        'num_classes': 10,
        'numerical_features': ['energy-kcal_100g', 'fat_100g', 'saturated-fat_100g'],
        'max_seq_length': 10,
        'batch_size': 4,
        'learning_rate': 0.001,
        'epochs': 2,
        'model_save_path': 'model.pth'
    }
    
    data = {
        'image_nutrition_url': ['https://example.com/image1.jpg', 'https://example.com/image2.jpg'],
        'ingredients_text': ['Sugar, Salt, Fat', 'Protein, Fiber, Carbohydrates'],
        'energy-kcal_100g': [100, 200],
        'fat_100g': [10, 20],
        'saturated-fat_100g': [5, 15],
        'labels': [1, 0]
    }
    
    return config, data

def test_data_loader(sample_data):
    """
    Test the DataLoader to ensure it processes images and text correctly.
    """
    config, data = sample_data
    loader = DataLoader(config)
    images, texts, numericals = loader.load_multimodal_data(data)
    
    assert len(images) == 2
    assert len(texts) == 2
    assert len(numericals) == 2

def test_model_forward_pass(sample_data):
    """
    Test the forward pass of the MultimodalNutritionModel.
    """
    config, data = sample_data
    model = MultimodalNutritionModel(config)
    
    # Create dummy tensors for input
    images = torch.randn(2, 3, config['image_size'], config['image_size'])
    texts = torch.randint(0, config['vocab_size'], (2, config['max_seq_length']))
    numericals = torch.randn(2, len(config['numerical_features']))
    
    outputs = model(images, texts, numericals)
    
    assert outputs.shape == (2, config['num_classes'])

def test_trainer(sample_data):
    """
    Test the training process of the model.
    """
    config, data = sample_data
    loader = DataLoader(config)
    train_images, train_texts, train_numericals = loader.load_multimodal_data(data)
    train_loader = get_data_loader(train_images, train_texts, train_numericals, data['labels'], config['batch_size'])
    
    model = MultimodalNutritionModel(config)
    trainer = Trainer(model, train_loader, config)
    
    trainer.train_one_epoch()  # Run a single epoch
    assert trainer.model is not None

def test_inference_pipeline(sample_data):
    """
    Test the InferencePipeline to ensure it generates predictions.
    """
    config, data = sample_data
    loader = DataLoader(config)
    test_images, test_texts, test_numericals = loader.load_multimodal_data(data)
    test_loader = get_data_loader(test_images, test_texts, test_numericals, data['labels'], config['batch_size'])
    
    model = MultimodalNutritionModel(config)
    inference_pipeline = InferencePipeline(model, test_loader, config)
    
    predictions = inference_pipeline.predict()
    
    assert len(predictions) == 2
    assert all(isinstance(pred, int) for pred in predictions)
