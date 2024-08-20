import yaml
from data_loader import DataSplitter, DataLoader, get_data_loader
from model import MultimodalNutritionModel
from train import Trainer
from inference import InferencePipeline

def main():
    # Load configuration
    with open('src/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Data splitting and loading
    splitter = DataSplitter(config['data_path'], config['split_ratios'], config['seed'])
    data = splitter.load_data()
    train_data, val_data, test_data = splitter.split_data(data)

    # Initialize data loaders
    loader = DataLoader(config)
    train_images, train_texts, train_numericals = loader.load_multimodal_data(train_data)
    val_images, val_texts, val_numericals = loader.load_multimodal_data(val_data)
    train_loader = get_data_loader(train_images, train_texts, train_numericals, train_data['labels'], config['batch_size'])
    val_loader = get_data_loader(val_images, val_texts, val_numericals, val_data['labels'], config['batch_size'])

    # Initialize and train the model
    model = MultimodalNutritionModel(config)
    trainer = Trainer(model, train_loader, config)
    trainer.fit(train_loader, val_loader)

    # Run inference on test data
    test_images, test_texts, test_numericals = loader.load_multimodal_data(test_data)
    test_loader = get_data_loader(test_images, test_texts, test_numericals, test_data['labels'], config['batch_size'])
    inference_pipeline = InferencePipeline(model, test_loader, config)
    predictions = inference_pipeline.predict()

    # Output predictions
    print(predictions)

if __name__ == "__main__":
    main()
