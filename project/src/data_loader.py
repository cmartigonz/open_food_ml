import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms
from PIL import Image

class DataSplitter:
    def __init__(self, data_path, labels_path, split_ratios, seed):
        """
        Initialize the splitter with data paths, split ratios, and a random seed.
        """
        self.data_path = data_path
        self.labels_path = labels_path
        self.split_ratios = split_ratios
        self.seed = seed

    def load_data(self):
        """
        Load the data from CSV files.
        Return: data, labels
        """
        data = pd.read_csv(self.data_path)
        labels = pd.read_csv(self.labels_path)
        return data, labels

    def split_data(self, data, labels):
        """
        Split the data into training, validation, and test sets based on the provided ratios.
        Return: train_data, val_data, test_data
        """
        # Split the data into training, validation, and test sets
        train_data, test_data, train_labels, test_labels = train_test_split(
            data, labels, test_size=self.split_ratios['test'], random_state=self.seed)
        train_data, val_data, train_labels, val_labels = train_test_split(
            train_data, train_labels, test_size=self.split_ratios['val'], random_state=self.seed)
        return train_data, val_data, test_data, train_labels, val_labels, test_labels

class DataLoader:
    def __init__(self, config):
        """
        Initialize the data loader with configuration settings.
        """
        self.config = config

    def preprocess_image(self, image_path):
        """
        Preprocess the image (resize, normalize, etc.).
        Return: processed_image
        """
        image = Image.open(image_path)
        # Define the image transformations
        preprocess = transforms.Compose([
            transforms.Resize((self.config['image_size'], self.config['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config['image_mean'], std=self.config['image_std'])
        ])
        return preprocess(image)

    def preprocess_text(self, text):
        """
        Preprocess the text (tokenize, convert to indices, and pad).
        Return: processed_text (a tensor with padded indices)
        """        
        # Tokenize the text
        tokens = self.tokenizer.tokenize(text)
        
        # Convert tokens to indices
        token_indices = [self.tokenizer.convert_tokens_to_ids(token) for token in tokens]
        
        # Pad sequences to a fixed length
        max_length = self.config['max_seq_length']
        if len(token_indices) < max_length:
            # Pad with zeros (or any specified padding index)
            token_indices += [0] * (max_length - len(token_indices))
        else:
            # Truncate the sequence
            token_indices = token_indices[:max_length]
        
        # Convert the list of indices to a tensor (for PyTorch)
        processed_text = torch.tensor(token_indices)
        
        return processed_text

    def load_multimodal_data(self, image_paths, texts):
        """
        Load and preprocess multimodal data (images and text).
        Return: processed_images, processed_texts, numerical data
        """
        processed_images = [self.preprocess_image(url) for url in data['image_nutrition_url']] 
        processed_texts = [self.preprocess_text(text) for text in data['ingredients_text']]
        numerical_data = data[['energy-kcal_100g', 'fat_100g', 'saturated-fat_100g', 'carbohydrates_100g',
                               'sugars_100g', 'fiber_100g', 'proteins_100g', 'salt_100g',
                               'vitamin-a_100g', 'vitamin-c_100g', 'calcium_100g', 'iron_100g',
                               'energy_100g', 'sodium_100g', 'polyols_100g', 'trans-fat_100g',
                               'serum-proteins_100g', 'casein_100g']]
        return processed_images, processed_texts, numerical_data


class NutritionDataset(Dataset):
    def __init__(self, images, texts, numerical_data, labels, transform=None):
        """
        Initialize the dataset with images, texts, numerical data, labels, and optional transforms.
        """
        self.images = images
        self.texts = texts
        self.numerical_data = numerical_data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        """
        Return the total number of samples.
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Retrieve a single data point and its label.
        Apply transformations if specified.
        """
        image = self.images[idx]
        text = self.texts[idx]
        numerical = self.numerical_data.iloc[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, text, numerical, label

def get_data_loader(images, texts, numerical_data, labels, batch_size, shuffle=True):
    """
    Initialize and return a PyTorch DataLoader.
    """
    dataset = NutritionDataset(images, texts, numerical_data, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

