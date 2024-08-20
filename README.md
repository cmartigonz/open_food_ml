# Nutritional Facts Extraction from Product Images

## Project Overview

This project aims to develop a machine learning pipeline for extracting nutritional information from product images. The pipeline combines image and text processing techniques to classify and tag nutritional facts such as calories, fat content, sugars, and more. The solution is designed to be modular, scalable, and ready for deployment in a production environment.

## Project Structure
```
project/
│
├── data/
│ ├── raw/ # Raw data files (images, annotations, etc.)
│ ├── processed/ # Preprocessed data for training and testing
│
├── src/
│ ├── data_loader.py # Data loading and preprocessing logic
│ ├── model.py # Model definitions
│ ├── train.py # Training script
│ ├── inference.py # Inference script
│ ├── utils.py # Utility functions (e.g., for data splitting, configuration)
│ ├── config.yaml # Configuration file for model hyperparameters, paths, etc.
│ ├── main.py # Main entry point for the pipeline
│
├── notebooks/ # Jupyter notebooks for EDA and experimentation
│
├── tests/ # Unit tests for different components
│
├── README.md # Project documentation
│
├── requirements.txt # Python dependencies
│
└── setup.py # Setup script for the package
```


## Setup Instructions

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)
- Git

### Installation

1. **Clone the Repository**:
    ```bash
    git clone <repository-url>
    cd project
    ```

2. **Create a Virtual Environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Install the Project Package**:
    ```bash
    pip install -e .
    ```

5. **Download and Prepare Data**:
    - Place raw data (images, annotations) in the `data/raw/` directory.
    - Run any preprocessing scripts provided in `src/` to prepare the data.

### Configuration

- The project uses a `config.yaml` file for managing hyperparameters, paths, and other settings.
- Modify the `config.yaml` file located in `src/` to adjust parameters such as batch size, learning rate, or model architecture.

## Usage

### Training the Model

To train the model, run the `main.py` script:

```bash
python src/main.py
```

This will:

- Load the data from data/processed/.
- Initialize the model and data loaders.
- Train the model based on the settings in config.yaml.
- Save the best-performing model to the specified directory.

### Running Inference
```bash
python src/inference.py --input_dir <path_to_images> --output_file <output_predictions.csv>
```

This will:

- Load the trained model.
- Process the images in the specified directory.
- Output the predictions to a CSV file.

## Branch Strategy
The project follows a Git branching strategy to manage development:

- main: This branch contains the stable, production-ready code. Direct commits to main are not allowed; all changes must go through a pull request from develop.
- develop: This branch contains the latest development code. New features and bug fixes are integrated here.
- feature/*: These branches are used to develop new features. They are branched off develop and merged back into develop after completion and testing.
- bugfix/*: These branches are for fixing bugs in the develop branch. They follow a similar workflow to feature branches.
- hotfix/*: These branches are for fixing critical issues in the main branch. After the fix, the branch is merged into both main and develop.
