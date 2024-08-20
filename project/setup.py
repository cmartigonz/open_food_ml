from setuptools import setup, find_packages

setup(
    name="nutrition_image_text_model",  # Name of your project/package
    version="0.1.0",  # Version of your project
    description="A multimodal machine learning pipeline for extracting nutritional information from images and text.",  # Short description
    author="Your Name",  # Your name
    author_email="your.email@example.com",  # Your email
    url="https://github.com/yourusername/nutrition_image_text_model",  # URL to the project's repository or website
    packages=find_packages(),  # Automatically find all packages in the src directory
    install_requires=[
        "torch>=1.8.0",  # PyTorch for deep learning
        "torchvision>=0.9.0",  # TorchVision for image processing
        "transformers>=4.0.0",  # Transformers for text processing (if using a pre-trained model)
        "pandas>=1.1.0",  # Pandas for data manipulation
        "scikit-learn>=0.24.0",  # Scikit-learn for data splitting and evaluation
        "Pillow>=8.0.0",  # Pillow for image handling
        "requests>=2.25.0",  # Requests for handling HTTP requests (if downloading images from URLs)
        "pyyaml>=5.3.0",  # PyYAML for configuration file handling
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",  # Pytest for running tests
            "flake8>=3.8.0",  # Flake8 for linting
            "black>=20.8b1",  # Black for code formatting
        ]
    },
    entry_points={
        "console_scripts": [
            "train_model=src.train:main",  # Command to run the training script
            "run_inference=src.inference:main",  # Command to run the inference script
        ]
    },
    python_requires=">=3.7",  # Minimum Python version required
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",  # License type
        "Operating System :: OS Independent",
    ],
)
