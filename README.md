# Next Word Prediction using Deep Learning

## Project Overview
This project implements a next word prediction model using deep learning techniques. The model is trained on a pizza-related text dataset (`pizza.txt`) to predict the most likely next word in a sequence.

## Features
- LSTM-based neural network for sequence prediction
- Trained on pizza-related vocabulary and phrases
- Interactive prediction interface
- Customizable prediction length

## Requirements
- Python 3.7+
- TensorFlow 2.x
- Keras
- Numpy
- NLTK

## Installation
1. Clone this repository
   ```bash
   git clone https://github.com/yourusername/next-word-prediction.git
   cd next-word-prediction
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

## Dataset
The model is trained on `pizza.txt` which contains:
- Pizza recipes
- Menu descriptions
- Cooking instructions
- Pizza-related reviews and comments

## Usage
1. Train the model:
   ```python
   python train.py --data pizza.txt --epochs 20
   ```

2. Run predictions:
   ```python
   python predict.py --text "I would like to order a"
   ```

## Model Architecture
- Embedding Layer
- LSTM Layers (128 units)
- Dense Output Layer with Softmax

## Results
The model achieves:
- Training accuracy: ~85%
- Validation accuracy: ~78%
- Perplexity: ~2.5

## Future Improvements
- Incorporate transformer architecture
- Expand training dataset
- Add user feedback mechanism
- Deploy as web service
