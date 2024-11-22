# Neural Collaborative Filtering

This repository contains an implementation of **Neural Collaborative Filtering (NCF)**, a state-of-the-art deep learning approach for recommendation systems. NCF combines matrix factorization techniques with neural networks to model user-item interactions and predict preferences effectively.

## Features

- Implementation of Neural Collaborative Filtering
- Flexible architecture for embedding layers and deep layers
- Customizable hyperparameters for experiments
- Support for user-item datasets in standard formats
- Evaluation metrics like RMSE, MAE, or Top-N recommendations

### Prerequisites

To run the code, you need the following installed:

- Python 3.7 or higher
- TensorFlow/PyTorch (depending on the framework used)
- NumPy
- Pandas
- Scikit-learn
- Matplotlib (for visualizations)

## Functions in ncf.py
### 1. build_model(num_users, num_items, layers, embedding_dim)
-Purpose: Constructs the neural network model for NCF.
-Inputs:
   -num_users: Total number of unique users.
   -num_items: Total number of unique items.
   -layers: List of layer sizes for the MLP (Multi-Layer Perceptron) portion.
   -embedding_dim: Dimension of the embedding vectors for users and items.
-Output: A compiled deep learning model with user and item embedding layers and dense layers for predictions.

### 2. train_model(model, train_data, epochs, batch_size, learning_rate)
-Purpose: Trains the NCF model on the provided dataset.
-Inputs:
   -model: The NCF model built using build_model.
   -train_data: Training dataset containing user-item interactions and labels.
   -epochs: Number of training iterations over the dataset.
   -batch_size: Number of samples per batch.
   -learning_rate: Step size for gradient descent.
-Output: Trained model with updated weights.

### 3. evaluate_model(model, test_data)
-Purpose: Evaluates the performance of the trained NCF model on a test dataset.
-Inputs:
   -model: Trained NCF model.
   -test_data: Test dataset containing user-item interactions and labels.
-Output: Metrics like RMSE, MAE, or precision@k depending on the task.

### 4. predict_top_n(model, user_id, n, item_pool)
-Purpose: Generates the top-N recommendations for a given user.
-Inputs:
   -model: Trained NCF model.
   -user_id: ID of the user for whom recommendations are generated.
   -n: Number of recommendations to generate.
   -item_pool: List of all possible items for recommendations.
-Output: List of the top-N recommended items for the user.

### 5. prepare_data(data, user_col, item_col, rating_col)
-Purpose: Prepares the dataset for training and testing by encoding user and item IDs and normalizing ratings.
-Inputs:
   -data: Original dataset as a DataFrame.
   -user_col: Column name for user IDs.
   -item_col: Column name for item IDs.
   -rating_col: Column name for ratings or feedback.
-Output: Processed data ready for training and evaluation.

## Project Structure
   -ncf.py: Contains core functions for building, training, and evaluating the NCF model.
   -train.py: Script for training the NCF model.
   -evaluate.py: Script for evaluating the trained model.
   -recommend.py: Script for generating recommendations.
   -models/: Folder containing neural network architecture definitions.
   -data/: Folder to store the input datasets.

## Acknowledgments

- [Neural Collaborative Filtering: He et al. (2017)](https://arxiv.org/abs/1708.05031)
- Inspiration from recommendation system research
