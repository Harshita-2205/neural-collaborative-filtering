
# Neural Collaborative Filtering (NCF)

This repository contains an implementation of **Neural Collaborative Filtering (NCF)**, a cutting-edge deep learning technique for building recommendation systems. NCF integrates matrix factorization with neural networks to capture user-item interactions and deliver highly accurate predictions.

---

## Features

- **Neural Collaborative Filtering**: Advanced implementation for user-item recommendations.
- **Flexible Architecture**: Configurable embedding layers and dense layers.
- **Hyperparameter Tuning**: Customizable for various experiments.
- **Dataset Compatibility**: Supports standard user-item datasets.
- **Evaluation Metrics**: Includes RMSE, MAE, and Top-N recommendation evaluation.

---

## Prerequisites

Ensure the following dependencies are installed:

- Python 3.7 or higher
- TensorFlow or PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib (for visualizations)

Install required libraries with:
```bash
pip install numpy pandas scikit-learn matplotlib tensorflow  # or pytorch
```

---

## Functions in `ncf.py`

### 1. `build_model(num_users, num_items, layers, embedding_dim)`
- **Purpose**: Constructs the NCF model.
- **Inputs**:
  - `num_users`: Number of unique users.
  - `num_items`: Number of unique items.
  - `layers`: List of layer sizes for the Multi-Layer Perceptron (MLP).
  - `embedding_dim`: Embedding vector dimension for users and items.
- **Output**: Compiled model with embedding and dense layers.

### 2. `train_model(model, train_data, epochs, batch_size, learning_rate)`
- **Purpose**: Trains the NCF model.
- **Inputs**:
  - `model`: Model built using `build_model`.
  - `train_data`: Dataset of user-item interactions.
  - `epochs`: Number of training iterations.
  - `batch_size`: Number of samples per batch.
  - `learning_rate`: Gradient descent step size.
- **Output**: Trained model.

### 3. `evaluate_model(model, test_data)`
- **Purpose**: Evaluates the trained model.
- **Inputs**:
  - `model`: Trained NCF model.
  - `test_data`: Test dataset with user-item interactions.
- **Output**: Performance metrics (e.g., RMSE, MAE, precision@k).

### 4. `predict_top_n(model, user_id, n, item_pool)`
- **Purpose**: Generates top-N recommendations for a user.
- **Inputs**:
  - `model`: Trained NCF model.
  - `user_id`: Target user ID.
  - `n`: Number of recommendations.
  - `item_pool`: Pool of items for recommendation.
- **Output**: List of top-N recommended items.

### 5. `prepare_data(data, user_col, item_col, rating_col)`
- **Purpose**: Processes dataset for training and evaluation.
- **Inputs**:
  - `data`: Original dataset (Pandas DataFrame).
  - `user_col`: Column name for user IDs.
  - `item_col`: Column name for item IDs.
  - `rating_col`: Column name for ratings or feedback.
- **Output**: Processed dataset ready for modeling.

---

## Project Structure

```
neural-collaborative-filtering
│
├── README.md                     # Project description and instructions
├── requirements.txt              # List of dependencies
├── Code                         
│   ├── ncf.py                    # NCF model definition
├── prompts.md                    #prompts to learn concept better
```
---

## Acknowledgments

- **Research Paper**: [Neural Collaborative Filtering by He et al. (2017)](https://arxiv.org/abs/1708.05031)
- **Inspiration**: Derived from cutting-edge research on recommendation systems.

---

Contributions are welcome! Feel free to submit pull requests or raise issues.
