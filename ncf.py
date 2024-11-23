"""
# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    Developer Details: 
        Name: Harshita
        Role: Developer
        Code ownership rights: Harshita

    Version:
        Version: V 1.0.0 (12 October, 2024)
            Developer: Harshita

    Description: Implementation of Neural Collaborative Filtering (NCF) for building recommendation systems. 
    Combines matrix factorization (Generalized Matrix Factorization - GMF) and deep neural networks 
    (Multi-Layer Perceptron - MLP) to model user-item interactions.

# TECHNICAL DETAILS - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    Programming Language: Python
    Framework: PyTorch
    Dependencies: 
        - Python 3.7+
        - PyTorch
        - NumPy
        - Pandas
        - Scikit-learn
        - Matplotlib

# DATA REQUIREMENTS - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    Input Data Format: User-item interaction data in CSV or similar tabular formats.
    Output: Predicted user-item preference scores or Top-N recommendations.

# USAGE INSTRUCTIONS - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    For installation, setup, and execution details, refer to the [README.md] file in the project repository.

# VERSION HISTORY - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    Version: 1.0.0
    Changes: Initial implementation of the Neural Collaborative Filtering (NCF) model.
    Date: 12 October, 2024

"""

import torch
import torch.nn as nn
import torch.optim as optim

# Define the Neural Collaborative Filtering (NCF) model
class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32, hidden_layers=[64, 32, 16, 8]):
        super(NCF, self).__init__()
        
        # Embedding layer for users
        self.user_embedding = nn.Embedding(num_users, embedding_dim)  # Learnable user representations
        # Embedding layer for items
        self.item_embedding = nn.Embedding(num_items, embedding_dim)  # Learnable item representations
        
        # Generalized Matrix Factorization (GMF) layer
        self.gmf = nn.Linear(embedding_dim, 1)  # Computes the dot product of user and item embeddings
        
        # Multi-Layer Perceptron (MLP) layers
        layers = []  # List to hold MLP layers
        input_size = embedding_dim * 2  # Input size for the first MLP layer (user + item embeddings)
        for hidden_size in hidden_layers:  # Loop through the sizes of hidden layers
            layers.append(nn.Linear(input_size, hidden_size))  # Fully connected layer
            layers.append(nn.ReLU())  # Activation function
            input_size = hidden_size  # Update input size for the next layer
        self.mlp = nn.Sequential(*layers)  # Combine all MLP layers sequentially
        
        # Final fusion and prediction layer
        self.fc = nn.Linear(hidden_layers[-1] + 1, 1)  # Combines GMF and MLP outputs
        self.sigmoid = nn.Sigmoid()  # Applies sigmoid activation to output a probability
    
    # Forward pass method
    def forward(self, user_ids, item_ids):
        # Get user embeddings
        user_embed = self.user_embedding(user_ids)
        # Get item embeddings
        item_embed = self.item_embedding(item_ids)
        
        # GMF component: element-wise product of user and item embeddings
        gmf_output = self.gmf(user_embed * item_embed)
        
        # MLP component: concatenate user and item embeddings
        mlp_input = torch.cat([user_embed, item_embed], dim=-1)
        mlp_output = self.mlp(mlp_input)  # Pass concatenated input through MLP
        
        # Combine GMF and MLP outputs
        concat_output = torch.cat([gmf_output, mlp_output], dim=-1)
        output = self.fc(concat_output)  # Final linear layer for prediction
        return self.sigmoid(output)  # Output probability

# Example usage
num_users, num_items = 1000, 1000  # Number of users and items in the dataset
model = NCF(num_users, num_items)  # Instantiate the NCF model

# Dummy input for a batch of user-item pairs
user_ids = torch.LongTensor([0, 1, 2])  # Example user IDs
item_ids = torch.LongTensor([50, 100, 150])  # Example item IDs

# Forward pass through the model
output = model(user_ids, item_ids)  # Predict user-item interactions
print(output)  # Print the output probabilities
