import torch
import torch.nn as nn
import torch.optim as optim

class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32, hidden_layers=[64, 32, 16, 8]):
        super(NCF, self).__init__()
        
        # Embedding layers for users and items
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Generalized Matrix Factorization (GMF) layer
        self.gmf = nn.Linear(embedding_dim, 1)
        
        # Multi-Layer Perceptron (MLP) layers
        layers = []
        input_size = embedding_dim * 2
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        self.mlp = nn.Sequential(*layers)
        
        # Final fusion and prediction layer
        self.fc = nn.Linear(hidden_layers[-1] + 1, 1)  
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, user_ids, item_ids):
        user_embed = self.user_embedding(user_ids)
        item_embed = self.item_embedding(item_ids)
        
        # GMF component
        gmf_output = self.gmf(user_embed * item_embed)
        
        # MLP component
        mlp_input = torch.cat([user_embed, item_embed], dim=-1)
        mlp_output = self.mlp(mlp_input)
        
        # Fusion of GMF and MLP
        concat_output = torch.cat([gmf_output, mlp_output], dim=-1)
        output = self.fc(concat_output)
        return self.sigmoid(output)

# Example usage
num_users, num_items = 1000, 1000
model = NCF(num_users, num_items)

# Dummy input for a batch of user-item pairs
user_ids = torch.LongTensor([0, 1, 2])
item_ids = torch.LongTensor([50, 100, 150])

# Forward pass
output = model(user_ids, item_ids)
print(output)
