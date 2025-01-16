import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class IVNet(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        return self.net(x)

class MLInterpolator:
    def __init__(self, x_data, y_data, smoothing_factor=0.3):
        # Convert to numpy arrays
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        
        # Sort and remove duplicates by averaging y values for same x
        unique_x, indices = np.unique(x_data, return_inverse=True)
        unique_y = np.array([y_data[indices == i].mean() for i in range(len(unique_x))])
        
        # Sort the unique values
        sort_idx = np.argsort(unique_x)
        self.x_data = unique_x[sort_idx]
        self.y_data = unique_y[sort_idx]
        
        # Convert to PyTorch tensors
        x_tensor = torch.FloatTensor(self.x_data.reshape(-1, 1))
        y_tensor = torch.FloatTensor(self.y_data.reshape(-1, 1))
        
        # Normalize data
        self.x_mean = x_tensor.mean()
        self.x_std = x_tensor.std()
        self.y_mean = y_tensor.mean()
        self.y_std = y_tensor.std()
        
        x_normalized = (x_tensor - self.x_mean) / self.x_std
        y_normalized = (y_tensor - self.y_mean) / self.y_std
        
        # Create and train the model
        self.model = IVNet()
        self._train_model(x_normalized, y_normalized)
        
        # Set model to evaluation mode
        self.model.eval()

    def _train_model(self, x, y, epochs=1000):
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = self.model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

    def __call__(self, x):
        """Evaluate the model at x."""
        with torch.no_grad():
            if isinstance(x, (list, np.ndarray)):
                x = np.array(x)
                x_tensor = torch.FloatTensor(x.reshape(-1, 1))
            else:
                x_tensor = torch.FloatTensor([[x]])
            
            # Normalize input
            x_normalized = (x_tensor - self.x_mean) / self.x_std
            
            # Get prediction and denormalize
            y_normalized = self.model(x_normalized)
            y_pred = y_normalized * self.y_std + self.y_mean
            
            return y_pred.numpy().flatten()