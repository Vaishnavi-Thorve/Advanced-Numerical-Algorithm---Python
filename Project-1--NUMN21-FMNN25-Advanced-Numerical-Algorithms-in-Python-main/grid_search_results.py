import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors

# Load the data from the JSON file
file_path = 'merged_search_output.json'
with open(file_path, 'r') as f:
    data = json.load(f)

# Extract hyperparameters and validation loss
batch_size = [entry['batch_size'] for entry in data]
learning_rate = [entry['learning_rate'] for entry in data]
hidden_size = [entry['hidden_size'] for entry in data]
val_loss = [entry['val_loss'] for entry in data]

# Find the best hyperparameters (lowest validation loss)
min_loss_index = np.argmin(val_loss)
best_hyperparams = {
    "batch_size": batch_size[min_loss_index],
    "learning_rate": learning_rate[min_loss_index],
    "hidden_size": hidden_size[min_loss_index],
    "val_loss": val_loss[min_loss_index]
}

# Print the best hyperparameters
print(f"Best Hyperparameters: {best_hyperparams}")

# Plot 3D scatter plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Create scatter plot
sc = ax.scatter(batch_size, learning_rate, hidden_size, c=val_loss, norm=colors.LogNorm(vmin=min(val_loss), vmax=max(val_loss)),
                   cmap='PuBu_r')

# Labels and title
ax.set_xlabel('Batch Size')
ax.set_ylabel('Learning Rate')
ax.set_zlabel('Hidden Size')
plt.title('3D Plot of Grid Search Results (Validation Loss)')

# Color bar
plt.colorbar(sc, label='Validation Loss')

# Show plot
plt.show()