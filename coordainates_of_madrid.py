import numpy as np
import pandas as pd

# Bounding box around Madrid
min_lat, max_lat = 40.310, 40.520
min_lon, max_lon = -3.830, -3.530

# Step size for the grid (in degrees, ~0.01 ~ 1km)
step = 0.01

# Generate grid
lats = np.arange(min_lat, max_lat, step)
lons = np.arange(min_lon, max_lon, step)

# Create all combinations
grid = [(lat, lon) for lat in lats for lon in lons]
df_grid = pd.DataFrame(grid, columns=["Latitude", "Longitude"])

# Save if needed
df_grid.to_csv("madrid_coordinate_grid.csv", index=False)
