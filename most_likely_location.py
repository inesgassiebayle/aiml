from datasets import load_dataset
from pyproj import Transformer
import numpy as np



# Step 1: Load the dataset
print("Loading data sets...")
train = load_dataset("ieuniversity/competition_ai_ml_24_train")
test = load_dataset("ieuniversity/competition_ai_ml_24_test")

df_train = train["train"].to_pandas()

# Step 2: Extract X and Y as numpy arrays
x_values = np.array(df_train["X"])
y_values = np.array(df_train["Y"])

# Step 3: Compute the centroid (mean coordinates)
x_mean = np.mean(x_values)
y_mean = np.mean(y_values)

# Step 4: Convert from EPSG:25830 to EPSG:4326 (longitude/latitude)
transformer = Transformer.from_crs("EPSG:25830", "EPSG:4326", always_xy=True)
lon, lat = transformer.transform(x_mean, y_mean)

# Step 5: Print result
print(f"📍 Exact Location Coordinates:")
print(f"Latitude: {lat:.6f}, Longitude: {lon:.6f}")

