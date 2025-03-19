import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset

# Load Data
print("Loading data sets...")
train = load_dataset("ieuniversity/competition_ai_ml_24_train")
test = load_dataset("ieuniversity/competition_ai_ml_24_test")

df_train = train["train"].to_pandas()
df_test = test["train"].to_pandas()

# Plot the data
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_train, x="X", y="Y", alpha=0.5)

# Add labels and title
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Scatter Plot of X vs Y")

# Show the plot
plt.show()
