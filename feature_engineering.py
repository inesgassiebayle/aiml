import plotly.express as px
from datasets import load_dataset

# Load Data
print("Loading data sets...")
train = load_dataset("ieuniversity/competition_ai_ml_24_train")
test = load_dataset("ieuniversity/competition_ai_ml_24_test")

df_train = train["train"].to_pandas()
df_test = test["train"].to_pandas()

fig = px.scatter(
    df_train,
    x="X",
    y="Y",
    title="Interactive Scatter Plot of X vs Y",
    labels={"X": "Longitude", "Y": "Latitude"},
    hover_data=["X", "Y"]
)

# Show the plot
fig.show()
