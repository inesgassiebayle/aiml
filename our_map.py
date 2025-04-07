from datasets import load_dataset
import plotly.express as px

# Cargar dataset
train = load_dataset("ieuniversity/competition_ai_ml_24_train")
df_train = train["train"].to_pandas()

# Graficar con Plotly
fig = px.scatter(
    df_train,
    x="X",
    y="Y",
    title="Scatter Plot of X and Y from df_train (Interactive)",
    labels={"X": "Longitude", "Y": "Latitude"},
    width=800,
    height=600
)

# Mostrar el gráfico interactivo
fig.show()
