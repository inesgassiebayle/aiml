import osmnx as ox
import plotly.express as px

# Obtener el grafo de Madrid
city = "Madrid, Spain"
G = ox.graph_from_place(city, network_type="all")

# Convertir los nodos del grafo a GeoDataFrame
df_madrid = ox.graph_to_gdfs(G, nodes=True, edges=False)

# Extraer coordenadas
df_madrid["X"] = df_madrid.geometry.x
df_madrid["Y"] = df_madrid.geometry.y

# Graficar con Plotly
fig = px.scatter(
    df_madrid,
    x="X",
    y="Y",
    title="Madrid Road Network Nodes (Interactive)",
    labels={"X": "Longitude", "Y": "Latitude"},
    width=800,
    height=600
)

# Mostrar gráfico interactivo
fig.show()
