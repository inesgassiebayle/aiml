# Example: Calculate distance to the nearest school (using external geospatial data)
X['Distance_to_School'] = calculate_distance(X['X'], X['Y'], school_lat, school_lon)


import osmnx as ox
import geopandas as gpd

# Define the city or region
place_name = "Athens, Greece"

# Query OpenStreetMap for schools
schools = ox.geometries_from_place(place_name, tags={'amenity': 'school'})

# Convert to a list of school latitudes and longitudes
school_coords = schools[['geometry']].dropna()
school_lat_lon = [(point.y, point.x) for point in school_coords.geometry]


from geopy.distance import geodesic

def calculate_distance(lat1, lon1, locations_list):
    """
    Given a latitude & longitude, find the minimum distance to a list of locations.
    """
    distances = [geodesic((lat1, lon1), (lat2, lon2)).km for lat2, lon2 in locations_list]
    return min(distances) if distances else None  # Return nearest distance

# Apply distance calculation to dataset
X['Distance_to_School'] = X.apply(lambda row: calculate_distance(row['Y'], row['X'], school_lat_lon), axis=1)
