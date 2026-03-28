import cdsapi
import time

# All unique locations (coords rounded to 1 decimal place)
locations = {
    # --- Cádiz ---
    "cadiz_puerto_real":     {"lat": 36.5, "lon": -6.3},
    "cadiz_amazon_arco":     {"lat": 36.7, "lon": -6.1},
    "cadiz_las_quinientas":  {"lat": 36.6, "lon": -6.1},  # same rounded as amazon_arco — see note
    "cadiz_miramundo":       {"lat": 36.5, "lon": -6.1},
    "cadiz_cartuja":         {"lat": 36.6, "lon": -6.1},  # same rounded as las_quinientas
    "cadiz_el_yarte_group":  {"lat": 36.6, "lon": -5.8},  # covers El Yarte, La Guita, Arenosas
    # --- Zaragoza ---
    "zaragoza_barrica":      {"lat": 40.9, "lon": -1.3},
    "zaragoza_tico":         {"lat": 41.2, "lon": -1.0},
    "zaragoza_los_belos":    {"lat": 41.5, "lon": -1.1},
    "zaragoza_amazon":       {"lat": 41.7, "lon": -0.8},
    "zaragoza_azaila":       {"lat": 41.3, "lon": -0.5},  # same rounded as alizarsun/talento
    "zaragoza_talento_group":{"lat": 41.2, "lon": -0.4},  # covers Alizarsun, Talento, Esplendor
}

# De-duplicate by (lat, lon) — ERA5-Land is 0.1° grid so identical rounded coords = same cell
seen = {}
unique_locations = {}
for name, coords in locations.items():
    key = (coords["lat"], coords["lon"])
    if key not in seen:
        seen[key] = name
        unique_locations[name] = coords
    else:
        print(f"  SKIP {name} — same ERA5 cell as '{seen[key]}' ({key})")

print(f"\n→ {len(unique_locations)} unique ERA5 grid cells to download\n")

dataset = "reanalysis-era5-land-timeseries"
variables = [
    "2m_dewpoint_temperature",
    "2m_temperature",
    "surface_pressure",
    "total_precipitation",
    "surface_solar_radiation_downwards",
    "surface_thermal_radiation_downwards",
]
date_range = "2023-01-01/2025-12-31"

client = cdsapi.Client()

for name, coords in unique_locations.items():
    filename = f"era5_{name}_{coords['lat']}N_{coords['lon']}E.csv"
    print(f"Requesting: {name} → lat={coords['lat']}, lon={coords['lon']}")

    request = {
        "variable": variables,
        "location": {"longitude": coords["lon"], "latitude": coords["lat"]},
        "date": [date_range],
        "data_format": "csv",
    }

    try:
        client.retrieve(dataset, request).download(filename)
        print(f"  ✓ Saved: {filename}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    time.sleep(2)  # for politeness to the API

print("\nAll done!")