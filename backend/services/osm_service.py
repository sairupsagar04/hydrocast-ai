# services/osm_service.py
import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY", "")
client = genai.Client(api_key=gemini_key) if gemini_key else None

def fetch_roads_near_point(lat: float, lng: float, radius_meters: int = 2000, place_name: str = "") -> dict:
    resolved_place = place_name.strip() if place_name.strip() else f"[{lat:.4f}, {lng:.4f}]"
    is_urban = any(
        k in resolved_place.lower() 
        for k in ["kondapur", "madhapur", "hyderabad", "patancheru", "mumbai", "dadar", "bengaluru", "delhi", "sector", "ward"]
    )
    
    if not client:
        return get_fast_fallback(lat, lng, resolved_place, is_urban)

    zone_type = f"Urban Municipal Sector ({resolved_place})" if is_urban else f"Rural Catchment / Outskirts ({resolved_place})"

    prompt = f"""
    Generate an urban hydrologic infrastructure layout for geographic center Lat: {lat}, Lng: {lng}, Named: "{resolved_place}".
    Urban Status: {is_urban}. Zone Classification: "{zone_type}".

    Generate 3 to 5 realistic roadway segments, each having a 'coordinates' list of 3-5 multi-point [lat, lng] waypoints representing curved road geometries.
    If Urban: generate 4 to 8 stormwater conduits and interconnected manhole junctions.
    If Rural: conduits and manholes must be empty arrays [].

    Ensure the output strictly conforms to this JSON structure:
    {{
      "zone_classification": "{zone_type}",
      "is_urban": {str(is_urban).lower()},
      "streets": [
        {{
          "id": "s-1",
          "name": "Main Arterial Corridor",
          "lat": {lat},
          "lng": {lng},
          "coordinates": [[{lat - 0.003}, {lng - 0.003}], [{lat}, {lng}], [{lat + 0.003}, {lng + 0.002}]],
          "is_subway": false,
          "surface": "asphalt"
        }}
      ],
      "conduits": [
        {{
          "id": "c-1",
          "name": "Stormwater Trunk 1",
          "from_node": "m-1",
          "to_node": "m-2",
          "length_m": 160,
          "diameter_mm": 900,
          "slope": 0.005,
          "coordinates": [[{lat}, {lng}], [{lat + 0.002}, {lng + 0.002}]],
          "capacity_used_pct": 55,
          "status": "normal"
        }}
      ],
      "manholes": [
        {{
          "id": "m-1",
          "name": "Node A",
          "lat": {lat},
          "lng": {lng},
          "type": "manhole",
          "capacity_used_pct": 50,
          "flow_rate_m3s": 3.0,
          "surcharge_status": "normal",
          "is_surcharging": false,
          "is_blocked_inlet": false
        }}
      ]
    }}
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )
        return json.loads(response.text.strip())
    except Exception as e:
        print(f"[!] Warning: Gemini infrastructure generation failed ({e}). Reverting to fast fallback.")
        return get_fast_fallback(lat, lng, resolved_place, is_urban)


def get_fast_fallback(lat: float, lng: float, place_name: str, is_urban: bool) -> dict:
    streets = [
        {
            "id": "f-1",
            "name": f"Main Arterial - {place_name}",
            "lat": lat,
            "lng": lng,
            "coordinates": [
                [lat - 0.003, lng - 0.003],
                [lat - 0.001, lng - 0.001],
                [lat, lng],
                [lat + 0.002, lng + 0.001],
                [lat + 0.004, lng + 0.003]
            ],
            "is_subway": is_urban,
            "surface": "asphalt" if is_urban else "dirt"
        },
        {
            "id": "f-2",
            "name": f"Bypass Ridge Way - {place_name}",
            "lat": lat + 0.002,
            "lng": lng - 0.002,
            "coordinates": [
                [lat - 0.002, lng + 0.003],
                [lat + 0.001, lng + 0.001],
                [lat + 0.002, lng - 0.002],
                [lat + 0.005, lng - 0.004]
            ],
            "is_subway": False,
            "surface": "asphalt"
        },
        {
            "id": "f-3",
            "name": f"Local Access Route - {place_name}",
            "lat": lat - 0.002,
            "lng": lng + 0.002,
            "coordinates": [
                [lat - 0.004, lng - 0.001],
                [lat - 0.002, lng + 0.002],
                [lat, lng + 0.003]
            ],
            "is_subway": False,
            "surface": "asphalt" if is_urban else "unpaved"
        }
    ]

    conduits = [
        {
            "id": "fc-1",
            "name": "Conduit Trunk 1",
            "from_node": "m-1",
            "to_node": "m-2",
            "length_m": 120,
            "diameter_mm": 900,
            "slope": 0.005,
            "coordinates": [[lat, lng], [lat + 0.002, lng + 0.002]],
            "capacity_used_pct": 50,
            "status": "normal"
        },
        {
            "id": "fc-2",
            "name": "Conduit Trunk 2",
            "from_node": "m-2",
            "to_node": "m-3",
            "length_m": 180,
            "diameter_mm": 1200,
            "slope": 0.004,
            "coordinates": [[lat + 0.002, lng + 0.002], [lat + 0.004, lng + 0.003]],
            "capacity_used_pct": 75,
            "status": "normal"
        }
    ] if is_urban else []

    manholes = [
        {
            "id": "m-1",
            "name": f"{place_name} Junction A",
            "lat": lat,
            "lng": lng,
            "type": "manhole",
            "capacity_used_pct": 45,
            "flow_rate_m3s": 2.5,
            "surcharge_status": "normal",
            "is_surcharging": False,
            "is_blocked_inlet": False
        },
        {
            "id": "m-2",
            "name": f"{place_name} Junction B",
            "lat": lat + 0.002,
            "lng": lng + 0.002,
            "type": "manhole",
            "capacity_used_pct": 78,
            "flow_rate_m3s": 4.0,
            "surcharge_status": "warning",
            "is_surcharging": False,
            "is_blocked_inlet": False
        },
        {
            "id": "m-3",
            "name": f"{place_name} Junction C",
            "lat": lat + 0.004,
            "lng": lng + 0.003,
            "type": "manhole",
            "capacity_used_pct": 92,
            "flow_rate_m3s": 5.2,
            "surcharge_status": "overflow",
            "is_surcharging": True,
            "is_blocked_inlet": True
        }
    ] if is_urban else []

    return {
        "zone_classification": f"Urban Municipal Sector ({place_name})" if is_urban else f"Rural Outskirts ({place_name})",
        "is_urban": is_urban,
        "streets": streets,
        "conduits": conduits,
        "manholes": manholes
    }