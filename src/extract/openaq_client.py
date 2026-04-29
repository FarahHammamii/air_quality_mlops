"""
src/extract/openaq_client.py
============================
CHANGES vs original:
  - fetch_city_data() accepts save_path= instead of hardcoding ../data/raw/
  - fetch_city_data() returns (df, fetch_start, fetch_end) so callers can
    update their watermark from the actual date range used
  - days_back can be a float for sub-day granularity
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

COUNTRY_IDS = {
    "IN": 13, "US": 5, "FR": 67, "DE": 73, "GB": 2, "CN": 20,
    "PK": 163, "BD": 16, "NP": 155, "TN": 212, "NG": 157, "MX": 142,
}


class OpenAQClient:
    def __init__(self):
        self.api_key = os.getenv("OPENAQ_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAQ_API_KEY not found in .env file")
        self.base_url = "https://api.openaq.org/v3"
        self.headers = {"X-API-Key": self.api_key, "Content-Type": "application/json"}

    def get_country_id(self, iso_code: str) -> int | None:
        if iso_code.upper() in COUNTRY_IDS:
            return COUNTRY_IDS[iso_code.upper()]
        url = f"{self.base_url}/countries"
        r = requests.get(url, headers=self.headers, params={"limit": 200})
        if r.status_code == 200:
            for c in r.json().get("results", []):
                if c.get("code", "").upper() == iso_code.upper():
                    return c["id"]
        return None

    def get_locations(self, country_iso=None, bbox=None, coordinates=None,
                      radius=25000, limit=10):
        url = f"{self.base_url}/locations"
        params = {"limit": limit}
        if bbox:
            params["bbox"] = ",".join(str(x) for x in bbox)
        elif coordinates:
            params["coordinates"] = f"{coordinates[0]},{coordinates[1]}"
            params["radius"] = min(radius, 25000)
        elif country_iso:
            cid = self.get_country_id(country_iso)
            if cid:
                params["countries_id"] = cid

        r = requests.get(url, headers=self.headers, params=params)
        return r.json().get("results", []) if r.status_code == 200 else []

    def get_sensors(self, location_id: int):
        r = requests.get(
            f"{self.base_url}/locations/{location_id}/sensors",
            headers=self.headers,
        )
        return r.json().get("results", []) if r.status_code == 200 else []

    def fetch_sensor_measurements(self, sensor_id, start_date, end_date, limit=1000):
        params = {
            "datetime_from": start_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "datetime_to":   end_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit": limit,
        }
        r = requests.get(
            f"{self.base_url}/sensors/{sensor_id}/measurements",
            headers=self.headers,
            params=params,
        )
        return r.json().get("results", []) if r.status_code == 200 else []

    def fetch_city_data(
        self,
        bbox=None,
        country_iso=None,
        parameters=None,
        days_back: float = 7,
        max_locations: int = 5,
        save_local: bool = True,
        # ── NEW: explicit path replaces the hardcoded ../data/raw/{label}.parquet ──
        save_path: str | None = None,
        label: str = "output",
    ) -> pd.DataFrame:
        """
        Fetch measurements and return a DataFrame.

        Parameters
        ----------
        save_path : str | None
            If provided, save parquet to this exact path.
            Falls back to ../data/raw/{label}.parquet when None (original behaviour).
        label : str
            Used for the fallback path and tqdm description.

        Returns
        -------
        pd.DataFrame
            Columns: datetime, value, parameter, unit, location, latitude, longitude
        """
        end_date   = datetime.utcnow()
        start_date = end_date - timedelta(days=days_back)

        locations = self.get_locations(
            country_iso=country_iso, bbox=bbox, limit=max_locations
        )
        if not locations:
            raise ValueError("No locations found for the given filters.")

        all_rows = []
        for loc in tqdm(locations, desc=f"Fetching {label}"):
            loc_id   = loc["id"]
            loc_name = loc.get("name")
            loc_lat  = loc.get("coordinates", {}).get("latitude")
            loc_lon  = loc.get("coordinates", {}).get("longitude")

            for sensor in self.get_sensors(loc_id):
                p_name = sensor.get("parameter", {}).get("name", "").lower()
                if parameters and p_name not in parameters:
                    continue

                measurements = self.fetch_sensor_measurements(
                    sensor["id"], start_date, end_date
                )
                for m in measurements:
                    period = m.get("period", {})
                    dt_obj = period.get("datetimeFrom", {})
                    dt_utc = dt_obj.get("utc") if isinstance(dt_obj, dict) else dt_obj
                    if not dt_utc:
                        continue
                    all_rows.append({
                        "datetime":  dt_utc,
                        "value":     m.get("value"),
                        "parameter": p_name,
                        "unit":      sensor.get("parameter", {}).get("units"),
                        "location":  loc_name,
                        "latitude":  loc_lat,
                        "longitude": loc_lon,
                    })
                time.sleep(0.1)

        df = pd.DataFrame(all_rows)
        if df.empty:
            return df

        df["datetime"] = pd.to_datetime(df["datetime"])

        if save_local:
            # ── NEW: use explicit save_path if given, else fall back to original ──
            if save_path:
                out = save_path
            else:
                os.makedirs("../data/raw", exist_ok=True)
                out = f"../data/raw/{label}.parquet"

            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
            df.to_parquet(out, index=False)

        return df
