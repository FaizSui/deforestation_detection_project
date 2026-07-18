"""
Step 6: Download Sentinel-2 RGB imagery of Rondonia (BR-364 corridor, near
Porto Velho) for two time periods, using the Copernicus Data Space
Ecosystem (CDSE) Catalog + Process APIs.

What this script does, in plain English:
1. Reads your API credentials from a local .env file (never typed into
   chat, never uploaded anywhere) and exchanges them for a temporary
   access token -- standard OAuth2 "client credentials" login.
2. Defines a fixed geographic bounding box over the BR-364 corridor
   southeast of Porto Velho, Rondonia. Both downloads use the EXACT same
   box, which is what lets us compare them pixel-for-pixel later.
3. For each time period (July-Aug 2018, July-Aug 2024), first SEARCHES
   the Catalog API for every individual Sentinel-2 pass over that box in
   that window, and prints each one's date and scene-wide cloud cover
   percentage.

   This replaces our first attempt, which asked the Process API to build
   a "least cloud cover" MOSAIC across the whole 2-month window. That
   approach quietly blended pixels from multiple different satellite
   passes into one image -- which produced a visible seam and a hazy/
   smoky patch in the 2024 image, because it stitched together two
   different dates with different atmospheric conditions. Searching first
   and downloading a single specific date avoids that: every pixel in the
   final image comes from the same satellite pass, so there's nothing to
   stitch.
4. The catalog's "cloud cover" percentage turned out not to catch
   everything -- one 0.0%-cloud-cover date still came back visibly hazy
   (thin atmospheric haze/cirrus that the automated cloud mask doesn't
   flag as "cloud"). So instead of trusting that number blindly, we
   fetch a small, cheap low-resolution PREVIEW of the several least-cloudy
   candidate dates and measure their actual pixel contrast (haze flattens
   contrast -- it washes everything toward a similar mid-brightness gray).
   Whichever candidate has the highest contrast is the clearest, and only
   THAT one gets downloaded at full resolution.
5. Saves the result as ./data/sentinel2_rondonia_<year>.png

Note on processing level: this uses Sentinel-2 L1C (top-of-atmosphere
reflectance, NOT atmospherically corrected), not L2A. This matters
because EuroSAT -- the dataset our classifier was trained on -- was built
from L1C imagery. L2A (atmospherically-corrected) imagery has a visibly
different color balance (atmospheric haze brightens the blue channel in
L1C), which caused systematic misclassification when we first tried L2A.
Matching the same processing level as the training data avoids that
train/serve mismatch.

Note on coordinate system: the actual image request is made in a LOCAL
UTM PROJECTION (meters), not raw latitude/longitude degrees. Degrees of
longitude and latitude don't correspond to a fixed number of meters --
that distance shrinks toward the poles and even varies slightly with
which direction you're measuring. Requesting a bbox directly in degrees
and dividing by a pixel count only gives an *approximate* meters/pixel
resolution (an earlier version of this script was off by about 3% in the
east-west direction because of exactly this). Working in UTM meters
avoids the approximation entirely: we pick a center point, define the
bbox as exactly +/-8000m and +/-8320m in projected meters, and request
exactly 1600x1664 pixels -- giving an EXACT 10.000 m/pixel resolution,
matching EuroSAT's stated resolution precisely rather than approximately.
"""

import io
import os
import sys

import numpy as np
import requests
from PIL import Image
from pyproj import Transformer
from dotenv import load_dotenv

load_dotenv()  # reads CDSE_CLIENT_ID / CDSE_CLIENT_SECRET from a .env file in this folder or a parent folder

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CATALOG_URL = "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

# Center point over the BR-364 corridor southeast of Porto Velho, Rondonia.
# Everything below is derived from this single point, so shifting the
# study area is just a matter of changing these two numbers.
CENTER_LON = -63.55
CENTER_LAT = -9.05

# At 10 meters/pixel (matching EuroSAT's resolution, so our trained model
# sees imagery at the same scale it learned on), our 25x26-patch grid
# (64px patches) needs exactly 1600 x 1664 pixels.
WIDTH_PX = 1600   # 25 patches of 64px -> exactly 16,000 m at 10 m/pixel
HEIGHT_PX = 1664  # 26 patches of 64px -> exactly 16,640 m at 10 m/pixel

# UTM zone number = floor((lon + 180) / 6) + 1. EPSG:327xx is WGS84 UTM
# zone xx, southern hemisphere (327 prefix; 326 would be northern).
_UTM_ZONE = int((CENTER_LON + 180) / 6) + 1
PROCESS_CRS_EPSG = 32700 + _UTM_ZONE
PROCESS_CRS_URI = f"http://www.opengis.net/def/crs/EPSG/0/{PROCESS_CRS_EPSG}"

_to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{PROCESS_CRS_EPSG}", always_xy=True)
_center_easting, _center_northing = _to_utm.transform(CENTER_LON, CENTER_LAT)

_half_width_m = WIDTH_PX * 10 / 2
_half_height_m = HEIGHT_PX * 10 / 2

# The bbox actually sent to the Process API: exact meters, exact 10m/pixel.
PROCESS_BBOX = [
    _center_easting - _half_width_m,
    _center_northing - _half_height_m,
    _center_easting + _half_width_m,
    _center_northing + _half_height_m,
]

# The Catalog API's bbox parameter is always expected in plain lon/lat
# (WGS84), regardless of the imagery's native CRS -- and it's only used
# to find overlapping scenes, so it doesn't need to be pixel-exact. We
# derive an approximate lon/lat box from the same UTM box for that search.
_to_lonlat = Transformer.from_crs(f"EPSG:{PROCESS_CRS_EPSG}", "EPSG:4326", always_xy=True)
_lon_min, _lat_min = _to_lonlat.transform(PROCESS_BBOX[0], PROCESS_BBOX[1])
_lon_max, _lat_max = _to_lonlat.transform(PROCESS_BBOX[2], PROCESS_BBOX[3])
CATALOG_BBOX = [_lon_min, _lat_min, _lon_max, _lat_max]

SEARCH_WINDOWS = {
    2018: ("2018-07-01T00:00:00Z", "2018-08-31T23:59:59Z"),
    2024: ("2024-07-01T00:00:00Z", "2024-08-31T23:59:59Z"),
}

# Standard Sentinel-2 true-color rendering (same visual convention used by
# Sentinel Hub's EO Browser): read the Red, Green, Blue bands (each a
# dimensionless top-of-atmosphere reflectance value, nominally 0-1) and
# apply a 2.5x brightness gain so the image isn't too dark -- vegetation
# and soil reflect well under half of incoming light, so without this
# gain a raw reflectance render looks nearly black.
#
# sampleType "AUTO" then converts each gained value to an 8-bit pixel via
# a documented, exact rule (Sentinel Hub Evalscript V3 spec): values are
# linearly stretched from [0, 1] to [0, 255] and clamped at both ends --
# NO gamma correction is applied. Combined with our 2.5x gain, the full
# formula from raw reflectance to output pixel value is:
#
#     pixel_value = round(clip(2.5 * reflectance, 0, 1) * 255)
#
# which means any raw reflectance >= 0.4 in a band saturates that
# channel to pure 255. This is worth knowing explicitly: very bright
# surfaces (bare soil, rooftops, thin cloud) can hit this ceiling and
# lose relative brightness differences above it.
EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: ["B02", "B03", "B04"],
    output: { bands: 3, sampleType: "AUTO" }
  };
}
function evaluatePixel(sample) {
  return [2.5 * sample.B04, 2.5 * sample.B03, 2.5 * sample.B02];
}
"""


def get_access_token(client_id, client_secret):
    response = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
    )
    response.raise_for_status()
    return response.json()["access_token"]


def find_best_date(token, time_from, time_to):
    """Search the catalog for every Sentinel-2 pass over our bbox in the
    given window, print each one's date and cloud cover, and return the
    date string (YYYY-MM-DD) of the least-cloudy pass."""

    request_body = {
        "collections": ["sentinel-2-l1c"],
        "bbox": CATALOG_BBOX,
        "datetime": f"{time_from}/{time_to}",
        "limit": 100,
    }

    response = requests.post(
        CATALOG_URL,
        headers={"Authorization": f"Bearer {token}"},
        json=request_body,
    )
    response.raise_for_status()
    features = response.json()["features"]

    if not features:
        raise RuntimeError(f"No Sentinel-2 scenes found between {time_from} and {time_to}")

    # Each feature is one satellite pass. properties["eo:cloud_cover"] is
    # the cloud cover percentage over the WHOLE scene (Sentinel-2 tiles
    # are about 110km x 110km), not just our small bbox -- so it's a
    # useful proxy but not a perfect one. Good enough to rank candidates.
    candidates = []
    for feature in features:
        date = feature["properties"]["datetime"][:10]
        cloud_cover = feature["properties"].get("eo:cloud_cover", 100.0)
        candidates.append((date, cloud_cover))

    # A single date can appear twice if two overlapping orbits pass over
    # the area on the same day -- keep the lowest cloud cover per date.
    best_per_date = {}
    for date, cloud_cover in candidates:
        if date not in best_per_date or cloud_cover < best_per_date[date]:
            best_per_date[date] = cloud_cover

    sorted_dates = sorted(best_per_date.items(), key=lambda x: x[1])

    print(f"  Found {len(sorted_dates)} candidate dates:")
    for date, cloud_cover in sorted_dates:
        print(f"    {date}: {cloud_cover:.1f}% scene cloud cover")

    return sorted_dates


def request_image(token, date, width, height):
    """Request a true-color image for a single specific day (00:00 to
    23:59 UTC on that date), so the result is guaranteed to come from one
    satellite pass instead of being blended across multiple dates.
    Returns the raw PNG bytes."""

    time_from = f"{date}T00:00:00Z"
    time_to = f"{date}T23:59:59Z"

    request_body = {
        "input": {
            "bounds": {
                "bbox": PROCESS_BBOX,
                "properties": {"crs": PROCESS_CRS_URI},
            },
            "data": [
                {
                    "type": "sentinel-2-l1c",
                    "dataFilter": {
                        "timeRange": {"from": time_from, "to": time_to},
                    },
                }
            ],
        },
        "output": {
            "width": width,
            "height": height,
            "responses": [{"identifier": "default", "format": {"type": "image/png"}}],
        },
        "evalscript": EVALSCRIPT,
    }

    response = requests.post(
        PROCESS_URL,
        headers={"Authorization": f"Bearer {token}"},
        json=request_body,
    )

    if response.status_code != 200:
        print(f"Request failed ({response.status_code}): {response.text}")
        response.raise_for_status()

    return response.content


def measure_contrast(png_bytes):
    """Higher standard deviation across pixel values means more contrast
    (a healthy mix of dark forest, mid-tone fields, bright clearings).
    Haze/thin cirrus washes everything toward the same mid-gray, which
    lowers this number even when the official cloud mask reports 0%."""
    image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    arr = np.array(image)
    return arr.std()


def pick_clearest_date(token, candidates, num_to_check=5):
    """Fetch a small, cheap preview of the top N least-cloudy candidate
    dates and return whichever one has the highest actual pixel contrast."""

    preview_size = 200  # small on purpose -- just enough to judge haze, cheap to request
    results = []

    print(f"  Checking previews of the {min(num_to_check, len(candidates))} least-cloudy candidates for haze...")
    for date, cloud_cover in candidates[:num_to_check]:
        preview_bytes = request_image(token, date, preview_size, preview_size)
        contrast = measure_contrast(preview_bytes)
        print(f"    {date}: contrast={contrast:.1f} (cloud cover {cloud_cover:.1f}%)")
        results.append((date, contrast))

    best_date, best_contrast = max(results, key=lambda x: x[1])
    print(f"  Selected {best_date} (highest contrast: {best_contrast:.1f})")
    return best_date


def download_image(token, date, out_path):
    png_bytes = request_image(token, date, WIDTH_PX, HEIGHT_PX)

    with open(out_path, "wb") as f:
        f.write(png_bytes)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"Saved {out_path} ({size_kb:.0f} KB)")


def main():
    client_id = os.environ.get("CDSE_CLIENT_ID")
    client_secret = os.environ.get("CDSE_CLIENT_SECRET")

    if not client_id or not client_secret:
        print("Missing credentials. Create a .env file in the project root with:")
        print("  CDSE_CLIENT_ID=your-client-id")
        print("  CDSE_CLIENT_SECRET=your-client-secret")
        sys.exit(1)

    os.makedirs(DATA_DIR, exist_ok=True)

    print("Requesting access token...")
    token = get_access_token(client_id, client_secret)
    print("Got access token.")

    for year, (time_from, time_to) in SEARCH_WINDOWS.items():
        print(f"\nSearching {year} window ({time_from} to {time_to})...")
        candidates = find_best_date(token, time_from, time_to)
        best_date = pick_clearest_date(token, candidates)

        print(f"Downloading full-resolution image for {best_date}...")
        out_path = os.path.join(DATA_DIR, f"sentinel2_rondonia_{year}.png")
        download_image(token, best_date, out_path)


if __name__ == "__main__":
    main()
