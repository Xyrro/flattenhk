"""Registry of every external dataset used by the project.

Each entry records the URL, the access date, resolution/vintage, licence and
the limitations that matter for this analysis.  ``python -m hk_flat_routes
sources`` prints this table, and it is the single source of truth for the
data-provenance section of the README.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

#: Date on which every URL below was last fetched and verified.
ACCESS_DATE = "2026-10-13"

#: Overture Maps release every published figure was built from.  Overture
#: keeps only its latest few releases on the bucket (three on the access
#: date, the oldest of them 55 days old), so a fresh ``download`` works from
#: this one only while it is still hosted; ``HKFR_OVERTURE_RELEASE`` names
#: another when it is not.  A different release is different data: the
#: figures move with it, and its schema may have moved too (see
#: ``download.PLACE_COLUMNS``).
OVERTURE_RELEASE_PINNED = "2026-09-23.1"
OVERTURE_RELEASE = os.environ.get("HKFR_OVERTURE_RELEASE") or OVERTURE_RELEASE_PINNED
OVERTURE_BUCKET = "https://overturemaps-us-west-2.s3.amazonaws.com"
OVERTURE_PREFIX = f"release/{OVERTURE_RELEASE}/theme=transportation"
#: Places and base themes of the same release, used only for the route
#: page's offline place search.  The addresses theme is not read: its Hong
#: Kong rows are a copy of the government's Address Lookup Service file in
#: English only, so the file itself is fetched instead (``ALS_URL``).
OVERTURE_PLACES_PREFIX = f"release/{OVERTURE_RELEASE}/theme=places"
OVERTURE_BASE_PREFIX = f"release/{OVERTURE_RELEASE}/theme=base"
#: Divisions theme: the SAR boundary (land and waters) that clips the network.
OVERTURE_DIVISIONS_PREFIX = f"release/{OVERTURE_RELEASE}/theme=divisions"

#: CEDD territory-wide airborne LiDAR survey of 2020: the bare-earth DTM at
#: 0.5 m, published tile by tile on the government's spatial data portal.
CSDI_FILE_API = "https://portal.csdi.gov.hk/csdi-webpage/file-api"
LIDAR_DATASET_ID = "cedd_rcd_1629267205233_87895"
#: Grid index: one polygon per 750 m x 600 m tile, with its download URL.
DTM_INDEX_URL = (f"{CSDI_FILE_API}?dataset_id={LIDAR_DATASET_ID}"
                 "&format=geojson&layer_name=DTM_2020")
DTM_TILE_BASE = "https://bulkdata.csdi.gov.hk/cedd/lidar/2020dtm"

#: Neighborhood polygons: the Town Planning Board's statutory planning scheme
#: areas, one polygon per Outline Zoning Plan, named in English and Chinese.
NEIGHBORHOOD_URL = (
    "https://www.ozp.tpb.gov.hk/arcgis2/rest/services/DATA/OZP_PLAN_CSDI/"
    "FeatureServer/0/query?where=1%3D1&outFields=*&outSR=4326&f=geojson"
)

#: Lands Department 5 m terrain model: one file for the whole territory,
#: photogrammetric and independent of the LiDAR survey.  Cross-check only.
DTM_5M_URL = ("https://www.landsd.gov.hk/landsd_psi_data/SMO/data/"
              "Whole_HK_DTM_5m.zip")

#: Survey and Mapping Office bench marks: levelled heights above the
#: Principal Datum, the ground truth of the validation stage.
BENCHMARKS_URL = "https://www.geodetic.gov.hk/common/data/GSCS/BM.xml"
#: The terms the bench marks are used under, stated here once; the README
#: and LICENSE repeat this wording.  The file above is the Lands
#: Department's own copy of its Geodetic Survey Control Station Database,
#: and the page that offers it points to the same database on the CSDI
#: Portal (dataset ``landsd_rcd_1637226129981_91080``, which DATA.GOV.HK
#: lists as ``hk-landsd-openmap-geodetic-database``), so the portal's terms
#: are the ones relied on.
BENCHMARKS_TERMS = (
    "CSDI Portal Terms and Conditions of Use (the Geodetic Survey Control "
    "Station Database is a dataset of the CSDI Portal, fetched here from "
    "the Lands Department's own copy): free for commercial and "
    "non-commercial use, the Government and the CSDI Portal to be named as "
    "the source. Used for validation only and not redistributed beyond "
    "the stations quoted in the validation report.")

# ---- place search (route page only; see places.py) ------------------------
#: Lands Department gazetteer of place names: one layer of points and one of
#: the names that belong to them, joined on GEO_NAME_ID.
PLACE_NAME_DATASET_ID = "landsd_rcd_1648571595120_89752"
PLACE_NAME_POINTS_URL = (f"{CSDI_FILE_API}?dataset_id={PLACE_NAME_DATASET_ID}"
                         "&format=geojson&layer_name=GEO_PLACE_NAME")
PLACE_NAME_NAMES_URL = (f"{CSDI_FILE_API}?dataset_id={PLACE_NAME_DATASET_ID}"
                        "&format=geojson&layer_name=PLACE_NAME")
#: Lands Department building names: a table without geometry, whose
#: BuildingCSUID carries the HK1980 Grid position of each building.
BUILDING_DATASET_ID = "landsd_rcd_1637211194312_35158"
BUILDING_NAME_URL = (f"{CSDI_FILE_API}?dataset_id={BUILDING_DATASET_ID}"
                     "&format=geojson&layer_name=BuildingName")
#: Lands Department iGeoCom points of interest, as one CSV in a zip.
IGEOCOM_URL = ("https://open.hkmapservice.gov.hk/OpenData/directDownload"
               "?productName=iGeoCom&sheetName=iGeoCom&productFormat=CSV")
#: The download host balances between two servers and one of them answers
#: with a redirect to an internal error page; this cookie asks for the other.
IGEOCOM_COOKIE = {"SERVERID": "web1-app1-OpenData"}
#: Address Lookup Service: every address the service knows, by district.
ALS_URL = "https://www.als.gov.hk/data/ALS-GeoJSON.zip"
#: Transport Department stop lists, one feature per stop of every route.
TD_STOPS_BASE = "https://static.data.gov.hk/td/routes-fares-geojson"
TD_STOPS_URLS = {
    "tram": f"{TD_STOPS_BASE}/JSON_TRAM.json",
    "peak_tram": f"{TD_STOPS_BASE}/JSON_PTRAM.json",
    "ferry": f"{TD_STOPS_BASE}/JSON_FERRY.json",
}


@dataclass(frozen=True)
class Dataset:
    key: str
    title: str
    publisher: str
    url: str
    accessed: str
    resolution: str
    licence: str
    limitations: str
    role: str
    local: str = ""
    notes: str = ""
    optional: bool = False
    substituted: bool = False
    substitution_reason: str = ""


DATASETS: tuple[Dataset, ...] = (
    Dataset(
        key="overture_segments",
        title=f"Overture Maps transportation segments (release {OVERTURE_RELEASE})",
        publisher="Overture Maps Foundation (derived from OpenStreetMap)",
        url=f"{OVERTURE_BUCKET}/{OVERTURE_PREFIX}/type=segment/",
        accessed=ACCESS_DATE,
        resolution="Vector linestrings; OSM-equivalent positional accuracy (~1-5 m)",
        licence="ODbL 1.0 (OpenStreetMap contributors); Overture schema CDLA-Permissive 2.0",
        role="Routable street network: geometry, road class, per-mode access "
             "restrictions, bridge/tunnel flags, bilingual names and "
             "connector topology.",
        local="data/raw/overture_segments_hk.parquet",
        limitations=(
            "OSM-derived, so completeness and tagging quality vary by area. "
            "Consecutive ways are joined into one segment, so bridge and "
            "tunnel flags, access rules and even street names apply to "
            "ranges along a segment and have to be read per stretch. Trunk "
            "roads range from Gloucester Road, with pavements, to tunnels; "
            "a few road tunnels (Tate's Cairn among them) carry no ban on "
            "walking and are closed by rule here. Sidewalks are mapped as "
            "separate ways in some districts and not in others. Nothing "
            "marks an escalator or a lift: the Central-Mid-Levels escalator "
            "is a covered footway with steps. About a fifth of the "
            "segments in the bounding box are in Shenzhen and are clipped "
            "away with the SAR boundary. Class 'unknown' is not used at all: "
            "it holds OpenStreetMap's roads under construction and the "
            "2,148 segments of the extract that Overture takes from TomTom, "
            "which carry no class, name or access rule, so nothing in the "
            "network is TomTom's."
        ),
        notes="Read with Parquet row-group bbox pruning: 19 of 16,384 "
              "global row groups intersect Hong Kong, so the whole extract "
              "costs about a minute and 40 MB instead of 72 GB. Overture "
              "hosts only its latest few releases; see OVERTURE_RELEASE.",
    ),
    Dataset(
        key="overture_connectors",
        title=f"Overture Maps transportation connectors (release {OVERTURE_RELEASE})",
        publisher="Overture Maps Foundation (derived from OpenStreetMap)",
        url=f"{OVERTURE_BUCKET}/{OVERTURE_PREFIX}/type=connector/",
        accessed=ACCESS_DATE,
        resolution="Vector points",
        licence="ODbL 1.0; Overture schema CDLA-Permissive 2.0",
        role="Authoritative intersection nodes. Using connector IDs for graph "
             "topology avoids geometric snapping tolerances entirely.",
        local="data/raw/overture_connectors_hk.parquet",
        limitations="Connectors exist only where OSM ways share a node; "
                    "grade-separated crossings correctly do not connect.",
    ),
    Dataset(
        key="overture_divisions",
        title=f"Overture Maps divisions: division areas (release {OVERTURE_RELEASE})",
        publisher="Overture Maps Foundation (derived from OpenStreetMap)",
        url=f"{OVERTURE_BUCKET}/{OVERTURE_DIVISIONS_PREFIX}/type=division_area/",
        accessed=ACCESS_DATE,
        resolution="Polygons: the Hong Kong SAR (land, and land with "
                   "territorial waters) and its eighteen districts",
        licence="ODbL 1.0 (OpenStreetMap contributors)",
        role="The SAR boundary the street network is clipped to, which is "
             "what keeps Shenzhen's streets out.",
        local="data/raw/overture_division_area_hk.parquet",
        limitations="The polygon with waters is used, so piers, breakwaters "
                    "and the bridges to the airport are kept; a strip of "
                    "the Shenzhen Bay Port on the far shore is inside it "
                    "and falls away as fragments.",
    ),
    Dataset(
        key="overture_land_use",
        title=f"Overture Maps base theme: land use (release {OVERTURE_RELEASE})",
        publisher="Overture Maps Foundation (derived from OpenStreetMap)",
        url=f"{OVERTURE_BUCKET}/{OVERTURE_BASE_PREFIX}/type=land_use/",
        accessed=ACCESS_DATE,
        resolution="Mapped outlines with names and OSM-derived classes",
        licence="ODbL 1.0 (OpenStreetMap contributors)",
        role="The gated sites of the street network: service roads inside "
             "military land, the container terminals and the power stations "
             "are closed to walkers and cyclists (config.CLOSED_LAND_NAMES). "
             "Read again, with the rest of the base theme, for the place "
             "search.",
        local="data/raw/overture_land_use_hk.parquet",
        limitations="The theme has no class for a port or a power station, "
                    "so those are picked out by name; a gated yard that is "
                    "not mapped as land use, or is mapped under another "
                    "name, stays open, and the Frontier Closed Area is not "
                    "in the theme at all. A network built without this "
                    "file leaves every such road open, so download fetches "
                    "it with the street network and stops if it cannot.",
    ),
    Dataset(
        key="dem_05m",
        title="Digital Terrain Model from the 2020 territory-wide airborne "
              "LiDAR survey",
        publisher="Civil Engineering and Development Department (CEDD), "
                  "Geotechnical Engineering Office, HKSAR Government",
        url=DTM_INDEX_URL,
        accessed=ACCESS_DATE,
        resolution="0.5 m grid; HK1980 Grid (EPSG:2326); float32 metres "
                   "above the Hong Kong Principal Datum (mPD); 3,253 tiles "
                   "of 750 m x 600 m; flown 2020",
        licence="Free for commercial and non-commercial use with "
                "attribution (data.gov.hk / CSDI terms of use)",
        role="Primary elevation source for all grade and climbing metrics.",
        local="data/raw/dem/*.zip, data/raw/dem_index_2020.geojson",
        limitations=(
            "Bare earth: buildings, vegetation and elevated structures are "
            "removed, so under a footbridge, flyover or the Mid-Levels "
            "escalator the model gives the ground below, and in a tunnel "
            "the hill above -- handled by interpolating elevation across "
            "segments flagged is_bridge or is_tunnel. The sea is nodata. "
            "Podium-level walkways through buildings take the ground "
            "elevation. Residual decimetre noise is handled by the Gaussian "
            "pre-filter, Savitzky-Golay smoothing and the gain dead-band. "
            "14.1 GB zipped, in 3,253 files; the server serves a few MB/s."
        ),
    ),
    Dataset(
        key="neighborhoods",
        title="Statutory planning scheme areas (Outline Zoning Plan boundaries)",
        publisher="Town Planning Board / Planning Department, HKSAR Government",
        url=NEIGHBORHOOD_URL,
        accessed=ACCESS_DATE,
        resolution="151 polygons, one per plan, named in English and Chinese; "
                   "58 are used, as 56 neighborhoods (see "
                   "config.NEIGHBORHOOD_PLANS and NEIGHBORHOOD_MERGE)",
        licence="Town Planning Board terms of use: free for commercial and "
                "non-commercial use with attribution and acknowledgement of "
                "the Board's intellectual property rights",
        role="Neighborhood names and extents for origin/destination "
             "selection and corridor attribution.",
        local="data/raw/ozp_plan_scheme_areas.geojson",
        limitations=(
            "Planning areas, not neighborhoods as residents draw them: Sham "
            "Shui Po is inside 'Cheung Sha Wan', Happy Valley inside 'Wong "
            "Nai Chung', Kowloon City inside 'Ma Tau Kok'. They do not tile "
            "the territory (most of the countryside is in no plan), so they are not "
            "what the network is clipped to. New-town plans are large "
            "(Sha Tin is 28 km2) and each gets a single access point. "
            "Boundaries change when a plan is amended."
        ),
    ),
    Dataset(
        key="dtm_5m",
        title="Digital Terrain Model, 5 m grid, whole territory",
        publisher="Survey and Mapping Office, Lands Department, HKSAR Government",
        url=DTM_5M_URL,
        accessed=ACCESS_DATE,
        resolution="5 m grid, integer metres above HKPD, HK1980 Grid; stated "
                   "accuracy +/- 5 m; from 2015 aerial photography",
        licence="DATA.GOV.HK terms of use: free for commercial and "
                "non-commercial use with attribution",
        role="Independent cross-check on the LiDAR elevations (validation "
             "only -- far too coarse for street grades).",
        local="data/raw/Whole_HK_DTM_5m.zip",
        limitations="Photogrammetric: it carries tree canopy and elevated "
                    "roads, the sea is 0 rather than nodata, and heights are "
                    "whole metres.",
        optional=True,
    ),
    Dataset(
        key="bench_marks",
        title="Geodetic Survey Control Station database: bench marks",
        publisher="Survey and Mapping Office, Lands Department, HKSAR Government",
        url=BENCHMARKS_URL,
        accessed=ACCESS_DATE,
        resolution="1,862 levelled points; position on the HK1980 Grid to "
                   "1 m, height above HKPD to 0.5 mm",
        licence=BENCHMARKS_TERMS,
        role="Ground truth for the validation stage: absolute street "
             "elevations, mean gradients between marks on the same road, and "
             "the heights of the gaps (validation only).",
        local="data/raw/smo_bench_marks.xml",
        limitations="Heights are to the top of the mark, which stands about "
                    "0.2 m above the ground. Positions are rounded to 1 m "
                    "and a mark is beside the road, not on its centreline, "
                    "so one next to a retaining wall can be metres off the "
                    "carriageway. The locality is free text, not a street "
                    "key.",
        optional=True,
    ),
    Dataset(
        key="place_names",
        title="Place Name (gazetteer of Hong Kong place names)",
        publisher="Lands Department, HKSAR Government (via the CSDI Portal)",
        url=PLACE_NAME_POINTS_URL,
        accessed=ACCESS_DATE,
        resolution="2,706 points with 2,826 names in English and Chinese "
                   "(official names and aliases): areas, towns, villages, "
                   "hills, bays, capes and islands",
        licence="Free for commercial and non-commercial use; the Government "
                "and the CSDI Portal are to be named as the source "
                "(CSDI Portal terms of use)",
        role="The first layer of the route page's offline place search: the "
             "names of districts and neighbourhoods (Mong Kok / 旺角), which "
             "outrank everything else on an equal match.",
        local="data/raw/landsd_place_name_{points,names}.geojson",
        limitations="A point per place, not an outline. 'Central' is "
                    "recorded as 'Central District'. Reefs, channels and "
                    "other features of the sea are left out. Not used by "
                    "the analysis itself.",
        optional=True,
    ),
    Dataset(
        key="overture_base",
        title=f"Overture Maps base theme: land use, infrastructure, land (release {OVERTURE_RELEASE})",
        publisher="Overture Maps Foundation (derived from OpenStreetMap)",
        url=f"{OVERTURE_BUCKET}/{OVERTURE_BASE_PREFIX}/",
        accessed=ACCESS_DATE,
        resolution="Mapped outlines and points with names and OSM-derived classes",
        licence="ODbL 1.0 (OpenStreetMap contributors)",
        role="Mapped parks, housing estates, commercial complexes, schools, "
             "hospitals, bus termini, piers, bridges, viewpoints, peaks and "
             "beaches for the route page's offline search, each placed on "
             "its mapped outline.",
        local="data/raw/overture_{land_use,infrastructure,land}_hk.parquet",
        limitations="Only named features in a fixed class list are used. "
                    "OpenStreetMap writes both languages into one name "
                    "('維多利亞公園 Victoria Park'), which is split here. The "
                    "extract covers a bounding box that takes in Shenzhen "
                    "and is clipped to the SAR. The infrastructure and land "
                    "layers are not used by the analysis itself; the land "
                    "use is (overture_land_use above), and is not optional.",
        optional=True,
    ),
    Dataset(
        key="igeocom",
        title="GeoCommunity Database (iGeoCom) v5.0",
        publisher="Lands Department, Survey and Mapping Office, HKSAR "
                  "Government (Hong Kong Map Service open data)",
        url=IGEOCOM_URL,
        accessed=ACCESS_DATE,
        resolution="37,379 points of interest in 18 classes, 97% named in "
                   "both languages; HK1980 Grid coordinates to 1 m; updated "
                   "quarterly",
        licence="Free for commercial and non-commercial use with "
                "attribution of the Government as the source "
                "(DATA.GOV.HK terms of use)",
        role="MTR stations and their exits, Light Rail stops, piers, malls, "
             "parks, hospitals, libraries, museums, markets, hotels, schools "
             "and scenic spots (the Mid-Levels escalator, the Noonday Gun) "
             "for the route page's offline search. Station exits come from "
             "nowhere else.",
        local="data/raw/landsd_igeocom_csv.zip",
        limitations="Carries no general building or estate names. Station "
                    "names are verbose ('Mass Transit Railway Central "
                    "Station-A Access') and are rewritten to the way people "
                    "type them. Post boxes, substations, refuse points, "
                    "pavilions, car parks and other records with generic "
                    "names are dropped. The download host had two faults "
                    "when this was fetched: one of its two servers answered "
                    "with an error page, and its certificate chain lacks "
                    "the intermediate certificate, so Python cannot verify "
                    "it. download.py asks for the working server and, "
                    "rather than skip verification, completes the chain "
                    "with that intermediate, which is shipped in vendor/, "
                    "falling back to the system's curl; where neither "
                    "verifies the host this layer is left out, and download "
                    "says so when it ends. Not used by the analysis itself.",
        optional=True,
    ),
    Dataset(
        key="building_names",
        title="Building dataset, BuildingName layer",
        publisher="Lands Department, HKSAR Government (via the CSDI Portal)",
        url=BUILDING_NAME_URL,
        accessed=ACCESS_DATE,
        resolution="90,666 names of 68,185 buildings, 91% in both languages; "
                   "the position is carried in the building's identifier, "
                   "to 1 m on the HK1980 Grid",
        licence="Free for commercial and non-commercial use; the Government "
                "and the CSDI Portal are to be named as the source "
                "(CSDI Portal terms of use)",
        role="Building names for the route page's offline search: in Hong "
             "Kong people navigate by building and estate name far more "
             "than by street number.",
        local="data/raw/landsd_building_names.geojson",
        limitations="A table without geometry: each point is decoded from "
                    "the building's identifier and is its reference point, "
                    "not an entrance. Estates are listed block by block; the "
                    "blocks of one estate are thinned to a single record. "
                    "Chinese names use full-width digits and letters, which "
                    "are normalised. Not used by the analysis itself.",
        optional=True,
    ),
    Dataset(
        key="als_addresses",
        title="Addresses from the Address Lookup Service (bulk GeoJSON)",
        publisher="Digital Policy Office, HKSAR Government",
        url=ALS_URL,
        accessed=ACCESS_DATE,
        resolution="182,221 address points in 18 district files, with "
                   "street or village name, number, building and estate "
                   "name in English and Chinese; updated monthly",
        licence="Free for commercial and non-commercial use with "
                "attribution of the Government as the source "
                "(DATA.GOV.HK terms of use)",
        role="Offline street-address search in the route page ('123 Nathan "
             "Road', '彌敦道123號'), and the names of housing estates.",
        local="data/raw/als_addresses_geojson.zip",
        limitations="The publisher states that the list is not exhaustive. "
                    "One point per street (or village) and number is kept; "
                    "letter suffixes and flats are dropped. English names "
                    "arrive in upper case and are title-cased here. The "
                    "zip's 622 MB file of public rental housing flats is "
                    "not read. This is the source of Overture's Hong Kong "
                    "addresses, which carry the English half only. Not used "
                    "by the analysis itself.",
        optional=True,
    ),
    Dataset(
        key="td_stops",
        title="Routes and fares of public transport: tram, Peak Tram and "
              "ferry stops",
        publisher="Transport Department, HKSAR Government (DATA.GOV.HK)",
        url=TD_STOPS_URLS["tram"],
        accessed=ACCESS_DATE,
        resolution="117 tram stops, 6 Peak Tram stations and 68 ferry piers, "
                   "named in English and Chinese, WGS84",
        licence="Free for commercial and non-commercial use with "
                "attribution of the Government as the source "
                "(DATA.GOV.HK terms of use)",
        role="Tram stops and ferry piers for the route page's offline search.",
        local="data/raw/td_stops_{tram,peak_tram,ferry}.json",
        limitations="One record per stop of every route, deduplicated here. "
                    "Tram stops are named after the street they stand on, "
                    "once for each direction of travel; the pair is indexed "
                    "as one stop ('Pedder Street Tram Stop'). Bus and "
                    "minibus stops are deliberately not indexed: there are "
                    "over ten thousand, with near-identical names. Not used "
                    "by the analysis itself.",
        optional=True,
    ),
    Dataset(
        key="overture_places",
        title=f"Overture Maps places (release {OVERTURE_RELEASE})",
        publisher="Overture Maps Foundation (Meta, Microsoft, Foursquare "
                  "and AllThePlaces POI data)",
        url=f"{OVERTURE_BUCKET}/{OVERTURE_PLACES_PREFIX}/type=place/",
        accessed=ACCESS_DATE,
        resolution="Point features with a name, a category and a confidence score",
        licence="CDLA Permissive 2.0 (Meta, Microsoft), Apache 2.0 "
                "(Foursquare), CC0 1.0 (AllThePlaces), per record",
        role="The last layer of the route page's offline search: shops, "
             "restaurants, cafes and hotels, which no government list carries.",
        local="data/raw/overture_places_hk.parquet",
        limitations="Names are in one language only (about half English, a "
                    "third Chinese) and the same name recurs at several "
                    "points, some of them far from the real thing. Only "
                    "records with confidence >= 0.9 in everyday categories "
                    "are kept, never a chain (a name at three or more "
                    "sites) and never where a mapped or government record "
                    "carries the name. Not used by the analysis itself.",
        optional=True,
    ),
)

DATASETS_BY_KEY = {d.key: d for d in DATASETS}


def format_table() -> str:
    """Human-readable provenance report."""
    lines = [f"Data sources (all URLs verified {ACCESS_DATE})", "=" * 78]
    for d in DATASETS:
        flag = " [OPTIONAL]" if d.optional else ""
        flag += " [SUBSTITUTED]" if d.substituted else ""
        lines += [
            f"\n{d.key}{flag}",
            f"  title       : {d.title}",
            f"  publisher   : {d.publisher}",
            f"  url         : {d.url}",
            f"  accessed    : {d.accessed}",
            f"  resolution  : {d.resolution}",
            f"  licence     : {d.licence}",
            f"  local cache : {d.local}",
            f"  role        : {d.role}",
            f"  limitations : {d.limitations}",
        ]
        if d.notes:
            lines.append(f"  notes       : {d.notes}")
        if d.substitution_reason:
            lines.append(f"  substitution: {d.substitution_reason}")
    return "\n".join(lines)


def markdown_table() -> str:
    """Compact markdown table for the README."""
    rows = ["| Dataset | Publisher | Resolution / vintage | Licence | Role |",
            "|---|---|---|---|---|"]
    for d in DATASETS:
        rows.append(
            f"| {d.title} | {d.publisher} | {d.resolution} | {d.licence} | {d.role} |"
        )
    return "\n".join(rows)
