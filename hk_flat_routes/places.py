"""Offline place index for the route page: places, addresses, and the base.

A static page cannot call a geocoding API without carrying a billed key in
public, and should not need one for a single city anyway, so the index is
built here and shipped with the page.

Upstream built San Francisco's index from two Overture themes.  Hong Kong
needs other sources, for three reasons:

* People there find their way by **name** -- of the area, the MTR station
  and its exit, the housing estate, the building -- far more than by street
  number, and Overture's place feed carries those names badly: in one
  language only, and with the same name at several points.
* Every name comes in **two languages** and either may be typed, so each
  record carries both: ``names`` holds the English name where the place has
  one, and ``alt`` the Chinese.
* The bounding box that holds Hong Kong also holds much of **Shenzhen**, so
  every layer is clipped to the SAR itself.

The layers, in the order in which they win a name:

1. **Place names** from the Lands Department's gazetteer: areas, towns,
   villages, hills, bays and islands ("Mong Kok", "旺角").
2. **The railway** from the Lands Department's iGeoCom: every MTR station
   and each of its exits, and the Light Rail stops, renamed from the
   official form to the one people type ("Central Station Exit A").
3. **Mapped features** from Overture's base theme, which is OpenStreetMap
   data: parks, housing estates, commercial complexes, schools, hospitals,
   piers, peaks and beaches, each placed on its mapped outline.
4. **Points of interest**, the rest of iGeoCom: malls, markets, libraries,
   museums, hospitals, schools, places of worship, scenic spots.
5. **Estates** from the Address Lookup Service and **buildings** from the
   Lands Department's building names, thinned so that the forty blocks of
   one estate make one record.
6. **Tram stops and ferry piers** from the Transport Department.
7. **Shops, restaurants and hotels** from Overture's places theme, last and
   only where no layer above carries the name: that feed is noisy.

A name already present within a few hundred metres is not added again, so
a park that three sources know appears once.  Every source is optional:
the index is built from whichever of them were downloaded.  Two more kinds
of result need nothing from this list:

* **Street junctions** are derived in the browser from the graph itself
  ("Nathan Road & Jordan Road"); nothing to pack.
* **Addresses** come from the Address Lookup Service, one point per street
  (or village) and number, with the street table in both languages.

The hillshade base is rendered from the same lidar DEM the analysis uses,
then palette-quantised: it is a quiet grey image and does not need 24-bit
colour.
"""
from __future__ import annotations

import base64
import functools
import io
import json
import re
import unicodedata
import zipfile

import numpy as np
import pandas as pd

from .config import CRS_PROJECTED, PROCESSED_DIR, STUDY_BBOX
from .download import (ALS_ZIP, BASE_PARQUETS, BUILDING_NAMES_GEOJSON,
                       DIVISIONS_PARQUET, IGEOCOM_ZIP, PLACE_NAME_NAMES_GEOJSON,
                       PLACE_NAME_POINTS_GEOJSON, PLACES_PARQUET, TD_STOPS_JSON)
from .network import split_name
from .utils import get_logger, step

log = get_logger("hk_flat_routes.places")

HILLSHADE_PNG = PROCESSED_DIR / "hillshade_light.png"
#: Width of the hillshade before cropping to the land: the territory is
#: 65 km across, so this is about 18 m a pixel -- coarser than San
#: Francisco's 10 m, and as large as a phone will comfortably decode.
HILLSHADE_WIDTH_PX = 3600
#: Vertical exaggeration of the relief, and the height (m) at which the
#: tint is darkest.  Hong Kong's hills are several times San Francisco's,
#: and upstream's values (1.8 and 260 m) turn them black.
HILLSHADE_Z_FACTOR = 0.9
HILLSHADE_TINT_M = 700.0

#: The raw file behind ``build_addresses``: the Address Lookup Service zip.
#: The route page packs addresses only when it exists.
ADDRESSES_PATH = ALS_ZIP

# --------------------------------------------------------------------------
# kinds
# --------------------------------------------------------------------------
#: Every kind a record can carry (the label shown beside a suggestion), with
#: its rank on an equal text match.  0 is what a bare name most often means:
#: typing "Central" should offer the area and the station before the forty
#: buildings named after them.  1 is the rest of what a map names -- transit
#: stops, natural features, parks, estates and the larger institutions; 2 is
#: single buildings and neighbourhood facilities; 3 is everyday shops and
#: food, which are many and share their names with everything around them.
KIND_RANK = {
    "area": 0, "station": 0,
    "station exit": 1, "light rail stop": 1, "tram stop": 1, "peak tram": 1,
    "cable car": 1, "pier": 1, "bus terminus": 1, "airport": 1,
    "control point": 1,
    "village": 1, "island": 1, "peak": 1, "pass": 1, "bay": 1, "cape": 1,
    "beach": 1, "river": 1, "valley": 1, "cave": 1, "rock": 1, "peninsula": 1,
    "park": 1, "country park": 1, "estate": 1, "complex": 1, "mall": 1,
    "market": 1, "hospital": 1, "university": 1, "museum": 1, "landmark": 1,
    "stadium": 1, "bridge": 1, "viewpoint": 1,
    "building": 2, "school": 2, "college": 2, "clinic": 2, "library": 2,
    "garden": 2, "playground": 2, "plaza": 2, "sports centre": 2,
    "swimming pool": 2, "golf course": 2, "marina": 2,
    "church": 2, "temple": 2, "mosque": 2, "synagogue": 2, "cemetery": 2,
    "police station": 2, "fire station": 2, "post office": 2, "court": 2,
    "consulate": 2, "community centre": 2, "theatre": 2, "hotel": 2,
    "food": 3, "shop": 3,
}

#: Lands Department gazetteer PLACE_TYPE -> kind.  Reefs, straits, channels
#: and harbours are left out: their points lie in open water, and nobody
#: walks to a strait.
GAZETTEER_KINDS = {
    "Area": "area", "Town": "area", "Village": "village", "Hill": "peak",
    "Island": "island", "Islands": "island", "Bay": "bay", "Cape": "cape",
    "Pass": "pass", "Peninsula": "peninsula", "Rock": "rock",
    "Valley": "valley", "Cave": "cave", "River": "river", "Stream": "river",
}
#: Gazetteer names rewritten to the way they are said.  The gazetteer files
#: the Central neighbourhood as "Central District" (research_places.txt);
#: on the street, on the MTR map and in an address it is "Central".
GAZETTEER_RENAMES = {"Central District": "Central"}

#: Overture base-theme (OpenStreetMap) classes kept, with the kind shown in
#: the search list. Keyed by (type, class).  The first block is upstream's
#: list, less its military land and its observation and communication
#: towers (here those are barracks, fire lookouts and hilltop transmitters,
#: not Sutro Tower); the second is what Hong Kong adds.
BASE_CLASSES = {
    ("land_use", "park"): "park", ("land_use", "dog_park"): "park",
    ("land_use", "playground"): "playground", ("land_use", "garden"): "garden",
    ("land_use", "allotments"): "garden", ("land_use", "plaza"): "plaza",
    ("land_use", "pedestrian"): "plaza", ("land_use", "school"): "school",
    ("land_use", "kindergarten"): "school", ("land_use", "university"): "university",
    ("land_use", "college"): "college", ("land_use", "hospital"): "hospital",
    ("land_use", "golf_course"): "golf course", ("land_use", "stadium"): "stadium",
    ("land_use", "marina"): "marina", ("land_use", "recreation_ground"): "park",
    ("infrastructure", "railway_station"): "station",
    ("infrastructure", "subway_station"): "station",
    ("infrastructure", "ferry_terminal"): "pier", ("infrastructure", "pier"): "pier",
    ("infrastructure", "bridge"): "bridge", ("infrastructure", "viewpoint"): "viewpoint",
    ("land", "peak"): "peak", ("land", "hill"): "peak", ("land", "beach"): "beach",
    ("land", "island"): "island", ("land", "islet"): "island",
    # Hong Kong.  A named residential outline is a housing estate or a
    # private development, which is how most people there name where they
    # live; a named commercial outline is a complex such as the
    # International Finance Centre or Times Square.  The country parks are
    # mapped as national parks.  Bus termini are destinations in their own
    # right.  A saddle is a named pass on a hill path.  (Cable car stations
    # are left to iGeoCom: OpenStreetMap calls Ocean Park's "Summit" and
    # "Waterfront".)
    ("land_use", "residential"): "estate", ("land_use", "commercial"): "complex",
    ("land_use", "retail"): "mall", ("land_use", "cemetery"): "cemetery",
    ("land_use", "national_park"): "country park",
    ("land_use", "protected_landscape_seascape"): "country park",
    ("land_use", "theme_park"): "landmark", ("land_use", "zoo"): "landmark",
    ("land_use", "resort"): "landmark",            # Ocean Park is mapped as one
    ("infrastructure", "viaduct"): "bridge",
    ("infrastructure", "bus_station"): "bus terminus",
    ("land", "saddle"): "pass",
}
#: Kinds named by geography.  Several hills really are called Kwun Yam Shan
#: and several passes Pak Kung Au, so a name shared by many of these is not
#: the generic label it would be on a shop or a playground.
_NATURAL_KINDS = frozenset(("area", "village", "peak", "pass", "island", "bay",
                            "cape", "beach", "river", "valley", "cave", "rock",
                            "peninsula"))
#: A bridge in OpenStreetMap usually carries the name of the road on it
#: ("沙宣道 Sassoon Road"), which would put a street name in the list as if
#: it were a place.  Only bridges named as bridges are kept.
_BRIDGE_RE = re.compile(r"bridge|flyover|viaduct|橋", re.I)
#: The same for a mapped pier.  OpenStreetMap piers without such a word are
#: either harbour works ("Mooring Dolphin 3", "Ramp") or carry the bare name
#: of the place they serve, so that "Hong Kong" would be a pier.
_PIER_RE = re.compile(r"pier|ferry|terminal|jetty|landing|wharf|quay|dock|"
                      r"碼頭|渡輪|渡頭|埗頭", re.I)

#: iGeoCom TYPE -> kind, for the types kept.  Everything else is dropped:
#: post boxes, substations, refuse points, pavilions, toilets, car parks,
#: beacons, convenience stores, kindergartens, guest houses and sports
#: pitches either carry generic names ("Pavilion", "Football Field") or
#: are too many and too small to be where a walk begins.  Peak Tram stops
#: come from the Transport Department's list instead, and minibus termini
#: are left out because their names describe a kerb ("Pai Tau Street,
#: outside Sha Tin Station").  The scenic spots (SIG) are kept: about a
#: hundred places a visitor asks for by name, most of them in no other
#: layer -- the Central to Mid-Levels Escalator, the Blue House and the
#: Noonday Gun among them.
IGEOCOM_KINDS = {
    # transit
    "RSN": "station", "HSN": "station", "MTA": "station exit",
    "HSA": "station exit", "LRA": "light rail stop", "FER": "pier",
    "FET": "pier", "CBC": "cable car", "BUS": "bus terminus", "AIR": "airport",
    "ICP": "control point",
    # open space
    "PAR": "park", "RGD": "garden", "PLG": "playground", "CPB": "country park",
    "CSA": "country park", "SPA": "country park", "BBS": "beach",
    "VPT": "viewpoint", "STD": "stadium", "ZOO": "landmark", "AMK": "landmark",
    "RCO": "landmark", "SIG": "landmark", "GCO": "golf course",
    "SPL": "swimming pool", "IGH": "sports centre",
    # civic
    "HOS": "hospital", "CLI": "clinic", "LIB": "library", "EXB": "museum",
    "MKT": "market", "CFS": "market", "CHL": "landmark", "PFM": "theatre",
    "DMT": "landmark", "POF": "post office", "PSN": "police station",
    "FSN": "fire station", "JUD": "court", "CST": "consulate",
    "CMC": "community centre", "CEM": "cemetery",
    # worship
    "CHU": "church", "TMP": "temple", "MON": "temple", "MOS": "mosque",
    "SYN": "synagogue",
    # schools
    "PRS": "school", "SES": "school", "SEC": "school", "TEI": "college",
    "SCM": "university", "VTI": "college",
    # commerce
    "MAL": "mall", "HTL": "hotel",
}

#: Transport Department stop files -> kind.
STOP_KINDS = {"tram": "tram stop", "peak_tram": "peak tram", "ferry": "pier"}

#: Overture places kept, by the kind shown.  The names on the right are
#: nodes of the feed's taxonomy; a record takes the kind of the deepest
#: node of its hierarchy that is listed, so a shopping mall is a mall, not
#: a shop.  Upstream's list is not used: it named ``categories.primary``
#: values, a column this release no longer has, and it took parks,
#: stations, schools and hospitals from the feed, which in Hong Kong all
#: come from better sources (the feed files over a thousand "hospitals"
#: there).
#: Two nodes are left out for what a sample of them held: plain "lodging"
#: (cafes and villages) and "stadium_arena" (five-a-side pitches).
CATEGORY_GROUPS = {
    "food": ["food_and_drink"],
    "shop": ["shopping"],
    "mall": ["shopping_mall"],
    "hotel": ["hotel", "hostel", "resort", "service_apartment", "inn",
              "bed_and_breakfast", "guest_house"],
    "landmark": ["museum", "art_museum", "history_museum", "science_museum",
                 "childrens_museum", "contemporary_art_museum", "aquarium",
                 "zoo", "amusement_park", "movie_theater", "theatre_venue",
                 "performing_arts_venue", "art_gallery", "observatory",
                 "cultural_center"],
}
_GROUP_OF = {c: g for g, cs in CATEGORY_GROUPS.items() for c in cs}
#: Overture's confidence that a place exists.  Upstream asked for 0.6.  The
#: feed has 145,000 records in Hong Kong's bounding box and its names are
#: the least reliable in the index, so only the records it is surest of are
#: taken: about 12,000 survive.  At 0.8 the index gains some 7,000 records
#: (a tenth of its packed size) and a sample of them was visibly rougher.
POI_MIN_CONFIDENCE = 0.9
#: A name found at this many separate sites is a chain ("7-Eleven" is at
#: nearly 400).  The list can show one record per name, and one branch
#: chosen at random is worse than none.
POI_CHAIN_SITES = 3

#: Two records share a place when they share a name within this distance
#: (metres), by the kind of the newcomer.  Natural features and areas are
#: labelled wherever the cartographer found room, so they get more.
_SAME_PLACE_M = {"area": 1500.0, "peak": 1500.0, "island": 3000.0,
                 "bay": 1500.0, "cape": 1500.0, "beach": 800.0,
                 "village": 800.0, "country park": 5000.0, "park": 600.0,
                 "estate": 600.0, "station": 600.0, "university": 2000.0,
                 "hospital": 1000.0, "landmark": 2000.0}
_SAME_PLACE_DEFAULT_M = 300.0
#: Kinds whose name belongs to one institution, however many buildings and
#: plots it is spread over (see ``_merge_layers``), and how far apart those
#: can lie.  Parks are not among them: "Central Park" is a park in one
#: district and a housing estate in another.
_COMPOUND_KINDS = frozenset(("university", "college", "hospital", "landmark",
                             "stadium", "museum", "airport"))
_COMPOUND_REACH_M = 10_000.0

#: Short forms that everybody uses and no source lists, as (full English
#: name, short form).  Each becomes a record of its own at the place of the
#: full name, and only if the full name is in the index: nothing here
#: supplies a position.  Deliberately few -- the ones the place research
#: found wanting (research_places.txt: IFC, TST, 維園) and HKU -- because
#: every entry is a claim that no dataset backs.
ALIASES = (
    ("International Finance Centre", "IFC"),
    ("The University of Hong Kong", "HKU"),
    ("Tsim Sha Tsui", "TST"),
    ("Victoria Park", "維園"),
)


# --------------------------------------------------------------------------
# names
# --------------------------------------------------------------------------
_CJK = r"\u2e80-\u9fff\uf900-\ufaff\U00020000-\U0002ffff"
_CJK_RE = re.compile(f"[{_CJK}]")
_LATIN_RE = re.compile(r"[A-Za-z]")
#: Characters Hong Kong writes in two ways, folded when names are compared
#: (never when they are shown): 邨/村, 臺/台, 滙/匯, 峯/峰, 裏/里, 綫/線.
_VARIANTS = str.maketrans("邨臺滙峯裏綫", "村台匯峰里線")


def _clean(value) -> str:
    """A name as it will be shown, or '' for none.

    Unicode NFKC folds the full-width digits, letters and brackets that the
    government's Chinese names are typed with ("重慶大廈Ａ座", "第６座") into
    the plain ones a visitor will type.
    """
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()


def _key(name: str) -> str:
    """The form in which two names are compared: letters and digits only.

    'Queen's Road Central' and 'QUEENS ROAD CENTRAL' are one name, and so
    are 美孚新邨 and 美孚新村.
    """
    s = unicodedata.normalize("NFKC", name).casefold().translate(_VARIANTS)
    return "".join(ch for ch in s if ch.isalnum())


def _pair(en, zh) -> tuple[str, str]:
    """Clean a record's two names and put each in its own language's slot.

    The sources do not always: iGeoCom repeats 'KK HOTEL' as the Chinese
    name of a hotel that has none, and files some Chinese-only names under
    English.
    """
    en, zh = _clean(en), _clean(zh)
    if zh and not _CJK_RE.search(zh):
        en, zh = en or zh, ""
    if en and _CJK_RE.search(en) and not _LATIN_RE.search(en):
        en, zh = "", zh or en
    if _key(en) == _key(zh):
        en, zh = (("", zh) if _CJK_RE.search(zh) else (en, ""))
    return en, zh


#: Words left in capitals when an upper-case name is title-cased.
_ACRONYMS = frozenset((
    "HK HKU HKUST MTR KCR KCRC YMCA YWCA HSBC IFC ICC AIA CLP PCCW TVB HKT "
    "UA DBS NWS KMB CCC SKH PLK HKFYG ELCHK TWGHS YHA SPCA RTHK HKSKH HKBU "
    "CUHK HKMA ICAC CITIC BOC NT TST II III IV VI VII VIII IX XI XII").split())
#: Words written small inside a name ("Bank of China Tower").  Not "on",
#: "to" or "in": in Hong Kong those are syllables (Tai On Building, To Kwa
#: Wan), far more often than they are English.
_SMALL_WORDS = frozenset(("of", "the", "and"))
_WORD_RE = re.compile(r"[0-9A-Za-z]+(?:['’][A-Za-z]+)*")


def _title_en(raw: str) -> str:
    """'QUEEN'S ROAD CENTRAL' -> 'Queen's Road Central'.

    The Address Lookup Service and the Rating and Valuation Department
    write English names in capitals.  Python's own ``title`` and
    ``capitalize`` get Hong Kong's street names wrong at the apostrophe
    ("Queen'S", "D'aguilar"): after one, an ending stays small ("Queen's",
    "Hiram's") and a name after a single letter keeps its capital
    ("D'Aguilar", "O'Brien").  Ordinals stay small ("1st"), single letters
    and a few acronyms stay capital ("Block A", "MTR"), and a name that
    already has lower-case letters is left exactly as its publisher wrote
    it.
    """
    s = str(raw)
    if any(ch.islower() for ch in s):
        return s

    def word(m: re.Match) -> str:
        tok = m.group(0)
        if tok[0].isdigit():                       # '1ST' -> '1st'; '23A' stays
            return tok.lower() if re.fullmatch(r"\d+(?:ST|ND|RD|TH)", tok) else tok
        if any(ch.isdigit() for ch in tok) or len(tok) == 1 or tok in _ACRONYMS:
            return tok                             # 'K11', 'A', 'MTR'
        if tok.lower() in _SMALL_WORDS and m.start() and s[m.start() - 1] == " ":
            return tok.lower()
        parts = re.split(r"(['’])", tok)
        out = [parts[0].capitalize()]
        for i in range(2, len(parts), 2):
            follows_initial = len(parts[i - 2]) == 1
            out += [parts[i - 1],
                    parts[i].capitalize() if follows_initial else parts[i].lower()]
        return "".join(out)

    return _WORD_RE.sub(word, s)


_MTR_EN_RE = re.compile(r"^(?:Mass Transit Railway|MTR)\s+", re.I)
_MTR_ZH_RE = re.compile(r"^(?:香港鐵路|港鐵)")
_EXIT_EN_RE = re.compile(r"\s*[-–]\s*([A-Za-z]?\d*[A-Za-z]?)\s*(?:Access|Exit|Entrance)$", re.I)
_EXIT_ZH_RE = re.compile(r"\s*[-–]\s*([A-Za-z]?\d*[A-Za-z]?)\s*(?:進出口|出入口|出口|入口)$")
_LR_EN_RE = re.compile(r"^LR\s*[-–]\s*", re.I)
_LR_ZH_RE = re.compile(r"^輕鐵\s*[-–—－]\s*")


def _station_names(en: str, zh: str, kind: str) -> tuple[str, str]:
    """A railway record's names, the way people type them.

    iGeoCom writes 'Mass Transit Railway Central Station' / '香港鐵路中環站',
    an exit as '...Central Station-A Access' / '...中環站-A進出口' and a
    Light Rail stop as 'LR-Fung Tei' / '輕鐵－鳳地'.  Nobody types those:
    this gives 'Central Station' / '中環站', 'Central Station Exit A' /
    '中環站A出口' and 'Fung Tei Light Rail Stop' / '鳳地輕鐵站'.

    OpenStreetMap names a station by its bare name ('Central' / '中環'),
    which is also the name of the area around it, so a station from there
    gains the word: 'Central Station' / '中環站'.
    """
    en, zh = _clean(en), _clean(zh)
    if kind == "light rail stop":
        en, zh = _LR_EN_RE.sub("", en), _LR_ZH_RE.sub("", zh)
        if en and not re.search(r"\b(?:stop|station)$", en, re.I):
            en += " Light Rail Stop"
        if zh and not zh.endswith("站"):
            zh += "輕鐵站"
        return en, zh
    en, zh = _MTR_EN_RE.sub("", en), _MTR_ZH_RE.sub("", zh)
    if kind == "station exit":
        m = _EXIT_EN_RE.search(en)
        if m:
            en = f"{en[:m.start()]} Exit {m.group(1).upper()}".strip()
        m = _EXIT_ZH_RE.search(zh)
        if m:
            zh = f"{zh[:m.start()]}{m.group(1).upper()}出口"
        return en, zh
    if en and not re.search(r"\b(?:station|terminus|terminal)$", en, re.I):
        en += " Station"
    if zh and not zh.endswith(("站", "總站")):
        zh += "站"
    return en, zh


def _terminus_names(en: str, zh: str) -> tuple[str, str]:
    """'Wong Tai Sin' / '黃大仙' -> 'Wong Tai Sin Bus Terminus' / '黃大仙巴士總站'.

    Both sources name most bus termini after the place they serve and
    nothing else, which is the name of an area, an estate, a street or --
    worst -- the railway station next door ('Tsing Yi Station'), which
    would then pass for the station itself.
    """
    if en and not re.search(r"bus|terminus|interchange", en, re.I):
        en += " Bus Terminus"
    if zh and not re.search(r"巴士|總站|交匯處|轉車站", zh):
        zh += "巴士總站"
    return en, zh


_BLOCK_ID = (r"(?:[a-z]{0,2}\d{1,3}[a-z]?|[a-z]|one|two|three|four|five|six|"
             r"seven|eight|nine|ten|[ivx]{1,4})")
#: 'Block 42', '(Blk 6)', 'Tower One', 'Phase II', 'House Number 15' at the
#: end of a name: a block that is spelt out.
_BLOCK_WORD_EN = (r"[\s,\-]*\(?\b(?:block|blk|tower|house|hse|phase|stage)\.?\s*"
                  r"(?:(?:no\.?|number)\s*)?" + _BLOCK_ID + r"\)?\s*$")
#: The same in Chinese: '第二十四座', 'A座', '(第6座)', '二期', '32號屋'.
_BLOCK_WORD_ZH = (r"[\s(]*第?[0-9A-Za-z一二三四五六七八九十廿]+"
                  r"(?:座|幢|期|號屋|號洋房|號別墅)\)?\s*$")
_WORD_EN_RE = re.compile(_BLOCK_WORD_EN, re.I)
_WORD_ZH_RE = re.compile(_BLOCK_WORD_ZH)
#: A spelt-out block, or a bare house code at the end ('Regalia Bay C10' /
#: '富豪海灣C10'), which is how the houses of a low-rise development are
#: listed.
_BLOCK_EN_RE = re.compile(_BLOCK_WORD_EN + r"|\s+[a-z]{0,2}\d{1,3}[a-z]?$", re.I)
_BLOCK_ZH_RE = re.compile(_BLOCK_WORD_ZH + r"|[A-Za-z]{0,2}\d{1,3}[A-Za-z]?$")
#: A bare code is taken for a house number only when this many names end
#: in one after the same stem (see ``_fold_blocks``).
_MIN_HOUSES = 3


def _strip_block(name: str, pattern: re.Pattern) -> str:
    """Take block, tower and phase designators off the end of a name.

    What joined two of them goes as well: 'Block B1 & B2' and 'B1-B9座'
    would otherwise leave 'Block B1 &' and 'B1-' behind.
    """
    for _ in range(3):                             # 'Phase 1 Block 10'
        m = pattern.search(name)
        head = name[:m.start()].rstrip(" \t,-–&及").strip() if m else ""
        if len(head) < 2:
            break
        name = head
    return name


def _estate_stem(en: str, zh: str, codes: bool = True) -> tuple[str, str]:
    """('Marina Cove Block 42', '匡湖居42座') -> ('Marina Cove', '匡湖居').

    The building list names every block of an estate, and one estate can
    have hundreds: Marina Cove has 465 rows.  The block is dropped only
    when both names carry one, so that 'One International Finance Centre'
    / '國際金融中心一期', where only the Chinese does, stays whole.  With
    ``codes`` off, only a block that is spelt out is dropped and a bare
    code ('C10') is left on the name.
    """
    pattern_en, pattern_zh = ((_BLOCK_EN_RE, _BLOCK_ZH_RE) if codes
                              else (_WORD_EN_RE, _WORD_ZH_RE))
    e, z = _strip_block(en, pattern_en), _strip_block(zh, pattern_zh)
    if en and zh and (e != en) != (z != zh):
        return en, zh
    return e, z


def _fold_blocks(names: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Fold every (English, Chinese) building name of a list into its estate.

    A block that is spelt out ('Block 42', '第6座') is dropped wherever it
    stands.  A bare code is another matter.  'Regalia Bay C10' is house C10
    of Regalia Bay, but 'Central 88', 'Stanley 11' and 'Soho 38' are whole
    names -- new buildings in Hong Kong are often named after their street
    and number -- and without the number they would be listed as a
    building called Central, Stanley or Soho.  Two things tell them apart.
    Where the other language spells the block out ('Harbour View 1' /
    '海濱大樓一座'), the code is that block.  Where neither does, the list
    itself decides: a development lists its houses one by one, so the code
    is dropped only where at least ``_MIN_HOUSES`` different names end in
    one after the same stem.
    """
    def key(pair: tuple[str, str]) -> str:
        return _key(pair[0]) or "\0" + _key(pair[1])

    def spelt(pair: tuple[str, str]) -> bool:
        return (_strip_block(pair[0], _WORD_EN_RE) != pair[0]
                or _strip_block(pair[1], _WORD_ZH_RE) != pair[1])

    whole = [_estate_stem(en, zh, codes=False) for en, zh in names]
    cut = [_estate_stem(en, zh) for en, zh in names]
    # names the bare-code rule alone would shorten, by the stem it leaves
    houses: dict[str, set[str]] = {}
    for name, w, c in zip(names, whole, cut):
        if c != w and not spelt(name):
            houses.setdefault(key(c), set()).add(key(name))
    return [c if c == w or spelt(name) or len(houses[key(c)]) >= _MIN_HOUSES else w
            for name, w, c in zip(names, whole, cut)]


#: Building names that are plant, not places: the list records every
#: substation, pump house and public toilet that has a roof.  The last few
#: patterns are the Rating and Valuation Department's notes on a tenement
#: ("(Dwelling Portion)", "Part"), which are filed as names too.
_PLANT_EN_RE = re.compile(
    r"sub-?station|transformer|pump(?:ing)? ?(?:house|station|room)|toilet|"
    r"latrine|refuse|car ?park|guard ?(?:house|room)|switch ?(?:room|house)|"
    r"meter room|plant room|store ?room|generator|sewage|water tank|"
    r"aqua privy|bathhouse|bus terminus|^rcp$|^\(|\bportion\b|^part$", re.I)
_PLANT_ZH_RE = re.compile(r"變電站|電力站|泵房|抽水站|公廁|廁所|垃圾|停車場|"
                          r"警衛室|污水|機房|電錶房|儲物室")


# --------------------------------------------------------------------------
# positions
# --------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _grid_transformer():
    from pyproj import Transformer
    return Transformer.from_crs("EPSG:4326", CRS_PROJECTED, always_xy=True)


def _to_grid(lon, lat) -> tuple[np.ndarray, np.ndarray]:
    """Longitude and latitude to HK1980 Grid metres."""
    x, y = _grid_transformer().transform(np.asarray(lon, float), np.asarray(lat, float))
    return np.asarray(x, float), np.asarray(y, float)


def _inside(boundary, x, y) -> np.ndarray:
    """Which grid positions lie inside ``boundary`` (a polygon on the grid).

    The index is clipped to the SAR polygon and not to the study bounding
    box, because the box takes in Shenzhen north of the river: 5,300 of the
    145,000 Overture places in it are across the border.
    """
    import shapely
    shapely.prepare(boundary)
    return shapely.contains_xy(boundary, np.asarray(x, float), np.asarray(y, float))


def _sar_boundary():
    """The SAR polygon everything is clipped to (``FileNotFoundError`` if absent)."""
    if not DIVISIONS_PARQUET.exists():
        raise FileNotFoundError(
            f"{DIVISIONS_PARQUET.name} is missing: the place index is clipped "
            "to the SAR boundary (run `python -m hk_flat_routes download`)")
    from .neighborhoods import city_boundary
    return city_boundary()


_LAYER_COLUMNS = ["en", "zh", "group", "lon", "lat", "x", "y"]


def _layer(rows, boundary) -> pd.DataFrame:
    """A layer's records with clean names, on the grid, inside the SAR.

    ``rows`` has columns en, zh, group, lon, lat (any others are kept).
    """
    df = pd.DataFrame(rows).reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(columns=_LAYER_COLUMNS)
    names = [_pair(e, z) for e, z in zip(df["en"], df["zh"])]
    df["en"] = [n[0] for n in names]
    df["zh"] = [n[1] for n in names]
    lon = pd.to_numeric(df["lon"], errors="coerce").to_numpy(float)
    lat = pd.to_numeric(df["lat"], errors="coerce").to_numpy(float)
    ok = np.isfinite(lon) & np.isfinite(lat)
    x = np.full(len(df), np.nan); y = np.full(len(df), np.nan)
    x[ok], y[ok] = _to_grid(lon[ok], lat[ok])
    df["lon"], df["lat"], df["x"], df["y"] = lon, lat, x, y
    keep = ok & ((df["en"] != "") | (df["zh"] != "")).to_numpy()
    keep[keep] = _inside(boundary, x[keep], y[keep])
    return df[keep].reset_index(drop=True)


def _thin(df: pd.DataFrame, radius_m: float, pick: str = "central") -> pd.DataFrame:
    """One record for every cluster of records that share a name.

    Records are one cluster when they carry the same name and each is
    within ``radius_m`` of another: the blocks of an estate, the pieces a
    park is mapped in, the two carriageways of a bridge.  ``pick`` chooses
    which record stands for the cluster: "central", the one nearest the
    middle (so that an estate is a point on one of its own blocks), or
    "first", the earliest in the frame's order (so that a caller can sort
    by area and keep the largest outline).  The survivor takes a missing
    other-language name from the rest of its cluster.
    """
    from scipy.spatial import cKDTree

    if len(df) < 2:
        return df.reset_index(drop=True)
    df = df.reset_index(drop=True)
    en = df["en"].tolist(); zh = df["zh"].tolist()
    xy = df[["x", "y"]].to_numpy(float)
    groups: dict[str, list[int]] = {}
    for i, (e, z) in enumerate(zip(en, zh)):
        groups.setdefault(_key(e) or "\0" + _key(z), []).append(i)
    keep: list[int] = []
    for members in groups.values():
        if len(members) == 1:
            keep.append(members[0])
            continue
        pts = xy[members]
        parent = list(range(len(members)))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for a, b in cKDTree(pts).query_pairs(radius_m):
            parent[find(a)] = find(b)
        clusters: dict[int, list[int]] = {}
        for k in range(len(members)):
            clusters.setdefault(find(k), []).append(k)
        for ks in clusters.values():
            if pick == "first":
                best = min(ks)
            else:
                mid = np.median(pts[ks], axis=0)
                best = ks[int(np.argmin(((pts[ks] - mid) ** 2).sum(axis=1)))]
            i = members[best]
            for col in (en, zh):
                if not col[i]:
                    col[i] = next((col[members[k]] for k in ks if col[members[k]]), "")
            keep.append(i)
    keep.sort()
    out = df.iloc[keep].copy()
    out["en"] = [en[i] for i in keep]
    out["zh"] = [zh[i] for i in keep]
    return out.reset_index(drop=True)


def _drop_generic(df: pd.DataFrame, at_least: int = 4) -> pd.DataFrame:
    """Drop names that many places share and that therefore name none.

    iGeoCom has 208 piers called 'Jetty' and 181 playgrounds called
    'Playground'.  Call it on a thinned frame, so that the count is of
    separate sites and not of the rows one place is listed under.
    """
    keys = [_key(e) or "\0" + _key(z) for e, z in zip(df["en"], df["zh"])]
    count = pd.Series(keys).value_counts()
    generic = set(count[count >= at_least].index)
    return df[[k not in generic for k in keys]].reset_index(drop=True)


def _merge_layers(layers: list[pd.DataFrame]) -> pd.DataFrame:
    """Stack layers in priority order, dropping names already taken nearby.

    A record is dropped when an earlier one carries the same English or the
    same Chinese name within the distance its kind allows
    (``_SAME_PLACE_M``): the earlier layer is the better source.  The
    record that stays takes the other-language name from the one dropped
    when it had none, which is how an English-only OpenStreetMap name
    gains the government's Chinese one.

    Two refinements, both learnt from the data:

    * The same Chinese name under a *different* English one is kept.
      OpenStreetMap has 重慶大廈 as "Chungking Mansion" and the Lands
      Department as "Chungking Mansions"; whoever types the one must not
      be told there is no such place because the index chose the other.
    * An estate or building named exactly like a campus, a hospital or a
      landmark (``_COMPOUND_KINDS``) is dropped anywhere within
      ``_COMPOUND_REACH_M`` of it.  The address and building lists file
      every outlying block of the University of Hong Kong and every shed
      in Ocean Park under the institution's own name, and a search for
      the name should land on the institution.
    """
    filled = [d[_LAYER_COLUMNS] for d in layers if len(d)]
    if not filled:                     # every source on disk, none with a record
        return pd.DataFrame(columns=_LAYER_COLUMNS)
    df = pd.concat(filled, ignore_index=True)
    en = df["en"].tolist(); zh = df["zh"].tolist(); kinds = df["group"].tolist()
    x = df["x"].to_numpy(float); y = df["y"].to_numpy(float)
    reach = [_SAME_PLACE_M.get(k, _SAME_PLACE_DEFAULT_M) for k in kinds]
    ken = [_key(n) for n in en]; kzh = [_key(n) for n in zh]
    taken: dict[str, list[int]] = {}
    compounds: dict[str, list[int]] = {}
    keep: list[int] = []
    for i in range(len(df)):
        ke, kz = ken[i], kzh[i]
        if kinds[i] in ("estate", "building") and any(
                (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 < _COMPOUND_REACH_M ** 2
                for j in compounds.get(ke or kz, ())):
            continue
        hit = -1
        for lang, k in (("e", ke), ("z", kz)):
            for j in taken.get(lang + k, ()) if k else ():
                if lang == "z" and ke and ken[j] and ke != ken[j]:
                    continue                   # another English name for it
                r = max(reach[i], reach[j])
                if (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 < r * r:
                    hit = j
                    break
            if hit >= 0:
                break
        if hit >= 0:
            if not zh[hit] and kz:
                zh[hit], kzh[hit] = zh[i], kz
                taken.setdefault("z" + kz, []).append(hit)
            if not en[hit] and ke:
                en[hit], ken[hit] = en[i], ke
                taken.setdefault("e" + ke, []).append(hit)
            continue
        keep.append(i)
        if ke:
            taken.setdefault("e" + ke, []).append(i)
        if kz:
            taken.setdefault("z" + kz, []).append(i)
        if kinds[i] in _COMPOUND_KINDS:
            # under its English name where it has one ("Ocean Park Hong
            # Kong" also as "Ocean Park"), else under its Chinese
            for k in {ke, _key(_core(en[i]))} - {""} or {kz}:
                compounds.setdefault(k, []).append(i)
    out = df.iloc[keep].copy()
    out["en"] = [en[i] for i in keep]
    out["zh"] = [zh[i] for i in keep]
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------
# layers
# --------------------------------------------------------------------------
def _read_json(path) -> dict:
    """A JSON file as the government publishes it (some carry a byte-order mark)."""
    with open(path, encoding="utf-8-sig") as fh:
        return json.load(fh)


def build_gazetteer(boundary=None) -> pd.DataFrame:
    """Named places from the Lands Department gazetteer.

    Two layers: points with a class, and the names that belong to them
    (an official name and sometimes an alias, in either language).  An
    alias becomes a record of its own at the same point, with the official
    name of the other language beside it.
    """
    boundary = _sar_boundary() if boundary is None else boundary
    where = {}
    for f in _read_json(PLACE_NAME_POINTS_GEOJSON)["features"]:
        p, g = f["properties"], f.get("geometry") or {}
        kind = GAZETTEER_KINDS.get(p.get("PLACE_TYPE"))
        if kind and g.get("coordinates"):
            where[p["GEO_NAME_ID"]] = (kind, g["coordinates"][0], g["coordinates"][1])
    names = [f["properties"] for f in _read_json(PLACE_NAME_NAMES_GEOJSON)["features"]]
    official = {p["GEO_NAME_ID"]: (_clean(p.get("NAME_EN")), _clean(p.get("NAME_TC")))
                for p in names if p.get("NAME_STATUS") == "Official"}
    rows = []
    for p in names:
        gid = p["GEO_NAME_ID"]
        if gid not in where:
            continue
        kind, lon, lat = where[gid]
        en, zh = _clean(p.get("NAME_EN")), _clean(p.get("NAME_TC"))
        if p.get("NAME_STATUS") != "Official":
            base_en, base_zh = official.get(gid, ("", ""))
            en, zh = en or base_en, zh or base_zh
        en = GAZETTEER_RENAMES.get(en, en)
        # areas first, so that an area wins a name it shares with a village
        rows.append({"en": en, "zh": zh, "group": kind, "lon": lon, "lat": lat,
                     "order": 2 * KIND_RANK[kind] + (p.get("NAME_STATUS") != "Official")})
    df = _layer(rows, boundary)
    df = df.sort_values("order", kind="stable").drop(columns="order")
    df = df.drop_duplicates(["en", "zh", "lon", "lat"]).reset_index(drop=True)
    log.info("place names: %d (%s)", len(df), _counts(df))
    return df


def build_base(boundary=None) -> pd.DataFrame:
    """Named OpenStreetMap features from the Overture base theme.

    Returns en, zh, group, lon, lat (and grid x, y): upstream's single
    ``name`` column is split, because OpenStreetMap writes Hong Kong names
    as "維多利亞公園 Victoria Park" in one string.  Two-character names are
    kept (upstream dropped anything under three): 中環 is a whole name.
    """
    import pyarrow.parquet as pq
    import shapely
    boundary = _sar_boundary() if boundary is None else boundary
    frames = []
    for typ, path in BASE_PARQUETS.items():
        if not path.exists():
            continue
        t = pq.read_table(path, columns=["names", "class", "geometry"]).to_pandas()
        kind = pd.Series([BASE_CLASSES.get((typ, c)) for c in t["class"]], index=t.index)
        named = t["names"].map(lambda n: isinstance(n, dict) and bool(n.get("primary")))
        t = t[kind.notna() & named]
        kind = kind[t.index]
        if t.empty:
            continue
        pairs = [split_name(n) for n in t["names"]]
        geom = shapely.from_wkb(t["geometry"].values)
        pts = shapely.get_coordinates(shapely.point_on_surface(geom))
        area = np.where(shapely.get_type_id(geom) >= 3, shapely.area(geom), 0.0)
        frames.append(pd.DataFrame({
            "en": [p[0] or "" for p in pairs], "zh": [p[1] or "" for p in pairs],
            "group": kind.to_numpy(), "lon": pts[:, 0], "lat": pts[:, 1], "area": area,
        }))
    if not frames:
        return pd.DataFrame(columns=_LAYER_COLUMNS + ["area"])
    df = _layer(pd.concat(frames, ignore_index=True), boundary)
    bridge = df["group"] == "bridge"
    named_bridge = (df["en"] + " " + df["zh"]).map(lambda n: bool(_BRIDGE_RE.search(n)))
    pier = df["group"] == "pier"
    named_pier = (df["en"] + " " + df["zh"]).map(lambda n: bool(_PIER_RE.search(n)))
    df = df[(~bridge | named_bridge) & (~pier | named_pier)].copy()
    for kind, rename in (("station", lambda e, z: _station_names(e, z, "station")),
                         ("bus terminus", _terminus_names)):
        rows = df["group"] == kind
        fixed = [rename(e, z) for e, z in zip(df.loc[rows, "en"], df.loc[rows, "zh"])]
        df.loc[rows, "en"] = [f[0] for f in fixed]
        df.loc[rows, "zh"] = [f[1] for f in fixed]
    # a bridge is mapped once per carriageway and a park once per polygon
    # ring: keep the largest outline under each name
    df = df.sort_values("area", ascending=False, kind="stable")
    out = []
    for kind, part in df.groupby("group", sort=False):
        out.append(_thin(part, max(1000.0, _SAME_PLACE_M.get(kind, 0.0)), pick="first"))
    df = pd.concat(out, ignore_index=True)
    # petrol stations are mapped as retail land under the brand alone, so
    # there are nine places called "Shell"
    natural = df["group"].isin(_NATURAL_KINDS)
    df = pd.concat([df[natural], _drop_generic(df[~natural])], ignore_index=True)
    log.info("base features: %d named (%s)", len(df), _counts(df))
    return df


def build_igeocom(boundary=None) -> pd.DataFrame:
    """Stations, exits, malls and civic places from the Lands Department's iGeoCom."""
    from pyproj import Transformer
    boundary = _sar_boundary() if boundary is None else boundary
    with zipfile.ZipFile(IGEOCOM_ZIP) as z:
        member = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        with z.open(member) as fh:
            t = pd.read_csv(fh, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    t = t[t["TYPE"].isin(IGEOCOM_KINDS)].copy()
    kind = t["TYPE"].map(IGEOCOM_KINDS)
    to_ll = Transformer.from_crs(CRS_PROJECTED, "EPSG:4326", always_xy=True)
    lon, lat = to_ll.transform(pd.to_numeric(t["EASTING"], errors="coerce").to_numpy(float),
                               pd.to_numeric(t["NORTHING"], errors="coerce").to_numpy(float))
    en = [_title_en(_clean(n)) for n in t["ENGLISHNAME"]]
    zh = [_clean(n) for n in t["CHINESENAME"]]
    rows = []
    for e, z, k, x, y in zip(en, zh, kind, lon, lat):
        if k in ("station", "station exit", "light rail stop"):
            e, z = _station_names(e, z, k)
        elif k == "bus terminus":
            e, z = _terminus_names(e, z)
        elif k == "library" and "Mobile Library" in e:
            continue                   # a kerb the library van stops at
        rows.append({"en": e, "zh": z, "group": k, "lon": x, "lat": y})
    df = _drop_generic(_thin(_layer(rows, boundary), _SAME_PLACE_DEFAULT_M))
    log.info("iGeoCom: %d of %d kept (%s)", len(df), len(t), _counts(df))
    return df


@functools.lru_cache(maxsize=1)
def _load_als() -> pd.DataFrame:
    """Every record of the Address Lookup Service bulk file, flattened.

    One row per premises: its point, the street (or, in the New
    Territories, the village) it is numbered on, the number range, the
    locality and district, and the estate it belongs to.  Strings are ''
    where the record has none.  The zip holds one GeoJSON per district and
    a nineteenth file of 622 MB that lists every flat of the public rental
    estates; that one is skipped, because the district files already carry
    its blocks.  Cached: both builders read it and it takes a while.
    """
    if not ADDRESSES_PATH.exists():
        raise FileNotFoundError(
            f"{ADDRESSES_PATH.name} is missing: no Address Lookup Service "
            "file to build addresses from (run `python -m hk_flat_routes download`)")
    rows = []
    with step("reading the Address Lookup Service file", log):
        with zipfile.ZipFile(ADDRESSES_PATH) as z:
            for member in sorted(z.namelist()):
                if not member.rsplit("/", 1)[-1].startswith("als_addresses_("):
                    continue           # the 3D public-rental file, and anything else
                with z.open(member) as fh:
                    features = json.load(fh)["features"]
                for f in features:
                    xy = (f.get("geometry") or {}).get("coordinates") or ()
                    if len(xy) < 2:    # two records have a point without one
                        continue
                    pa = f["properties"]["Address"]["PremisesAddress"]
                    e = pa.get("EngPremisesAddress") or {}
                    c = pa.get("ChiPremisesAddress") or {}
                    es, cs = e.get("EngStreet"), c.get("ChiStreet")
                    village = not (es or cs)
                    if village:
                        es, cs = e.get("EngVillage"), c.get("ChiVillage")
                    es, cs = es or {}, cs or {}
                    rows.append((
                        xy[0], xy[1],
                        es.get("StreetName") or es.get("VillageName") or "",
                        cs.get("StreetName") or cs.get("VillageName") or "",
                        es.get("BuildingNoFrom") or cs.get("BuildingNoFrom") or "",
                        es.get("BuildingNoTo") or cs.get("BuildingNoTo") or "",
                        es.get("LocationName") or "", cs.get("LocationName") or "",
                        village,
                        (e.get("EngEstate") or {}).get("EstateName") or "",
                        (c.get("ChiEstate") or {}).get("EstateName") or "",
                        e.get("EngDistrict") or "", c.get("ChiDistrict") or ""))
    df = pd.DataFrame(rows, columns=[
        "lon", "lat", "en", "zh", "no_from", "no_to", "loc_en", "loc_zh",
        "village", "estate_en", "estate_zh", "district_en", "district_zh"])
    log.info("Address Lookup Service: %d records", len(df))
    return df


def build_estates(boundary=None) -> pd.DataFrame:
    """Housing estates named in the Address Lookup Service.

    An estate is a few dozen address records that share an estate name;
    it becomes one record, on the address nearest the middle of them.
    """
    boundary = _sar_boundary() if boundary is None else boundary
    a = _load_als()
    a = a[(a["estate_en"] != "") | (a["estate_zh"] != "")]
    title = {n: _title_en(_clean(n)) for n in a["estate_en"].unique()}
    df = _layer({"en": a["estate_en"].map(title).to_numpy(),
                 "zh": a["estate_zh"].to_numpy(), "group": "estate",
                 "lon": a["lon"].to_numpy(), "lat": a["lat"].to_numpy()}, boundary)
    df = _thin(df, 400.0)
    log.info("estates: %d", len(df))
    return df


_CSUID_RE = re.compile(r"^(\d{5})(\d{5})[A-Z]\d{8}$")


def _csuid_position(csuid) -> tuple[float, float] | None:
    """A building's HK1980 Grid position, read out of its identifier.

    The building-name table has no geometry, but a BuildingCSUID is five
    digits of easting and five of northing (each less 800,000 m), a letter
    and a date.  Decoded this way, 99,917 identifiers shared with the
    Address Lookup Service land a median 0.4 m from that service's own
    coordinates (research_places.txt).
    """
    m = _CSUID_RE.match(csuid) if isinstance(csuid, str) else None
    if not m:
        return None
    return 800000.0 + int(m.group(1)), 800000.0 + int(m.group(2))


def build_buildings(boundary=None) -> pd.DataFrame:
    """Building names from the Lands Department, one record per estate.

    The table has a row for every name of every building: an official one
    from the Lands Department, often the same again in capitals from the
    Rating and Valuation Department, sometimes an alias.  Blocks are
    folded into their estate (``_fold_blocks``), plant is dropped
    (``_PLANT_EN_RE``), and what remains is thinned to one record per name
    and place.
    """
    from pyproj import Transformer
    boundary = _sar_boundary() if boundary is None else boundary
    names, where = [], []
    for f in _read_json(BUILDING_NAMES_GEOJSON)["features"]:
        p = f["properties"]
        pos = _csuid_position(p.get("BuildingCSUID"))
        if pos is None or p.get("NameStatus") == "Proposed":
            continue
        en, zh = _pair(_title_en(_clean(p.get("BuildingNameEN"))), p.get("BuildingNameTC"))
        # a name that is nothing but a block ("Block A", "第1A座") names nothing
        if len(_BLOCK_EN_RE.sub("", en)) < 2 or sum(c.isalpha() for c in en) < 3:
            en = ""
        if len(_BLOCK_ZH_RE.sub("", zh)) < 2:
            zh = ""
        names.append((en, zh))
        where.append(pos)
    rows = []
    # folded over the whole list at once: whether a code at the end of a
    # name is a house number shows only in the names around it
    for (en, zh), pos in zip(_fold_blocks(names), where):
        if (en and _PLANT_EN_RE.search(en)) or (zh and _PLANT_ZH_RE.search(zh)):
            continue
        rows.append((en, zh, pos[0], pos[1]))
    t = pd.DataFrame(rows, columns=["en", "zh", "e", "n"])
    to_ll = Transformer.from_crs(CRS_PROJECTED, "EPSG:4326", always_xy=True)
    lon, lat = to_ll.transform(t["e"].to_numpy(float), t["n"].to_numpy(float))
    df = _layer({"en": t["en"].to_numpy(), "zh": t["zh"].to_numpy(),
                 "group": "building", "lon": lon, "lat": lat}, boundary)
    # six sites and not iGeoCom's four: unrelated buildings do share names
    # (there is more than one Wing On Building), but not by the half-dozen
    df = _drop_generic(_thin(df, _SAME_PLACE_DEFAULT_M), at_least=6)
    log.info("buildings: %d names from %d rows", len(df), len(t))
    return df


_BOUND_RE = re.compile(r"\s*\([^()]*(?:bound|方向)[^()]*\)\s*$", re.I)


def build_stops(boundary=None) -> pd.DataFrame:
    """Tram stops, Peak Tram stations and ferry piers (Transport Department).

    The files list every stop once per route.  A tram stop is named after
    the street it stands on, once for each direction ('Chai Wan Road (West
    bound)' / '柴灣道 (西行方向)'); the pair becomes one 'Chai Wan Road Tram
    Stop' / '柴灣道電車站', so that it cannot be mistaken for the street.
    """
    boundary = _sar_boundary() if boundary is None else boundary
    rows = []
    for mode, path in TD_STOPS_JSON.items():
        if not path.exists():
            continue
        seen = set()
        for f in _read_json(path)["features"]:
            p, xy = f["properties"], (f.get("geometry") or {}).get("coordinates") or ()
            if p.get("stopId") in seen or len(xy) < 2:
                continue
            seen.add(p.get("stopId"))
            en, zh = _clean(p.get("stopNameE")), _clean(p.get("stopNameC"))
            if mode == "tram":
                en, zh = _BOUND_RE.sub("", en), _BOUND_RE.sub("", zh)
                if en and not re.search(r"terminus|tram", en, re.I):
                    en += " Tram Stop"
                elif en and "tram" not in en.lower():
                    en = re.sub(r"(?i)\bterminus\b", "Tram Terminus", en)
                if zh and "電車" not in zh:
                    zh = zh[:-2] + "電車總站" if zh.endswith("總站") else zh + "電車站"
            elif mode == "peak_tram":
                if en and "peak tram" not in en.lower():
                    en = re.sub(r"(?i)\bstation\b", "Peak Tram Station", en)
                if zh and "纜車" not in zh:
                    zh = zh[:-1] + "山頂纜車站" if zh.endswith("站") else zh
            rows.append({"en": en, "zh": zh, "group": STOP_KINDS[mode],
                         "lon": xy[0], "lat": xy[1]})
    df = _thin(_layer(rows, boundary), _SAME_PLACE_DEFAULT_M)
    log.info("transit stops: %d (%s)", len(df), _counts(df))
    return df


# --------------------------------------------------------------------------
# the POI feed
# --------------------------------------------------------------------------
def _support(names: pd.Series, x: np.ndarray, y: np.ndarray,
             all_names: pd.Series, all_x: np.ndarray, all_y: np.ndarray,
             radius_m: float = 300.0) -> np.ndarray:
    """How many nearby records mention each name.

    'Times Square' at the real complex is surrounded by 'Times Square Food
    Forum', 'Lane Crawford Times Square' and so on; a stray 'Times Square'
    dropped in a village has none of that.  The feed carries such strays
    with full confidence, so the name alone cannot pick the right one.
    Positions are HK1980 Grid metres (upstream worked in degrees, scaled
    for San Francisco's latitude).
    """
    grid: dict[tuple[int, int], list[int]] = {}
    low = all_names.str.lower().to_numpy()
    for i, (px, py) in enumerate(zip(all_x, all_y)):
        grid.setdefault((int(px // radius_m), int(py // radius_m)), []).append(i)
    out = np.zeros(len(names), dtype=int)
    for k, (n, px, py) in enumerate(zip(names.str.lower(), x, y)):
        cx, cy = int(px // radius_m), int(py // radius_m)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in grid.get((cx + dx, cy + dy), ()):
                    o = low[j]
                    if o != n and n in o and abs(all_x[j] - px) < radius_m \
                            and abs(all_y[j] - py) < radius_m:
                        out[k] += 1
    return out


_CITY_SUFFIXES = {"hk", "hong kong", "hongkong", "hong kong sar", "hk sar",
                  "hong kong china", "kowloon", "kowloon hong kong", "china",
                  "香港", "九龍", "新界", "中國香港"}
# 'X Hong Kong' and 'X HK' are city suffixes even without a comma; a bare
# trailing 'Kowloon' or 'China' only after one ('Bank of China').
_CORE_RE = re.compile(
    r"(?:[\s,\-/]+(?:hong\s?kong|hk|香港)(?:[\s,]+(?:sar|china))?"
    r"|[,\-/]\s*(?:kowloon|china|九龍|新界))\s*$", re.I)
_HEAD_RE = re.compile(r"\s*[,\-/(（|@]\s*")


def _core(name: str) -> str:
    """'Rosewood, Hong Kong' -> 'rosewood'."""
    return _CORE_RE.sub("", str(name)).strip().lower()


def _head(name: str) -> str:
    """'Genki Sushi (Yau Tong)' -> 'genki sushi': the name before a qualifier."""
    return _HEAD_RE.split(_core(name), maxsplit=1)[0]


def _prune_variants(df: pd.DataFrame, radius_m: float = 500.0) -> pd.DataFrame:
    """Drop 'Lan Fong Yuen, Hong Kong' when 'Lan Fong Yuen' is 200 m away.

    The places feed carries many user-typed variants of the same name. A
    record is dropped when a better-supported kept name is a prefix of it
    (at a word boundary) and the two points are within ``radius_m``.
    Positions are grid metres in columns x and y.
    """
    df = df.sort_values(["support", "conf"], ascending=False).reset_index(drop=True)
    kept_idx = []
    by_first: dict[str, list[int]] = {}
    x = df["x"].to_numpy(); y = df["y"].to_numpy()
    names = df["name"].tolist(); groups = df["group"].tolist()
    support = df["support"].to_numpy()
    for i, n in enumerate(names):
        # the leading run of the name; a Chinese name has no spaces, so it
        # is cut at the first bracket or comma as well
        first = re.split(r"[\s,\-/(（]", n.strip(), maxsplit=1)[0].lower()
        dup = False
        for j in by_first.get(first, ()):
            k = names[j]
            if not (n.startswith(k) and len(n) > len(k) and n[len(k)] in " ,-/(（"):
                continue
            # a city suffix never marks a different place; anything else
            # (a branch, a sub-area) only when it is close by
            rest = n[len(k):].strip(" ,-/()（）").lower()
            if rest in _CITY_SUFFIXES:
                r = 6000.0                       # 'X, Hong Kong' anywhere
            elif groups[i] != groups[j]:
                continue                         # 'Times Square Food Forum' is food
            elif support[j] >= 3 and support[i] == 0:
                r = 6000.0                       # a same-kind variant of a well-known name
            else:
                r = radius_m
            if abs(x[i] - x[j]) < r and abs(y[i] - y[j]) < r:
                dup = True
                break
        if not dup:
            kept_idx.append(i)
            by_first.setdefault(first, []).append(i)
    out = df.iloc[kept_idx].reset_index(drop=True)
    log.info("places: %d near-duplicate name variants pruned", len(df) - len(out))
    return out


def _poi_names(names) -> tuple[str, str]:
    """(English, Chinese) of a POI record.

    The feed has one name per record and no language tags.  A name written
    Chinese first ('大快活 Fairwood') follows OpenStreetMap's convention and
    is split by ``network.split_name`` like a street's.  One written
    English first is left whole, in the Chinese slot because it contains
    Chinese: what follows the English there is as often a branch or a hall
    as a translation ('Nara Thai Cuisine 海港城店'), and a whole name is
    still found by either half.  The dash or comma that stood between the
    two halves ('赤鬼居酒屋 - Akaoni') is left on neither.
    """
    return _pair(*((n or "").strip(" \t-–—|,·•:;/") for n in split_name(names)))


def build_poi(taken: set[str] | None = None, boundary=None) -> pd.DataFrame:
    """Shops, restaurants, hotels and venues from Overture's places theme.

    ``taken`` is every name (as ``_key`` makes it) that a better layer
    already has anywhere in Hong Kong; a POI of that name is the feed's
    copy of the same place, or a stray, and is dropped either way.

    Upstream leaned on this feed for landmarks and used the records around
    a name to tell the real one from the strays.  Here the landmarks come
    from the layers above, so the feed supplies only what nothing else
    has, and the same evidence is used more bluntly: a chain is dropped
    outright, and of two records with one name the better-supported stays.
    """
    import pyarrow.parquet as pq
    import shapely
    boundary = _sar_boundary() if boundary is None else boundary
    taken = taken or set()
    t = pq.read_table(PLACES_PARQUET, columns=[
        "names", "taxonomy", "confidence", "operating_status", "geometry"]).to_pandas()
    t = t[t["operating_status"].astype(object) != "permanently_closed"]
    pts = shapely.get_coordinates(shapely.from_wkb(t["geometry"].values))

    def group_of(tax) -> str | None:
        if not isinstance(tax, dict):
            return None
        path = tax.get("hierarchy")
        path = [] if path is None else list(path)
        for node in reversed(path + [tax.get("primary")]):
            if node in _GROUP_OF:
                return _GROUP_OF[node]
        return None

    pairs = [_poi_names(n) for n in t["names"]]
    df = _layer({"en": [p[0] for p in pairs], "zh": [p[1] for p in pairs],
                 "group": t["taxonomy"].map(group_of).to_numpy(),
                 "lon": pts[:, 0], "lat": pts[:, 1],
                 "conf": pd.to_numeric(t["confidence"], errors="coerce").fillna(0).to_numpy()},
                boundary)
    df["name"] = df["en"].where(df["en"] != "", df["zh"])
    df = df[df["name"].str.len().between(2, 60)].reset_index(drop=True)
    n_hk = len(df)

    # chains, counted over every record of the name whatever its confidence
    head = df["name"].map(lambda n: _key(_head(n)))
    cell = (df["x"] // 300).astype(int).astype(str) + ":" + (df["y"] // 300).astype(int).astype(str)
    sites = pd.DataFrame({"head": head, "cell": cell}).drop_duplicates().groupby("head").size()
    chains = set(sites[sites >= POI_CHAIN_SITES].index)

    keep = df["group"].notna() & (df["conf"] >= POI_MIN_CONFIDENCE) & (head != "")
    keep &= ~head.isin(chains)
    # the better layers win the name outright, with or without a qualifier
    core = df["name"].map(lambda n: _key(_core(n)))
    keep &= ~(head.isin(taken) | core.isin(taken))
    keep &= ~df["zh"].map(_key).isin(taken - {""})
    # ... and so does the English that leads a mixed name.  'Langham Place
    # 朗豪坊' sits whole in the Chinese slot (see ``_poi_names``), where the
    # tests above compare all of it and find nothing; the feed has one of
    # that name more than 4 km from Langham Place.  (Three letters at
    # least: one or two before the Chinese, as in 'I素食', are not a name
    # of their own.)
    lead = df["name"].map(lambda n: _key(_CJK_RE.split(n, maxsplit=1)[0]))
    keep &= ~((df["en"] == "") & lead.isin(taken) & (lead.str.len() >= 3))
    kept = df[keep].reset_index(drop=True)
    kept["support"] = _support(kept["name"], kept["x"].to_numpy(), kept["y"].to_numpy(),
                               df["name"], df["x"].to_numpy(), df["y"].to_numpy())
    kept["key"] = kept["name"].map(_key)
    kept = (kept.sort_values(["support", "conf"], ascending=False)
                .drop_duplicates("key").reset_index(drop=True))
    kept = _prune_variants(kept)
    log.info("places: %d kept of %d POI records in Hong Kong (%d chain names dropped)",
             len(kept), n_hk, len(chains))
    return kept


def _counts(df: pd.DataFrame, top: int = 10) -> str:
    return ", ".join(f"{g} {n}" for g, n in df["group"].value_counts().head(top).items())


def build_places() -> dict:
    """Compact place list: names in two languages, display kind, coordinates.

    Keys: ``names`` (the name shown first: English where the place has
    one, otherwise Chinese), ``alt`` (the other-language name, '' when
    there is only one), ``group`` (index into ``groups``), ``groups`` (the
    kinds), ``group_rank`` (the rank of each kind on an equal text match,
    0 first; see ``KIND_RANK``), ``lon`` and ``lat``.  Records are sorted
    by name.  Upstream's index had no ``alt`` and no ``group_rank``.

    Each source is optional -- any of the downloads can fail -- and the
    index is built from the ones on disk.  ``FileNotFoundError`` when none
    is, or when the SAR boundary to clip them with is missing.
    """
    boundary = _sar_boundary()
    # (layer, its files, whether it needs all of them or any, builder)
    sources = [
        ("place names", [PLACE_NAME_POINTS_GEOJSON, PLACE_NAME_NAMES_GEOJSON],
         all, build_gazetteer),
        ("mapped features", list(BASE_PARQUETS.values()), any, build_base),
        ("iGeoCom", [IGEOCOM_ZIP], all, build_igeocom),
        ("estates", [ADDRESSES_PATH], all, build_estates),
        ("buildings", [BUILDING_NAMES_GEOJSON], all, build_buildings),
        ("transit stops", list(TD_STOPS_JSON.values()), any, build_stops),
    ]
    layers: dict[str, pd.DataFrame] = {}
    with step("building the place index", log):
        for label, paths, need, build in sources:
            if not need(p.exists() for p in paths):
                log.warning("places: no %s (%s not downloaded)", label,
                            ", ".join(p.name for p in paths if not p.exists()))
                continue
            layers[label] = build(boundary)
        if "iGeoCom" in layers:
            # the railway comes from the government's list when it is there,
            # and ahead of the mapped features: the list has every station
            # once, under one form of name, with its exits.  OpenStreetMap's
            # stations are the fallback.
            poi = layers["iGeoCom"]
            rail = poi["group"].isin(("station", "station exit", "light rail stop"))
            layers["railway"], layers["iGeoCom"] = poi[rail], poi[~rail]
            if "mapped features" in layers:
                base = layers["mapped features"]
                layers["mapped features"] = base[base["group"] != "station"]
        order = ("place names", "railway", "mapped features", "iGeoCom", "estates",
                 "buildings", "transit stops")
        layers = {label: layers[label] for label in order if label in layers}
        if PLACES_PARQUET.exists():
            taken = set()
            for d in layers.values():
                for col in ("en", "zh"):
                    taken.update(_key(n) for n in d[col])
                taken.update(_key(_head(n)) for n in d["en"])
            taken.discard("")
            layers["POI feed"] = build_poi(taken, boundary)
        else:
            log.warning("places: no POI feed (%s not downloaded)", PLACES_PARQUET.name)
        if not layers:
            raise FileNotFoundError(
                "no place data under data/raw: run `python -m hk_flat_routes "
                "download` to fetch it (the route page searches junctions "
                "only without it)")
        df = _merge_layers(list(layers.values()))
        # the short forms (ALIASES): a copy of the full name's record, kind
        # and position, under the short name alone
        full = df.drop_duplicates("en").set_index("en")
        short = [dict(full.loc[name], en=alias if not _CJK_RE.search(alias) else "",
                      zh=alias if _CJK_RE.search(alias) else "")
                 for name, alias in ALIASES if name in full.index]
        if short:
            df = pd.concat([df, pd.DataFrame(short)], ignore_index=True)
    df["name"] = df["en"].where(df["en"] != "", df["zh"])
    df["alt"] = df["zh"].where(df["en"] != "", "")
    df = df.sort_values(["name", "alt"], kind="stable").reset_index(drop=True)
    log.info("places: %d records from %d layers (%s); %.0f%% carry both languages",
             len(df), len(layers), _counts(df, 14), 100 * (df["alt"] != "").mean())
    groups = sorted(set(df["group"]))
    index = {g: i for i, g in enumerate(groups)}
    return {
        "names": df["name"].tolist(),
        "alt": df["alt"].tolist(),
        "group": [index[g] for g in df["group"]],
        "groups": groups,
        "group_rank": [KIND_RANK.get(g, 2) for g in groups],
        "lon": np.round(df["lon"].to_numpy(), 5).tolist(),
        "lat": np.round(df["lat"].to_numpy(), 5).tolist(),
    }


# --------------------------------------------------------------------------
# addresses
# --------------------------------------------------------------------------
#: One number on one street lies within this of itself (metres).  The same
#: number further apart under one name is two streets that share the name.
_ADDR_SAME_M = 300.0
#: Two stretches under one name are one street when they come this close.
_ADDR_JOIN_M = 1000.0


def _same_zh(a: str, b: str) -> bool:
    """Could two Chinese names be one street's?

    The same, or nearly: the file writes 恆富街 in one district and 恒富街
    in the next, and 麒麟圍第一巷 beside 麒麟圍1巷.  Castle Peak Road's
    青山道 and 青山公路 are two roads end to end, and are not.
    """
    from difflib import SequenceMatcher

    a, b = _key(a), _key(b)
    if not a or not b or a == b:
        return True
    if len(a) == len(b) and sum(p != q for p, q in zip(a, b)) == 1:
        return True
    return SequenceMatcher(None, a, b).ratio() >= 0.7


def _join_stretches(units: list[dict]) -> list[int]:
    """Which stretches of one English street name are the same street.

    The Address Lookup Service numbers a street within a district and a
    locality, and a name alone does not say whether two such stretches
    belong together.  Prince Edward Road West runs from one district into
    the next with its numbers carrying on; Castle Peak Road is more than
    a dozen sections from Kwai Chung to the border, each numbered from 1;
    and several unrelated villages are all called Tin Sam.  Stretches are joined
    when they come within ``_ADDR_JOIN_M`` of each other, their Chinese
    names agree, and the numbers they share are the same doors.  (Most of
    them, not all: one number can cover an estate hundreds of metres
    across, and its records then lie further apart than two streets do.)

    Each unit is a dict with ``zh``, ``pts`` (n x 2 grid metres) and
    ``nums`` (number -> position).  Returns a cluster label per unit.
    """
    from scipy.spatial import cKDTree

    clusters = [{"members": [i], "zh": [u["zh"]], "pts": u["pts"], "nums": dict(u["nums"])}
                for i, u in enumerate(units)]

    def joinable(a: dict, b: dict) -> bool:
        if not all(_same_zh(p, q) for p in a["zh"] for q in b["zh"]):
            return False
        small, large = (a, b) if len(a["nums"]) < len(b["nums"]) else (b, a)
        apart = [np.hypot(px - large["nums"][num][0], py - large["nums"][num][1])
                 for num, (px, py) in small["nums"].items() if num in large["nums"]]
        if apart and np.median(apart) > _ADDR_SAME_M:
            return False
        return cKDTree(a["pts"]).query(b["pts"], k=1)[0].min() <= _ADDR_JOIN_M

    merged = True
    while merged and len(clusters) > 1:
        merged = False
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                a, b = clusters[i], clusters[j]
                if joinable(a, b):
                    a["members"] += b["members"]
                    a["zh"] += b["zh"]
                    a["pts"] = np.vstack([a["pts"], b["pts"]])
                    for num, pos in b["nums"].items():
                        a["nums"].setdefault(num, pos)
                    del clusters[j]
                    merged = True
                    break
            if merged:
                break
    label = [0] * len(units)
    for c, cluster in enumerate(clusters):
        for i in cluster["members"]:
            label[i] = c
    return label


def _leading_number(series: pd.Series) -> pd.Series:
    """'23E' -> 23; anything without a leading number -> NaN."""
    return pd.to_numeric(series.astype(str).str.extract(r"^\s*(\d+)")[0], errors="coerce")


def _fits_uint16(values: np.ndarray, what: str) -> np.ndarray:
    """``values`` as little-endian uint16, or ``ValueError`` if any does not fit.

    Upstream clipped offsets into range.  A clipped offset is an address
    drawn in the wrong place, so here it is an error.
    """
    values = np.asarray(values)
    if len(values) and (values.min() < 0 or values.max() > 65535):
        raise ValueError(f"address {what} does not fit in 16 bits "
                         f"(range {values.min()}..{values.max()})")
    return values.astype("<u2")


def build_addresses() -> dict:
    """One point per (street, number), with a street table in two languages.

    Keys: ``streets`` (English street or village names, title-cased),
    ``streets_zh`` (the Chinese name of each, or ''), ``street`` (index
    into ``streets`` per address), ``number`` (the leading house number),
    ``lon`` and ``lat`` (offsets from ``origin`` in 1e-5 degree steps),
    all four as little-endian uint16, and ``origin`` (the south-west
    corner of the study bounding box).  Rows are sorted by street, then by
    number, which is what the page's binary search needs.

    What differs from upstream, which read Overture's addresses theme:

    * The source is the Address Lookup Service itself.  Overture's Hong
      Kong addresses are a copy of it in English only.
    * Half the records are village addresses ("28 A Kung Ngam"): a house
      number within a named village, no street.  A village is a street
      here.
    * A name is not a key.  Stretches that share one are joined or kept
      apart by ``_join_stretches``; where a name is left on more than one
      street, each is labelled with its locality: "Castle Peak Road (Ting
      Kau)" / "青山公路 (汀九段)".
    * A range ("8-12 Hennessy Road") is indexed at both of its ends.

    ``FileNotFoundError`` when the file has not been downloaded.
    """
    a = _load_als()
    a = a[(a["en"] != "") | (a["zh"] != "")].copy()
    # positions are packed as steps from the corner of the study box, so a
    # point outside the box cannot be written.  The file is republished
    # every month: a stray coordinate should cost that record, not the
    # whole address search (``_fits_uint16`` would refuse the lot).
    inside = (a["lon"].between(STUDY_BBOX[0], STUDY_BBOX[1])
              & a["lat"].between(STUDY_BBOX[2], STUDY_BBOX[3]))
    if not inside.all():
        log.info("addresses: %d records outside the study area left out",
                 int((~inside).sum()))
        a = a[inside].copy()
    a["num"] = _leading_number(a["no_from"])
    a["num_to"] = _leading_number(a["no_to"])
    a = a[a["num"].notna()].copy()
    title = {n: _title_en(_clean(n)) for n in a["en"].unique()}
    a["en"] = a["en"].map(title)
    a["zh"] = a["zh"].map(_clean)
    a["name"] = a["en"].where(a["en"] != "", a["zh"])
    a["x"], a["y"] = _to_grid(a["lon"].to_numpy(), a["lat"].to_numpy())
    # a plain number is a better point for "23" than "23E" is
    a["plain"] = a["no_from"].astype(str).str.fullmatch(r"\s*\d+\s*")
    a = a.sort_values(["name", "num", "plain"], ascending=[True, True, False],
                      kind="stable").reset_index(drop=True)

    # stretches: one name in one district and locality
    a["unit"] = a.groupby(["name", "zh", "district_en", "loc_en"], sort=False).ngroup()
    unit_name = a.groupby("unit")["name"].first()
    unit_rows = a.groupby("unit").indices
    xy = a[["x", "y"]].to_numpy(float)
    num = a["num"].to_numpy().astype(int)
    zh_of = a["zh"].to_numpy()
    street_of_unit: dict[int, int] = {}
    next_id = 0
    for name, unit_ids in unit_name.groupby(unit_name).groups.items():
        unit_ids = list(unit_ids)
        if len(unit_ids) == 1:
            labels = [0]
        else:
            units = []
            for u in unit_ids:
                rows = unit_rows[u]
                nums: dict[int, tuple[float, float]] = {}
                for r in rows:                 # sorted: a plain number comes first
                    nums.setdefault(int(num[r]), (xy[r, 0], xy[r, 1]))
                units.append({"zh": zh_of[rows[0]], "pts": xy[rows], "nums": nums})
            labels = _join_stretches(units)
        for u, lab in zip(unit_ids, labels):
            street_of_unit[u] = next_id + lab
        next_id += max(labels) + 1
    a["street"] = a["unit"].map(street_of_unit)

    # label each street; a name left on several streets gains a locality
    def mode(series: pd.Series) -> str:
        series = series[~series.isin(("", "\t"))]
        return series.mode().iloc[0] if len(series) else ""

    q_en = a["loc_en"].str.split(",").str[0].str.strip()
    q_zh = a["loc_zh"].map(_clean).str.split(" ").str[-1]
    d_en = a["district_en"].str.replace(r"\s+DISTRICT$", "", regex=True)
    # the commonest locality and district are each picked as a pair of
    # names.  Picked a language at a time, a tie gives the English of one
    # locality and the Chinese of another: Hiram's Highway came out as
    # "(Ho Chung)" beside "(打鼓嶺)", which is Ta Ku Ling.
    a["q"] = q_en + "\t" + q_zh
    a["d"] = d_en + "\t" + a["district_zh"]
    by_street = a.groupby("street")
    locality = by_street["q"].agg(mode).str.partition("\t")
    district = by_street["d"].agg(mode).str.partition("\t")
    table = pd.DataFrame({
        "name": by_street["name"].first(), "zh": by_street["zh"].agg(mode),
        "q_en": locality[0], "q_zh": locality[2],
        "d_en": district[0], "d_zh": district[2],
    })
    shared = table["name"].map(table["name"].value_counts()) > 1
    q_en = table["q_en"].where(table["q_en"] != "", table["d_en"]).map(_title_en)
    q_zh = table["q_zh"].where(table["q_zh"] != "", table["d_zh"])
    # two sections in one locality: say the district as well, if it says more
    d_en = table["d_en"].map(_title_en)
    again = shared & (table["name"] + "|" + q_en).duplicated(keep=False) & (q_en != d_en)
    q_en = q_en.where(~again, q_en + ", " + d_en)
    q_zh = q_zh.where(~again, table["d_zh"] + " " + q_zh)
    table["label"] = table["name"].where(~shared, table["name"] + " (" + q_en + ")")
    table["label_zh"] = table["zh"].where(~shared | (table["zh"] == ""),
                                          table["zh"] + " (" + q_zh + ")")
    # an English-only table keeps '' beside a name that is itself Chinese
    table.loc[table["zh"] == table["name"], "label_zh"] = ""
    a["label"] = a["street"].map(table["label"])

    # both ends of a range, the stated start first
    ends = a[a["num_to"].notna() & (a["num_to"] != a["num"])].copy()
    ends["num"] = ends["num_to"]
    ends["plain"] = False
    a["end"] = False
    ends["end"] = True
    rows = pd.concat([a, ends], ignore_index=True)
    big = rows["num"] >= 65536
    if big.any():
        log.info("addresses: %d numbers above 65,535 left out", int(big.sum()))
    rows = rows[~big]
    rows = (rows.sort_values(["label", "num", "end", "plain"],
                             ascending=[True, True, True, False], kind="stable")
                .drop_duplicates(["label", "num"]).reset_index(drop=True))

    labels = table.drop_duplicates("label").sort_values("label")
    streets = labels["label"].tolist()
    sidx = {s: i for i, s in enumerate(streets)}
    if len(streets) > 65536:
        raise ValueError(f"{len(streets)} streets do not fit a 16-bit index")
    rows["sid"] = rows["label"].map(sidx)
    rows = rows.sort_values(["sid", "num"], kind="stable").reset_index(drop=True)
    log.info("addresses: %d unique street numbers on %d streets and villages "
             "(%d names shared by more than one)", len(rows), len(streets),
             int(table.loc[shared, "name"].nunique()))
    lon0, lat0 = STUDY_BBOX[0], STUDY_BBOX[2]
    return {
        "streets": streets,
        "streets_zh": labels["label_zh"].tolist(),
        "street": _fits_uint16(rows["sid"].to_numpy(), "street index"),
        "number": _fits_uint16(rows["num"].to_numpy().astype(np.int64), "number"),
        # 1e-5 degree offsets from the bbox corner fit in uint16 (~1 m)
        "lon": _fits_uint16(np.round((rows["lon"].to_numpy() - lon0) / 1e-5).astype(np.int64),
                            "longitude offset"),
        "lat": _fits_uint16(np.round((rows["lat"].to_numpy() - lat0) / 1e-5).astype(np.int64),
                            "latitude offset"),
        "origin": [lon0, lat0],
    }


def build_hillshade(width_px: int = HILLSHADE_WIDTH_PX) -> dict:
    """Quiet shaded relief in Web Mercator, as a palette PNG with bounds.

    The image is warped to the map's own projection, not to plain
    longitude and latitude: Leaflet stretches an image overlay evenly
    between its corners in Mercator, and over Hong Kong's 50 km from south
    to north an evenly spaced latitude grid would sit up to 18 m off the
    streets drawn over it.  The sea is transparent (the terrain model has
    no data there), which is what draws the coastline.
    """
    import rasterio
    from PIL import Image
    from pyproj import Transformer
    from rasterio.enums import Resampling
    from rasterio.warp import calculate_default_transform, reproject

    from .elevation import DEM_MOSAIC, build_dem_overview
    from .viz_static import hillshade

    with step("rendering the hillshade base", log):
        with rasterio.open(build_dem_overview()) as src:
            transform, w, h = calculate_default_transform(
                src.crs, "EPSG:3857", src.width, src.height, *src.bounds)
            scale = width_px / w
            w2, h2 = int(w * scale), int(h * scale)
            transform = transform * transform.scale(w / w2, h / h2)
            dem = np.full((h2, w2), np.nan, dtype="float32")
            reproject(rasterio.band(src, 1), dem, dst_transform=transform,
                      dst_crs="EPSG:3857", dst_nodata=np.nan,
                      resampling=Resampling.bilinear)
        valid = np.isfinite(dem)
        # crop to the land, with a small margin
        rows = np.flatnonzero(valid.any(axis=1)); cols = np.flatnonzero(valid.any(axis=0))
        r0, r1 = max(rows[0] - 4, 0), min(rows[-1] + 5, h2)
        c0, c1 = max(cols[0] - 4, 0), min(cols[-1] + 5, w2)
        dem = dem[r0:r1, c0:c1]; valid = valid[r0:r1, c0:c1]
        transform = transform * transform.translation(c0, r0)
        h2, w2 = dem.shape
        # shade over a surface that carries on past the coast, so the shore
        # does not read as a cliff
        from scipy.ndimage import distance_transform_edt
        idx = distance_transform_edt(~valid, return_distances=False, return_indices=True)
        filled = dem[tuple(idx)]
        mid_lat = 0.5 * (STUDY_BBOX[2] + STUDY_BBOX[3])
        px_m = abs(transform.a) * np.cos(np.radians(mid_lat))   # ground metres
        hs = hillshade(filled, res=px_m, z_factor=HILLSHADE_Z_FACTOR)
        shade = (0.72 + 0.28 * hs)[..., None]
        tint = np.clip(filled / HILLSHADE_TINT_M, 0, 1)[..., None]
        base = np.array([243, 242, 238], float); dark = np.array([196, 194, 186], float)
        rgb = (base * (1 - tint * 0.45) + dark * (tint * 0.45)) * shade
        img = np.zeros((h2, w2, 4), np.uint8)
        img[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
        img[..., 3] = np.where(valid, 255, 0)
        im = Image.fromarray(img, "RGBA").quantize(colors=96, method=Image.Quantize.FASTOCTREE)
        buf = io.BytesIO(); im.save(buf, "PNG", optimize=True)
        im.save(HILLSHADE_PNG)
        to_ll = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        west, north = to_ll.transform(transform.c, transform.f)
        east, south = to_ll.transform(transform.c + transform.a * w2,
                                      transform.f + transform.e * h2)
        log.info("  hillshade %dx%d (%.0f m a pixel), %.0f KB", w2, h2, px_m,
                 len(buf.getvalue()) / 1024)
    return {"bounds": [[south, west], [north, east]], "png": buf.getvalue(),
            "data_uri": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}
