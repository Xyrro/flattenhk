"""The route page's offline place index, built from Hong Kong sources.

The first half needs no data: it exercises the name handling, the clipping
and the merging on a few records typed in here.  The second half reads the
raw downloads and is skipped where they are absent.
"""
from __future__ import annotations

import gzip
import json
import zipfile

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from hk_flat_routes import network, places
from hk_flat_routes.config import STUDY_BBOX
from hk_flat_routes.download import (ADDRESSES_PARQUET, BASE_PARQUETS,
                                     BUILDING_NAMES_GEOJSON, DIVISIONS_PARQUET,
                                     IGEOCOM_ZIP, PLACE_NAME_NAMES_GEOJSON,
                                     PLACE_NAME_POINTS_GEOJSON, PLACES_PARQUET,
                                     TD_STOPS_JSON)

#: Central and the harbour on the HK1980 Grid; Shenzhen is far to the north.
CENTRAL_BOX = box(830000, 812000, 840000, 822000)
CENTRAL = (114.1588, 22.2820)
SHA_TIN = (114.1911, 22.3892)
LO_WU_SHENZHEN = (114.1190, 22.5400)


# --------------------------------------------------------------------------
# names
# --------------------------------------------------------------------------
def test_upper_case_names_are_title_cased_around_the_apostrophe():
    t = places._title_en
    assert t("QUEEN'S ROAD CENTRAL") == "Queen's Road Central"
    assert t("D'AGUILAR STREET") == "D'Aguilar Street"
    assert t("O'BRIEN ROAD") == "O'Brien Road"
    assert t("ST. STEPHEN'S LANE") == "St. Stephen's Lane"
    assert t("PARK LANE SHOPPER'S BOULEVARD") == "Park Lane Shopper's Boulevard"
    assert t("BEL-AIR AVENUE") == "Bel-Air Avenue"


def test_title_casing_knows_what_hong_kong_names_are_made_of():
    t = places._title_en
    # ordinals, single letters and acronyms
    assert t("FAIRVIEW PARK SECT C 1ST STREET") == "Fairview Park Sect C 1st Street"
    assert t("ON CHIU HOUSE (BLK 6)") == "On Chiu House (Blk 6)"
    assert t("HSBC MAIN BUILDING") == "HSBC Main Building"
    assert t("JAZZ GARDEN PHASE II") == "Jazz Garden Phase II"
    assert t("K11 MUSEA") == "K11 Musea"
    # 'of' and 'the' go small; 'on' and 'to' are syllables here, not words
    assert t("THE UNIVERSITY OF HONG KONG") == "The University of Hong Kong"
    assert t("TAI ON BUILDING") == "Tai On Building"
    assert t("TO KWA WAN ROAD") == "To Kwa Wan Road"
    # a name its publisher already cased is left alone
    assert t("Times Square Tower One") == "Times Square Tower One"
    assert t("iSQUARE") == "iSQUARE"


def test_names_are_nfkc_normalised_and_compared_without_punctuation():
    # full-width letters, digits and brackets, as the building list types them
    assert places._clean("重慶大廈Ａ座") == "重慶大廈A座"
    assert places._clean("安潮樓（第６座）") == "安潮樓(第6座)"
    assert places._clean("  Mei  Foo\u3000Sun Chuen ") == "Mei Foo Sun Chuen"
    assert places._clean(None) == "" and places._clean(float("nan")) == ""
    assert places._key("Queen's Road Central") == places._key("QUEENS ROAD CENTRAL")
    # the two ways Hong Kong writes 'estate'
    assert places._key("美孚新邨") == places._key("美孚新村")
    assert places._key("Ａ座") == "a座"


def test_each_name_goes_in_its_own_languages_slot():
    assert places._pair("Chungking Mansions", "重慶大廈Ａ座") == ("Chungking Mansions", "重慶大廈A座")
    assert places._pair("KK HOTEL", "KK HOTEL") == ("KK HOTEL", "")       # no Chinese name at all
    assert places._pair("大王爺", "") == ("", "大王爺")                    # Chinese filed as English
    assert places._pair(None, float("nan")) == ("", "")
    # two-character Chinese names are whole names
    assert places._pair("Central", "中環") == ("Central", "中環")


def test_bilingual_names_are_split_by_the_networks_own_function():
    assert places.split_name is network.split_name
    assert places._poi_names({"primary": "大快活 Fairwood"}) == ("Fairwood", "大快活")
    assert places._poi_names({"primary": "蘭香麵家"}) == ("", "蘭香麵家")
    assert places._poi_names({"primary": "Dead Poets HK"}) == ("Dead Poets HK", "")
    # English first is as often a branch as a translation: left whole
    assert places._poi_names({"primary": "Nara Thai Cuisine 海港城店"}) == ("", "Nara Thai Cuisine 海港城店")
    # the dash or comma between the two halves belongs to neither
    assert places._poi_names({"primary": "赤鬼居酒屋 - Akaoni"}) == ("Akaoni", "赤鬼居酒屋")
    assert places._poi_names({"primary": "利希慎音樂廳, CUHK"}) == ("CUHK", "利希慎音樂廳")


def test_stations_are_named_the_way_people_type_them():
    s = places._station_names
    assert s("Mass Transit Railway Central Station", "香港鐵路中環站", "station") == \
        ("Central Station", "中環站")
    assert s("Mass Transit Railway Central Station-D1 Access", "香港鐵路中環站-D1進出口",
             "station exit") == ("Central Station Exit D1", "中環站D1出口")
    assert s("MTR Lok Ma Chau Station-A Exit", "港鐵落馬洲站-A出口", "station exit") == \
        ("Lok Ma Chau Station Exit A", "落馬洲站A出口")
    # the hyphen inside a station's own name is not the exit's
    assert s("Mass Transit Railway AsiaWorld-Expo Station-A Access", "香港鐵路博覽館站-A進出口",
             "station exit") == ("AsiaWorld-Expo Station Exit A", "博覽館站A出口")
    assert s("Hong Kong West Kowloon Station", "香港西九龍站", "station") == \
        ("Hong Kong West Kowloon Station", "香港西九龍站")
    assert s("LR-Fung Tei", "輕鐵－鳳地", "light rail stop") == \
        ("Fung Tei Light Rail Stop", "鳳地輕鐵站")
    # OpenStreetMap's bare name is also the area's
    assert s("Central", "中環", "station") == ("Central Station", "中環站")


def test_a_bus_terminus_cannot_pass_for_the_place_it_serves():
    t = places._terminus_names
    assert t("Wong Tai Sin", "黃大仙") == ("Wong Tai Sin Bus Terminus", "黃大仙巴士總站")
    assert t("Tsing Yi Station", "青衣站") == ("Tsing Yi Station Bus Terminus", "青衣站巴士總站")
    assert t("Tin Fu Bus Terminus", "天富總站") == ("Tin Fu Bus Terminus", "天富總站")


def test_blocks_fold_into_their_estate():
    e = places._estate_stem
    assert e("Marina Cove Block J16", "匡湖居J16座") == ("Marina Cove", "匡湖居")
    assert e("City One Shatin Block 24", "沙田第一城第二十四座") == ("City One Shatin", "沙田第一城")
    assert e("Chungking Mansions Block A", "重慶大廈A座") == ("Chungking Mansions", "重慶大廈")
    assert e("Times Square Tower One", "時代廣場一座") == ("Times Square", "時代廣場")
    assert e("On Chiu House (Blk 6)", "安潮樓(第6座)") == ("On Chiu House", "安潮樓")
    assert e("Windsor Park Phase 1 Block 10", "雲景花園第一期第十座") == ("Windsor Park", "雲景花園")
    assert e("Regalia Bay C10", "富豪海灣C10") == ("Regalia Bay", "富豪海灣")
    assert e("Mount Nicholson Tower B", "") == ("Mount Nicholson", "")


def test_a_number_that_is_part_of_the_name_stays():
    e = places._estate_stem
    assert e("Wing On House", "永安大廈") == ("Wing On House", "永安大廈")
    assert e("Tower 535", "") == ("Tower 535", "")
    # only one of the two names carries a block: neither is cut
    assert e("One International Finance Centre", "國際金融中心一期") == \
        ("One International Finance Centre", "國際金融中心一期")
    assert e("Central Pier 8", "中環8號碼頭") == ("Central Pier 8", "中環8號碼頭")
    assert e("8 Finance Street", "金融街8號") == ("8 Finance Street", "金融街8號")


def test_a_bare_code_is_a_house_number_only_in_a_development():
    """'Regalia Bay C10' is a house; 'Central 88' and 'Stanley 11' are names."""
    houses = [(f"Regalia Bay C{i}", f"富豪海灣C{i}") for i in (10, 11, 12)]
    names = houses + [
        ("Central 88", "中環88"), ("Stanley 11", ""), ("Soho 38", ""), ("Soho 69", ""),
        ("Harbour View 1", "海濱大樓一座"),            # the Chinese spells the block out
        ("Chuen Yiu Terrace House Number 15", "傳耀臺第十五座"),
        ("Marina Cove Block J16", "匡湖居J16座"),
        ("Wing On House", "永安大廈")]
    assert places._fold_blocks(names) == [("Regalia Bay", "富豪海灣")] * 3 + [
        ("Central 88", "中環88"), ("Stanley 11", ""), ("Soho 38", ""), ("Soho 69", ""),
        ("Harbour View", "海濱大樓"),
        ("Chuen Yiu Terrace", "傳耀臺"),
        ("Marina Cove", "匡湖居"),
        ("Wing On House", "永安大廈")]
    # the same row twice is one house, not two
    assert places._fold_blocks([("Soho 38", "")] * 4) == [("Soho 38", "")] * 4


def test_a_building_identifier_carries_its_position():
    assert places._csuid_position("3715522478T20050430") == (837155.0, 822478.0)
    assert places._csuid_position("not an id") is None
    assert places._csuid_position(None) is None


# --------------------------------------------------------------------------
# positions and merging
# --------------------------------------------------------------------------
def test_grid_positions_are_hk1980():
    x, y = places._to_grid([CENTRAL[0]], [CENTRAL[1]])
    assert 833000 < x[0] < 835500 and 815000 < y[0] < 817500


def test_clipping_keeps_hong_kong_and_drops_shenzhen():
    x, y = places._to_grid([CENTRAL[0], LO_WU_SHENZHEN[0]], [CENTRAL[1], LO_WU_SHENZHEN[1]])
    assert places._inside(CENTRAL_BOX, x, y).tolist() == [True, False]
    df = places._layer([
        {"en": "Statue Square", "zh": "皇后像廣場", "group": "park", "lon": CENTRAL[0], "lat": CENTRAL[1]},
        {"en": "", "zh": "罗湖口岸", "group": "station", "lon": LO_WU_SHENZHEN[0], "lat": LO_WU_SHENZHEN[1]},
        {"en": "", "zh": "", "group": "park", "lon": CENTRAL[0], "lat": CENTRAL[1]},       # no name
        {"en": "Nowhere", "zh": "", "group": "park", "lon": None, "lat": None},            # no point
    ], CENTRAL_BOX)
    assert df["en"].tolist() == ["Statue Square"]
    assert set(places._LAYER_COLUMNS) <= set(df.columns)


def _frame(rows):
    """(en, zh, kind, x, y) rows as a layer frame near Central."""
    df = pd.DataFrame(rows, columns=["en", "zh", "group", "x", "y"])
    df["lon"], df["lat"] = CENTRAL
    return df


def test_thinning_leaves_one_record_per_estate_on_one_of_its_blocks():
    blocks = _frame([("Marina Cove", "匡湖居", "building", 1000.0 + 60 * i, 500.0) for i in range(9)]
                    + [("Marina Cove", "匡湖居", "building", 9000.0, 500.0),      # a namesake 8 km off
                       ("Hebe Villa", "", "building", 1000.0, 520.0),
                       ("Hebe Villa", "白沙灣別墅", "building", 1030.0, 520.0)])
    out = places._thin(blocks, 300.0)
    cove = out[out["en"] == "Marina Cove"].sort_values("x")
    assert cove["x"].tolist() == [1240.0, 9000.0]          # the middle block, and the namesake
    # the survivor takes the Chinese name its neighbour had
    assert out.loc[out["en"] == "Hebe Villa", "zh"].tolist() == ["白沙灣別墅"]
    # 'first' keeps the earliest record instead: the caller's largest outline
    assert places._thin(blocks, 300.0, pick="first").query("en == 'Marina Cove'")["x"].min() == 1000.0


def test_names_shared_by_many_sites_are_dropped_as_generic():
    df = _frame([("Jetty", "渡頭", "pier", 5000.0 * i, 0.0) for i in range(4)]
                + [("Blake Pier", "卜公碼頭", "pier", 100.0, 0.0),
                   ("Lok On Pai Pier", "", "pier", 0.0, 0.0), ("Lok On Pai Pier", "", "pier", 5000.0, 0.0)])
    assert sorted(places._drop_generic(df)["en"]) == ["Blake Pier", "Lok On Pai Pier", "Lok On Pai Pier"]


def test_the_higher_layer_wins_a_name_and_gains_the_other_language():
    osm = _frame([("Victoria Park", "", "park", 0.0, 0.0),
                  ("Chungking Mansion", "重慶大廈", "estate", 5000.0, 0.0)])
    gov = _frame([("Victoria Park", "維多利亞公園", "garden", 150.0, 0.0),      # the same park
                  ("Victoria Park", "維多利亞公園", "garden", 9000.0, 0.0),     # a namesake far away
                  ("Chungking Mansions", "重慶大廈", "building", 5010.0, 0.0),  # another English name
                  ("", "重慶大廈", "building", 5020.0, 0.0)])                  # nothing new
    out = places._merge_layers([osm, gov])
    got = sorted(zip(out["en"], out["zh"], out["group"], out["x"]))
    assert got == [
        ("Chungking Mansion", "重慶大廈", "estate", 5000.0),
        ("Chungking Mansions", "重慶大廈", "building", 5010.0),
        ("Victoria Park", "維多利亞公園", "garden", 9000.0),
        ("Victoria Park", "維多利亞公園", "park", 0.0),        # OSM's record, the government's Chinese
    ]


def test_an_institutions_name_is_not_left_on_its_outbuildings():
    campus = _frame([("The University of Hong Kong", "香港大學", "university", 0.0, 0.0)])
    blocks = _frame([("The University of Hong Kong", "香港大學", "estate", 7000.0, 0.0),
                     ("The University of Hong Kong", "", "building", 9000.0, 0.0),
                     ("Wing On House", "永安大廈", "building", 9000.0, 0.0)])
    out = places._merge_layers([campus, blocks])
    assert sorted(out["group"]) == ["building", "university"]
    assert "Wing On House" in set(out["en"])


def test_core_strips_a_trailing_city_name():
    assert places._core("Rosewood, Hong Kong") == "rosewood"
    assert places._core("Dead Poets HK") == "dead poets"
    assert places._core("Lan Fong Yuen - Hong Kong") == "lan fong yuen"
    assert places._core("蘭芳園 香港") == "蘭芳園"
    assert places._core("Star Ferry") == "star ferry"
    # a suffix that is part of the name is not a city suffix
    assert places._core("Bank of China") == "bank of china"
    assert places._head("Genki Sushi (Yau Tong)") == "genki sushi"


def test_support_counts_nearby_records_that_mention_the_name():
    names = pd.Series(["Times Square", "Times Square Food Forum", "Lane Crawford Times Square",
                       "Times Square", "Nowhere"])
    x = np.array([0.0, 60.0, -40.0, 9000.0, 20000.0])
    y = np.array([0.0, 30.0, 50.0, 4000.0, 0.0])
    sup = places._support(names, x, y, names, x, y)
    assert sup[0] == 2          # the real complex: two neighbours mention it
    assert sup[3] == 0          # the stray copy across town: none


def test_variant_pruning_keeps_different_kinds_and_distant_namesakes():
    df = pd.DataFrame({
        "name": ["Lan Fong Yuen", "Lan Fong Yuen, Hong Kong", "Lan Fong Yuen Bakery",
                 "蘭芳園", "蘭芳園(中環店)", "Sing Heung Yuen", "Sing Heung Yuen - Annex"],
        "group": ["food", "food", "shop", "food", "food", "food", "food"],
        "x": [0.0, 3000.0, 100.0, 0.0, 200.0, 20000.0, 26000.0],
        "y": [0.0, 2000.0, 100.0, 0.0, 100.0, 0.0, 0.0],
        "conf": [0.97, 0.92, 0.99, 0.97, 0.91, 0.98, 0.9],
        "support": [3, 0, 0, 3, 0, 1, 0],
    })
    kept = set(places._prune_variants(df)["name"])
    assert "Lan Fong Yuen" in kept and "蘭芳園" in kept
    assert "Lan Fong Yuen, Hong Kong" not in kept        # city suffix, any distance
    assert "Lan Fong Yuen Bakery" in kept                # a different kind of place
    assert "蘭芳園(中環店)" not in kept                    # a Chinese name is cut at its bracket
    assert "Sing Heung Yuen - Annex" in kept             # a same-kind variant, but 6 km away


def test_the_poi_feed_gives_way_to_names_the_better_layers_hold(monkeypatch, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from shapely import to_wkb
    from shapely.geometry import Point

    def row(name, node, conf=0.95):
        return {"names": {"primary": name},
                "taxonomy": {"primary": node, "hierarchy": ["somewhere", node]},
                "confidence": conf, "operating_status": "open",
                "geometry": to_wkb(Point(*CENTRAL))}
    path = tmp_path / "places.parquet"
    pq.write_table(pa.Table.from_pylist([
        row("Times Square", "shopping_mall"),             # a better layer has it
        row("Langham Place 朗豪坊", "shopping_mall"),      # ... and this, English first
        row("Black Bar 尖沙咀", "food_and_drink"),          # a branch in an area, not the area
        row("Lan Fong Yuen", "food_and_drink"),
        row("Quiet Corner", "food_and_drink", conf=0.5),  # the feed is not sure of it
        row("Some Bank", "bank"),                         # not a kind that is kept
    ]), path)
    monkeypatch.setattr(places, "PLACES_PARQUET", path)
    taken = {places._key(n) for n in ("Times Square", "Langham Place", "朗豪坊",
                                      "Tsim Sha Tsui", "尖沙咀")}
    kept = places.build_poi(taken, CENTRAL_BOX)
    assert sorted(kept["name"]) == ["Black Bar 尖沙咀", "Lan Fong Yuen"]
    assert kept["group"].tolist() == ["food", "food"]


# --------------------------------------------------------------------------
# the whole index, from files written here
# --------------------------------------------------------------------------
def _write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def _point(lon, lat, **props):
    return {"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": props}


def _no_raw_files(monkeypatch, tmp_path):
    """Point every raw input of the index at a path that does not exist."""
    for name in ("PLACE_NAME_POINTS_GEOJSON", "PLACE_NAME_NAMES_GEOJSON", "IGEOCOM_ZIP",
                 "BUILDING_NAMES_GEOJSON", "PLACES_PARQUET", "ADDRESSES_PATH"):
        monkeypatch.setattr(places, name, tmp_path / f"absent_{name}")
    monkeypatch.setattr(places, "BASE_PARQUETS", {"land_use": tmp_path / "absent_base"})
    monkeypatch.setattr(places, "TD_STOPS_JSON", {"tram": tmp_path / "absent_tram"})
    monkeypatch.setattr(places, "_sar_boundary", lambda: CENTRAL_BOX)
    places._load_als.cache_clear()


def test_missing_raw_files_raise_file_not_found(monkeypatch, tmp_path):
    _no_raw_files(monkeypatch, tmp_path)
    with pytest.raises(FileNotFoundError, match="no place data"):
        places.build_places()
    with pytest.raises(FileNotFoundError, match="Address Lookup Service"):
        places.build_addresses()
    places._load_als.cache_clear()


def test_the_boundary_is_required(monkeypatch, tmp_path):
    monkeypatch.setattr(places, "DIVISIONS_PARQUET", tmp_path / "absent.parquet")
    with pytest.raises(FileNotFoundError, match="SAR boundary"):
        places._sar_boundary()


def test_the_index_is_built_from_whatever_sources_are_on_disk(monkeypatch, tmp_path):
    """Gazetteer, building names and tram stops only: three small files."""
    _no_raw_files(monkeypatch, tmp_path)
    pts, names = tmp_path / "points.geojson", tmp_path / "names.geojson"
    _write(pts, {"features": [
        _point(*CENTRAL, GEO_NAME_ID=1, PLACE_TYPE="Area"),
        _point(*SHA_TIN, GEO_NAME_ID=2, PLACE_TYPE="Town"),               # outside the box
        _point(114.1600, 22.2850, GEO_NAME_ID=3, PLACE_TYPE="Strait")]})  # open water: not kept
    _write(names, {"features": [
        {"properties": {"GEO_NAME_ID": 1, "NAME_EN": "Central District", "NAME_TC": "中環",
                        "NAME_STATUS": "Official"}},
        {"properties": {"GEO_NAME_ID": 1, "NAME_EN": "Chung Wan", "NAME_TC": None,
                        "NAME_STATUS": "Alias"}},
        {"properties": {"GEO_NAME_ID": 2, "NAME_EN": "Sha Tin", "NAME_TC": "沙田",
                        "NAME_STATUS": "Official"}},
        {"properties": {"GEO_NAME_ID": 3, "NAME_EN": "Sulphur Channel", "NAME_TC": "硫磺海峽",
                        "NAME_STATUS": "Official"}}]})
    buildings = tmp_path / "buildings.geojson"
    _write(buildings, {"features": [
        {"properties": {"BuildingCSUID": f"34{i:03d}16000T20050430", "NameStatus": "Default",
                        "BuildingNameEN": f"Exchange Square Block {i}",
                        "BuildingNameTC": f"交易廣場第{'一二三'[i - 1]}座"}} for i in (1, 2, 3)]
        + [{"properties": {"BuildingCSUID": "3400016000T20050430", "NameStatus": "Default",
                           "BuildingNameEN": "EXCHANGE SQUARE SUBSTATION", "BuildingNameTC": "交易廣場變電站"}},
           {"properties": {"BuildingCSUID": "3410016100T20050430", "NameStatus": "Proposed",
                           "BuildingNameEN": "Not Yet Built", "BuildingNameTC": None}}]})
    tram = tmp_path / "tram.json"
    tram.write_text("\ufeff" + json.dumps({"features": [
        _point(114.1575, 22.2826, stopId=1, stopNameE="Pedder Street (West bound)", stopNameC="畢打街 (西行方向)"),
        _point(114.1577, 22.2824, stopId=2, stopNameE="Pedder Street (East bound)", stopNameC="畢打街 (東行方向)"),
        _point(114.1577, 22.2824, stopId=2, stopNameE="Pedder Street (East bound)", stopNameC="畢打街 (東行方向)"),
    ]}, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(places, "PLACE_NAME_POINTS_GEOJSON", pts)
    monkeypatch.setattr(places, "PLACE_NAME_NAMES_GEOJSON", names)
    monkeypatch.setattr(places, "BUILDING_NAMES_GEOJSON", buildings)
    monkeypatch.setattr(places, "TD_STOPS_JSON", {"tram": tram})

    index = places.build_places()
    assert set(index) == {"names", "alt", "group", "groups", "group_rank", "lon", "lat"}
    rows = sorted(zip(index["names"], index["alt"], (index["groups"][g] for g in index["group"])))
    assert rows == [
        ("Central", "中環", "area"),                        # 'Central District', as it is said
        ("Chung Wan", "中環", "area"),                      # the alias, with the official Chinese
        ("Exchange Square", "交易廣場", "building"),         # three blocks, one record; no substation
        ("Pedder Street Tram Stop", "畢打街電車站", "tram stop"),   # both directions, one stop
    ]
    assert index["names"] == sorted(index["names"])
    assert dict(zip(index["groups"], index["group_rank"])) == {"area": 0, "building": 2, "tram stop": 1}
    assert all(round(v, 5) == v for v in index["lon"] + index["lat"])


def _als_feature(lon, lat, district, street=None, village=None, number=None, to=None,
                 loc=None, estate=None):
    """One Address Lookup Service record: names are (English, Chinese) pairs."""
    eng, chi = {"EngDistrict": district[0]}, {"ChiDistrict": district[1]}
    for key, name in (("Street", street), ("Village", village)):
        if name:
            e = {f"{key}Name": name[0]}
            c = {f"{key}Name": name[1]}
            if number:
                e["BuildingNoFrom"] = c["BuildingNoFrom"] = number
            if to:
                e["BuildingNoTo"] = c["BuildingNoTo"] = to
            if loc:
                e["LocationName"], c["LocationName"] = loc
            eng[f"Eng{key}"], chi[f"Chi{key}"] = e, c
    if estate:
        eng["EngEstate"], chi["ChiEstate"] = {"EstateName": estate[0]}, {"EstateName": estate[1]}
    feat = _point(lon, lat, Address={"PremisesAddress": {
        "EngPremisesAddress": eng, "ChiPremisesAddress": chi}})
    if lon is None:
        feat["geometry"] = {"type": "Point"}
    return feat


@pytest.fixture
def small_als(monkeypatch, tmp_path):
    ytm, kc = ("YAU TSIM MONG DISTRICT", "油尖旺區"), ("KOWLOON CITY DISTRICT", "九龍城區")
    tp, yl = ("TAI PO DISTRICT", "大埔區"), ("YUEN LONG DISTRICT", "元朗區")
    pe = ("PRINCE EDWARD ROAD WEST", "太子道西")
    files = {
        "als_addresses_(yau_tsim_mong_district).geojson": [
            _als_feature(114.17173, 22.29964, ytm, street=("NATHAN ROAD", "彌敦道"), number="123"),
            _als_feature(114.17150, 22.29800, ytm, street=("NATHAN ROAD", "彌敦道"), number="100A"),
            _als_feature(114.17152, 22.29802, ytm, street=("NATHAN ROAD", "彌敦道"), number="100"),
            _als_feature(114.17200, 22.30100, ytm, street=("NATHAN ROAD", "彌敦道"), number="132", to="134"),
            _als_feature(114.17000, 22.32400, ytm, street=pe, number="100"),
            _als_feature(114.17300, 22.29700, ytm, street=("QUEEN'S ROAD CENTRAL", "皇后大道中"), number="9"),
            _als_feature(None, None, ytm, street=("NATHAN ROAD", "彌敦道"), number="1"),      # no point
            # a stray coordinate west of the study box: left out, and the rest still packs
            _als_feature(113.5, 22.3, ytm, street=("NATHAN ROAD", "彌敦道"), number="999"),
            _als_feature(114.17300, 22.29700, ytm, estate=("CHUNGKING MANSIONS", "重慶大廈")),   # no street
        ],
        "als_addresses_(kowloon_city_district).geojson": [
            # the same street carrying on into the next district
            _als_feature(114.17500, 22.32450, kc, street=pe, number="200"),
        ],
        "als_addresses_(tai_po_district).geojson": [
            _als_feature(114.15468, 22.47015, tp, village=("TAI WO", "大窩"), number="5", loc=("TAI PO", "大埔")),
        ],
        "als_addresses_(yuen_long_district).geojson": [
            # another village of the same name, 10 km away
            _als_feature(114.07485, 22.41884, yl, village=("TAI WO", "大窩"), number="5",
                         loc=("PAT HEUNG, YUEN LONG", "元朗 八鄉")),
        ],
        # every flat of the public rental estates: must not be read
        "als_addresses_3d_(public_rental_housing).geojson": "not json at all",
    }
    path = tmp_path / "als.zip"
    with zipfile.ZipFile(path, "w") as z:
        for name, feats in files.items():
            z.writestr(name, feats if isinstance(feats, str) else json.dumps({"features": feats},
                                                                             ensure_ascii=False))
    monkeypatch.setattr(places, "ADDRESSES_PATH", path)
    places._load_als.cache_clear()
    yield path
    places._load_als.cache_clear()


def test_addresses_pack_streets_and_villages_in_two_languages(small_als):
    a = places.build_addresses()
    assert a["streets"] == ["Nathan Road", "Prince Edward Road West", "Queen's Road Central",
                            "Tai Wo (Pat Heung)", "Tai Wo (Tai Po)"]
    assert a["streets_zh"] == ["彌敦道", "太子道西", "皇后大道中", "大窩 (八鄉)", "大窩 (大埔)"]
    assert a["origin"] == [STUDY_BBOX[0], STUDY_BBOX[2]]
    for k in ("street", "number", "lon", "lat"):
        assert a[k].dtype == np.dtype("<u2"), k
    rows = list(zip(a["street"].tolist(), a["number"].tolist()))
    assert rows == sorted(rows)
    # Nathan Road: 100 once (the plain number, not 100A), 123, and both ends of 132-134
    assert [n for s, n in rows if s == 0] == [100, 123, 132, 134]
    i = rows.index((0, 100))
    assert a["lon"][i] == round((114.17152 - STUDY_BBOX[0]) / 1e-5)
    assert a["lat"][i] == round((22.29802 - STUDY_BBOX[2]) / 1e-5)
    # one street across two districts; two villages of one name kept apart
    assert [n for s, n in rows if s == 1] == [100, 200]
    assert [n for s, n in rows if s in (3, 4)] == [5, 5]


def test_a_shared_names_locality_is_one_place_in_both_languages(monkeypatch, tmp_path):
    """Two localities tie on one street: the label must not mix them.

    Hiram's Highway had as many records in Ho Chung as in Ta Ku Ling and
    came out as "(Ho Chung)" in English beside "(打鼓嶺)" in Chinese.
    """
    sk, yl = ("SAI KUNG DISTRICT", "西貢區"), ("YUEN LONG DISTRICT", "元朗區")
    hh = ("HIRAM'S HIGHWAY", "西貢公路")
    ho_chung, ta_ku_ling = ("HO CHUNG, SAI KUNG", "西貢 蠔涌"), ("TA KU LING, SAI KUNG", "西貢 打鼓嶺")
    files = {
        "als_addresses_(sai_kung_district).geojson": [
            _als_feature(114.2512, 22.3530, sk, street=hh, number="1", loc=ho_chung),
            _als_feature(114.2513, 22.3531, sk, street=hh, number="3", loc=ho_chung),
            _als_feature(114.2530, 22.3470, sk, street=hh, number="5", loc=ta_ku_ling),
            _als_feature(114.2531, 22.3471, sk, street=hh, number="7", loc=ta_ku_ling)],
        # a namesake across the territory, so that the name needs a locality at all
        "als_addresses_(yuen_long_district).geojson": [
            _als_feature(114.0300, 22.4450, yl, street=hh, number="1")],
    }
    path = tmp_path / "als.zip"
    with zipfile.ZipFile(path, "w") as z:
        for name, feats in files.items():
            z.writestr(name, json.dumps({"features": feats}, ensure_ascii=False))
    monkeypatch.setattr(places, "ADDRESSES_PATH", path)
    places._load_als.cache_clear()
    try:
        a = places.build_addresses()
    finally:
        places._load_als.cache_clear()
    assert dict(zip(a["streets"], a["streets_zh"])) == {
        "Hiram's Highway (Ho Chung)": "西貢公路 (蠔涌)",
        "Hiram's Highway (Yuen Long)": "西貢公路 (元朗區)"}


def test_estates_come_from_the_address_file(small_als):
    df = places.build_estates(box(800000, 800000, 870000, 850000))
    assert list(zip(df["en"], df["zh"], df["group"])) == [("Chungking Mansions", "重慶大廈", "estate")]


def test_stretches_of_a_street_join_only_when_they_are_one_street():
    def unit(zh, x0, numbers):
        pts = np.array([[x0 + 10.0 * n, 0.0] for n in numbers])
        return {"zh": zh, "pts": pts, "nums": {n: tuple(p) for n, p in zip(numbers, pts)}}
    # carrying on with new numbers 200 m further: one street
    assert places._join_stretches([unit("太子道西", 0, [1, 2, 3]), unit("太子道西", 200, [4, 5])]) == [0, 0]
    # the same numbers 5 km away: another street of the same name
    assert places._join_stretches([unit("大窩", 0, [1, 2, 3]), unit("大窩", 5000, [1, 2, 3])]) == [0, 1]
    # next to each other but under different Chinese names: two roads end to end
    assert places._join_stretches([unit("青山道", 0, [1, 2]), unit("青山公路", 100, [3, 4])]) == [0, 1]
    # a variant character does not make a second street
    assert places._join_stretches([unit("恆富街", 0, [1, 2]), unit("恒富街", 100, [3, 4])]) == [0, 0]
    assert places._same_zh("麒麟圍第一巷", "麒麟圍1巷") and not places._same_zh("大埔道", "大埔公路")


def test_values_that_do_not_fit_sixteen_bits_are_an_error_not_a_clip():
    assert places._fits_uint16(np.array([0, 65535]), "offset").dtype == np.dtype("<u2")
    with pytest.raises(ValueError, match="longitude offset"):
        places._fits_uint16(np.array([10, 65536]), "longitude offset")
    with pytest.raises(ValueError):
        places._fits_uint16(np.array([-1, 5]), "offset")


# --------------------------------------------------------------------------
# the real index
# --------------------------------------------------------------------------
PLACE_FILES = [PLACE_NAME_POINTS_GEOJSON, PLACE_NAME_NAMES_GEOJSON, IGEOCOM_ZIP,
               BUILDING_NAMES_GEOJSON, PLACES_PARQUET, ADDRESSES_PARQUET, DIVISIONS_PARQUET,
               *BASE_PARQUETS.values(), *TD_STOPS_JSON.values()]
needs_places = pytest.mark.skipif(not all(p.exists() for p in PLACE_FILES),
                                  reason="place sources not downloaded")
needs_addresses = pytest.mark.skipif(not ADDRESSES_PARQUET.exists(),
                                     reason="Address Lookup Service file not downloaded")


@pytest.fixture(scope="module")
def index():
    places._load_als.cache_clear()
    return places.build_places()


@pytest.fixture(scope="module")
def records(index):
    """{(name, kind): (alt, lon, lat)} -- the first record of each."""
    out = {}
    for n, a, g, lo, la in zip(index["names"], index["alt"], index["group"],
                               index["lon"], index["lat"]):
        out.setdefault((n, index["groups"][g]), (a, lo, la))
    return out


def _near(got, lon, lat, tol=0.004):
    return abs(got[1] - lon) < tol and abs(got[2] - lat) < tol


@needs_places
def test_place_index_is_compact_sorted_and_inside_hong_kong(index):
    n = len(index["names"])
    assert 30_000 < n < 100_000
    assert len(index["alt"]) == len(index["group"]) == len(index["lon"]) == len(index["lat"]) == n
    assert max(index["group"]) < len(index["groups"]) == len(index["group_rank"])
    assert index["groups"] == sorted(index["groups"])
    assert all(g == g.lower() for g in index["groups"])
    assert set(index["group_rank"]) == {0, 1, 2, 3}
    rank = dict(zip(index["groups"], index["group_rank"]))
    assert rank["area"] == rank["station"] == 0 and rank["food"] == rank["shop"] == 3
    assert index["names"] == sorted(index["names"])
    assert all(index["names"]) and all(isinstance(a, str) for a in index["alt"])
    # inside the SAR itself, not just the bounding box that takes in Shenzhen
    x, y = places._to_grid(index["lon"], index["lat"])
    assert places._inside(places._sar_boundary().buffer(5.0), x, y).all()
    assert min(index["lon"]) >= STUDY_BBOX[0] and max(index["lon"]) <= STUDY_BBOX[1]
    assert min(index["lat"]) >= STUDY_BBOX[2] and max(index["lat"]) <= STUDY_BBOX[3]
    # most records carry both languages, and the page's budget is kept
    assert sum(1 for a in index["alt"] if a) > 0.6 * n
    packed = gzip.compress(json.dumps(index, separators=(",", ":"), ensure_ascii=False).encode())
    assert len(packed) < 1_500_000


@needs_places
def test_names_are_normalised(index):
    full_width = [n for n in index["names"] + index["alt"] if any("\uff01" <= c <= "\uff5e" for c in n)]
    assert not full_width, full_width[:5]
    # two-character Chinese names are kept
    assert sum(1 for a in index["alt"] if len(a) == 2) > 500


@needs_places
def test_areas_come_from_the_gazetteer(records):
    alt, lon, lat = records[("Mong Kok", "area")]
    assert alt == "旺角" and _near((alt, lon, lat), 114.1715, 22.3201)
    assert records[("Central", "area")][0] == "中環"
    assert records[("Tsim Sha Tsui", "area")][0] == "尖沙咀"
    assert records[("Lion Rock", "peak")][0] == "獅子山"


@needs_places
def test_every_mtr_station_is_there_once_with_its_exits(index, records):
    got = records[("Central Station", "station")]
    assert got[0] == "中環站" and _near(got, 114.1577, 22.2822)
    assert records[("Central Station Exit A", "station exit")][0] == "中環站A出口"
    assert _near(records[("Central Station Exit A", "station exit")], 114.158, 22.2825)
    assert records[("Mong Kok Station", "station")][0] == "旺角站"
    assert records[("Hong Kong West Kowloon Station", "station")][0] == "香港西九龍站"
    assert ("Siu Hong Light Rail Stop", "light rail stop") in records
    kinds = [index["groups"][g] for g in index["group"]]
    stations = [n for n, k in zip(index["names"], kinds) if k == "station"]
    # 98 MTR stations and the high-speed rail terminus when this was written
    # (research_places.txt).  The railway grows and the file is refreshed
    # every quarter, so the count is a floor with a little room above it.
    assert len(stations) == len(set(stations))
    assert 99 <= len(stations) <= 110
    assert not [n for n in stations if "Mass Transit" in n or n.startswith("MTR ")]
    assert kinds.count("station exit") > 600
    # a bus terminus beside a station does not take the station's name
    assert ("Tsing Yi Station", "bus terminus") not in records


@needs_places
def test_estates_buildings_and_complexes_are_found_under_both_names(records):
    assert records[("Mei Foo Sun Chuen", "estate")][0] == "美孚新邨"
    assert _near(records[("Mei Foo Sun Chuen", "estate")], 114.139, 22.337)
    got = records[("International Finance Centre", "complex")]
    assert got[0] == "國際金融中心" and _near(got, 114.1585, 22.285)
    got = records[("Times Square", "complex")]
    assert got[0] == "時代廣場" and _near(got, 114.182, 22.278)
    got = records[("Chungking Mansions", "building")]
    assert got[0] == "重慶大廈" and _near(got, 114.1727, 22.2964)
    got = records[("Victoria Park", "park")]
    assert got[0] == "維多利亞公園" and _near(got, 114.1885, 22.282)
    # the blocks of an estate are one record
    assert not [k for k in records if k[0].startswith("Marina Cove Block")]
    assert ("Marina Cove", "estate") in records
    # ... but a number that ends a building's name stays on it: there is no
    # building called Stanley in Central
    assert ("Stanley 11", "building") in records and ("Stanley", "building") not in records
    assert records[("Central 88", "building")][0] == "中環88"


@needs_places
def test_an_institutions_name_leads_to_the_institution(index, records):
    hits = [index["groups"][g] for n, g in zip(index["names"], index["group"])
            if n == "The University of Hong Kong"]
    assert hits == ["university"]
    assert _near(records[("The University of Hong Kong", "university")], 114.1375, 22.283)
    assert ("Hong Kong International Airport", "airport") in records
    assert ("Hong Kong International Airport", "estate") not in records
    # the short forms people type
    assert _near(records[("IFC", "complex")], 114.1585, 22.285)
    assert ("HKU", "university") in records and ("TST", "area") in records


@needs_places
def test_transit_stops_and_the_poi_feed(index, records):
    assert ("Pedder Street Tram Stop", "tram stop") in records
    assert records[("Kennedy Road Peak Tram Station", "peak tram")][0] == "堅尼地道山頂纜車站"
    assert any(k == "pier" and "Star Ferry" in n for n, k in records)
    # the scenic spots of iGeoCom, which no other layer names
    got = records[("The Central to Mid-Levels Escalator", "landmark")]
    assert got[0] == "中環至半山自動扶梯" and _near(got, 114.1549, 22.2836)
    # the feed's copies of a name a better layer holds are not kept, whichever
    # language leads (one "Langham Place 朗豪坊" sat over 4 km from Langham Place)
    assert not [n for n in index["names"] if n in ("Langham Place 朗豪坊", "Elements 圓方")]
    kinds = [index["groups"][g] for g in index["group"]]
    assert kinds.count("food") > 3000 and kinds.count("shop") > 1000
    # chains are left out: one branch of 480 is not a destination
    lowered = {n.lower() for n in index["names"]}
    assert not lowered & {"7-eleven", "starbucks", "circle k", "mcdonald's"}


@needs_addresses
def test_addresses_pack_into_sorted_uint16_offsets():
    a = places.build_addresses()
    n = len(a["number"])
    assert n > 100_000
    assert len(a["streets"]) == len(a["streets_zh"]) == len(set(a["streets"])) > 4000
    assert a["streets"] == sorted(a["streets"])
    assert a["street"].dtype == np.dtype("<u2") and a["number"].dtype == np.dtype("<u2")
    assert a["lon"].dtype == np.dtype("<u2") and a["lat"].dtype == np.dtype("<u2")
    assert len(a["street"]) == len(a["lon"]) == len(a["lat"]) == n
    assert int(a["street"].max()) == len(a["streets"]) - 1
    # sorted by street then number, so the browser can binary-search
    key = a["street"].astype(np.int64) * 100_000 + a["number"]
    assert np.all(np.diff(key) > 0)
    assert a["origin"] == [STUDY_BBOX[0], STUDY_BBOX[2]]
    lon = a["origin"][0] + a["lon"] * 1e-5
    lat = a["origin"][1] + a["lat"] * 1e-5
    assert lon.min() >= STUDY_BBOX[0] and lon.max() <= STUDY_BBOX[1]
    assert lat.min() >= STUDY_BBOX[2] and lat.max() <= STUDY_BBOX[3]

    zh = dict(zip(a["streets"], a["streets_zh"]))
    assert zh["Nathan Road"] == "彌敦道"
    assert zh["Queen's Road Central"] == "皇后大道中"
    assert zh["D'Aguilar Street"] == "德己立街"
    assert zh["A Kung Ngam"] == "阿公岩"                    # a village address
    assert sum(1 for v in zh.values() if v) > 0.99 * len(zh)
    # 123 Nathan Road, checked against the service's own answer in the research
    i = a["streets"].index("Nathan Road")
    rows = np.flatnonzero(a["street"] == i)
    k = rows[np.searchsorted(a["number"][rows], 123)]
    assert a["number"][k] == 123
    assert abs(lon[k] - 114.17173) < 2e-5 and abs(lat[k] - 22.29964) < 2e-5
    # Castle Peak Road is a dozen sections, each numbered from 1
    sections = [s for s in a["streets"] if s.startswith("Castle Peak Road (")]
    assert len(sections) >= 10 and "Castle Peak Road" not in zh
    assert zh["Castle Peak Road (Ting Kau)"] == "青山公路 (汀九段)"
    assert zh["Castle Peak Road (Sham Shui Po)"].startswith("青山道")
