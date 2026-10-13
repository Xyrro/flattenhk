"""Tests for access-rule interpretation and graph construction."""
import numpy as np
import pytest

from hk_flat_routes.network import _as_dict, _as_list, _flags, evaluate_access


def rule(access_type, mode=None, heading=None, during=None, between=None,
         recognized=None, using=None):
    return {"access_type": access_type, "between": between,
            "when": {"mode": mode, "heading": heading, "during": during,
                     "recognized": recognized, "using": using}}


def test_no_restrictions_means_allowed():
    r = evaluate_access(None, "foot")
    assert r["allowed"] and not r["oneway_forward_only"]


def test_one_way_is_a_backward_denial_for_all_modes():
    res = evaluate_access([rule("denied", heading="backward")], "bicycle")
    assert res["allowed"] and res["oneway_forward_only"]


def test_contraflow_bike_lane_cancels_the_one_way():
    rules = [rule("denied", heading="backward"),
             rule("allowed", mode=["bicycle"], heading="backward")]
    assert not evaluate_access(rules, "bicycle")["oneway_forward_only"]
    # ... but motor-vehicle-style one-way still stands for other modes
    assert evaluate_access([rule("denied", heading="backward")],
                           "foot")["oneway_forward_only"]


def test_a_one_way_drawn_against_its_direction_closes_forward_travel():
    """OpenStreetMap's oneway=-1 arrives as a denial of the forward heading."""
    res = evaluate_access([rule("denied", heading="forward")], "bicycle")
    assert res["allowed"] and res["oneway_backward_only"]
    assert not res["oneway_forward_only"]
    assert not evaluate_access([rule("denied", heading="backward")],
                               "bicycle")["oneway_backward_only"]
    # a permit for the closed direction lifts it, as it does the other way
    rules = [rule("denied", heading="forward"),
             rule("allowed", mode=["bicycle"], heading="forward")]
    assert not evaluate_access(rules, "bicycle")["oneway_backward_only"]


def test_mode_specific_denial_only_affects_that_mode():
    rules = [rule("denied", mode=["bicycle"])]
    assert not evaluate_access(rules, "bicycle")["allowed"]
    assert evaluate_access(rules, "foot")["allowed"]


def test_designated_counts_as_permitted():
    assert evaluate_access([rule("designated", mode=["bicycle"])],
                           "bicycle")["allowed"]


def test_private_access_is_not_routable_for_through_travel():
    r = evaluate_access([rule("allowed", recognized=["as_private"])], "foot")
    assert not r["allowed"] and r["restricted"]


def test_destination_only_access_is_not_routable():
    r = evaluate_access([rule("allowed", using=["at_destination"])], "foot")
    assert not r["allowed"]


def test_time_conditional_rules_are_ignored():
    rules = [rule("denied", mode=["bicycle"], during="Mo-Fr 07:00-09:00")]
    assert evaluate_access(rules, "bicycle")["allowed"]


def test_partial_segment_rules_do_not_veto_the_whole_edge():
    rules = [rule("denied", mode=["foot"], between=[0.0, 0.4])]
    r = evaluate_access(rules, "foot")
    assert r["allowed"] and r["partial_rules"]


def test_later_rule_overrides_an_earlier_general_one():
    rules = [rule("denied"), rule("allowed", mode=["foot"])]
    assert evaluate_access(rules, "foot")["allowed"]


def test_a_mode_specific_permit_outranks_a_later_general_restriction():
    """The Slow Street shape, exactly as Overture carries it for Cabrillo
    Street: an all-modes destination-only rule comes *after* the explicit
    foot and bicycle permits and must not override them."""
    rules = [rule("allowed", mode=["foot"]),
             rule("designated", mode=["bicycle"]),
             rule("allowed", mode=["motor_vehicle"], using=["at_destination"]),
             rule("allowed", using=["at_destination"])]
    for mode in ("foot", "bicycle"):
        r = evaluate_access(rules, mode)
        assert r["allowed"], mode
        assert not r["restricted"], mode


def test_a_general_restriction_still_applies_without_a_mode_permit():
    rules = [rule("allowed", mode=["motor_vehicle"], using=["at_destination"]),
             rule("allowed", using=["at_destination"])]
    assert not evaluate_access(rules, "foot")["allowed"]


def test_a_mode_specific_denial_outranks_an_earlier_general_permit():
    rules = [rule("allowed"), rule("denied", mode=["foot"])]
    assert not evaluate_access(rules, "foot")["allowed"]
    assert evaluate_access(rules, "bicycle")["allowed"]


def test_motor_vehicle_denial_does_not_block_walking_or_cycling():
    rules = [rule("allowed", mode=["foot", "bicycle"]),
             rule("denied", mode=["motor_vehicle"])]
    assert evaluate_access(rules, "foot")["allowed"]
    assert evaluate_access(rules, "bicycle")["allowed"]


# ------------------------------------------- Arrow / numpy coercion helpers
def test_nested_arrow_columns_arrive_as_ndarrays():
    """``to_pandas`` turns Arrow lists into object ndarrays, whose truthiness
    raises; every nested access must go through the coercion helpers."""
    arr = np.array([rule("denied", heading="backward")], dtype=object)
    res = evaluate_access(arr, "bicycle")
    assert res["oneway_forward_only"]


def test_as_list_handles_none_nan_scalars_and_arrays():
    assert _as_list(None) == []
    assert _as_list(float("nan")) == []
    assert _as_list([1, 2]) == [1, 2]
    assert _as_list(np.array([1, 2])) == [1, 2]


def test_as_dict_rejects_non_mappings():
    assert _as_dict(None) == {}
    assert _as_dict(float("nan")) == {}
    assert _as_dict({"a": 1}) == {"a": 1}


def test_flags_collects_values_from_nested_arrays():
    rf = np.array([{"values": np.array(["is_bridge"], dtype=object)},
                   {"values": ["is_link"]}], dtype=object)
    assert _flags(rf) == {"is_bridge", "is_link"}


def test_flags_of_empty_input():
    assert _flags(None) == set()


# ------------------------------------------------ rules for part of a segment
def test_a_partial_rule_applies_where_it_covers_and_nowhere_else():
    """Overture joins ways into one segment and keeps a ban on the tunnel in
    the middle of it as a range.  Asked about a position, the rule applies
    there and not on the street either side."""
    rules = [rule("denied", mode=["foot"], between=[0.2, 0.8])]
    assert not evaluate_access(rules, "foot", at=0.5)["allowed"]
    assert evaluate_access(rules, "foot", at=0.1)["allowed"]
    assert evaluate_access(rules, "foot", at=0.9)["allowed"]
    assert evaluate_access(rules, "bicycle", at=0.5)["allowed"]


def test_a_vehicle_dimension_limit_bars_nobody_on_foot():
    """A headroom or weight limit arrives as a denial with no mode."""
    r = {"access_type": "denied", "between": None,
         "when": {"mode": None, "heading": None, "during": None,
                  "recognized": None, "using": None,
                  "vehicle": [{"dimension": "height", "comparison": "greater_than",
                               "value": 4.0, "unit": "m"}]}}
    assert evaluate_access([r], "foot")["allowed"]
    assert evaluate_access([r], "bicycle")["allowed"]


def test_no_vehicles_means_no_bicycles():
    rules = [rule("denied", mode=["vehicle"])]
    assert not evaluate_access(rules, "bicycle")["allowed"]
    assert evaluate_access(rules, "foot")["allowed"]


def test_a_permissive_way_is_open_to_the_public():
    """Wan Chai Gap Road above Kennedy Road, as Overture carries it: no
    bicycles, private to motor vehicles, permissive to everyone else."""
    rules = [rule("denied", mode=["bicycle"]),
             rule("allowed", mode=["motor_vehicle"], recognized=["as_private"]),
             rule("allowed", recognized=["as_permitted"])]
    assert evaluate_access(rules, "foot")["allowed"]
    assert not evaluate_access(rules, "bicycle")["allowed"]
    assert not evaluate_access([rule("allowed", recognized=["as_private"])], "foot")["allowed"]


def test_only_a_rule_naming_the_mode_is_an_explicit_permit():
    """Bicycles need one on a Hong Kong footway; a general permit is not it."""
    assert evaluate_access([rule("allowed", mode=["bicycle"])], "bicycle")["explicit"]
    assert evaluate_access([rule("designated", mode=["bicycle"])], "bicycle")["explicit"]
    assert not evaluate_access([rule("allowed")], "bicycle")["explicit"]
    assert not evaluate_access(None, "bicycle")["explicit"]


def test_flags_are_read_per_stretch():
    from hk_flat_routes.network import _flag_ranges, _flags_at
    rf = [{"values": ["is_bridge"], "between": [0.4, 0.6]},
          {"values": ["is_link"], "between": None}]
    ranges = _flag_ranges(rf)
    assert _flags_at(ranges, 0.5) == {"is_bridge", "is_link"}
    assert _flags_at(ranges, 0.1) == {"is_link"}


# ------------------------------------------------------------ bilingual names
def test_names_split_into_english_and_chinese():
    from hk_flat_routes.network import split_name
    assert split_name({"primary": "彌敦道 Nathan Road", "common": None}) == \
        ("Nathan Road", "彌敦道")
    assert split_name({"primary": "皇后大道中 Queen's Road Central",
                       "common": [("en", "Queen's Road Central"), ("zh", "皇后大道中")]}) == \
        ("Queen's Road Central", "皇后大道中")
    # digits and dashes inside the Chinese part stay with it
    assert split_name({"primary": "麥理浩徑第10段 MacLehose Trail Section 10"}) == \
        ("MacLehose Trail Section 10", "麥理浩徑第10段")
    assert split_name({"primary": "龍脊"}) == (None, "龍脊")
    assert split_name({"primary": "Tsing Ying Road"}) == ("Tsing Ying Road", None)
    assert split_name(None) == (None, None)


def test_a_segment_that_changes_name_is_named_by_position():
    """One Overture segment runs as Nathan Road and then as Austin Road."""
    from hk_flat_routes.network import split_name
    names = {"primary": "彌敦道 Nathan Road", "common": None, "rules": [
        {"variant": "common", "language": None, "value": "彌敦道 Nathan Road",
         "between": [0.07, 0.542]},
        {"variant": "common", "language": None, "value": "柯士甸道 Austin Road",
         "between": [0.542, 1.0]}]}
    assert split_name(names, at=0.3) == ("Nathan Road", "彌敦道")
    assert split_name(names, at=0.8) == ("Austin Road", "柯士甸道")


def test_a_stretch_no_rule_names_is_unnamed():
    """Segment b3ceb93e of the extract: Yuen Wo Road for part of its length,
    joined by Overture to an unnamed cycle track.  ``primary`` and the
    name map belong to the road, not to the track."""
    from hk_flat_routes.network import split_name
    names = {"primary": "源禾路 Yuen Wo Road", "common": [("zh", "源禾路")], "rules": [
        {"variant": "common", "language": None, "value": "源禾路 Yuen Wo Road",
         "between": [0.0, 0.4]},
        {"variant": "common", "language": "en", "value": "Yuen Wo Road",
         "between": [0.0, 0.4]}]}
    assert split_name(names, at=0.2) == ("Yuen Wo Road", "源禾路")
    assert split_name(names, at=0.6) == (None, None)
    # asked about the segment as a whole, it has the one name there is
    assert split_name(names) == ("Yuen Wo Road", "源禾路")
    # a stretch named in one language only keeps what it has
    only_en = {"primary": "Tsing Luk Street", "common": None, "rules": [
        {"variant": "common", "language": "en", "value": "Tsing Luk Street",
         "between": [0.1, 1.0]}]}
    assert split_name(only_en, at=0.5) == ("Tsing Luk Street", None)
    assert split_name(only_en, at=0.05) == (None, None)


def test_a_ranged_rule_in_one_language_speaks_for_that_language_only():
    """Tuen Mun Road as Overture carries it: the road's name runs the whole
    segment, and a Simplified Chinese name is tagged on part of it.  That
    part is still Tuen Mun Road, 屯門公路."""
    from hk_flat_routes.network import split_name
    names = {"primary": "屯門公路 Tuen Mun Road",
             "common": [("en", "Tuen Mun Road"), ("zh-Hant", "屯門公路")],
             "rules": [
                 {"variant": "common", "language": None,
                  "value": "屯門公路 Tuen Mun Road", "between": None},
                 {"variant": "common", "language": "zh-Hans", "value": "屯门公路",
                  "between": [0.0, 0.853]}]}
    assert split_name(names, at=0.4) == ("Tuen Mun Road", "屯門公路")
    assert split_name(names, at=0.9) == ("Tuen Mun Road", "屯門公路")
    # with no Traditional name anywhere, the Simplified one is better than none
    del names["common"][1]
    names["rules"][0]["value"] = "Tuen Mun Road"
    names["primary"] = "Tuen Mun Road"
    assert split_name(names, at=0.4) == ("Tuen Mun Road", "屯门公路")
    assert split_name(names, at=0.9) == ("Tuen Mun Road", None)


def test_a_segment_is_cut_where_its_name_really_changes():
    from hk_flat_routes.network import _name_cuts
    nathan = {"primary": "彌敦道 Nathan Road", "common": None, "rules": [
        {"variant": "common", "language": None, "value": "彌敦道 Nathan Road",
         "between": [0.0, 0.542]},
        {"variant": "common", "language": None, "value": "柯士甸道 Austin Road",
         "between": [0.542, 1.0]}]}
    assert _name_cuts(nathan) == [0.542]
    # the same name in another language on two stretches is no change
    tat_tung = {"primary": "達東路 Tat Tung Road",
                "common": [("en", "Tat Tung Road"), ("zh", "達東路")], "rules": [
                    {"variant": "common", "language": None,
                     "value": "達東路 Tat Tung Road", "between": None},
                    {"variant": "common", "language": "zh-Hant", "value": "達東路",
                     "between": [0.0, 0.4868]},
                    {"variant": "common", "language": "zh-Hant", "value": "達東路",
                     "between": [0.7058, 1.0]}]}
    assert _name_cuts(tat_tung) == []
    # a named stretch ends where an unnamed one begins
    half = {"primary": "北山路", "common": None, "rules": [
        {"variant": "common", "language": None, "value": "北山路",
         "between": [0.0, 0.4957]}]}
    assert _name_cuts(half) == [0.4957]
    assert _name_cuts({"primary": "Peel Street"}) == [] and _name_cuts(None) == []


def _in_a_row(n, spacing=100.0):
    """Ends and middle of ``n`` edges laid end to end along the x axis."""
    x = spacing * np.arange(n)[:, None] + spacing * np.array([0.0, 0.5, 1.0])
    return np.stack([x, np.zeros_like(x)], axis=-1)


def test_every_stretch_of_a_street_takes_the_same_pair_of_names():
    import pandas as pd
    from hk_flat_routes.network import harmonise_names
    name = pd.Series(["Hennessy Road", "軒尼詩道", "Hennessy Road", "Peel Street", None, "無名里"])
    zh = pd.Series(["軒尼詩道", None, None, None, None, None])
    name, zh = harmonise_names(name, zh, _in_a_row(6))
    assert list(name[:3]) == ["Hennessy Road"] * 3
    assert list(zh[:3]) == ["軒尼詩道"] * 3
    assert name[3] == "Peel Street" and pd.isna(zh[3])
    assert name[5] == "無名里" and pd.isna(zh[5])       # no English known: left alone


def test_a_name_is_paired_only_with_a_street_nearby():
    """Two towns each have a 新安街, under different English names; a third
    has one that nobody has given an English name at all."""
    import pandas as pd
    from hk_flat_routes.network import harmonise_names
    name = pd.Series(["San On Street", "新安街", "Sun On Street", "新安街",
                      "Sun On Street", "新安街"])
    zh = pd.Series(["新安街", None, "新安街", None, "新安街", None])
    points = _in_a_row(6)
    points[2:5, :, 0] += 30_000.0          # the second town, 30 km east
    points[5, :, 0] += 15_000.0            # the third, far from both
    name, zh = harmonise_names(name, zh, points)
    assert name[1] == "San On Street" and zh[1] == "新安街"
    assert name[3] == "Sun On Street" and zh[3] == "新安街"
    assert name[5] == "新安街" and pd.isna(zh[5])
    # an English-only stretch is completed the same way, from nearby only
    name = pd.Series(["Club Street", "Club Street", "Club Street"])
    zh = pd.Series(["會所街", None, None])
    points = _in_a_row(3)
    points[2, :, 0] += 20_000.0
    name, zh = harmonise_names(name, zh, points)
    assert zh[1] == "會所街" and pd.isna(zh[2])


# ------------------------------------------------------ sidewalks and crossings
def test_a_footbridge_that_lands_on_sidewalks_is_joined_at_both_ends():
    """Street A-B-C-D; a footbridge P-Q reached only by sidewalk and crossing
    runs at either end.  Both runs are kept (so the bridge is a way through,
    not a dead end); the stub of sidewalk that leads nowhere is not."""
    from hk_flat_routes.network import select_sidepaths
    u = ["A", "B", "C", "P", "A", "s1", "Q", "s2", "B"]
    v = ["B", "C", "D", "Q", "s1", "P", "s2", "D", "s3"]
    length = [100, 100, 100, 60, 10, 40, 40, 10, 15]
    core = [1, 1, 1, 1, 0, 0, 0, 0, 0]
    side = [0, 0, 0, 0, 1, 1, 1, 1, 1]
    keep = select_sidepaths(u, v, length, core, side)
    assert keep.astype(int).tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 0]


def test_a_sidewalk_beside_its_own_street_is_left_out():
    from hk_flat_routes.network import select_sidepaths
    u = ["A", "B", "C", "A", "t1", "t2"]
    v = ["B", "C", "D", "t1", "t2", "D"]
    length = [100, 100, 100, 8, 300, 8]
    keep = select_sidepaths(u, v, length, [1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1])
    assert not keep.any()


def test_a_sidewalk_that_is_a_real_short_cut_is_kept():
    """The street goes the long way round (600 m); the path is 100 m."""
    from hk_flat_routes.network import select_sidepaths
    u = ["A", "B", "C", "A"]
    v = ["B", "C", "D", "D"]
    keep = select_sidepaths(u, v, [200, 200, 200, 100], [1, 1, 1, 0], [0, 0, 0, 1])
    assert keep.tolist() == [False, False, False, True]


def test_a_long_sidewalk_beside_a_closed_carriageway_is_kept():
    """Tsing Tsuen Bridge in small.  The carriageway is closed to walkers,
    so the centrelines join the two ends only by 5 km round; the pavement
    is 500 m in four pieces, and no short search finds its far end."""
    from hk_flat_routes.network import select_sidepaths
    u = ["A", "X", "Y", "A", "s1", "s2", "s3"]
    v = ["X", "Y", "B", "s1", "s2", "s3", "B"]
    length = [2000, 1000, 2000, 125, 125, 125, 125]
    core = [1, 1, 1, 0, 0, 0, 0]
    side = [0, 0, 0, 1, 1, 1, 1]
    keep = select_sidepaths(u, v, length, core, side)
    assert keep.astype(int).tolist() == [0, 0, 0, 1, 1, 1, 1]
    # and a pavement that long beside an open carriageway is still left out
    u2 = u + ["A"]; v2 = v + ["B"]
    keep = select_sidepaths(u2, v2, length + [480], core + [1], side + [0])
    assert not keep.any()


def test_the_second_pass_joins_at_any_length_and_can_be_left_out():
    from hk_flat_routes.network import select_sidepaths
    u = ["A", "B", "C"]; v = ["B", "C", "D"]
    args = (u, v, [100, 400, 100], [1, 0, 1], [0, 1, 0])
    assert select_sidepaths(*args, radius=50).tolist() == [False, True, False]
    assert not select_sidepaths(*args, radius=50, join_stranded=False).any()


def test_long_edges_are_cut_to_fit_the_payload():
    import geopandas as gpd
    from shapely.geometry import LineString
    from hk_flat_routes.config import MAX_EDGE_LENGTH_M
    from hk_flat_routes.network import _split_long_edges
    g = gpd.GeoDataFrame({"u": ["a"], "v": ["b"], "segment_id": ["s"], "part": [0],
                          "length_m": [5000.0]},
                         geometry=[LineString([(0, 0), (5000, 0)])], crs="EPSG:2326")
    out = _split_long_edges(g)
    assert len(out) == 3 and out["length_m"].max() <= MAX_EDGE_LENGTH_M
    assert abs(out["length_m"].sum() - 5000.0) < 1e-6
    # the pieces chain from a to b through nodes of their own
    assert out["u"].iloc[0] == "a" and out["v"].iloc[-1] == "b"
    assert out["v"].iloc[0] == out["u"].iloc[1] and out["v"].iloc[1] == out["u"].iloc[2]


# ------------------------------------------------------------ cutting segments
def _segments(rows):
    """Overture segments as ``load_segments`` returns them, from lines given
    in HK1980 Grid metres: ``(id, class, points, connectors, options)``."""
    import pandas as pd
    from pyproj import Transformer
    from shapely import to_wkb
    from shapely.geometry import LineString
    to_lonlat = Transformer.from_crs("EPSG:2326", "EPSG:4326", always_xy=True)
    out = []
    for sid, cls, pts, conns, *rest in rows:
        opt = rest[0] if rest else {}
        out.append({
            "id": sid, "subtype": "road", "class": cls,
            "subclass": opt.get("subclass"),
            "geometry": to_wkb(LineString([to_lonlat.transform(x, y) for x, y in pts])),
            "connectors": [{"connector_id": c, "at": at} for c, at in conns],
            "road_flags": opt.get("flags"), "names": opt.get("names"),
            "access_restrictions": opt.get("rules")})
    return pd.DataFrame(out)


def test_connectors_are_cut_at_their_position_in_metres():
    """A segment that runs a kilometre east and then a kilometre north, with
    a connector at the corner: half-way along, in metres.  A degree of
    longitude is shorter than one of latitude here, so half-way along the
    line in degrees is 36 m short of the corner."""
    from hk_flat_routes.network import _base_filter, _split_geometries
    corner = (836000.0, 820000.0)
    df = _segments([("s", "residential",
                     [(835000.0, 820000.0), corner, (836000.0, 821000.0)],
                     [("a", 0.0), ("m", 0.5), ("b", 1.0)])])
    first, second = _split_geometries(_base_filter(df))
    assert (first["u"], first["v"], second["u"], second["v"]) == ("a", "m", "m", "b")
    for end in (first["geometry"].coords[-1], second["geometry"].coords[0]):
        assert np.hypot(end[0] - corner[0], end[1] - corner[1]) < 0.01
    assert abs(first["geometry"].length - 1000.0) < 0.01


def test_segments_are_cut_where_a_flag_or_a_name_changes():
    from hk_flat_routes.network import _base_filter, _split_geometries
    names = {"primary": "彌敦道 Nathan Road", "common": None, "rules": [
        {"variant": "common", "language": None, "value": "彌敦道 Nathan Road",
         "between": [0.0, 0.5]},
        {"variant": "common", "language": None, "value": "柯士甸道 Austin Road",
         "between": [0.5, 1.0]}]}
    df = _segments([
        ("n", "primary", [(835000.0, 820000.0), (835400.0, 820000.0)],
         [("a", 0.0), ("b", 1.0)], {"names": names}),
        ("r", "trunk", [(835000.0, 821000.0), (835400.0, 821000.0)],
         [("c", 0.0), ("d", 1.0)],
         {"flags": [{"values": ["is_link"], "between": [0.0, 0.25]}]})])
    rows = _split_geometries(_base_filter(df))
    assert [(r["u"], r["v"]) for r in rows] == [
        ("a", "cut:n:0.500000"), ("cut:n:0.500000", "b"),
        ("c", "cut:r:0.250000"), ("cut:r:0.250000", "d")]
    assert [r["name"] for r in rows[:2]] == ["Nathan Road", "Austin Road"]
    assert [r["name_zh"] for r in rows[:2]] == ["彌敦道", "柯士甸道"]
    assert [sorted(r["flags"]) for r in rows[2:]] == [["is_link"], []]
    assert abs(rows[2]["geometry"].length - 100.0) < 0.01


def test_class_unknown_is_not_a_street():
    """Bare ``highway=construction`` ways and unverifiable TomTom segments."""
    import pandas as pd
    from hk_flat_routes.config import MODES, NEVER_ROUTABLE_CLASSES
    from hk_flat_routes.network import _base_filter
    assert "unknown" in NEVER_ROUTABLE_CLASSES
    assert not any("unknown" in mode.allowed_classes for mode in MODES.values())
    df = pd.DataFrame({"subtype": ["road"] * 3, "subclass": [None] * 3,
                       "class": ["unknown", None, "residential"]})
    assert _base_filter(df)["cls"].tolist() == ["residential"]


def test_a_piece_of_a_few_decimetres_is_contracted_not_deleted():
    """Two ways meet through a link 0.3 m long.  Deleting the link leaves
    them apart; contracting it makes its two ends one node."""
    from shapely.geometry import LineString
    from hk_flat_routes.network import _contract_short_pieces, _join_ends
    u = ["a", "b", "c", "b"]
    v = ["b", "c", "d", "c"]
    u2, v2, keep = _contract_short_pieces(u, v, [50.0, 0.3, 40.0, 5.0])
    # the link goes, and so does a longer piece between the same two ends
    assert keep.tolist() == [True, False, True, False]
    assert v2[0] == u2[2] == "b"
    # the ends that meet at the merged node are brought onto one point
    lines = [LineString([(0, 0), (50, 0)]), LineString([(50.3, 0), (90.3, 0)])]
    joined, moved = _join_ends(lines, u2[keep], v2[keep])
    assert joined[0].coords[-1] == joined[1].coords[0] == (50.15, 0.0)
    assert joined[0].coords[0] == (0.0, 0.0) and joined[1].coords[-1] == (90.3, 0.0)
    assert moved.max() == pytest.approx(0.15)
    # nothing is contracted that is longer than the limit
    assert _contract_short_pieces(u, v, [50.0, 0.6, 40.0, 5.0])[2].all()


def test_ways_joined_by_a_stub_stay_joined(monkeypatch):
    """A footway and a stair mapped 0.3 m apart, with a stub between their
    two connectors: after cutting they share a node, at one point."""
    from hk_flat_routes import network
    x, y = 835000.0, 820000.0
    df = _segments([
        ("path", "footway", [(x, y), (x + 50, y)], [("a", 0.0), ("b", 1.0)]),
        ("stub", "footway", [(x + 50, y), (x + 50.3, y)], [("b", 0.0), ("c", 1.0)]),
        ("stair", "steps", [(x + 50.3, y), (x + 90, y)], [("c", 0.0), ("d", 1.0)])])
    monkeypatch.setattr(network, "load_segments", lambda path: df)
    pieces = network.cut_pieces("segments.parquet", clip_to_city=False)
    assert pieces["segment_id"].tolist() == ["path", "stair"]
    assert pieces["v"].iloc[0] == pieces["u"].iloc[1] == "b"
    assert pieces.geometry.iloc[0].coords[-1] == pieces.geometry.iloc[1].coords[0]
    assert pieces.crs.to_epsg() == 2326
    assert pieces["length_m"].sum() == pytest.approx(90.0, abs=0.01)


# --------------------------------------------------------------- who may pass
def _pieces(rows):
    """A table shaped like ``cut_pieces``'s: ``(u, v, class, points, options)``."""
    import geopandas as gpd
    from shapely.geometry import LineString
    out = []
    for k, (u, v, cls, pts, *rest) in enumerate(rows):
        opt = rest[0] if rest else {}
        out.append({"segment_id": opt.get("segment", f"s{k}"), "u": u, "v": v,
                    "part": opt.get("part", 0), "start_at": 0.0, "end_at": 1.0,
                    "mid_at": 0.5, "cls": cls, "subclass": opt.get("subclass"),
                    "name": opt.get("name"), "name_zh": None,
                    "flags": set(opt.get("flags", ())),
                    "access_restrictions": opt.get("rules"),
                    "geometry": LineString(pts)})
    g = gpd.GeoDataFrame(out, geometry="geometry", crs="EPSG:2326")
    g["length_m"] = g.geometry.length
    return g


def test_a_tunnel_is_judged_whole_not_piece_by_piece():
    """One tube cut into three, 1,150 m in all with a 50 m piece in the
    middle, is closed throughout; a street that runs under a building for
    95 m is a street."""
    from hk_flat_routes.network import resolve_access
    tube = {"flags": ["is_tunnel"]}
    rows = [("a", "b", "trunk", [(0, 0), (550, 0)], tube),
            ("b", "c", "trunk", [(550, 0), (600, 0)], tube),
            ("c", "d", "trunk", [(600, 0), (1150, 0)], tube),
            ("d", "e", "trunk", [(1150, 0), (1250, 0)]),
            ("p", "q", "secondary", [(0, 100), (95, 100)], tube),
            # a service road with a roof is not a road tunnel at any length
            ("r", "s", "service", [(0, 200), (1500, 200)], tube)]
    out = resolve_access(_pieces(rows))
    assert out["walk_ok"].tolist() == [False, False, False, True, True, True]
    assert out["bike_ok"].tolist() == [False, False, False, True, True, True]
    # what the data opens to a mode by name stays open to that mode
    signed = dict(tube, rules=[rule("allowed", mode=["foot"])])
    out = resolve_access(_pieces([(u, v, c, pts, signed) for u, v, c, pts, _ in rows[:3]]))
    assert out["walk_ok"].all() and not out["bike_ok"].any()


def test_a_tunnel_the_data_closes_in_part_is_closed_throughout():
    """Tseung Lam Highway in small.  One tube is three pieces, 370 m in all,
    and the data bars walkers and cyclists from the middle one; the other
    tube runs beside it, shares no node with it and is barred to nobody.  A
    tunnel is one thing.  The street under a building next to it is another
    thing, and so is a tunnel of the same name half a kilometre away."""
    from hk_flat_routes.network import resolve_access
    barred = [rule("denied", mode=["foot", "bicycle"])]

    def tube(name="Tseung Lam Highway", **opt):
        return dict(opt, flags=["is_tunnel"], name=name)
    rows = [("a", "b", "primary", [(0, 0), (100, 0)], tube()),
            ("b", "c", "primary", [(100, 0), (270, 0)], tube(rules=barred)),
            ("c", "d", "primary", [(270, 0), (370, 0)], tube()),
            ("p", "q", "primary", [(0, 15), (260, 15)], tube()),
            ("r", "s", "secondary", [(0, 40), (95, 40)], tube("Tuen Fat Road")),
            ("x", "y", "primary", [(0, 500), (200, 500)], tube()),
            # the road outside the tunnel is as the data has it
            ("d", "e", "primary", [(370, 0), (470, 0)], {"name": "Tseung Lam Highway"})]
    out = resolve_access(_pieces(rows))
    assert out["walk_ok"].tolist() == [False, False, False, False, True, True, True]
    assert out["bike_ok"].tolist() == [False, False, False, False, True, True, True]
    # barred to one mode, the tunnel is closed to that mode only
    rows[1] = rows[1][:4] + (tube(rules=[rule("denied", mode=["bicycle"])]),)
    out = resolve_access(_pieces(rows))
    assert out["walk_ok"].all()
    assert out["bike_ok"].tolist() == [False, False, False, False, True, True, True]
    # a piece the data opens to walkers by name stays open to them
    rows[1] = rows[1][:4] + (tube(rules=barred),)
    rows[3] = rows[3][:4] + (tube(rules=[rule("allowed", mode=["foot"])]),)
    out = resolve_access(_pieces(rows))
    assert out["walk_ok"].tolist() == [False, False, False, True, True, True, True]
    # two tunnels with no name are not one tunnel for lying side by side
    out = resolve_access(_pieces([
        ("a", "b", "primary", [(0, 0), (100, 0)], tube(None, rules=barred)),
        ("p", "q", "primary", [(0, 15), (100, 15)], tube(None))]))
    assert out["walk_ok"].tolist() == [False, True]


def test_trunk_slip_roads_and_viaducts_are_closed_to_walkers():
    from hk_flat_routes.network import resolve_access
    out = resolve_access(_pieces([
        ("a", "b", "trunk", [(0, 0), (100, 0)]),
        ("b", "c", "trunk", [(100, 0), (200, 0)], {"flags": ["is_link"]}),
        ("c", "d", "trunk", [(200, 0), (300, 0)], {"flags": ["is_bridge"]}),
        ("d", "e", "trunk", [(300, 0), (400, 0)],
         {"flags": ["is_bridge"], "rules": [rule("designated", mode=["foot"])]}),
        ("e", "f", "primary", [(400, 0), (500, 0)], {"flags": ["is_bridge", "is_link"]}),
    ]))
    assert out["walk_ok"].tolist() == [True, False, False, True, True]
    assert out["bike_ok"].all()


def test_a_slip_road_piece_inside_a_junction_of_streets_stays_open():
    """Pui Ching Road in small: a street of two carriageways that a trunk
    slip road crosses.  The 12 m of slip road between the carriageways is
    all that joins them there, and is part of the junction; the slip road
    beyond it is a slip road, and so is one a footpath lands on."""
    from hk_flat_routes.network import resolve_access
    link = {"flags": ["is_link"]}
    rows = [("a", "b", "secondary", [(0, 0), (100, 0)]),
            ("c", "d", "secondary", [(100, 12), (0, 12)]),
            ("b", "c", "trunk", [(100, 0), (100, 12)], link),
            ("c", "e", "trunk", [(100, 12), (100, 112)], link),
            ("e", "f", "trunk", [(100, 112), (300, 112)]),
            ("b", "g", "trunk", [(100, 0), (100, -50)], link),
            ("g", "h", "footway", [(100, -50), (150, -50)])]
    out = resolve_access(_pieces(rows))
    assert out["walk_ok"].tolist() == [True, True, True, False, True, False, True]
    assert out["bike_ok"].tolist() == [True] * 6 + [False]
    # the junction of a deck is a junction too
    rows[2] = rows[2][:4] + ({"flags": ["is_link", "is_bridge"]},)
    assert resolve_access(_pieces(rows))["walk_ok"].iloc[2]
    # a carriageway the data closes to walkers is no street to join, and
    # neither is a service road: past the gate of a yard, a slip road is
    # still a slip road
    other = rows[1]
    rows[1] = other + ({"rules": [rule("denied", mode=["foot"])]},)
    assert not resolve_access(_pieces(rows))["walk_ok"].iloc[2]
    rows[1] = other[:2] + ("service",) + other[3:]
    assert not resolve_access(_pieces(rows))["walk_ok"].iloc[2]


def test_service_roads_inside_a_closed_site_are_closed():
    import geopandas as gpd
    from shapely.geometry import box
    from hk_flat_routes.network import resolve_access
    land = gpd.GeoDataFrame({"name": ["Power Station"]},
                            geometry=[box(0, 0, 100, 100)], crs="EPSG:2326")
    pieces = _pieces([
        ("a", "b", "service", [(10, 50), (90, 50)]),
        ("b", "c", "service", [(90, 50), (300, 50)]),       # mostly outside
        ("a", "d", "footway", [(10, 50), (10, 90)]),        # a path: as mapped
        ("d", "e", "service", [(10, 90), (90, 90)],
         {"rules": [rule("allowed", mode=["foot"])]}),
        ("e", "f", "residential", [(90, 90), (90, 60)]),
    ])
    out = resolve_access(pieces, land)
    assert out["walk_ok"].tolist() == [False, True, True, True, True]
    assert out["bike_ok"].tolist() == [False, True, False, False, True]
    assert resolve_access(pieces)["walk_ok"].all()


def test_closed_land_is_military_land_and_the_named_industrial_sites(tmp_path):
    import pandas as pd
    from shapely import to_wkb
    from shapely.geometry import LineString, box
    from hk_flat_routes.network import closed_land

    def site(subtype, cls, name, geom=box(114.0, 22.3, 114.01, 22.31)):
        return {"subtype": subtype, "class": cls, "geometry": to_wkb(geom),
                "names": None if name is None else {"primary": name}}
    path = tmp_path / "land_use.parquet"
    pd.DataFrame([
        site("military", "barracks", "石崗軍營 Shek Kong Barracks"),
        site("military", "range", "青山操炮區 Castle Peak Firing Range"),
        site("military", "military", None),
        site("developed", "industrial", "葵青貨櫃碼頭 Kwai Tsing Container Terminals"),
        site("developed", "industrial", "南丫發電廠 Lamma Power Station"),
        site("developed", "industrial", "內河貨運碼頭 River Trade Terminal"),
        site("developed", "industrial", "南豐工業城 Nan Fung Industrial City"),
        site("developed", "commercial", "Power Station Plaza"),
        site("military", "barracks", "A fence", LineString([(114.0, 22.3), (114.01, 22.3)])),
    ]).to_parquet(path)
    land = closed_land(path)
    assert sorted(land["name"]) == ["", "Kwai Tsing Container Terminals",
                                    "Lamma Power Station", "River Trade Terminal",
                                    "Shek Kong Barracks"]
    assert land.crs.to_epsg() == 2326 and (land.area > 1e5).all()
    # a data directory fetched before the rule has no extract: nothing is
    # closed, and the build says so
    assert len(closed_land(tmp_path / "missing.parquet")) == 0


def test_a_short_footway_link_is_opened_to_pushed_bicycles():
    """A cycle track and a street 20 m apart, joined at one end by a kerb
    link.  The link is pushed across; steps, a footbridge, a subway and a
    footway the data closes to bicycles are not."""
    from hk_flat_routes.network import resolve_access
    track = ("a", "b", "cycleway", [(0, 0), (500, 0)])
    street = ("c", "d", "residential", [(0, 20), (500, 20)])

    def link(cls, **opt):
        return resolve_access(_pieces([track, street,
                                       ("b", "d", cls, [(500, 0), (500, 20)], opt)]))
    out = link("footway")
    assert out["bike_push"].tolist() == [False, False, True]
    assert out["bike_ok"].all() and out["walk_ok"].all()
    assert link("footway", subclass="crosswalk")["bike_push"].iloc[2]
    assert link("pedestrian")["bike_push"].iloc[2]
    for refused in (link("steps"), link("footway", flags=["is_bridge"]),
                    link("footway", flags=["is_tunnel"]),
                    link("footway", rules=[rule("denied", mode=["bicycle"])])):
        assert not refused["bike_push"].any() and not refused["bike_ok"].iloc[2]
    # a footway signed for cycling is ridden, not pushed
    ridden = link("footway", rules=[rule("designated", mode=["bicycle"])])
    assert ridden["bike_ok"].iloc[2] and not ridden["bike_push"].any()
    # whoever pushes a bicycle is on foot, and a one-way sign is not for them
    signed = link("footway", rules=[rule("denied", heading="backward")])
    assert signed["bike_push"].iloc[2] and not signed["bike_oneway"].iloc[2]


def test_nothing_longer_than_the_limit_is_pushed():
    from hk_flat_routes.config import BIKE_PUSH_MAX_M
    from hk_flat_routes.network import resolve_access
    assert BIKE_PUSH_MAX_M == 50.0

    def gap(*lengths):
        """A cycle track and a street joined only by a run of footways."""
        y = np.concatenate([[0.0], np.cumsum(lengths)])
        nodes = ["b"] + [f"f{k}" for k in range(1, len(lengths))] + ["c"]
        run = [(nodes[k], nodes[k + 1], "footway", [(500, y[k]), (500, y[k + 1])])
               for k in range(len(lengths))]
        return resolve_access(_pieces([
            ("a", "b", "cycleway", [(0, 0), (500, 0)]),
            ("c", "d", "residential", [(500, y[-1]), (0, y[-1])])] + run))
    assert gap(20.0, 25.0)["bike_push"].tolist() == [False, False, True, True]
    # 60 m is not pushed, in one piece or in two, though nothing else joins
    # the track to the street
    assert not gap(60.0)["bike_push"].any()
    assert not gap(30.0, 30.0)["bike_push"].any()


def test_a_footway_that_saves_no_riding_is_not_pushed():
    """Pushing is charged three times its length, so a link is opened only
    where riding round is more than three times as far."""
    from hk_flat_routes.network import resolve_access
    out = resolve_access(_pieces([
        ("a", "b", "residential", [(0, 0), (0, 30)]),
        ("b", "c", "residential", [(0, 30), (40, 30), (40, 0)]),
        ("a", "c", "footway", [(0, 0), (40, 0)]),           # 40 m against 100 m
        ("a", "e", "residential", [(0, 0), (0, -200)]),
        ("e", "f", "residential", [(0, -200), (-30, -200)]),
        ("f", "g", "residential", [(-30, -200), (-30, 0)]),
        ("a", "g", "footway", [(0, 0), (-30, 0)]),          # 30 m against 430 m
    ]))
    assert out["bike_push"].tolist() == [False] * 6 + [True]


def test_a_one_way_drawn_backwards_is_turned_round():
    """Two pieces of a segment tagged oneway=-1 and an ordinary one-way
    street: afterwards every one-way runs from u to v."""
    from hk_flat_routes.network import resolve_access
    back = [rule("denied", heading="forward")]
    out = resolve_access(_pieces([
        ("a", "b", "residential", [(0, 0), (100, 0)], {"segment": "s", "part": 0, "rules": back}),
        ("b", "c", "residential", [(100, 0), (200, 0)], {"segment": "s", "part": 1, "rules": back}),
        ("c", "d", "residential", [(200, 0), (300, 0)],
         {"segment": "t", "rules": [rule("denied", heading="backward")]}),
        ("d", "e", "residential", [(300, 0), (400, 0)], {"segment": "w"}),
    ]))
    assert list(zip(out["u"], out["v"])) == [("b", "a"), ("c", "b"), ("c", "d"), ("d", "e")]
    assert out["bike_oneway"].tolist() == [True, True, True, False]
    assert out["bike_ok"].all() and out["walk_ok"].all()
    for geom, u in zip(out.geometry, [(100, 0), (200, 0), (200, 0), (300, 0)]):
        assert geom.coords[0] == u
    # pieces of a segment are numbered in the order they are now travelled,
    # so that consecutive parts still follow one another end to end
    assert out["part"].tolist() == [-1, -2, 0, 0]
    order = out[out["segment_id"] == "s"].sort_values("part")
    assert order["v"].iloc[0] == order["u"].iloc[1] == "b"


def test_a_street_closed_in_both_directions_is_closed_to_bicycles():
    from hk_flat_routes.network import resolve_access
    out = resolve_access(_pieces([
        ("a", "b", "residential", [(0, 0), (100, 0)],
         {"rules": [rule("denied", heading="forward"), rule("denied", heading="backward")]})]))
    assert not out["bike_ok"].iloc[0] and out["walk_ok"].iloc[0]


# ------------------------------------------------------------- the whole build
def test_build_edges_on_a_small_extract(tmp_path, monkeypatch):
    """A street and a cycle track beside it, joined by a crossing at one
    place and across a paved square at another; the street's own pavement,
    a path signed for cycling, a long plain path and a road of class
    ``unknown``.  The table is written beside the one it replaces, so that
    a symbolic link standing there is replaced and the file it points at is
    left alone."""
    from hk_flat_routes import network
    x, y = 835000.0, 820000.0
    shared = [rule("designated", mode=["bicycle"])]
    df = _segments([
        ("street", "residential", [(x, y), (x + 200, y)],
         [("a", 0.0), ("b", 0.5), ("c", 1.0)]),
        ("track", "cycleway", [(x, y + 15), (x + 200, y + 15)],
         [("d", 0.0), ("e", 0.5), ("f", 1.0)]),
        ("zebra", "footway", [(x + 100, y), (x + 100, y + 15)], [("b", 0.0), ("e", 1.0)],
         {"subclass": "crosswalk"}),
        ("pavement", "footway", [(x, y), (x, y - 3), (x + 100, y - 3), (x + 100, y)],
         [("a", 0.0), ("b", 1.0)], {"subclass": "sidewalk"}),
        ("square", "pedestrian", [(x + 200, y), (x + 200, y + 15)], [("c", 0.0), ("f", 1.0)]),
        ("signed", "footway", [(x + 200, y), (x + 300, y)], [("c", 0.0), ("g", 1.0)],
         {"rules": shared}),
        ("plain", "footway", [(x, y), (x, y - 150)], [("a", 0.0), ("h", 1.0)]),
        ("site", "unknown", [(x, y), (x - 100, y)], [("a", 0.0), ("i", 1.0)]),
    ])
    main = tmp_path / "main" / "edges_raw.parquet"
    run = tmp_path / "run" / "edges_raw.parquet"
    main.parent.mkdir(); run.parent.mkdir()
    main.write_bytes(b"the main build")
    run.symlink_to(main)
    monkeypatch.setattr(network, "EDGES_PARQUET", run)
    monkeypatch.setattr(network, "MIN_COMPONENT_NODES", 2)
    monkeypatch.setattr(network, "load_segments", lambda path: df)
    edges = network.build_edges(tmp_path / "segments.parquet", force=True,
                                clip_to_city=False,
                                land_use_path=tmp_path / "no_land_use.parquet")
    assert main.read_bytes() == b"the main build"
    assert not run.is_symlink() and [p.name for p in run.parent.iterdir()] == [run.name]
    import geopandas as gpd
    assert len(gpd.read_parquet(run)) == len(edges) == 8
    e = edges.set_index(edges["segment_id"] + edges["part"].astype(str))
    # the pavement offers nothing the street does not, and goes; the
    # crossing stays, as the link a bicycle is pushed across
    assert "site0" not in e.index and "pavement0" not in e.index
    assert e["bike_push"].to_dict() == {
        "street0": False, "street1": False, "track0": False, "track1": False,
        "zebra0": True, "square0": True, "signed0": False, "plain0": False}
    assert e["bike_ok"].drop("plain0").all() and not e.loc["plain0", "bike_ok"]
    # a shared path is one a bicycle may ride: not every footway, and no
    # link it is pushed across is a cycling facility of any kind
    assert e["bike_facility"].to_dict() == {
        "street0": "", "street1": "", "track0": "dedicated_cycleway",
        "track1": "dedicated_cycleway", "zebra0": "", "square0": "",
        "signed0": "shared_path", "plain0": ""}
    assert e["low_stress"].to_dict() == {
        "street0": True, "street1": True, "track0": True, "track1": True,
        "zebra0": False, "square0": True, "signed0": True, "plain0": False}
    assert list(edges["edge_id"]) == list(range(8))


# ------------------------------------------------- the built graph (if present)
from hk_flat_routes.config import PROCESSED_DIR  # noqa: E402

_BUILT = (PROCESSED_DIR / "edges_metrics.parquet").exists()


def _edges(columns):
    import pandas as pd
    return pd.read_parquet(PROCESSED_DIR / "edges_metrics.parquet", columns=columns)


@pytest.mark.skipif(not _BUILT, reason="processed network not built")
def test_no_main_road_tunnel_is_open_to_walkers_or_cyclists():
    """Pedestrians and cyclists are barred from every vehicular tunnel in
    Hong Kong, and an open one would be the flattest route through any hill
    (OpenStreetMap leaves a tube of Tate's Cairn Tunnel open, and 51 m of a
    slip road of Kai Tak Tunnel)."""
    from hk_flat_routes.config import ROAD_TUNNEL_CLASSES, ROAD_TUNNEL_MAX_M
    from hk_flat_routes.network import _run_lengths
    e = _edges(["u", "v", "name", "cls", "is_tunnel", "walk_ok", "bike_ok", "length_m"])
    t = e[e["is_tunnel"] & e["cls"].isin(ROAD_TUNNEL_CLASSES)]
    for mode in ("walk_ok", "bike_ok"):
        # no piece named as a tunnel is open, however short, nor any piece
        # of the three short tunnels of which the data closes one tube ...
        named = (t["name"].fillna("").str.contains("Tunnel|Harbour Crossing")
                 | t["name"].isin(["Tseung Lam Highway", "Ma On Shan Bypass",
                                   "Smithfield"]))
        assert not t.loc[named, mode].any(), sorted(set(t.loc[named & t[mode], "name"]))
        # ... and what is open under a roof is nowhere a long way through
        run = _run_lengths(t["u"].to_numpy(), t["v"].to_numpy(),
                           t["length_m"].to_numpy(), t[mode].to_numpy())
        assert run.max(initial=0.0) <= ROAD_TUNNEL_MAX_M


@pytest.mark.skipif(not _BUILT, reason="processed network not built")
def test_streets_under_a_building_or_a_flyover_stay_open():
    """Tagged as tunnels, and ordinary streets: Cheung Tung Road at Sunny
    Bay is the only road along north Lantau a bicycle may use.  Each has
    some 93 m under its roof in one piece, which a limit of 80 m a piece
    took out of the network."""
    from hk_flat_routes.config import ROAD_TUNNEL_CLASSES
    e = _edges(["name", "cls", "is_tunnel", "walk_ok", "bike_ok", "length_m"])
    e = e[e["cls"].isin(ROAD_TUNNEL_CLASSES)]
    for street in ("Cheung Tung Road", "Tuen Hi Road", "Tuen Fat Road"):
        if not (e["name"] == street).any():
            continue                    # outside this run's area
        s = e[(e["name"] == street) & e["is_tunnel"]]
        assert s["length_m"].max() > 80.0, street
        assert s["walk_ok"].all() and s["bike_ok"].all(), street


@pytest.mark.skipif(not _BUILT, reason="processed network not built")
def test_the_tram_streets_are_walkable_and_bikeable():
    e = _edges(["name", "cls", "walk_ok", "bike_ok"])
    for street in ("Hennessy Road", "Nathan Road", "Des Voeux Road West"):
        s = e[(e["name"] == street) & e["cls"].isin(["primary", "secondary", "tertiary"])]
        if not len(s):
            continue                    # outside this run's area
        assert s["walk_ok"].mean() > 0.95, (street, s["walk_ok"].mean())
        assert s["bike_ok"].mean() > 0.9, (street, s["bike_ok"].mean())


@pytest.mark.skipif(not _BUILT, reason="processed network not built")
def test_bicycles_stay_off_footways_and_steps():
    e = _edges(["cls", "bike_ok"])
    assert not e.loc[e["cls"] == "steps", "bike_ok"].any()
    # only footways signed for cycling are open: a small minority
    assert e.loc[e["cls"] == "footway", "bike_ok"].mean() < 0.05


@pytest.mark.skipif(not _BUILT, reason="processed network not built")
def test_street_names_are_english_with_chinese_alongside():
    e = _edges(["name", "name_zh"])
    named = e[e["name"].notna()]
    assert (named["name"] == "Nathan Road").any() or (named["name"] == "Hennessy Road").any()
    assert named["name_zh"].notna().mean() > 0.8
    # the combined "中文 English" strings have all been split; what is left
    # mixed is a handful of hikers' notes ("深石脊 (Tingsi Trail)")
    import re
    combined = re.compile("[\u4e00-\u9fff].* [A-Za-z]")
    assert named["name"].map(lambda n: bool(combined.search(n))).mean() < 0.001
