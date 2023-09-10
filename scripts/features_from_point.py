from typing import List, Union
import warnings

import geopandas as gpd
import pandas as pd
from osmnx import utils_geo, _downloader, settings
from osmnx.features import (
    _parse_node_to_coords,
    _parse_node_to_point,
    _parse_relation_to_multipolygon,
    _parse_way_to_linestring_or_polygon,
    _buffer_invalid_geometries,
    _filter_gdf_by_polygon_and_tags,
)
from osmnx._errors import EmptyOverpassResponse
from shapely.geometry import Polygon, MultiPolygon

from utils import get_logger

logger = get_logger("features_to_point")


def features_from_point(center_point, tags, dist=1000):
    north, south, east, west = utils_geo.bbox_from_point(center_point, dist)

    # convert the bounding box to a polygon
    polygon = utils_geo.bbox_to_poly(north, south, east, west)

    # create GeoDataFrame of features within this polygon
    gdf = features_from_polygon(polygon, tags)

    return gdf


def features_from_polygon(polygon, tags):
    if not polygon.is_valid:
        raise ValueError("The geometry of `polygon` is invalid")
    if not isinstance(polygon, (Polygon, MultiPolygon)):
        raise TypeError(
            "Boundaries must be a shapely Polygon or MultiPolygon. If you requested "
            "features from place name, make sure your query resolves to a Polygon or "
            "MultiPolygon, and not some other geometry, like a Point. See OSMnx "
            "documentation for details."
        )

    # download the data from OSM
    logger.info("Start downloading from OSM")
    response_jsons = _downloader._osm_features_download(polygon, tags)
    logger.info("Download completed")

    logger.info("Parsing response_jsons to geometries")
    # Parse geometries from the downloaded data
    geometries = parse_response_jsons_to_geometries(response_jsons, polygon, tags)
    logger.info("Parsing finished")

    logger.info("Start combining parsed geometries into GeoDataFrames")
    gdf = combine_geometries_into_gdf(geometries, polygon, tags)
    logger.info("Combining geometries into GeoDataFrames finished")
    return gdf


def combine_geometries_into_gdf(
    geometries: List[dict], polygon: Union[Polygon, MultiPolygon], tags: dict
) -> gpd.GeoDataFrame:
    
    from pprint import pprint;breakpoint()
    pass
    geodataframes = []
    for geometry_chunk in divide_into_chunks(geometries, 15):
        gdf = gpd.GeoDataFrame.from_dict(geometry_chunk, orient="index")
        if "geometry" not in gdf.columns:
            # if there is no geometry column, create a null column
            gdf = gdf.set_geometry([None] * len(gdf))
        gdf = gdf.set_crs(settings.default_crs)

        # Apply .buffer(0) to any invalid geometries
        gdf = _buffer_invalid_geometries(gdf)

        # Filter final gdf to requested tags and query polygon
        gdf = _filter_gdf_by_polygon_and_tags(gdf, polygon=polygon, tags=tags)

        # bug in geopandas <0.9 raises a TypeError if trying to plot empty
        # geometries but missing geometries (gdf['geometry'] = None) cannot be
        # projected e.g. gdf.to_crs(). Remove rows with empty (e.g. Point())
        # or missing (e.g. None) geometry, and suppress gpd warning caused by
        # calling gdf["geometry"].isna() on GeoDataFrame with empty geometries
        if not gdf.empty:
            warnings.filterwarnings("ignore", "GeoSeries.isna", UserWarning)
            gdf = gdf[~(gdf["geometry"].is_empty | gdf["geometry"].isna())].copy()
            warnings.resetwarnings()

        geodataframes.append(gdf)

    return pd.concat(geodataframes)


def parse_response_jsons_to_geometries(response_jsons, polygon, tags):
    elements_count = sum(len(rj["elements"]) for rj in response_jsons)

    # make sure we got data back from the server request(s)
    if elements_count == 0:
        msg = "There are no data elements in the server response. Check log and query location/tags."
        raise EmptyOverpassResponse(msg)

    # Dictionaries to hold nodes and complete geometries
    coords = {}
    geometries = {}

    # Set to hold the unique IDs of elements that do not have tags
    untagged_element_ids = set()

    # identify which relation types to parse to (multi)polygons
    relation_types = {"boundary", "multipolygon"}

    # extract geometries from the downloaded osm data
    for response_json in response_jsons:
        # Parses the JSON of OSM nodes, ways and (multipolygon) relations
        # to dictionaries of coordinates, Shapely Points, LineStrings,
        # Polygons and MultiPolygons
        for element in response_json["elements"]:
            # id numbers are only unique within element types
            # create unique id from combination of type and id
            unique_id = f"{element['type']}/{element['id']}"

            # add elements that are not nodes and that are without tags or
            # with empty tags to the untagged_element_ids set (untagged
            # nodes are not added to the geometries dict at all)
            if (element["type"] != "node") and (
                ("tags" not in element) or (not element["tags"])
            ):
                untagged_element_ids.add(unique_id)

            if element["type"] == "node":
                # Parse all nodes to coords
                coords[element["id"]] = _parse_node_to_coords(element=element)

                # If the node has tags and the tags are not empty parse it
                # to a Point. Empty check is necessary for JSONs created
                # from XML where nodes without tags are assigned tags={}
                if "tags" in element and len(element["tags"]) > 0:
                    point = _parse_node_to_point(element=element)
                    geometries[unique_id] = point

            elif element["type"] == "way":
                # Parse all ways to linestrings or polygons
                linestring_or_polygon = _parse_way_to_linestring_or_polygon(
                    element=element, coords=coords
                )
                geometries[unique_id] = linestring_or_polygon

            elif (
                element["type"] == "relation"
                and element.get("tags").get("type") in relation_types
            ):
                # parse relations to (multi)polygons
                multipolygon = _parse_relation_to_multipolygon(
                    element=element, geometries=geometries
                )
                geometries[unique_id] = multipolygon

    # remove untagged elements from the final dict of geometries
    for untagged_element_id in untagged_element_ids:
        geometries.pop(untagged_element_id, None)

    return geometries


def divide_into_chunks(l, n):
    """Yield n number of striped chunks from l."""
    for i in range(0, n):
        yield l[i::n]
