import math
from tqdm import tqdm
import numpy as np
import json
import os
from nuplan.database.maps_db.gpkg_mapsdb import GPKGMapsDB
import uuid
import argparse

import warnings
warnings.filterwarnings("ignore")

def main(location="us-ma-boston", map_root= "/data/nuplan/dataset/maps", output_file="output_map_data.json"):
    maps_db = GPKGMapsDB(
        map_version="nuplan-maps-v1.0",
        map_root=map_root
    )

    vector_layer_names = maps_db.vector_layer_names(location)

    # Init output json data
    out_json_data = {
        "version": "1.3",
        "polygon": [],
        "line": [],
        "node": [],
        "drivable_area": [],           # Polygons
        "ped_crossing": [],            # Polygon
        "road_divider": [],            # LineString
        "lane_divider": [],            # LineString
        "road_segment": [],            # Polygon
        "lane": [],                    # Polygon
        "traffic_light": [],
        "road_block": [],
        "walkway": [],
        "stop_line": [],
        "connectivity": {},
        "arcline_path_3": {},
        "lane_connector": [],
        "carpark_area": [],
        "canvas_edge": []
    }

    # crosswalks -> ped_crossing
    vector_layer = maps_db.load_vector_layer(location, 'crosswalks')

    new_polygon_list = []
    new_node_list = []
    new_ped_crossing_list = []

    for i in range(len(vector_layer)):
        token = vector_layer['creator_id'][i]
        geometry = vector_layer['geometry'][i]
        exterior_coords = list(geometry.exterior.coords)
        exterior_node_tokens = [str(uuid.uuid4()) for coord in exterior_coords]
        polygon_token = str(uuid.uuid4())
        
        new_node_list.extend([{
            "token": exterior_node_token,
            "x": coord[0],
            "y": coord[1]
        } for exterior_node_token, coord in zip(exterior_node_tokens, exterior_coords)])
        
        holes = [] # TODO: fill in the holes
        new_polygon_list.append({
            "token": polygon_token,
            "exterior_node_tokens": exterior_node_tokens,
            "holes": holes
        })
        new_ped_crossing_list.append({
            "token": token,
            "polygon_token": polygon_token,
            "road_segment_token" : None # TODO: fill in the road segment token
        })

    out_json_data['polygon'].extend(new_polygon_list)
    out_json_data['node'].extend(new_node_list)
    out_json_data['ped_crossing'].extend(new_ped_crossing_list)

    # road segments -> road_segment
    vector_layer = maps_db.load_vector_layer(location, 'road_segments')

    new_polygon_list = []
    new_node_list = []
    new_road_segment_list = []

    for i in range(len(vector_layer)):
        token = str(uuid.uuid4())
        geometry = vector_layer['geometry'][i]
        exterior_coords = list(geometry.exterior.coords)
        exterior_node_tokens =[]

        polygon_token = str(uuid.uuid4())

        for exterior_coord in exterior_coords:
            exterior_node_token = str(uuid.uuid4())
            exterior_node_tokens.append(exterior_node_token)
            new_node_list.append({
                "token": exterior_node_token,
                "x": exterior_coord[0],
                "y": exterior_coord[1]
            })
        holes = [] # TODO: fill in the holes
        new_polygon_list.append({
            "token": polygon_token,
            "exterior_node_tokens": exterior_node_tokens,
            "holes": holes
        })
        new_road_segment_list.append({
            "token": token,
            "polygon_token": polygon_token,
            "is_intersection": False, # TODO: check if it is an intersection
        })

    out_json_data['polygon'].extend(new_polygon_list)
    out_json_data['node'].extend(new_node_list)
    out_json_data['road_segment'].extend(new_road_segment_list)

    # generic_drivable_areas -> lane
    vector_layer = maps_db.load_vector_layer(location, 'generic_drivable_areas')

    new_polygon_list = []
    new_node_list = []
    new_lane_list = []

    for i in range(len(vector_layer)):
        token = vector_layer['creator_id'][i]
        geometry = vector_layer['geometry'][i]
        exterior_coords = list(geometry.exterior.coords)
        exterior_node_tokens = []
        
        polygon_token = str(uuid.uuid4())
        
        for exterior_coord in exterior_coords:
            exterior_node_token = str(uuid.uuid4())
            exterior_node_tokens.append(exterior_node_token)
            new_node_list.append({
                "token": exterior_node_token,
                "x": exterior_coord[0],
                "y": exterior_coord[1]
            })
        
        holes = [] # TODO: 如果需要,填充孔洞信息
        
        new_polygon_list.append({
            "token": polygon_token,
            "exterior_node_tokens": exterior_node_tokens,
            "holes": holes
        })
        
        new_lane_list.append({
            "token": token,
            "polygon_token": polygon_token,
            "lane_type": "CAR",  # TODO: 默认设置为 CAR了，应该还有 BUS、BICYCLE 的可选项
            "left_lane_divider_segments": [],
            "right_lane_divider_segments": []
        })

    out_json_data['polygon'].extend(new_polygon_list)
    out_json_data['node'].extend(new_node_list)
    out_json_data['lane'].extend(new_lane_list)

    # intersections -> lane
    vector_layer = maps_db.load_vector_layer(location, 'intersections')

    new_polygon_list = []
    new_node_list = []
    new_lane_list = []

    for i in range(len(vector_layer)):
        token = vector_layer['creator_id'][i]
        geometry = vector_layer['geometry'][i]
        exterior_coords = list(geometry.exterior.coords)
        exterior_node_tokens = []
        
        polygon_token = str(uuid.uuid4())
        
        for exterior_coord in exterior_coords:
            exterior_node_token = str(uuid.uuid4())
            exterior_node_tokens.append(exterior_node_token)
            new_node_list.append({
                "token": exterior_node_token,
                "x": exterior_coord[0],
                "y": exterior_coord[1]
            })
        
        holes = [] # TODO: 如果需要,填充孔洞信息
        
        new_polygon_list.append({
            "token": polygon_token,
            "exterior_node_tokens": exterior_node_tokens,
            "holes": holes
        })
        
        new_lane_list.append({
            "token": token,
            "polygon_token": polygon_token,
            "lane_type": "CAR",  # TODO: 默认设置为 CAR了，应该还有 BUS、BICYCLE 的可选项
            "left_lane_divider_segments": [],
            "right_lane_divider_segments": []
        })

    out_json_data['polygon'].extend(new_polygon_list)
    out_json_data['node'].extend(new_node_list)
    out_json_data['lane'].extend(new_lane_list)

    # boundaries[0] -> lane_divider
    vector_layer = maps_db.load_vector_layer(location, 'boundaries')

    new_line_list = []
    new_node_list = []
    new_boundary_list = []

    for i in range(len(vector_layer)):
        boundary_type_fid = vector_layer['boundary_type_fid'][i]
        if boundary_type_fid != 0:
            continue
        creator_id = vector_layer['creator_id'][i]
        geometry = vector_layer['geometry'][i]
        coords = list(geometry.coords)
        node_tokens = []
        line_token = str(uuid.uuid4())
        for coord in coords:
            node_token = str(uuid.uuid4())
            node_tokens.append(node_token)
            new_node_list.append({
                "token": node_token,
                "x": coord[0],
                "y": coord[1]
            })
        new_line_list.append({
            "token": line_token,
            "node_tokens": node_tokens
        })
        new_boundary_list.append({
            "token": str(creator_id),
            "line_token": line_token,
            "is_intersection": False, # TODO: 默认设为 False 了
        })

    out_json_data['line'].extend(new_line_list)
    out_json_data['node'].extend(new_node_list)
    out_json_data['lane_divider'].extend(new_boundary_list)

    # boundaries[2] -> road_divider
    vector_layer = maps_db.load_vector_layer(location, 'boundaries')
    new_line_list = []
    new_node_list = []
    new_boundary_list = []

    for i in range(len(vector_layer)):
        boundary_type_fid = vector_layer['boundary_type_fid'][i]
        if boundary_type_fid != 2:
            continue
        creator_id = vector_layer['creator_id'][i]
        geometry = vector_layer['geometry'][i]
        coords = list(geometry.coords)
        node_tokens = []
        line_token = str(uuid.uuid4())
        for coord in coords:
            node_token = str(uuid.uuid4())
            node_tokens.append(node_token)
            new_node_list.append({
                "token": node_token,
                "x": coord[0],
                "y": coord[1]
            })
        new_line_list.append({
            "token": line_token,
            "node_tokens": node_tokens
        })
        new_boundary_list.append({
            "token": str(creator_id),
            "line_token": line_token,
            "is_intersection": False, # TODO: 默认设为 False 了
        })

    out_json_data['line'].extend(new_line_list)
    out_json_data['node'].extend(new_node_list)
    # out_json_data['boundary'].extend(new_boundary_list)
    out_json_data['road_divider'].extend(new_boundary_list)
    
    # Dump json
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(out_json_data, json_file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert NuPlan GPKG to JSON")
    parser.add_argument("--location", default="us-pa-pittsburgh-hazelwood", help="Location for map data")
    parser.add_argument("--map_root", default="/data/nuplan/dataset/maps", help="Map root")
    parser.add_argument(
        "--output",
        default="/data/nuplan/maps/expansion/us-pa-pittsburgh-hazelwood.json",
        help="Output JSON file path"
    )
    args = parser.parse_args()

    main(args.location, args.map_root, args.output)