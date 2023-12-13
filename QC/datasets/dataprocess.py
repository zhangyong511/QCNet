import pandas as pd
from av2.utils.io import read_json_file
import math
import os
import pickle
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union
from urllib import request

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Dataset
from torch_geometric.data import HeteroData
from torch_geometric.data import extract_tar
from tqdm import tqdm
import sys
print(sys.path.append('/mnt/ve_share2/zy/QCNet'))
from utils import safe_list_index
from utils import side_to_directed_lineseg

try:
    from av2.geometry.interpolate import compute_midpoint_line
    from av2.map.map_api import ArgoverseStaticMap
    from av2.map.map_primitives import Polyline
    from av2.utils.io import read_json_file
except ImportError:
    compute_midpoint_line = object
    ArgoverseStaticMap = object
    Polyline = object
    read_json_file = object


def get_scenario_id(df: pd.DataFrame) -> str:
    return df['scenario_id'].values[0]

def get_city(df: pd.DataFrame) -> str:
    return df['city'].values[0]
def get_agent_features(df: pd.DataFrame) -> Dict[str, Any]:
    num_steps=110
    dim=3
    predict_unseen_agents = False
    if not predict_unseen_agents:  # filter out agents that are unseen during the historical time steps
        historical_df = df[df['timestep'] < 50]
        agent_ids = list(historical_df['track_id'].unique())
        df = df[df['track_id'].isin(agent_ids)]  ##把历史时刻出现过的agent保留下来
    else:
        agent_ids = list(df['track_id'].unique())

    num_agents = len(agent_ids)
    av_idx = agent_ids.index('AV')
    # initialization
    valid_mask = torch.zeros(num_agents, num_steps, dtype=torch.bool)
    current_valid_mask = torch.zeros(num_agents, dtype=torch.bool)
    predict_mask = torch.zeros(num_agents, num_steps, dtype=torch.bool)
    agent_id: List[Optional[str]] = [None] * num_agents
    agent_type = torch.zeros(num_agents, dtype=torch.uint8)
    agent_category = torch.zeros(num_agents, dtype=torch.uint8)
    position = torch.zeros(num_agents, num_steps, dim, dtype=torch.float)
    heading = torch.zeros(num_agents, num_steps, dtype=torch.float)
    velocity = torch.zeros(num_agents, num_steps, dim, dtype=torch.float)

    for track_id, track_df in df.groupby('track_id'):
        agent_idx = agent_ids.index(track_id)
        agent_steps = track_df['timestep'].values

        valid_mask[agent_idx, agent_steps] = True ##把出现过的agent的所有时刻都标记为True
        current_valid_mask[agent_idx] = valid_mask[agent_idx, 50 - 1]  ##观察的最后的时刻
        predict_mask[agent_idx, agent_steps] = True
        vector_repr=True
        if vector_repr:  # a time step t is valid only when both t and t-1 are valid
            valid_mask[agent_idx, 1: 50] = (
                    valid_mask[agent_idx, :50 - 1] &
                    valid_mask[agent_idx, 1: 50])
            valid_mask[agent_idx, 0] = False
        predict_mask[agent_idx, :50] = False
        if not current_valid_mask[agent_idx]:  #如果最后一个时刻不是valid，那么就不预测
            predict_mask[agent_idx, 50:] = False
        
        agent_types = ['vehicle', 'pedestrian', 'motorcyclist', 'cyclist', 'bus', 'static', 'background',
                        'construction', 'riderless_bicycle', 'unknown']
        agent_id[agent_idx] = track_id
        agent_type[agent_idx] = agent_types.index(track_df['object_type'].values[0])
        agent_category[agent_idx] = track_df['object_category'].values[0]
        position[agent_idx, agent_steps, :2] = torch.from_numpy(np.stack([track_df['position_x'].values,
                                                                            track_df['position_y'].values],
                                                                            axis=-1)).float()
        heading[agent_idx, agent_steps] = torch.from_numpy(track_df['heading'].values).float()
        velocity[agent_idx, agent_steps, :2] = torch.from_numpy(np.stack([track_df['velocity_x'].values,
                                                                            track_df['velocity_y'].values],
                                                                            axis=-1)).float()

  

    return {
        'num_nodes': num_agents, ## N
        'av_index': av_idx,  ##AV 的index
        'valid_mask': valid_mask,  ## N,110
        'predict_mask': predict_mask,  ##N,110
        'id': agent_id,
        'type': agent_type, #N agent对应的type的index 对应df 的object_type
        'category': agent_category,  #  N 对应df 的object_category
        'position': position,  #N,110,3
        'heading': heading, #N,110
        'velocity': velocity,  #N,110,3
    }

def get_map_features(
                        map_api: ArgoverseStaticMap,
                        centerlines: Mapping[str, Polyline]) -> Dict[Union[str, Tuple[str, str, str]], Any]:
    lane_segment_ids = map_api.get_scenario_lane_segment_ids()  ##所有的lane_segment的id
    cross_walk_ids = list(map_api.vector_pedestrian_crossings.keys())  ##所有的cross_walk的id
    polygon_ids = lane_segment_ids + cross_walk_ids
    num_polygons = len(lane_segment_ids) + len(cross_walk_ids) * 2  ##

    # initialization
    dim=3
    polygon_position = torch.zeros(num_polygons, dim, dtype=torch.float)
    polygon_orientation = torch.zeros(num_polygons, dtype=torch.float)
    polygon_height = torch.zeros(num_polygons, dtype=torch.float)
    polygon_type = torch.zeros(num_polygons, dtype=torch.uint8)
    polygon_is_intersection = torch.zeros(num_polygons, dtype=torch.uint8)
    point_position: List[Optional[torch.Tensor]] = [None] * num_polygons
    point_orientation: List[Optional[torch.Tensor]] = [None] * num_polygons
    point_magnitude: List[Optional[torch.Tensor]] = [None] * num_polygons
    point_height: List[Optional[torch.Tensor]] = [None] * num_polygons
    point_type: List[Optional[torch.Tensor]] = [None] * num_polygons
    point_side: List[Optional[torch.Tensor]] = [None] * num_polygons
    polygon_is_intersections = [True, False, None]
    polygon_types = ['VEHICLE', 'BIKE', 'BUS', 'PEDESTRIAN']
    point_types = ['DASH_SOLID_YELLOW', 'DASH_SOLID_WHITE', 'DASHED_WHITE', 'DASHED_YELLOW',
                            'DOUBLE_SOLID_YELLOW', 'DOUBLE_SOLID_WHITE', 'DOUBLE_DASH_YELLOW', 'DOUBLE_DASH_WHITE',
                            'SOLID_YELLOW', 'SOLID_WHITE', 'SOLID_DASH_WHITE', 'SOLID_DASH_YELLOW', 'SOLID_BLUE',
                            'NONE', 'UNKNOWN', 'CROSSWALK', 'CENTERLINE']
    point_sides = ['LEFT', 'RIGHT', 'CENTER']
    polygon_to_polygon_types = ['NONE', 'PRED', 'SUCC', 'LEFT', 'RIGHT']
    
    for lane_segment in map_api.get_scenario_lane_segments():
        lane_segment_idx = polygon_ids.index(lane_segment.id)
        centerline = torch.from_numpy(centerlines[lane_segment.id].xyz).float()
        polygon_position[lane_segment_idx] = centerline[0, :dim]
        polygon_orientation[lane_segment_idx] = torch.atan2(centerline[1, 1] - centerline[0, 1],
                                                            centerline[1, 0] - centerline[0, 0])
        polygon_height[lane_segment_idx] = centerline[1, 2] - centerline[0, 2]
        polygon_type[lane_segment_idx] = polygon_types.index(lane_segment.lane_type.value)
        polygon_is_intersection[lane_segment_idx] = polygon_is_intersections.index(
            lane_segment.is_intersection)

        left_boundary = torch.from_numpy(lane_segment.left_lane_boundary.xyz).float()
        right_boundary = torch.from_numpy(lane_segment.right_lane_boundary.xyz).float()
        point_position[lane_segment_idx] = torch.cat([left_boundary[:-1, :dim],
                                                        right_boundary[:-1, :dim],
                                                        centerline[:-1, :dim]], dim=0)
        left_vectors = left_boundary[1:] - left_boundary[:-1]
        right_vectors = right_boundary[1:] - right_boundary[:-1]
        center_vectors = centerline[1:] - centerline[:-1]
        point_orientation[lane_segment_idx] = torch.cat([torch.atan2(left_vectors[:, 1], left_vectors[:, 0]),
                                                            torch.atan2(right_vectors[:, 1], right_vectors[:, 0]),
                                                            torch.atan2(center_vectors[:, 1], center_vectors[:, 0])],
                                                        dim=0)
        point_magnitude[lane_segment_idx] = torch.norm(torch.cat([left_vectors[:, :2],
                                                                    right_vectors[:, :2],
                                                                    center_vectors[:, :2]], dim=0), p=2, dim=-1)
        point_height[lane_segment_idx] = torch.cat([left_vectors[:, 2], right_vectors[:, 2], center_vectors[:, 2]],
                                                    dim=0)
        left_type = point_types.index(lane_segment.left_mark_type.value)
        right_type = point_types.index(lane_segment.right_mark_type.value)
        center_type = point_types.index('CENTERLINE')
        point_type[lane_segment_idx] = torch.cat(
            [torch.full((len(left_vectors),), left_type, dtype=torch.uint8),
                torch.full((len(right_vectors),), right_type, dtype=torch.uint8),
                torch.full((len(center_vectors),), center_type, dtype=torch.uint8)], dim=0)
        point_side[lane_segment_idx] = torch.cat(
            [torch.full((len(left_vectors),), point_sides.index('LEFT'), dtype=torch.uint8),
                torch.full((len(right_vectors),), point_sides.index('RIGHT'), dtype=torch.uint8),
                torch.full((len(center_vectors),), point_sides.index('CENTER'), dtype=torch.uint8)], dim=0)
    
    for crosswalk in map_api.get_scenario_ped_crossings():
        crosswalk_idx = polygon_ids.index(crosswalk.id)
        edge1 = torch.from_numpy(crosswalk.edge1.xyz).float()
        edge2 = torch.from_numpy(crosswalk.edge2.xyz).float()
        start_position = (edge1[0] + edge2[0]) / 2
        end_position = (edge1[-1] + edge2[-1]) / 2
        polygon_position[crosswalk_idx] = start_position[:dim]
        polygon_position[crosswalk_idx + len(cross_walk_ids)] = end_position[:dim]
        polygon_orientation[crosswalk_idx] = torch.atan2((end_position - start_position)[1],
                                                            (end_position - start_position)[0])
        polygon_orientation[crosswalk_idx + len(cross_walk_ids)] = torch.atan2((start_position - end_position)[1],
                                                                                (start_position - end_position)[0])
        polygon_height[crosswalk_idx] = end_position[2] - start_position[2]
        polygon_height[crosswalk_idx + len(cross_walk_ids)] = start_position[2] - end_position[2]
        polygon_type[crosswalk_idx] = polygon_types.index('PEDESTRIAN')
        polygon_type[crosswalk_idx + len(cross_walk_ids)] = polygon_types.index('PEDESTRIAN')
        polygon_is_intersection[crosswalk_idx] = polygon_is_intersections.index(None)
        polygon_is_intersection[crosswalk_idx + len(cross_walk_ids)] = polygon_is_intersections.index(None)

        if side_to_directed_lineseg((edge1[0] + edge1[-1]) / 2, start_position, end_position) == 'LEFT':
            left_boundary = edge1
            right_boundary = edge2
        else:
            left_boundary = edge2
            right_boundary = edge1
        num_centerline_points = math.ceil(torch.norm(end_position - start_position, p=2, dim=-1).item() / 2.0) + 1
        centerline = torch.from_numpy(
            compute_midpoint_line(left_ln_boundary=left_boundary.numpy(),
                                    right_ln_boundary=right_boundary.numpy(),
                                    num_interp_pts=int(num_centerline_points))[0]).float()

        point_position[crosswalk_idx] = torch.cat([left_boundary[:-1, :dim],
                                                    right_boundary[:-1, :dim],
                                                    centerline[:-1, :dim]], dim=0)
        point_position[crosswalk_idx + len(cross_walk_ids)] = torch.cat(
            [right_boundary.flip(dims=[0])[:-1, :dim],
                left_boundary.flip(dims=[0])[:-1, :dim],
                centerline.flip(dims=[0])[:-1, :dim]], dim=0)
        left_vectors = left_boundary[1:] - left_boundary[:-1]
        right_vectors = right_boundary[1:] - right_boundary[:-1]
        center_vectors = centerline[1:] - centerline[:-1]
        point_orientation[crosswalk_idx] = torch.cat(
            [torch.atan2(left_vectors[:, 1], left_vectors[:, 0]),
                torch.atan2(right_vectors[:, 1], right_vectors[:, 0]),
                torch.atan2(center_vectors[:, 1], center_vectors[:, 0])], dim=0)
        point_orientation[crosswalk_idx + len(cross_walk_ids)] = torch.cat(
            [torch.atan2(-right_vectors.flip(dims=[0])[:, 1], -right_vectors.flip(dims=[0])[:, 0]),
                torch.atan2(-left_vectors.flip(dims=[0])[:, 1], -left_vectors.flip(dims=[0])[:, 0]),
                torch.atan2(-center_vectors.flip(dims=[0])[:, 1], -center_vectors.flip(dims=[0])[:, 0])], dim=0)
        point_magnitude[crosswalk_idx] = torch.norm(torch.cat([left_vectors[:, :2],
                                                                right_vectors[:, :2],
                                                                center_vectors[:, :2]], dim=0), p=2, dim=-1)
        point_magnitude[crosswalk_idx + len(cross_walk_ids)] = torch.norm(
            torch.cat([-right_vectors.flip(dims=[0])[:, :2],
                        -left_vectors.flip(dims=[0])[:, :2],
                        -center_vectors.flip(dims=[0])[:, :2]], dim=0), p=2, dim=-1)
        point_height[crosswalk_idx] = torch.cat([left_vectors[:, 2], right_vectors[:, 2], center_vectors[:, 2]],
                                                dim=0)
        point_height[crosswalk_idx + len(cross_walk_ids)] = torch.cat(
            [-right_vectors.flip(dims=[0])[:, 2],
                -left_vectors.flip(dims=[0])[:, 2],
                -center_vectors.flip(dims=[0])[:, 2]], dim=0)
        crosswalk_type = point_types.index('CROSSWALK')
        center_type = point_types.index('CENTERLINE')
        point_type[crosswalk_idx] = torch.cat([
            torch.full((len(left_vectors),), crosswalk_type, dtype=torch.uint8),
            torch.full((len(right_vectors),), crosswalk_type, dtype=torch.uint8),
            torch.full((len(center_vectors),), center_type, dtype=torch.uint8)], dim=0)
        point_type[crosswalk_idx + len(cross_walk_ids)] = torch.cat(
            [torch.full((len(right_vectors),), crosswalk_type, dtype=torch.uint8),
                torch.full((len(left_vectors),), crosswalk_type, dtype=torch.uint8),
                torch.full((len(center_vectors),), center_type, dtype=torch.uint8)], dim=0)
        point_side[crosswalk_idx] = torch.cat(
            [torch.full((len(left_vectors),), point_sides.index('LEFT'), dtype=torch.uint8),
                torch.full((len(right_vectors),), point_sides.index('RIGHT'), dtype=torch.uint8),
                torch.full((len(center_vectors),), point_sides.index('CENTER'), dtype=torch.uint8)], dim=0)
        point_side[crosswalk_idx + len(cross_walk_ids)] = torch.cat(
            [torch.full((len(right_vectors),), point_sides.index('LEFT'), dtype=torch.uint8),
                torch.full((len(left_vectors),), point_sides.index('RIGHT'), dtype=torch.uint8),
                torch.full((len(center_vectors),), point_sides.index('CENTER'), dtype=torch.uint8)], dim=0)

    num_points = torch.tensor([point.size(0) for point in point_position], dtype=torch.long)
    point_to_polygon_edge_index = torch.stack(
        [torch.arange(num_points.sum(), dtype=torch.long),
            torch.arange(num_polygons, dtype=torch.long).repeat_interleave(num_points)], dim=0)
    polygon_to_polygon_edge_index = []
    polygon_to_polygon_type = []
    for lane_segment in map_api.get_scenario_lane_segments():
        lane_segment_idx = polygon_ids.index(lane_segment.id)
        pred_inds = []
        for pred in lane_segment.predecessors:
            pred_idx = safe_list_index(polygon_ids, pred)
            if pred_idx is not None:
                pred_inds.append(pred_idx)
        if len(pred_inds) != 0:
            polygon_to_polygon_edge_index.append(
                torch.stack([torch.tensor(pred_inds, dtype=torch.long),
                                torch.full((len(pred_inds),), lane_segment_idx, dtype=torch.long)], dim=0))
            polygon_to_polygon_type.append(
                torch.full((len(pred_inds),), polygon_to_polygon_types.index('PRED'), dtype=torch.uint8))
        succ_inds = []
        for succ in lane_segment.successors:
            succ_idx = safe_list_index(polygon_ids, succ)
            if succ_idx is not None:
                succ_inds.append(succ_idx)
        if len(succ_inds) != 0:
            polygon_to_polygon_edge_index.append(
                torch.stack([torch.tensor(succ_inds, dtype=torch.long),
                                torch.full((len(succ_inds),), lane_segment_idx, dtype=torch.long)], dim=0))
            polygon_to_polygon_type.append(
                torch.full((len(succ_inds),), polygon_to_polygon_types.index('SUCC'), dtype=torch.uint8))
        if lane_segment.left_neighbor_id is not None:
            left_idx = safe_list_index(polygon_ids, lane_segment.left_neighbor_id)
            if left_idx is not None:
                polygon_to_polygon_edge_index.append(
                    torch.tensor([[left_idx], [lane_segment_idx]], dtype=torch.long))
                polygon_to_polygon_type.append(
                    torch.tensor([polygon_to_polygon_types.index('LEFT')], dtype=torch.uint8))
        if lane_segment.right_neighbor_id is not None:
            right_idx = safe_list_index(polygon_ids, lane_segment.right_neighbor_id)
            if right_idx is not None:
                polygon_to_polygon_edge_index.append(
                    torch.tensor([[right_idx], [lane_segment_idx]], dtype=torch.long))
                polygon_to_polygon_type.append(
                    torch.tensor([polygon_to_polygon_types.index('RIGHT')], dtype=torch.uint8))
    if len(polygon_to_polygon_edge_index) != 0:
        polygon_to_polygon_edge_index = torch.cat(polygon_to_polygon_edge_index, dim=1)
        polygon_to_polygon_type = torch.cat(polygon_to_polygon_type, dim=0)
    else:
        polygon_to_polygon_edge_index = torch.tensor([[], []], dtype=torch.long)
        polygon_to_polygon_type = torch.tensor([], dtype=torch.uint8)

    map_data = {
        'map_polygon': {},
        'map_point': {},
        ('map_point', 'to', 'map_polygon'): {},
        ('map_polygon', 'to', 'map_polygon'): {},
    }
    map_data['map_polygon']['num_nodes'] = num_polygons
    map_data['map_polygon']['position'] = polygon_position
    map_data['map_polygon']['orientation'] = polygon_orientation
    if dim == 3:
        map_data['map_polygon']['height'] = polygon_height
    map_data['map_polygon']['type'] = polygon_type
    map_data['map_polygon']['is_intersection'] = polygon_is_intersection
    if len(num_points) == 0:
        map_data['map_point']['num_nodes'] = 0
        map_data['map_point']['position'] = torch.tensor([], dtype=torch.float)
        map_data['map_point']['orientation'] = torch.tensor([], dtype=torch.float)
        map_data['map_point']['magnitude'] = torch.tensor([], dtype=torch.float)
        if dim == 3:
            map_data['map_point']['height'] = torch.tensor([], dtype=torch.float)
        map_data['map_point']['type'] = torch.tensor([], dtype=torch.uint8)
        map_data['map_point']['side'] = torch.tensor([], dtype=torch.uint8)
    else:
        map_data['map_point']['num_nodes'] = num_points.sum().item()
        map_data['map_point']['position'] = torch.cat(point_position, dim=0)
        map_data['map_point']['orientation'] = torch.cat(point_orientation, dim=0)
        map_data['map_point']['magnitude'] = torch.cat(point_magnitude, dim=0)
        if dim == 3:
            map_data['map_point']['height'] = torch.cat(point_height, dim=0)
        map_data['map_point']['type'] = torch.cat(point_type, dim=0)
        map_data['map_point']['side'] = torch.cat(point_side, dim=0)
    map_data['map_point', 'to', 'map_polygon']['edge_index'] = point_to_polygon_edge_index
    map_data['map_polygon', 'to', 'map_polygon']['edge_index'] = polygon_to_polygon_edge_index
    map_data['map_polygon', 'to', 'map_polygon']['type'] = polygon_to_polygon_type

    return map_data
'''
'map_polygon'：一个字典，包含地图多边形的信息。
'num_nodes'：地图多边形的数量。
'position'：一个形状为(num_polygons, 3)的张量，表示每个多边形的位置坐标。
'orientation'：一个形状为(num_polygons,)的张量，表示每个多边形的方向角度。
'height'：一个形状为(num_polygons,)的张量，表示每个多边形的高度（如果dim为3）。
'type'：一个形状为(num_polygons,)的张量，表示每个多边形的类型。
'is_intersection'：一个形状为(num_polygons,)的张量，表示每个多边形是否为交叉口。

'map_point'：一个字典，包含地图点的信息。
'num_nodes'：地图点的数量。
'position'：一个形状为(total_num_points, 3)的张量，表示所有点的位置坐标。
'orientation'：一个形状为(total_num_points,)的张量，表示所有点的方向角度。
'magnitude'：一个形状为(total_num_points,)的张量，表示所有点的大小。
'height'：一个形状为(total_num_points,)的张量，表示所有点的高度（如果dim为3）。
'type'：一个形状为(total_num_points,)的张量，表示所有点的类型。
'side'：一个形状为(total_num_points,)的张量，表示所有点的侧面。

('map_point', 'to', 'map_polygon')：一个字典，包含地图点到地图多边形的边的信息。
'edge_index'：一个形状为(2, total_num_points)的张量，表示地图点到地图多边形的边的索引。

('map_polygon', 'to', 'map_polygon')：一个字典，包含地图多边形到地图多边形的边的信息。
'edge_index'：一个形状为(2, total_num_edges)的张量，表示地图多边形到地图多边形的边的索引。
'type'：一个形状为(total_num_edges,)的张量，表示地图多边形到地图多边形的边的类型。
'''

raw='/mnt/ve_share2/zy/Argoverse_2_Motion_Forecasting_Dataset/raw/val/00010486-9a07-48ae-b493-cf4545855937/scenario_00010486-9a07-48ae-b493-cf4545855937.parquet'
df = pd.read_parquet(raw)

map_path ='/mnt/ve_share2/zy/Argoverse_2_Motion_Forecasting_Dataset/raw/val/00010486-9a07-48ae-b493-cf4545855937/log_map_archive_00010486-9a07-48ae-b493-cf4545855937.json'
map_path=Path(map_path)
map_data = read_json_file(map_path)

from IPython import embed;embed()

centerlines = {lane_segment['id']: Polyline.from_json_data(lane_segment['centerline'])
                for lane_segment in map_data['lane_segments'].values()}
map_api = ArgoverseStaticMap.from_json(map_path)
data = dict()
data['scenario_id'] =get_scenario_id(df)
data['city'] =get_city(df)
data['agent'] =get_agent_features(df)
data.update(get_map_features(map_api, centerlines))
# with open(os.path.join(.processed_dir, f'{raw_file_name}.pkl'), 'wb') as handle:
#     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)