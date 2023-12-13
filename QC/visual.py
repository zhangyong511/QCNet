import matplotlib.pyplot as plt
import numpy as np
import sys
# sys.path.append("/mnt/ve_share2/zy/HIVT")
# from argoverse_api.argoverse.map_representation.map_api import ArgoverseMap
from av2.map.map_api import ArgoverseStaticMap

from typing import Callable, Dict, List, Optional, Tuple, Union
from typing import Final, List, Optional, Sequence, Set, Tuple
from av2.utils.typing import NDArrayFloat, NDArrayInt
from pathlib import Path
_LANE_SEGMENT_COLOR: Final[str] = "#E0E0E0"
_DRIVABLE_AREA_COLOR: Final[str] = "#7A7A7A"
'''
过去轨迹 (Past Trajectory):

color="#ECA154"：这是过去轨迹的颜色，用浅橙色表示。
地面真实轨迹 (Ground Trut   ):

color="#d33e4c"：这是地面真实轨迹的颜色，用深红色表示。
预测轨迹 (Forecasted Trajectory):

color="#007672"：这是预测轨迹的颜色，用深绿色表示。
箭头和终点标记的颜色:

箭头和终点标记的颜色与相应轨迹的颜色一致，分别是橙色、红色和绿色。
'''
def plot_single_vehicle(
    sample_past_trajectory: np.ndarray,  # 过去轨迹的坐标数组，形状为 (1, 50, 2)
    sample_groundtruth: np.ndarray,  # 地面真实轨迹的坐标数组，形状为 (1, 60, 2)
    sample_forecasted_trajectories: List[np.ndarray],  # 预测轨迹的列表，每个数组形状为 (6,60, 2)
    scenario_id,
):

    plt.figure()
 
    min_x = min(
        np.min(sample_past_trajectory[:, :, 0]),
        np.min(sample_groundtruth[:, :, 0]),
        np.min([np.min(traj[:, 0]) for traj in sample_forecasted_trajectories])
    )
    max_x = max(
        np.max(sample_past_trajectory[:, :, 0]),
        np.max(sample_groundtruth[:, :, 0]),
        np.max([np.max(traj[:, 0]) for traj in sample_forecasted_trajectories])
    )
    min_y = min(
        np.min(sample_past_trajectory[:, :, 1]),
        np.min(sample_groundtruth[:, :, 1]),
        np.min([np.min(traj[:, 1]) for traj in sample_forecasted_trajectories])
    )
    max_y = max(
        np.max(sample_past_trajectory[:, :, 1]),
        np.max(sample_groundtruth[:, :, 1]),
        np.max([np.max(traj[:, 1]) for traj in sample_forecasted_trajectories])
    )

    x_buffer = 5
    y_buffer = 5

    plt.plot(
        sample_past_trajectory[0, :, 0],
        sample_past_trajectory[0, :, 1],
        color="#ECA154",
        label="Past Trajectory",
        alpha=1,
        linewidth=2,
        zorder=15,
        ls="--"
    )

    plt.plot(
        sample_groundtruth[0, :, 0],
        sample_groundtruth[0, :, 1],
        color="#d33e4c",
        label="Ground Truth",
        alpha=1,
        linewidth=2,
        zorder=20,
        ls="--"
    )

    for i, sample_forecasted_trajectory in enumerate(sample_forecasted_trajectories):
        plt.plot(
            sample_forecasted_trajectory[:, 0],
            sample_forecasted_trajectory[:, 1],
            color="#007672",
            label=f"Forecasted Trajectory {i + 1}",
            alpha=1,
            linewidth=2,
            zorder=20,
            ls="--"
        )

        # Plot the end marker for forecasted trajectories
        plt.arrow(
            sample_forecasted_trajectory[-2, 0],
            sample_forecasted_trajectory[-2, 1],
            sample_forecasted_trajectory[-1, 0] - sample_forecasted_trajectory[-2, 0],
            sample_forecasted_trajectory[-1, 1] - sample_forecasted_trajectory[-2, 1],
            color="#007672",
            label="Forecasted Trajectory",
            alpha=1,
            linewidth=3,
            zorder=25,
            head_width=0.3,
            head_length=0.3
        )

    # Plot the end marker for history
    plt.arrow(
        sample_past_trajectory[0, -2, 0],
        sample_past_trajectory[0, -2, 1],
        sample_past_trajectory[0, -1, 0] - sample_past_trajectory[0, -2, 0],
        sample_past_trajectory[0, -1, 1] - sample_past_trajectory[0, -2, 1],
        color="#ECA154",
        label="Past Trajectory",
        alpha=1,
        linewidth=2.5,
        zorder=25,
        head_width=0.1,
    )

    # Plot the end marker for ground truth
    plt.arrow(
        sample_groundtruth[0, -2, 0],
        sample_groundtruth[0, -2, 1],
        sample_groundtruth[0, -1, 0] - sample_groundtruth[0, -2, 0],
        sample_groundtruth[0, -1, 1] - sample_groundtruth[0, -2, 1],
        color="#d33e4c",
        label="Ground Truth",
        alpha=1,
        linewidth=3,
        zorder=30,
        head_width=0.1,
    )
    static_map_path=f"/mnt/ve_share2/zy/Argoverse_2_Motion_Forecasting_Dataset/raw/val/{scenario_id}/log_map_archive_{scenario_id}.json"
    static_map_path=Path(static_map_path)
    static_map = ArgoverseStaticMap.from_json(static_map_path)
    _plot_static_map_elements(static_map)

    plt.xlim(min_x - x_buffer, max_x + x_buffer)
    plt.ylim(min_y - y_buffer, max_y + y_buffer)
    plt.savefig(f'/mnt/ve_share2/zy/QCNet/zy/{scenario_id}.png')

def _plot_static_map_elements(
    static_map: ArgoverseStaticMap, show_ped_xings: bool = False
) -> None:
    """Plot all static map elements associated with an Argoverse scenario.

    Args:
        static_map: Static map containing elements to be plotted.
        show_ped_xings: Configures whether pedestrian crossings should be plotted.
    """
    # Plot drivable areas
    for drivable_area in static_map.vector_drivable_areas.values():
        _plot_polygons([drivable_area.xyz], alpha=0.5, color=_DRIVABLE_AREA_COLOR)

    # Plot lane segments
    for lane_segment in static_map.vector_lane_segments.values():
        _plot_polylines(
            [
                lane_segment.left_lane_boundary.xyz,
                lane_segment.right_lane_boundary.xyz,
            ],
            line_width=0.5,
            color=_LANE_SEGMENT_COLOR,
        )

    # Plot pedestrian crossings
    if show_ped_xings:
        for ped_xing in static_map.vector_pedestrian_crossings.values():
            _plot_polylines(
                [ped_xing.edge1.xyz, ped_xing.edge2.xyz],
                alpha=1.0,
                color=_LANE_SEGMENT_COLOR,
            )

def _plot_polylines(
    polylines: Sequence[NDArrayFloat],
    *,
    style: str = "-",
    line_width: float = 1.0,
    alpha: float = 1.0,
    color: str = "r",
) -> None:
    """Plot a group of polylines with the specified config.

    Args:
        polylines: Collection of (N, 2) polylines to plot.
        style: Style of the line to plot (e.g. `-` for solid, `--` for dashed)
        line_width: Desired width for the plotted lines.
        alpha: Desired alpha for the plotted lines.
        color: Desired color for the plotted lines.
    """
    for polyline in polylines:
        plt.plot(
            polyline[:, 0],
            polyline[:, 1],
            style,
            linewidth=line_width,
            color=color,
            alpha=alpha,
        )


def _plot_polygons(
    polygons: Sequence[NDArrayFloat], *, alpha: float = 1.0, color: str = "r"
) -> None:
    """Plot a group of filled polygons with the specified config.

    Args:
        polygons: Collection of polygons specified by (N,2) arrays of vertices.
        alpha: Desired alpha for the polygon fill.
        color: Desired color for the polygon.
    """
    for polygon in polygons:
        plt.fill(polygon[:, 0], polygon[:, 1], color=color, alpha=alpha)

