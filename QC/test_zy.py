# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>
"""Script to generate dynamic visualizations from a directory of Argoverse scenarios."""

from enum import Enum, unique
from pathlib import Path
from random import choices
from typing import Final

import click
from joblib import Parallel, delayed
from rich.progress import track

from av2.datasets.motion_forecasting import scenario_serialization
from zyvisual import visualize_scenario
# from zyvisual import visualize_zy
# from av2.datasets.motion_forecasting.viz.scenario_visualization import (
#     visualize_scenario,
# )
from av2.map.map_api import ArgoverseStaticMap

_DEFAULT_N_JOBS: Final[int] = -2  # Use all but one CPUs


@unique
class SelectionCriteria(str, Enum):
    """Valid criteria used to select Argoverse scenarios for visualization."""

    FIRST: str = "first"
    RANDOM: str = "random"


def generate_scenario_visualizations(
    argoverse_scenario_dir: Path,
    viz_output_dir: Path,
    num_scenarios: int,
    selection_criteria: SelectionCriteria,
    *,
    debug: bool = False,
    predicted_trajectories,
) -> None:
    """Generate and save dynamic visualizations for selected scenarios within `argoverse_scenario_dir`.

    Args:
        argoverse_scenario_dir: Path to local directory where Argoverse scenarios are stored.
        viz_output_dir: Path to local directory where generated visualizations should be saved.
        num_scenarios: Maximum number of scenarios for which to generate visualizations.
        selection_criteria: Controls how scenarios are selected for visualization.
        debug: Runs preprocessing in single-threaded mode when enabled.
    """
    Path(viz_output_dir).mkdir(parents=True, exist_ok=True)
    all_scenario_files = sorted(argoverse_scenario_dir.rglob("*.parquet"))
    scenario_file_list = (
        all_scenario_files[:num_scenarios]
        if selection_criteria == SelectionCriteria.FIRST
        else choices(all_scenario_files, k=num_scenarios)
    )  # Ignoring type here because type of "choice" is partially unknown.

    # Build inner function to generate visualization for a single scenario.
    def generate_scenario_visualization(scenario_path: Path,predicted_trajectories) -> None:
        """Generate and save dynamic visualization for a single Argoverse scenario.

        NOTE: This function assumes that the static map is stored in the same directory as the scenario file.

        Args:
            scenario_path: Path to the parquet file corresponding to the Argoverse scenario to visualize.
        """
        scenario_id = scenario_path.stem.split("_")[-1]
        static_map_path = (
            scenario_path.parents[0] / f"log_map_archive_{scenario_id}.json"
        )
        viz_save_path = viz_output_dir / f"{scenario_id}.mp4"

        scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)
        static_map = ArgoverseStaticMap.from_json(static_map_path)
        # visualize_zy(scenario, static_map, viz_save_path,predicted_trajectories=predicted_trajectories)
        visualize_scenario(scenario, static_map, viz_save_path,predicted_trajectories=predicted_trajectories)

    # Generate visualization for each selected scenario in parallel (except if running in debug mode)
    if debug:
        for scenario_path in track(scenario_file_list):
            generate_scenario_visualization(scenario_path,predicted_trajectories)
    else:
        Parallel(n_jobs=_DEFAULT_N_JOBS)(
            delayed(generate_scenario_visualization)(scenario_path)
            for scenario_path in track(scenario_file_list)
        )


def run_generate_scenario_visualizations(
    argoverse_scenario_dir: str,
    viz_output_dir: str,
    num_scenarios: int,
    selection_criteria: str,
    debug: bool,
    predicted_trajectories= None,
) -> None:
    """Click entry point for generation of Argoverse scenario visualizations."""
    generate_scenario_visualizations(
        Path(argoverse_scenario_dir),
        Path(viz_output_dir),
        num_scenarios,
        SelectionCriteria(selection_criteria.lower()),
        debug=debug,
        predicted_trajectories=predicted_trajectories,
    )


# run_generate_scenario_visualizations(
#     '/mnt/ve_share2/zy/Argoverse_2_Motion_Forecasting_Dataset/raw/val/f93736af-279f-42ad-80e7-20a42164fec8',
#     "/mnt/ve_share2/zy/QCNet/zy",
#     1,
#     'first',
#     True,
#     predicted_trajectories
# )