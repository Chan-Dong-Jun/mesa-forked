"""
Wraps the model class of mesa and extends it by caching functionality.

Core Objects: CacheableModel
"""

import glob
import itertools
import os

from pathlib import Path
from typing import Optional, Callable
import pandas as pd

from mesa import Model, Agent
from mesa.space import MultiGrid as Grid

import pyarrow as pa
import pyarrow.parquet as pq

class CacheableModel:
    """Class that takes a model and writes its steps to a cache file or reads them from a cache file."""
    def __init__(
        self,
        model: Model,
        cache_file_path: str,
        total_steps: int,
        cache_interval: int = 100,
        condition_function: Optional[Callable] = None,
    ) -> None:
        """Create a new caching wrapper around an existing mesa model instance.

        Attributes:
            model: mesa model
            cache_file_path: cache file to write to or read from
            total_steps: total steps simulation is to be run
            cache_interval: interval to cache results
            condition_function: condition function to detect critical points in simulation to cache
        """

        self.model = model
        self.cache_file_path = Path(cache_file_path)
        self._total_steps = total_steps
        self.step_count: int = 0

        self._cache_interval = cache_interval

        self._last_cached_step = 0  # inclusive since it is the bottom bound of slicing
        self.condition_function = condition_function

    def get_agent_vars_dataframe(self):
        """Create a pandas DataFrame from the agent variables.

        The DataFrame has one column for each variable, with two additional
        columns for tick and agent_id.
        """
        # Check if self.agent_reporters dictionary is empty, if so raise warning
        if not self.model.datacollector.agent_reporters:
            raise UserWarning(
                "No agent reporters have been defined in the DataCollector, returning empty DataFrame."
            )

        all_records = itertools.chain.from_iterable(
            self.model.datacollector._agent_records.values()
        )
        rep_names = list(self.model.datacollector.agent_reporters)

        df = pd.DataFrame.from_records(
            data=all_records,
            columns=["Step", "AgentID", *rep_names],
            index=["Step", "AgentID"],
        )
        sliced_df = df.loc[self._last_cached_step: self.model._steps]
        return sliced_df

    def get_model_vars_dataframe(self):
        """Create a pandas DataFrame from the model variables.
        The DataFrame has one column for each model variable, and the index is
        (implicitly) the model tick.
        """
        # Check if self.model_reporters dictionary is empty, if so raise warning
        if not self.model.datacollector.model_reporters:
            raise UserWarning(
                "No model reporters have been defined in the DataCollector, returning empty DataFrame."
            )

        return pd.DataFrame(self.model.datacollector.model_vars)[
            self._last_cached_step : self.model._steps
        ]

    def _save_to_parquet(self, model):
        """Save the current cache of data to a Parquet file and clear the cache."""
        model_df = self.get_model_vars_dataframe()
        agent_df = self.get_agent_vars_dataframe()
        padding = len(str(self._total_steps)) - 1

        model_file = f"{self.cache_file_path}/model_data_{-(self.model._steps // -self._cache_interval):0{padding}}.parquet"
        agent_file = f"{self.cache_file_path}/agent_data_{-(self.model._steps // -self._cache_interval):0{padding}}.parquet"

        self.cache_file_path.mkdir(parents=True, exist_ok=True)

        absolute_path = os.path.abspath(model_file)
        if os.path.exists(absolute_path):
            raise FileExistsError(
                f"A directory with the name {model_file} already exists."
            )
        if os.path.exists(model_file):
            raise FileExistsError(
                f"A directory with the name {model_file} already exists."
            )
        if os.path.exists(agent_file):
            raise FileExistsError(
                f"A directory with the name {agent_file} already exists."
            )

        if not model_df.empty:
            model_table = pa.Table.from_pandas(model_df)
            pq.write_table(model_table, model_file)

        if not agent_df.empty:
            agent_table = pa.Table.from_pandas(agent_df)
            pq.write_table(agent_table, agent_file)

    def get_agent_vars_dataframe(self):
        # Check if self.agent_reporters dictionary is empty, if so raise warning
        if not self.model.datacollector.agent_reporters:
            raise UserWarning(
                "No agent reporters have been defined in the DataCollector, returning empty DataFrame."
            )

        all_records = itertools.chain.from_iterable(
            self.model.datacollector._agent_records.values()
        )
        rep_names = list(self.model.datacollector.agent_reporters)

        df = pd.DataFrame.from_records(
            data=all_records,
            columns=["Step", "AgentID", *rep_names],
            index=["Step", "AgentID"],
        )
        sliced_df = df.loc[self._last_cached_step: self.model._steps]
        return sliced_df

    def get_model_vars_dataframe(self):
        # Check if self.model_reporters dictionary is empty, if so raise warning
        if not self.model.datacollector.model_reporters:
            raise UserWarning(
                "No model reporters have been defined in the DataCollector, returning empty DataFrame."
            )

        return pd.DataFrame(self.model.datacollector.model_vars)[
            self._last_cached_step : self.model._steps
        ]

    def cache(self):
        if (
                self.model._steps % self._cache_interval == 0
                or self.model._steps == self._total_steps
        ):
            self._save_to_parquet(self.model)
            self._last_cached_step = self.model._steps

        if self.condition_function and self.save_critical_result(
                self.condition_function
        ):
            pass

    def read_model_data(self):
        """Read and combine all model data Parquet files into a single DataFrame."""
        model_files = glob.glob(f"{self.cache_file_path}/model_data_*.parquet")
        model_dfs = []

        for model_file in model_files:
            table = pq.read_table(model_file)
            df = table.to_pandas()
            model_dfs.append(df)

        if model_dfs:
            model_df = pd.concat(model_dfs, ignore_index=True)
            return model_df
        else:
            raise FileNotFoundError("No model data files found.")

    def read_agent_data(self):
        """Read and combine all agent data Parquet files into a single DataFrame."""
        agent_files = glob.glob(f"{self.cache_file_path}/agent_data_*.parquet")
        agent_dfs = []

        for agent_file in agent_files:
            table = pq.read_table(agent_file)
            df = table.to_pandas()
            agent_dfs.append(df)

        if agent_dfs:
            agent_df = pd.concat(agent_dfs)
            return agent_df
        else:
            raise FileNotFoundError("No agent data files found.")

    def combine_dataframes(self):
        """Combine and return the model and agent DataFrames."""
        try:
            model_df = self.read_model_data()
            agent_df = self.read_agent_data()

            # Sort agent DataFrame by the multi-index (Step, AgentID) to ensure order
            agent_df = agent_df.sort_index()

            return model_df, agent_df
        except FileNotFoundError as e:
            print(e)
            return None, None

    def save_critical_result(self, condition_function: Callable[[dict], bool]):
        model_vars = self.model.datacollector.model_vars
        self.cache_file_path.mkdir(parents=True, exist_ok=True)

        current_step = self.model._steps
        special_results_file = f"{self.cache_file_path}/special_results.parquet"
        if condition_function(model_vars):
            step_data = {key: [value[-1]] for key, value in model_vars.items()}
            step_data["Step"] = current_step
            special_results_df = pd.DataFrame(step_data)

            # Append the current step data to the Parquet file
            if os.path.exists(special_results_file):
                existing_data = pq.read_table(special_results_file).to_pandas()
                combined_data = pd.concat(
                    [existing_data, special_results_df], ignore_index=True
                )
                special_results_table = pa.Table.from_pandas(combined_data)
            else:
                special_results_table = pa.Table.from_pandas(special_results_df)

            pq.write_table(special_results_table, special_results_file)

    def get_grid_dataframe(self, cache_file_path: str = None):
        grid_state = {
            'width': self.model.grid.width,
            'height': self.model.grid.height,
            'agents': []
        }
        for x in range(grid_state['width']):
            for y in range(grid_state['height']):
                cell_contents = self.model.grid._grid[x][y]
                if cell_contents:
                    if not hasattr(cell_contents, "__iter__"):
                        cell_contents = [cell_contents]
                    for agent in cell_contents:
                        agent_state = {
                            'pos_x': agent.pos[0],
                            'pos_y': agent.pos[1],
                            'unique_id': agent.unique_id,
                            'wealth': agent.wealth,
                            # **agent.__dict__
                        }
                        grid_state['agents'].append(agent_state)
        padding = len(str(self._total_steps)) - 1
        filename = f"{self.cache_file_path}/grid_data_{(self.model._steps):0{padding}}.parquet"

        # Convert to DataFrame
        df = pd.DataFrame(grid_state['agents'])

        # Save DataFrame to Parquet
        df.to_parquet(filename)

    @staticmethod
    def reconstruct_grid(filename, *attributes_list):
        # Load the DataFrame from Parquet
        df = pd.read_parquet(filename)

        # Create a new Grid instance
        width = df['pos_x'].max() + 1  # Assuming positions start from 0
        height = df['pos_y'].max() + 1  # Assuming positions start from 0
        grid = Grid(width, height, False)

        # Add agents to the grid
        for _, row in df.iterrows():
            agent = Agent(row['unique_id'], Model(100, 10, 10))
            agent.wealth = row["wealth"]
            grid.place_agent(agent, (row['pos_x'], row['pos_y']))

        return grid

    def get_continuous_space(self):
        pass

