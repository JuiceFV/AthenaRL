from typing import Tuple

import pandas as pd


def gen_discrete_mdp_pandas_df(dt: bool) -> Tuple[pd.DataFrame, str]:
    actions = ["1", "2", "3", "4"]
    next_actions = ["2", "3", "4", ""]
    rewards = [0, 1, 4, 5]
    metrics = [{"G": 0}, {"G": 1}, {"G": 4}, {"G": 5}]
    time_diffs = [1, 1, 1, 1] if dt else [1, 3, 1, 1]
    mdp_ids = ["0"] * 4
    sequence_numbers = [0, 1, 4, 5]
    sequence_number_ordinals = [1, 2, 3, 4]
    states = [{0: 1}, {1: 1}, {4: 1}, {5: 1}]
    next_states = [{1: 1}, {4: 1}, {5: 1}, {6: 1}]
    actions_probabilities = [0.3, 0.4, 0.5, 0.6]

    df = pd.DataFrame(
        {
            "mdp_id": mdp_ids,
            "sequence_number": sequence_numbers,
            "sequence_number_ordinal": sequence_number_ordinals,
            "state_features": states,
            "actions": actions,
            "actions_probability": actions_probabilities,
            "reward": rewards,
            "next_state_features": next_states,
            "next_actions": next_actions,
            "time_diff": time_diffs,
            "metrics": metrics,
            "ds_id": ["2022-09-01"] * 4
        }
    )
    return df


def gen_parametric_mdp_pandas_df(dt: bool) -> pd.DataFrame:
    actions = [{0: 1}, {3: 1}, {2: 1}, {1: 2}]
    next_actions = [{3: 1}, {2: 1}, {1: 2}, {}]
    rewards = [0, 1, 4, 5]
    metrics = [{"G": 0}, {"G": 1}, {"G": 4}, {"G": 5}]
    states = [{0: 1}, {1: 1}, {4: 1}, {5: 1}]
    next_states = [{1: 1}, {4: 1}, {5: 1}, {6: 1}]
    actions_probabilities = [0.3, 0.4, 0.5, 0.6]
    time_diffs = [1, 1, 1, 1] if dt else [1, 3, 1, 1]
    mdp_ids = ["0"] * 4
    sequence_numbers = [0, 1, 4, 5]
    sequence_number_ordinals = [1, 2, 3, 4]
    df = pd.DataFrame(
        {
            "mdp_id": mdp_ids,
            "sequence_number": sequence_numbers,
            "sequence_number_ordinal": sequence_number_ordinals,
            "state_features": states,
            "actions": actions,
            "actions_probability": actions_probabilities,
            "reward": rewards,
            "next_state_features": next_states,
            "next_actions": next_actions,
            "time_diff": time_diffs,
            "metrics": metrics,
            "ds_id": ["2022-09-01"] * 4
        }
    )
    return df
