package com.digi.spark.rl

object Constants {
    val RL_DATA_COLUMN_NAMES = Array(
        "ds_id",
        "mdp_id",
        "state_features",
        "actions",
        "actions_probability",
        "reward",
        "next_state_features",
        "next_actions",
        "sequence_number",
        "sequence_number_ordinal",
        "metrics"
    );

    val RL4R_DATA_COLUMN_NAMES = Array(
        "ds_id",
        "mdp_id",
        "sequence_number",
        "slate_reward",
        "item_reward",
        "actions",
        "actions_probability",
        "state_features",
        "state_sequence_features",
        "next_actions",
        "next_state_features",
        "next_state_sequence_features"
    );

    val DEFAULT_REWARD_COLUMNS = List[String](
        "reward",
        "metrics"
    );

    val DEFAULT_EXTRA_FEATURE_COLUMNS = List[String]();

    val DEFAULT_REWARD_TYPES = Map(
        "reward" -> "double",
        "metrics" -> "map<string, double>"
    );
}