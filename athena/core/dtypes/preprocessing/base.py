r"""
Describes basic types mandated by the data preprocessing process.
"""


class InputColumn:
    r"""
    Input column names after MDP operator is applied.

    .. note::
        In case you modify MDP operator s.t. it requires
        constantly use new columns, list these column names
        below. Don't hardcode it manually.
    """

    #: The features of the current step are independent of the actions.
    STATE_FEATURES = "state_features"

    #: In case the RL approach is used for the ranking problem, the features of a sequence.
    STATE_SEQUENCE_FEATURES = "state_sequence_features"

    #: The features of the subsequent step are independent actions.
    NEXT_STATE_FEATURES = "next_state_features"

    #: The features of the subsequent documents, in case the RL approach is used for the ranking problem.
    NEXT_STATE_SEQUENCE_FEATURES = "next_state_sequence_features"

    #: The actions were taken at the current step.
    ACTIONS = "actions"

    #: The actions were taken at the next step.
    NEXT_ACTIONS = "next_actions"

    #: The flag of an episode's step reaches a horizon.
    NOT_TERMINAL = "not_terminal"

    #: Episode's iteration.
    STEP = "step"

    #: Representing the number of states between the current state and one of the next :math:`n` state.
    #: If the input table consists of sub-sampled states will be missing. This column lets us know how many
    #: states are missing, which can be used to adjust the discount factor. Applicable in the n-step RL,
    #: especially Temporal Difference (TD) learning.
    TIME_DIFF = "time_diff"

    #: Representing the number of states between the current state and the first state of the recent episode.
    #: If the input table consists of sub-sampled states will be missing. This column lets us know the derivative
    #: (i.e., approach rate), which can be used to adjust the discount factor.
    TIME_SINCE_FIRST = "time_since_first"

    #: A unique ID of an episode.
    MDP_ID = "mdp_id"

    #: A number representing a location of the state in the episode before the sequence_number was converted to an
    #: ordinal number. Note that mdp_id + sequence_number makes a unique ID.
    SEQUENCE_NUMBER = "sequence_number"

    #: Long-term metric that is intended to be optimized.
    METRICS = "metrics"

    #: Short-term reward gained at a time step.
    REWARD = "reward"

    #: The probability that an action was taken.
    ACTIONS_PROBABILITY = "actions_probability"

    #: Used only in Reinforcement Learning for Ranking. The reward for the entire slate that commonly denoted
    #: as :math:`\mathcal{R}(x,y)`. It is equivalent of ``METRICS``.
    SLATE_REWARD = "slate_reward"

    #: Used only in Reinforcement Learning for Ranking. Item relevance may be whatever it is.
    #: E.g. dot product between query and document.
    ITEM_REWARD = "item_reward"
