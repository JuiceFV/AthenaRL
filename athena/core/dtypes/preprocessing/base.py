r"""
Describes basic types mandated by the data preprocessing process.
"""


class InputColumn:
    r"""
    Input column names after MDP operator applied.

    .. note::
        In case you modify MDP operator s.t. it requires
        constantly use new columns, list these column names
        below. Don't hardcode it manually.
    """

    #: The features of current step that are independent on the actions.
    STATE_FEATURES = "state_features"

    #: The features of documents, in case RL approach is used for the ranking problem.
    STATE_SEQUENCE_FEATURES = "state_sequence_features"

    #: The features of the subsequent step that are actions-independent.
    NEXT_STATE_FEATURES = "next_state_features"

    #: The features of the subsequent documents, in case RL approach is used for the ranking problem.
    NEXT_STATE_SEQUENCE_FEATURES = "next_state_sequence_features"

    #: The action(-s) taken at the current step.
    ACTIONS = "actions"

    #: The actions taken at the next step.
    NEXT_ACTIONS = "next_actions"

    #: The flag if an episode stem reaches horizon.
    NOT_TERMINAL = "not_terminal"

    #: Current step number
    STEP = "step"

    #: Representing the number of states between the
    #: current state and one of the next :math:`n` state. If the input table is
    #: sub-sampled states will be missing. This column allows us to know how
    #: many states are missing which can be used to adjust the discount factor.
    TIME_DIFF = "time_diff"

    #: Representing the number of states between
    #: current state and the very first state of the current episode. If the
    #: input table is sub-sampled states will be missing. This column allows us
    #: to know the derivative (i.e. approach rate) which can be used to adjust
    #: the discount factor.
    TIME_SINCE_FIRST = "time_since_first"

    #: A unique ID of episode.
    MDP_ID = "mdp_id"

    #: A number representing the location of the
    #: state in the episode before the sequence_number was converted to an
    #: ordinal number. Note, mdp_id + sequence_number makes unique ID.
    SEQUENCE_NUMBER = "sequence_number"

    #: The measure features used to calculate
    #: the reward. I.e. :math:`f(\text{metrics}) \sim \text{reward}`,
    #: the :math:`f` may be implicitly defined, but it's crucial that
    #: :math:`\frac{d\text{metrics}}{dt} = \frac{d\text{reward}}{dt}`.
    METRICS = "metrics"

    #: The reward at the current step.
    REWARD = "reward"

    #: The probability that this(-ese) action(-s) was(-ere) taken.
    ACTIONS_PROBABILITY = "actions_probability"

    #: Used only in RL4R. Reward for the entire slate (e.g. NDCG, Hit, MRR)
    SLATE_REWARD = "slate_reward"

    #: Used only in RL4R. Item relevance may be whatever it is.
    #: E.g. dot product between query and document.
    ITEM_REWARD = "item_reward"
