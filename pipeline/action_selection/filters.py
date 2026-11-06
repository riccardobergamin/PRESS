"""Choice labels do not require verified eventual possession outcomes."""


def select(actions):
    return actions.loc[actions.is_canonical & actions.state_eligible &
        actions.training_cohort_eligible].copy()
