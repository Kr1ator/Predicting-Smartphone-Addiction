"""Rules for staged feature screening."""

CV_STD = 0.000473

# delta = experiment_auc - baseline_auc

def delta_in_cv_std(delta):
    return delta / CV_STD


def screening_decision(delta):
    effect = delta_in_cv_std(delta)

    if effect < 0.25:
        return "Reject"
    if effect < 0.75:
        return "Maybe"
    if effect < 1.00:
        return "Promote"
    return "Strong candidate"


def fold2_confirmation(fold1_delta, fold2_delta):
    mean_delta = (fold1_delta + fold2_delta) / 2
    confirmed = fold2_delta > 0 and delta_in_cv_std(mean_delta) >= 0.50
    return "Confirmed" if confirmed else "Not confirmed"


def next_step(fold1_decision, fold1_delta, fold2_delta=None):
    if fold1_decision == "Reject":
        return "Stop"
    if fold1_decision == "Maybe":
        return "Hold"
    if fold2_delta is None:
        return "Run Fold 2"
    if fold2_confirmation(fold1_delta, fold2_delta) == "Confirmed":
        return "Run full 5-Fold"
    return "Reject"
