import numpy as np
import pandas as pd


def demographic_parity_difference(y_true, y_pred, sensitive_attribute):
    """
    Calcule la différence de parité démographique entre groupes.
    Retourne un dict avec les taux positifs par groupe et la différence.
    """
    groups = np.unique(sensitive_attribute)
    rates = {}
    for g in groups:
        mask = sensitive_attribute == g
        rates[g] = np.mean(y_pred[mask] == 1)

    values = list(rates.values())
    difference = max(values) - min(values)
    return {"rates": rates, "difference": difference, "groups": groups.tolist()}


def disparate_impact_ratio(y_true, y_pred, sensitive_attribute, unprivileged_value, privileged_value):
    """
    Calcule le ratio d'impact disproportionné (DI = taux_non_privilégié / taux_privilégié).
    Un ratio < 0.8 indique une discrimination selon la règle des 4/5.
    """
    mask_unpriv = sensitive_attribute == unprivileged_value
    mask_priv = sensitive_attribute == privileged_value

    rate_unpriv = np.mean(y_pred[mask_unpriv] == 1) if mask_unpriv.sum() > 0 else 0
    rate_priv = np.mean(y_pred[mask_priv] == 1) if mask_priv.sum() > 0 else 1

    ratio = rate_unpriv / rate_priv if rate_priv > 0 else 0
    return {
        "ratio": ratio,
        "rate_unprivileged": rate_unpriv,
        "rate_privileged": rate_priv,
        "unprivileged": unprivileged_value,
        "privileged": privileged_value,
    }


def equalized_odds_difference(y_true, y_pred, sensitive_attribute):
    """
    Calcule la différence d'égalisation des chances (TPR et FPR par groupe).
    """
    groups = np.unique(sensitive_attribute)
    tpr = {}
    fpr = {}
    for g in groups:
        mask = sensitive_attribute == g
        y_t = y_true[mask]
        y_p = y_pred[mask]
        tp = np.sum((y_t == 1) & (y_p == 1))
        fn = np.sum((y_t == 1) & (y_p == 0))
        fp = np.sum((y_t == 0) & (y_p == 1))
        tn = np.sum((y_t == 0) & (y_p == 0))
        tpr[g] = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr[g] = fp / (fp + tn) if (fp + tn) > 0 else 0

    tpr_diff = max(tpr.values()) - min(tpr.values())
    fpr_diff = max(fpr.values()) - min(fpr.values())
    return {"tpr": tpr, "fpr": fpr, "tpr_diff": tpr_diff, "fpr_diff": fpr_diff}
