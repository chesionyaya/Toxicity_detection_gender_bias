import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from src.Toxicity_classifier.classifier_toxicity import df_eval, female_comments_copy

LABEL_THRESHOLD = 0.5   #  let worker probability toxicity >= 0.5 as toxic(1)
PRED_THRESHOLD  = 0.5   # let model probability toxicity_predicted >= 0.5 as toxic(1)

def cm_and_metrics(df, group_name):
    y_true = (df["toxicity"].astype(float).to_numpy() >= LABEL_THRESHOLD).astype(int)
    y_pred = (df["toxicity_predicted"].astype(float).to_numpy() >= PRED_THRESHOLD).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else np.nan
    tpr = tp / (tp + fn) if (tp + fn) else np.nan
    fpr = fp / (fp + tn) if (fp + tn) else np.nan
    fnr = fn / (fn + tp) if (fn + tp) else np.nan

    print(f"\n===== {group_name} =====")
    print("Confusion Matrix [[TN, FP],[FN, TP]]:\n", cm)
    print(f"ACC={acc:.4f}  TPR={tpr:.4f}  FPR={fpr:.4f}  FNR={fnr:.4f}")

    return cm, {"ACC": acc, "TPR": tpr, "FPR": fpr, "FNR": fnr}

# df_eval (male) and female_comments_copy (female)
male_cm, male_m = cm_and_metrics(df_eval, "MALE")
female_cm, female_m = cm_and_metrics(female_comments_copy, "FEMALE")

# --- figure comparison ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ConfusionMatrixDisplay(male_cm, display_labels=["non-toxic", "toxic"]).plot(
    ax=axes[0], values_format="d", colorbar=False
)
axes[0].set_title("Male")

ConfusionMatrixDisplay(female_cm, display_labels=["non-toxic", "toxic"]).plot(
    ax=axes[1], values_format="d", colorbar=False
)
axes[1].set_title("Female")

plt.tight_layout()
plt.show()

#output audit difference between male and female
print("\n===== Metric gap (Female - Male) =====")
for k in ["ACC", "TPR", "FPR", "FNR"]:
    print(f"{k}: {female_m[k] - male_m[k]: .6f}   (female={female_m[k]:.6f}, male={male_m[k]:.6f})")
