import matplotlib.pyplot as plt
import statistics
from scipy import stats
import numpy as np
from patient import Patient
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# --- Load Data ---
Patient.all_patients.clear()
Patient.instantiate_from_csv("UpdatedLuminex.csv", "UpdatedMetaData.csv")

# --- Alzheimer’s patients only ---
alz_patients = [p for p in Patient.all_patients if p.cog_stat in ["Dementia", "Alzheimers disease", "Alzheimers Possible/ Probable"]]

# Groups
alz_with_injury = [p for p in alz_patients if p.head_inj == "Yes" and p.age_symp_on is not None and p.age_diag is not None]
alz_no_injury   = [p for p in alz_patients if p.head_inj == "No" and p.age_symp_on is not None and p.age_diag is not None]

# Data
x_yes = [p.age_symp_on for p in alz_with_injury]
y_yes = [p.age_diag for p in alz_with_injury]
x_no  = [p.age_symp_on for p in alz_no_injury]
y_no  = [p.age_diag for p in alz_no_injury]

# --- T-tests ---
# Onset
onset_yes = [p.age_symp_on for p in alz_with_injury]
onset_no  = [p.age_symp_on for p in alz_no_injury]
t_onset, p_onset = stats.ttest_ind(onset_yes, onset_no, equal_var=False)

# Progression
prog_yes = [p.age_diag - p.age_symp_on for p in alz_with_injury]
prog_no  = [p.age_diag - p.age_symp_on for p in alz_no_injury]
t_prog, p_prog = stats.ttest_ind(prog_yes, prog_no, equal_var=False)

# --- Scatter Plot with Regression ---
plt.figure(figsize=(7,6))
plt.scatter(x_yes, y_yes, color="orange", alpha=0.7, label="Head Injury")
plt.scatter(x_no, y_no, color="green", alpha=0.7, label="No Head Injury")

def add_regression(x, y, color, anchor, label):
    if len(x) > 2:
        X = np.array(x).reshape(-1,1)
        model = LinearRegression().fit(X, y)
        slope = model.coef_[0]
        intercept = model.intercept_
        r2 = r2_score(y, model.predict(X))
        _, _, r_val, _, p_val = stats.linregress(x, y)

        # Regression line
        x_line = np.linspace(min(x), max(x), 100).reshape(-1,1)
        plt.plot(x_line, model.predict(x_line), color=color, linestyle="--")

        # Place annotation
        if anchor == "topleft":
            xpos, ypos, va, ha = min(x), max(y), "top", "left"
        elif anchor == "bottomright":
            xpos, ypos, va, ha = max(x), min(y), "bottom", "right"
        else:
            xpos, ypos, va, ha = min(x), max(y), "top", "left"

        plt.text(xpos, ypos,
                 f"{label}\ny={slope:.2f}x+{intercept:.1f}\nR²={r2:.2f}, p={p_val:.4f}",
                 color=color, fontsize=9,
                 verticalalignment=va, horizontalalignment=ha,
                 bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

# Add regressions
add_regression(x_yes, y_yes, "orange", "topleft", "Head Injury")
add_regression(x_no, y_no, "green", "bottomright", "No Head Injury")


plt.xlabel("Age of Symptom Onset")
plt.ylabel("Age of Dementia Diagnosis")
plt.title("Onset vs Diagnosis Age (AD Patients)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()




# --- Age of Onset ---
onset_yes = [p.age_symp_on for p in alz_with_injury if p.age_symp_on is not None]
onset_no  = [p.age_symp_on for p in alz_no_injury if p.age_symp_on is not None]
t_onset, p_onset = stats.ttest_ind(onset_yes, onset_no, equal_var=False)

means_onset = [statistics.mean(onset_yes), statistics.mean(onset_no)]
stdevs_onset = [statistics.stdev(onset_yes), statistics.stdev(onset_no)]

plt.figure(figsize=(6,5))
plt.bar(groups, means_onset, yerr=stdevs_onset, capsize=10, color=["orange","green"])
plt.ylabel("Age of Symptom Onset (years)")
plt.title("Head Injury vs Age of Onset (AD Patients)")

# Place annotation just above highest bar (not too high)
y_annot_onset = max(means_onset[i] + stdevs_onset[i] for i in range(2)) + 0.5
plt.text(0.5, y_annot_onset,
         f"t={t_onset:.2f}, p={p_onset:.4f}\n(n={len(onset_yes)} vs {len(onset_no)})",
         ha="center", va="bottom", fontsize=10,
         bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

plt.tight_layout()
plt.show()


# --- Progression Duration (Onset → Diagnosis) ---
prog_yes = [p.age_diag - p.age_symp_on for p in alz_with_injury if p.age_symp_on is not None and p.age_diag is not None]
prog_no  = [p.age_diag - p.age_symp_on for p in alz_no_injury if p.age_symp_on is not None and p.age_diag is not None]
t_prog, p_prog = stats.ttest_ind(prog_yes, prog_no, equal_var=False)

means_prog = [statistics.mean(prog_yes), statistics.mean(prog_no)]
stdevs_prog = [statistics.stdev(prog_yes), statistics.stdev(prog_no)]

plt.figure(figsize=(6,5))
plt.bar(groups, means_prog, yerr=stdevs_prog, capsize=10, color=["orange","green"])
plt.ylabel("Progression Duration (years)\n(Onset → Diagnosis)")
plt.title("Head Injury vs Progression to Dementia (AD Patients)")

# Place annotation just above highest bar
y_annot_prog = max(means_prog[i] + stdevs_prog[i] for i in range(2)) + 0.2
plt.text(0.5, y_annot_prog,
         f"t={t_prog:.2f}, p={p_prog:.4f}\n(n={len(prog_yes)} vs {len(prog_no)})",
         ha="center", va="bottom", fontsize=10,
         bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

plt.tight_layout()
plt.show()
