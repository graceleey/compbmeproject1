import matplotlib.pyplot as plt
import numpy as np
import statistics
from scipy import stats
from patient import Patient

# Load data
Patient.instantiate_from_csv("UpdatedLuminex.csv", "UpdatedMetaData.csv")

# --- Filter Alzheimer's patients only ---
alzheimers = [p for p in Patient.all_patients if p.cog_stat == "Dementia"]

# Collect values
onset_yes = [p.age_symp_on for p in alzheimers if p.head_inj == "Yes" and p.age_symp_on is not None]
onset_no  = [p.age_symp_on for p in alzheimers if p.head_inj == "No"  and p.age_symp_on is not None]

prog_yes = [p.age_diag - p.age_symp_on for p in alzheimers if p.head_inj == "Yes" and p.age_diag and p.age_symp_on]
prog_no  = [p.age_diag - p.age_symp_on for p in alzheimers if p.head_inj == "No"  and p.age_diag and p.age_symp_on]

# Helper plotting function
def plot_with_ttest(data1, data2, labels, title, ylabel):
    means = [statistics.mean(data1), statistics.mean(data2)]
    stdevs = [statistics.stdev(data1), statistics.stdev(data2)]
    ns = [len(data1), len(data2)]

    # t-test
    t, p = stats.ttest_ind(data1, data2, equal_var=False)

    # Bar graph
    plt.figure(figsize=(6,5))
    bars = plt.bar(labels, means, yerr=stdevs, capsize=10, color=["orange","green"], alpha=0.7)


    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()
    
   # Print results
    print(f"{title}")
    print(f" {labels[0]} (n={ns[0]}): mean={means[0]:.1f}, stdev={stdevs[0]:.1f}")
    print(f" {labels[1]} (n={ns[1]}): mean={means[1]:.1f}, stdev={stdevs[1]:.1f}")
    print(f" t={t:.2f}, p={p:.4f}\n")

# --- Plot 1: Age of Onset ---
plot_with_ttest(onset_yes, onset_no,
                labels=["Head Injury", "No Head Injury"],
                title="Head Injury vs Age of Onset (AD Patients)",
                ylabel="Age at Onset (years)")

# --- Plot 2: Progression Duration ---
plot_with_ttest(prog_yes, prog_no,
                labels=["Head Injury", "No Head Injury"],
                title="Head Injury vs Progression Duration (AD Patients)",
                ylabel="Years from Onset → Diagnosis")
