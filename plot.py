#2) PRINT OUR LIST OF PATIENTS
from patient import Patient

Patient.instantiate_from_csv('UpdatedLuminex.csv', 'UpdatedMetaData.csv')

# for patient in Patient.all_patients:
#     print(patient)


#3) SORT OUR LIST OF PATIENTS

Patient.all_patients.sort(key=Patient.get_thal, reverse=False)

for patient in Patient.all_patients:
    print(patient)

#4) BUILD DICTIONARIES TO DO SORTING AND SUB-SORTING

from termcolor import colored

Patient.sort_ed()
Patient.subsort_thal()

for key in Patient.education_lvl:
    print(colored(key,"red"))
    for patient in Patient.education_lvl.get(key):
        print(patient)
    print()

#6) MAKE A FILTER TO PULL OUT PATIENTS WITH SPECIIFC ATTRIBUTES

@classmethod
def filter(cls, list, ABeta40:float ="any", ABeta42:float ="any", tTau:float= "any", pTau:float ="any", sex:str ="any", death_age:int ="any", ed_lvl:str ="any", cog_stat:str ="any", age_symp_on:int ="any", age_diag:int ="any", head_inj:str ="any", thal_score:int ="any"):
        all_patients = list
        remove_list = []
        attr_list = (
                    ABeta40,
                    ABeta42,
                    tTau,
                    pTau,
                    sex,
                    death_age,
                    ed_lvl,
                    cog_stat,
                    age_symp_on,
                    age_diag,
                    head_inj,
                    thal_score
                    )
        attr_name = (
                    "ABeta40",
                    "ABeta42",
                    "tTau",
                    "pTau",
                    "sex",
                    "death_age",
                    "ed_lvl",
                    "cog_stat",
                    "age_symp_on",
                    "age_diag",
                    "head_inj",
                    "thal_score"
                    )
        for attr in range(len(attr_list)):
            if attr_list[attr] != "any":
                for patient in all_patients:
                    if getattr(patient,attr_name[attr]) != attr_list[attr]:
                        remove_list.append(patient)
                all_patients = [patient for patient in all_patients if patient not in remove_list]
                remove_list.clear()
        
        return all_patients


fem_healty_patients = range(len(Patient.filter(Patient.all_patients, sex = "Female", cog_stat = "No dementia")))
male_healthy_patients = range(len(Patient.filter(Patient.all_patients, sex = "Male", cog_stat = "No dementia")))
fem_diseased_patients = range(len(Patient.filter(Patient.all_patients, sex = "Female", cog_stat = "Dementia")))
male_diseased_patients = range(len(Patient.filter(Patient.all_patients, sex = "Male", cog_stat = "Dementia")))

print(f'Female Healthy Patients = {len(fem_healty_patients)} | Male Healthy Patients = {len(male_healthy_patients)}')
print(f'Female Diseased Patients = {len(fem_diseased_patients)} | Male Diseased Patients = {len(male_diseased_patients)}')

#7) PLOT A BAR GRAPH OF THE SORTED DATA

from patient import Patient
from termcolor import colored
from matplotlib import pyplot as plt
from scipy import stats
import numpy as np
import statistics 


Patient.instantiate_from_csv('UpdatedLuminex.csv', 'UpdatedMetaData.csv')

tTau_fem_vals = []
tTau_male_vals = []

for patient in Patient.filter(Patient.all_patients, sex = "Female"):
     tTau_fem_vals.append(patient.tTau)
for patient in Patient.filter(Patient.all_patients, sex = "Male"):
     tTau_male_vals.append(patient.tTau)

x_fem_bar = (statistics.mean(tTau_fem_vals))
x_male_bar = (statistics.mean(tTau_male_vals))

tTau_fem_stdev = (statistics.stdev(tTau_fem_vals))
tTau_male_stdev = (statistics.stdev(tTau_male_vals))

print(f'x_fem_bar = {x_fem_bar}, tTau_fem_stdev {tTau_fem_stdev}')
print(f'x_male_bar = {x_male_bar}, tTau_male_stdev {tTau_male_stdev}')

x_fem_vals = range(len(Patient.filter(Patient.all_patients, sex = "Female")))
x_male_vals = range(len(Patient.filter(Patient.all_patients, sex = "Male")))

sex_cols = ['Female', 'Male']
mean_sex_ABeta42 = [x_fem_bar, x_male_bar]
stdev_sex_ABeta42 = [tTau_fem_stdev, tTau_male_stdev]
colors = ["pink", "blue"]
yerr = [np.zeros(len(mean_sex_ABeta42)), stdev_sex_ABeta42]

# Equal variance assumed
t_stat, p_val = stats.ttest_ind(tTau_fem_vals, tTau_male_vals)
print("t-statistic:", t_stat)
print("p-value:", p_val)

plt.bar(sex_cols, mean_sex_ABeta42, yerr=yerr, capsize=10, color=["red", "blue"])
plt.title("tTau Levels")
plt.xlabel("Sex")
plt.ylabel("tTau")
plt.show()


# #8) PLOT A BAR GRAPH OF THE FURTHER SORTED DATA

# ABeta42_fem_diseased_vals = []
# ABeta42_male_diseased_vals = []

# for patient in Patient.filter(Patient.all_patients, sex = "Female", cog_stat = "Dementia"):
#      ABeta42_fem_diseased_vals.append(patient.ABeta42)
# for patient in Patient.filter(Patient.all_patients, sex = "Male", cog_stat = "Dementia"):
#      ABeta42_male_diseased_vals.append(patient.ABeta42)

# x_fem_diseased_bar = (statistics.mean(ABeta42_fem_diseased_vals))
# x_male_diseased_bar = (statistics.mean(ABeta42_male_diseased_vals))

# ABeta42_fem_diseased_stdev = (statistics.stdev(ABeta42_fem_diseased_vals))
# ABeta42_male_diseased_stdev = (statistics.stdev(ABeta42_male_diseased_vals))

# print(f'x_fem_diseased_bar = {x_fem_diseased_bar}, ABeta42_fem_diseased_stdev {ABeta42_fem_diseased_stdev}')
# print(f'x_male_diseased_bar = {x_male_diseased_bar}, ABeta42_male_diseased_stdev {ABeta42_male_diseased_stdev}')

# x_fem_diseased_vals = range(len(Patient.filter(Patient.all_patients, sex = "Female")))
# x_male_diseased_vals = range(len(Patient.filter(Patient.all_patients, sex = "Male")))

# sex_diseased_cols = ['Diseased Females', 'Diseased Males']
# mean_sex_diseased_ABeta42 = [x_fem_diseased_bar, x_male_diseased_bar]
# stdev_sex_diseased_ABeta42 = [ABeta42_fem_diseased_stdev, ABeta42_male_diseased_stdev]
# colors = ["pink", "blue"]
# yerr = [np.zeros(len(mean_sex_diseased_ABeta42)), stdev_sex_diseased_ABeta42]

# # Equal variance assumed
# t_stat, p_val = stats.ttest_ind(ABeta42_fem_diseased_vals, ABeta42_male_diseased_vals)
# print("t-statistic:", t_stat)
# print("p-value:", p_val)

# plt.bar(sex_diseased_cols, mean_sex_diseased_ABeta42, yerr=yerr, capsize=10, color=["pink", "skyblue"])
# plt.title("ABeta42 Levels in Diseased Patients")
# plt.xlabel("Sex")
# plt.ylabel("Abeta42")
# plt.show()


# death_age_list = []
# ABeta42 = []

# for patient in Patient.all_patients:
#        death_age_list.append(patient.death_age)

# for patient in Patient.all_patients:
#        ABeta42.append(patient.ABeta42)

# X = [death_age_list]  # Independent variable
# y = [ABeta42]   # Dependent variable

# print(X)
# print(y)


# #10) VISUALIZE DATA ON A SCATTER PLOT

# plt.scatter(X, y, color='blue')
# plt.xlabel('Age of Death')
# plt.ylabel('ABeta42')
# plt.title('Scatter Plot of Age of Death vs ABeta42')
# plt.show()

# #11) EXPORT DATA TO A .csv FILE

import pandas as pd

# print(death_age_list)
# print(ABeta42)

# # Create a DataFrame
# df = pd.DataFrame({
#     'Age of Death': death_age_list,
#     'ABeta42': ABeta42
# })

# # Write to CSV
# df.to_csv('patient_data.csv', index=False)

# print("CSV file 'patient_data.csv' has been created.")

#12) LOAD LIBRARIES FOR A LINEAR REGRESSION
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#13) LOAD DATA SET FOR A LINEAR REGRESSION

df = pd.read_csv("patient_data.csv")

#14) Update these variable names to match EXACTLY your .csv file headers
# --- CHANGE THESE TO MATCH YOUR HEADERS EXACTLY ---
X_HEADER = "Age of Death"   # e.g. might actually be "death_age"
Y_HEADER = "ABeta42"        # e.g. might actually be "ABeta42(pg/ml)"

# --- CLEAN AND PREPARE DATA ---
# force to numeric (bad strings -> NaN)
df[[X_HEADER, Y_HEADER]] = df[[X_HEADER, Y_HEADER]].apply(pd.to_numeric, errors="coerce")
# drop rows with missing values
df_clean = df.dropna(subset=[X_HEADER, Y_HEADER]).copy()

# build numpy arrays
x = df_clean[[X_HEADER]].to_numpy()   # ensures 2D
y = df_clean[Y_HEADER].to_numpy()     # ensures 1D

#15) Perform the linear regression
model = LinearRegression()
model.fit(x, y)

slope = model.coef_[0]
intercept = model.intercept_
r2 = model.score(x, y)


#16) Make scatterplot
plt.scatter(x, y, label="Data")
plt.plot(x, model.predict(x), color="red")

# Annotate equation
equation = f"y = {slope:.2f}x + {intercept:.2f}\nR² = {r2:.2f}"
plt.text(x.min(), y.max(), equation, color="red", fontsize=12, verticalalignment='top')

# Annotate scatterplot with labeles and title
plt.xlabel("Age of Death")
plt.ylabel("ABeta42")
plt.title("Age of Death vs. ABeta42")
plt.show()

MMSE_injured_vals = []
MMSE_noinjury_vals = []

# Separate patients into groups based on head injury
for patient in Patient.filter(Patient.all_patients, head_inj="Yes"):
    if patient.MMSE is not None:  # check if score exists
        MMSE_injured_vals.append(patient.MMSE)

for patient in Patient.filter(Patient.all_patients, head_inj="No"):
    if patient.MMSE is not None:
        MMSE_noinjury_vals.append(patient.MMSE)

# Calculate means and standard deviations
mean_MMSE_injured = statistics.mean(MMSE_injured_vals)
mean_MMSE_noinjury = statistics.mean(MMSE_noinjury_vals)

stdev_MMSE_injured = statistics.stdev(MMSE_injured_vals)
stdev_MMSE_noinjury = statistics.stdev(MMSE_noinjury_vals)

print(f'Mean MMSE (Head Injury) = {mean_MMSE_injured}, Stdev = {stdev_MMSE_injured}')
print(f'Mean MMSE (No Injury) = {mean_MMSE_noinjury}, Stdev = {stdev_MMSE_noinjury}')

# Bar chart
groups = ['Head Injury', 'No Head Injury']
means = [mean_MMSE_injured, mean_MMSE_noinjury]
stdevs = [stdev_MMSE_injured, stdev_MMSE_noinjury]
yerr = [np.zeros(len(means)), stdevs]

plt.bar(groups, means, yerr=yerr, capsize=10, color=["orange", "green"])
plt.title("Head Injury vs Last MMSE Score")
plt.xlabel("Head Injury History")
plt.ylabel("Mean MMSE Score")
plt.show()

# Optional: run a t-test
from scipy import stats
t_stat, p_val = stats.ttest_ind(MMSE_injured_vals, MMSE_noinjury_vals)
print("t-statistic:", t_stat)
print("p-value:", p_val)


import matplotlib.pyplot as plt
import statistics
import numpy as np
from scipy import stats

MMSE_injured_vals = []
MMSE_noinjury_vals = []

# Separate patients into groups based on head injury
for patient in Patient.filter(Patient.all_patients, head_inj="Yes"):
    if patient.MMSE is not None:
        MMSE_injured_vals.append(patient.MMSE)

for patient in Patient.filter(Patient.all_patients, head_inj="No"):
    if patient.MMSE is not None:
        MMSE_noinjury_vals.append(patient.MMSE)

plt.figure(figsize=(7,6))

data = [MMSE_injured_vals, MMSE_noinjury_vals]
labels = ["Head Injury", "No Head Injury"]
colors = ["orange", "green"]

# Boxplot (no outlier symbols, so scatter can show them)
box = plt.boxplot(data, patch_artist=True, labels=labels, showfliers=False)

# Color each box
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.4)

# Overlay scatter (jitter x positions slightly for visibility)
for i, vals in enumerate(data):
    x_jitter = np.random.normal(i+1, 0.04, size=len(vals))  # jitter around 1,2
    plt.scatter(x_jitter, vals, color=colors[i], alpha=0.7, edgecolors="black")

# Add mean markers
plt.scatter(1, statistics.mean(MMSE_injured_vals), color="red", marker="D", s=100, edgecolors="black")
plt.scatter(2, statistics.mean(MMSE_noinjury_vals), color="red", marker="D", s=100, edgecolors="black")

# Labels and formatting
plt.title("Head Injury vs Last MMSE Scores")
plt.ylabel("MMSE Score")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# --- Optional: t-test ---
t_stat, p_val = stats.ttest_ind(MMSE_injured_vals, MMSE_noinjury_vals)
print("t-statistic:", t_stat)
print("p-value:", p_val)


x_vals = []
y_vals = []

# Collect patients who have both MMSE and Age of Dementia Diagnosis
for patient in Patient.all_patients:
    if patient.MMSE is not None and patient.age_diag is not None:
        x_vals.append(patient.age_diag)   # Age of Dementia Diagnosis
        y_vals.append(patient.MMSE)       # MMSE Score
# --- Fit linear regression ---
model = LinearRegression()
model.fit(x, y)

# Predictions for a smooth line
x_line = np.linspace(min(x_vals), max(x_vals), 100).reshape(-1, 1)
y_line = model.predict(x_line)

# R² score
r2 = r2_score(y, model.predict(x))

# --- Plot ---
plt.figure(figsize=(7,6))
plt.scatter(x, y, color="purple", alpha=0.7, edgecolors="black", label="Patients")
plt.plot(x_line, y_line, color="red", linewidth=2, label=f"Regression Line (R²={r2:.2f})")

plt.title("MMSE Score vs Age of Dementia Diagnosis")
plt.xlabel("Age of Dementia Diagnosis")
plt.ylabel("MMSE Score")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# --- Print regression stats ---
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("R²:", r2)