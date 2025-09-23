import csv
import warnings
import matplotlib.pyplot as plt

class Patient: 

    all_patients = []

    death_age = []

    education_lvl = {}

    def __init__(self, DonorID, ABeta40: float , ABeta42: float, tTau: float, pTau: float, MMSE: float = None):

        self.DonorID = DonorID
        self.ABeta40 = ABeta40
        self.ABeta42 = ABeta42
        self.tTau = tTau
        self.pTau = pTau
        self.sex = None
        self.death_age = None
        self.ed_lvl = None
        self.cog_stat = None
        self.age_symp_on = None
        self.age_diag = None 
        self.head_inj = None
        self.thal_score = None
        self.MMSE = MMSE
        Patient.all_patients.append(self)

    def __repr__(self):
        return f"{self.DonorID} | sex: {self.sex} | ABeta40 {self.ABeta40} | tTau {self.tTau} | pTau {self.pTau} | Death Age {self.death_age} | Thal Score {self.thal_score} | MMSE {self.MMSE}"

    def get_id(self):
        return self.DonorID

    def get_ABeta42(self):
        return self.ABeta42
    
    def get_thal(self):
        return self.thal_score
    
    def get_death_age(self):
        return self.death_age


    @classmethod
    def combine_data(cls, filename: str):
            with open(filename, encoding="utf8") as f:
                reader = csv.DictReader(f)
                rows_of_patients = list(reader)
                #for line in csv create object
                for row in range(len(rows_of_patients)):
                    if Patient.all_patients[row].DonorID == rows_of_patients[row]["Donor ID"]:
                        if rows_of_patients[row]["Sex"] != "":
                            Patient.all_patients[row].sex = rows_of_patients[row]["Sex"]

                        if rows_of_patients[row]["Age at Death"] != "":
                            Patient.all_patients[row].death_age = int(rows_of_patients[row]["Age at Death"])

                        if rows_of_patients[row]["Highest level of education"] != "":
                            Patient.all_patients[row].ed_lvl = rows_of_patients[row]["Highest level of education"]

                        if rows_of_patients[row]["Cognitive Status"] != "":
                            Patient.all_patients[row].cog_stat = rows_of_patients[row]["Cognitive Status"]

                        if rows_of_patients[row]["Age of onset cognitive symptoms"] != "":
                            Patient.all_patients[row].age_symp_on = int(rows_of_patients[row]["Age of onset cognitive symptoms"])

                        if rows_of_patients[row]["Age of Dementia diagnosis"] != "":
                            Patient.all_patients[row].age_diag = int(rows_of_patients[row]["Age of Dementia diagnosis"])

                        if rows_of_patients[row]["Known head injury"] != "":
                            Patient.all_patients[row].head_inj = rows_of_patients[row]["Known head injury"]

                        if rows_of_patients[row]["Thal"] != "":
                            Patient.all_patients[row].thal_score = int(rows_of_patients[row]["Thal"].split()[1])
                        if rows_of_patients[row]["Last MMSE Score"] != "":
                            try:
                                Patient.all_patients[row].MMSE = float(rows_of_patients[row]["Last MMSE Score"])
                            except ValueError:
                                Patient.all_patients[row].MMSE = None
            
                    else:
                        warnings.warn("IDs do not match.")

    @classmethod
    def sort_ed(cls):
        for patient in Patient.all_patients:
            Patient.education_lvl.update({patient.ed_lvl: []})
        for patient in Patient.all_patients:
            Patient.education_lvl[patient.ed_lvl].append(patient)

    @classmethod
    def subsort_thal(cls):
        for key in Patient.education_lvl:
            values = Patient.education_lvl.get(key)
            values.sort(key = Patient.get_thal)
            Patient.education_lvl.update({key: values})

    @classmethod
    def instantiate_from_csv(cls, filename: str, other_file: str):
    #open csv and create list of all rows
        with open(filename, encoding="utf8") as f:
            reader = csv.DictReader(f)
            rows_of_patients = list(reader)
            #for line in csv create object
            for row in rows_of_patients:
                Patient(
                    DonorID = row['Donor ID'],
                    ABeta40 = float(row['ABeta40 pg/ug']),
                    ABeta42 = float(row['ABeta42 pg/ug']),
                    tTau = float(row['tTAU pg/ug']),
                    pTau = float(row['pTAU pg/ug']),
                    
                )
            Patient.all_patients.sort(key = Patient.get_id)
            Patient.combine_data(other_file)
    
    @classmethod
    def filter(cls, patients, **criteria):
        """
        Return all patients from the list that match the given keyword arguments.
        Example: Patient.filter(Patient.all_patients, sex="Female", cog_stat="No dementia")
        """
        results = patients
        for attr, value in criteria.items():
            results = [p for p in results if getattr(p, attr) == value]
        return results


