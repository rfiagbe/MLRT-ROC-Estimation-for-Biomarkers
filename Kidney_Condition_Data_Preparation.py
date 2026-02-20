#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 11:24:05 2026

@author: roland
"""


#==============================================================================
#======================== IMPORTING RELEVANT LIBRARIES ========================
#==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os


#==============================================================================
#======================== SETTING WORKING DIRECTORY ===========================
#==============================================================================
## Set working directory
directory = '/Users/Roland/Desktop/UCF LIBRARY/DISSERTATION/REAL DATA APPLICATION/Kidney Datasets'
os.chdir(directory)
print(os.getcwd())


#==============================================================================
#================ IMPORTING ALL KIDNEY CONDTITION DATASETS ====================
#==============================================================================

# Make sure to set working directory to the folder that contains all the data files for 
# Kidney condition survey data
folder_path = '/Users/Roland/Desktop/UCF LIBRARY/DISSERTATION/REAL DATA APPLICATION/Kidney Datasets'
xpt_files = glob.glob(os.path.join(folder_path, "*.xpt"))
dfs = []
reference_cols = None

for file in xpt_files:
    df = pd.read_sas(file, format="xport")
    df = df.iloc[:, :2]

    if reference_cols is None:
        reference_cols = df.columns
    else:
        df.columns = reference_cols  # enforce consistency

    dfs.append(df)

combined_df = pd.concat(dfs, axis=0, ignore_index=True)

print(combined_df.head())
combined_df.shape


final_kidney_data = combined_df 
final_kidney_data.shape


#==============================================================================
#================ IMPORTING ALL CADMIUM AND LEAD DATASETS =====================
#==============================================================================

# Make sure to set working directory to the folder that contains all the data files for 
# the Cadmium, Lead, Mercury, Cotinine & Nutritional Biochemistries laboratory data
folder_path = '/Users/Roland/Desktop/UCF LIBRARY/DISSERTATION/REAL DATA APPLICATION/Cadmium and Lead Biomarker Datasets'

# Variables to keep
vars_to_keep = ["SEQN", "LBXBPB", "LBDBPBSI", "LBXBCD"]
xpt_files = glob.glob(os.path.join(folder_path, "*.xpt"))

dfs = []
reference_cols = vars_to_keep

for file in xpt_files:
    df = pd.read_sas(file, format="xport")
    
    # Keep only selected variables (if present)
    df_subset = df[vars_to_keep]
    
    # Enforce consistent column order
    df_subset = df_subset[reference_cols]
    
    dfs.append(df_subset)

# Concatenate all datasets row-wise
final_cadmium_data = pd.concat(dfs, axis=0, ignore_index=True)

print(final_cadmium_data.head())
print(final_cadmium_data.shape)

final_cadmium_data["SEQN"].value_counts()



#==============================================================================
#================ IMPORTING ALL METALS - URINE DATASETS =======================
#==============================================================================

# Make sure to set working directory to the folder that contains all the data files for 
# the Metals - Urine Laboratory data
folder_path = '/Users/Roland/Desktop/UCF LIBRARY/DISSERTATION/REAL DATA APPLICATION/Metals Urine Biomarker Datasets'

xpt_files = glob.glob(os.path.join(folder_path, "*.xpt"))

dfs = []

for file in xpt_files:
    df = pd.read_sas(file, format="xport")
    
    if not {"SEQN", "URDUCD", "URXUCD"}.intersection(df.columns):
        continue
    
    cols = [c for c in ["SEQN", "URDUCD", "URXUCD"] if c in df.columns]
    df_subset = df[cols]
    
    df_long = df_subset.melt(
        id_vars="SEQN",
        value_vars=[c for c in ["URDUCD", "URXUCD"] if c in df_subset.columns],
        var_name="cadmium_type",
        value_name="cadmium_value"
    )
    
    dfs.append(df_long)

final_metal_urine_data = pd.concat(dfs, axis=0, ignore_index=True)

print(final_metal_urine_data.head())

final_metal_urine_data.shape

final_metal_urine_data["SEQN"].value_counts().reset_index()


#==============================================================================
#========================= DATA PREP AND CLEANING =============================
#==============================================================================

# Count occurrences of each SEQN to make sure we have unique IDs
seqn_counts = combined_df["SEQN"].value_counts().reset_index()
seqn_counts.columns = ["SEQN", "count"]
seqn_counts_sorted = seqn_counts.sort_values(by="count", ascending=False)
seqn_counts_sorted


##  Missing Values Summary by Column
def missing_summary(df):
    """
    Returns a DataFrame with the count and percentage of missing values
    for each column in df.
    """
    missing_count = df.isna().sum()
    missing_percent = (missing_count / len(df)) * 100

    summary = pd.DataFrame({
        "missing_count": missing_count,
        "missing_percent": missing_percent.round(2)
    })

    return summary

missing_table = missing_summary(final_cadmium_data)
print(missing_table)


final_cadmium_data.shape
final_metal_urine_data.shape

missing_summary(final_cadmium_data)
missing_summary(final_metal_urine_data)


## Joining datasets
df_joined1 = pd.merge(final_cadmium_data, final_metal_urine_data, on="SEQN", how="inner")
df_joined1.shape


df_joined = pd.merge(df_joined1, final_kidney_data, on="SEQN", how="inner")

df_joined.shape
df_joined.head()
df_joined["KIQ022"].value_counts()



# Count occurrences of each SEQN to make sure we have unique IDs
seqn_counts = df_joined["SEQN"].value_counts().reset_index()
seqn_counts.columns = ["SEQN", "count"]
seqn_counts_sorted = seqn_counts.sort_values(by="count", ascending=False)
seqn_counts_sorted


missing_summary(df_joined)

df_joined.columns

df_joined = df_joined.rename(columns={"KIQ022": "group"})
df_joined.columns
df_joined["group"].value_counts()


final_kidney_condition_data = df_joined[['group', 'LBXBPB', 'cadmium_value']]
final_kidney_condition_data = final_kidney_condition_data[final_kidney_condition_data["group"].isin([1, 2])]

final_kidney_condition_data.shape
final_kidney_condition_data["group"].value_counts()


## Saving final data to directory
final_kidney_condition_data.to_csv("final_kidney_condition_data.csv", index=False)




#==============================================================================
#================= CHECKING DISTRIBUTION OF THE DATA & EDA ====================
#==============================================================================

## Partitioning the groups
disease_group_bio1 = final_kidney_condition_data[final_kidney_condition_data['group'] == 1][['LBXBPB']]
disease_group_bio2 = final_kidney_condition_data[final_kidney_condition_data['group'] == 1][['cadmium_value']]

healthy_group_bio1 = final_kidney_condition_data[final_kidney_condition_data['group'] == 2][['LBXBPB']]
healthy_group_bio2 = final_kidney_condition_data[final_kidney_condition_data['group'] == 2][['cadmium_value']]


# Histogram of Biomarker 1
fig, axes = plt.subplots(2, 2, figsize=(15, 8))
axes[0].hist(disease_group_bio1, bins=50, color='white', edgecolor='black', alpha=0.7)
axes[0].set_title("Distribution of Blood Lead Level for Disease Group", fontsize=14, fontweight='bold')
axes[0].set_xlabel("Blood Lead Levels", fontsize=12)
axes[0].set_ylabel("Frequency", fontsize=12)
axes[0].grid(alpha=0.3)

# Histogram of Biomarker 2
axes[1].hist(disease_group_bio2, bins=50, color='white', edgecolor='black', alpha=0.7)
axes[1].set_title("Distribution of Urine Cadmium for Disease Group", fontsize=14, fontweight='bold')
axes[1].set_xlabel("Urine Cadmium Values", fontsize=12)
axes[1].set_ylabel("Frequency", fontsize=12)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()



# Histogram of Biomarker 1
fig, axes = plt.subplots(1, 2, figsize=(15, 8))
axes[0].hist(healthy_group_bio1, bins=50, color='white', edgecolor='black', alpha=0.7)
axes[0].set_title("Distribution of Blood Lead Level for Healthy Group", fontsize=14, fontweight='bold')
axes[0].set_xlabel("Blood Lead Levels", fontsize=12)
axes[0].set_ylabel("Frequency", fontsize=12)
axes[0].grid(alpha=0.3)

# Histogram of Biomarker 2
axes[1].hist(healthy_group_bio2, bins=80, color='white', edgecolor='black', alpha=0.7)
axes[1].set_title("Distribution of Urine Cadmium for Healthy Group", fontsize=14, fontweight='bold')
axes[1].set_xlabel("Urine Cadmium Values", fontsize=12)
axes[1].set_ylabel("Frequency", fontsize=12)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()


