import pandas as pd
import os
import re

# ==============================
# CONFIG
# ==============================
INPUT_FILE = "Umesh.Mail.Reminder.Base.File.March.2026.xlsx"
OUTPUT_DIR = "output_files"

# ==============================
# CREATE OUTPUT FOLDER
# ==============================
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# LOAD DATA
# ==============================
df = pd.read_excel(INPUT_FILE)

# ==============================
# CLEAN COLUMN NAMES (strip spaces)
# ==============================
df.columns = df.columns.str.strip()

# ==============================
# CLEAN "ZBM Mailer list"
# ==============================
# Normalize values
col = df["ZBM Mailer list"]

# Remove Excel errors and string "#N/A"
col = col.replace("#N/A", pd.NA)

# Strip only non-null values (IMPORTANT FIX)
col = col.where(col.isna(), col.astype(str).str.strip())

# Convert empty strings to NA
col = col.replace("", pd.NA)

# Assign back
df["ZBM Mailer list"] = col

# Drop invalid rows
df = df.dropna(subset=["ZBM Mailer list"])

# ==============================
# GLOBAL UNIQUE COUNT (for log)
# ==============================
global_unique_db = df["ZBM Mailer list"].nunique()

# ==============================
# HELPER: CLEAN FILENAME
# ==============================
def clean_filename(text):
    return re.sub(r'[\\/*?:"<>|]', "_", str(text))

# ==============================
# OUTPUT COLUMN MAPPING
# ==============================
column_mapping = {
    "ZBM Mailer list": "DB Code",
    "Customer Name": "DB Name",
    "TBM Employ. Name": "TBM Name",
    "ABM Name": "ABM Name",
    "ZBM Name": "ZBM Name"
}

required_columns = list(column_mapping.keys())

# ==============================
# GROUPING & FILE CREATION
# ==============================
log_data = []

grouped = df.groupby("ZBM Emp code")

for zbm_code, group in grouped:

    # Extract ZBM Name safely
    zbm_name = group["ZBM Name"].dropna().iloc[0] if not group["ZBM Name"].dropna().empty else "Unknown"

    # ==============================
    # PREPARE OUTPUT DATA
    # ==============================
    output_df = group[required_columns].copy()
    output_df = output_df.rename(columns=column_mapping)
    # Fix DB Code format (remove .0)
    output_df["DB Code"] = output_df["DB Code"].apply(lambda x: str(int(float(x))) if pd.notna(x) else x)
    # ==============================
    # SAVE FILE
    # ==============================
    filename = f"{clean_filename(zbm_code)}_{clean_filename(zbm_name)}.xlsx"
    file_path = os.path.join(OUTPUT_DIR, filename)

    output_df.to_excel(file_path, index=False)

    # ==============================
    # LOG CALCULATIONS
    # ==============================
    log_entry = {
        "ZBM Emp code": zbm_code,
        "ZBM Name": zbm_name,
        "Unique ZBM Mailer list": group["ZBM Mailer list"].nunique(),
        "Unique TBM Employ. Code": group["TBM Employ. Code"].nunique(),
        "Unique ABM Code": group["ABM Code"].nunique(),
        "Total Unique ZBM Mailer list (Overall)": global_unique_db
    }

    log_data.append(log_entry)

# ==============================
# SAVE LOG FILE
# ==============================
log_df = pd.DataFrame(log_data)
log_file_path = os.path.join(OUTPUT_DIR, "ZBM_summary_log.xlsx")
log_df.to_excel(log_file_path, index=False)

# ==============================
# DONE
# ==============================
print("✅ Processing complete!")
print(f"📁 Files saved in: {OUTPUT_DIR}")
print(f"📊 Log file: {log_file_path}")