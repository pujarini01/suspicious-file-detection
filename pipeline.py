import pandas as pd
import hashlib
import mimetypes

# ---------- Settings ----------
SIZE_LIMIT_KB = 100_000
RISKY = ["/data/local/tmp/", "/.hidden/", "/.Trash/", "/tmp/"]
EMPTY_HASH = hashlib.sha256(b"").hexdigest()

# These columns MUST be in the uploaded file
REQUIRED = ["File_Name", "File_Size_KB", "File_Path",
            "Created_Time", "Modified_Time", "Accessed_Time"]

# Points given to each problem
WEIGHTS = {
    "Chk_Extension_Mismatch": 30, "Chk_Timestamp_Anomaly": 25,
    "Chk_Unusual_Location": 25, "Chk_Duplicate": 15,
    "Chk_Unusual_Size": 15, "Chk_Empty_File": 10, "Missing_Value_Flag": 10,
}
# Readable text for each problem
CHECK_NAMES = {
    "Chk_Extension_Mismatch": "Extension does not match actual file type",
    "Chk_Empty_File": "Empty file (size 0)",
    "Chk_Unusual_Size": "Unusually large file",
    "Chk_Unusual_Location": "Stored in a risky location",
    "Chk_Timestamp_Anomaly": "Timestamp order is unusual",
    "Chk_Duplicate": "Duplicate content (same hash as another file)",
    "Missing_Value_Flag": "Missing metadata value",
}


def run_pipeline(raw):
    # ---- A. Check that the important columns exist ----
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = raw.copy()

    # ---- B. Add optional columns if the file does not have them ----
    if "File_Type" in df.columns and "Expected_File_Type" not in df.columns:
        df = df.rename(columns={"File_Type": "Expected_File_Type"})
    if "Record_ID" not in df.columns:
        df.insert(0, "Record_ID", range(1, len(df) + 1))
    if "Expected_File_Type" not in df.columns:
        df["Expected_File_Type"] = df["File_Name"].apply(
            lambda n: mimetypes.guess_type(str(n))[0] or "unknown")
    if "Actual_File_Type" not in df.columns:
        df["Actual_File_Type"] = df["Expected_File_Type"]
    if "File_Hash" not in df.columns:
        df["File_Hash"] = None
    if "Device_Type" not in df.columns:
        df["Device_Type"] = None

    # ---- C. MODULE 2: cleaning ----
    for c in ["Created_Time", "Modified_Time", "Accessed_Time"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df["File_Size_KB"] = pd.to_numeric(df["File_Size_KB"], errors="coerce")
    for c in ["Device_Type", "File_Name", "File_Path"]:
        df[c] = df[c].astype("string").str.strip()

    # Only look at columns that really have data
    cols = ["Device_Type", "File_Name", "File_Size_KB", "File_Path",
            "Created_Time", "Modified_Time", "Accessed_Time"]
    cols = [c for c in cols if df[c].notna().any()]
    df["Missing_Value_Flag"] = df[cols].isnull().any(axis=1)

    # Remove rows that are exactly the same
    ignore = ["Record_ID", "Condition", "Source"]
    df = df.drop_duplicates(subset=[c for c in df.columns if c not in ignore])
    df = df.reset_index(drop=True)

    # ---- D. MODULE 3: file checks ----
    df["Chk_Extension_Mismatch"] = df["Expected_File_Type"] != df["Actual_File_Type"]
    df["Chk_Empty_File"] = df["File_Size_KB"] == 0
    df["Chk_Unusual_Size"] = df["File_Size_KB"] > SIZE_LIMIT_KB
    df["Chk_Unusual_Location"] = df["File_Path"].apply(
        lambda p: any(r.lower() in str(p).lower() for r in RISKY))
    df["Chk_Timestamp_Anomaly"] = (
        (df["Modified_Time"] < df["Created_Time"]) |
        (df["Accessed_Time"] < df["Modified_Time"]))

    # ---- E. MODULE 4: duplicate files using the hash ----
    h = df["File_Hash"]
    group_size = h.map(h.value_counts())
    df["Chk_Duplicate"] = h.notna() & (group_size > 1) & (h != EMPTY_HASH)

    # ---- F. MODULE 5: score and level ----
    df["Check_Reasons"] = df.apply(
        lambda row: "; ".join(m for c, m in CHECK_NAMES.items() if row[c]), axis=1)
    df["Suspicion_Score"] = sum(df[c].astype(int) * w for c, w in WEIGHTS.items())
    df["Suspicion_Score"] = df["Suspicion_Score"].clip(upper=100)
    df["Suspicion_Level"] = df["Suspicion_Score"].apply(
        lambda s: "High" if s >= 35 else "Medium" if s >= 15 else "Low")
    return df


# ---------- MODULE 6: timeline ----------
def build_timeline(df):
    ev = df.melt(
        id_vars=["Record_ID", "File_Name", "Device_Type",
                 "Suspicion_Score", "Suspicion_Level", "Check_Reasons"],
        value_vars=["Created_Time", "Modified_Time", "Accessed_Time"],
        var_name="Event", value_name="Event_Time")
    ev["Event"] = ev["Event"].str.replace("_Time", "", regex=False)
    ev = ev.dropna(subset=["Event_Time"]).sort_values("Event_Time")
    return ev.reset_index(drop=True)