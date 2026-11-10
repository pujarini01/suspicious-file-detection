import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pipeline import run_pipeline, build_timeline, CHECK_NAMES

st.set_page_config(page_title="Suspicious File Detection", layout="wide")
st.title("Suspicious File Detection in Mobile Device Forensics")
st.caption("A suspicious file is not proof of wrongdoing. "
           "Final verification is done by the forensic investigator.")

# ---------- Upload ----------
uploaded = st.file_uploader("Upload file metadata (CSV)", type=["csv"])
if uploaded is None:
    st.info("Upload a CSV file to start the analysis.")
    st.stop()

raw = pd.read_csv(uploaded)          # data exactly as uploaded (Module 1)
try:
    df = run_pipeline(raw)           # data after all modules
except ValueError as e:
    st.error(str(e))
    st.stop()

# ---------- Module menu (left side) ----------
MODULES = ["1. Data Collection", "2. Data Cleaning",
           "3. File Information Checking", "4. File Integrity Checking",
           "5. Suspicious File Identification", "6. Timeline Checking",
           "7. Result Display"]
FLOW = ["Collect", "Clean", "Check", "Hash", "Score", "Timeline", "Display"]

st.sidebar.header("Modules")
module = st.sidebar.radio("Select a module", MODULES)
idx = MODULES.index(module)

# ---------- Flow bar: current step is highlighted ----------
st.markdown("  →  ".join(f":red[**{n}**]" if i == idx else n
                         for i, n in enumerate(FLOW)))
st.divider()

# ==============================================================
# MODULE 1: DATA COLLECTION
# ==============================================================
if idx == 0:
    st.header("Module 1: Data Collection")
    st.write("File information collected from the mobile device forensic data.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total records", len(raw))
    c2.metric("Columns collected", raw.shape[1])
    if "Device_Type" in raw.columns:
        c3.metric("Device types", raw["Device_Type"].nunique())

    st.subheader("Information collected for each file")
    st.write(", ".join(raw.columns))

    if "Device_Type" in raw.columns:
        st.subheader("Records by device")
        st.dataframe(raw["Device_Type"].fillna("Missing").value_counts())

    st.subheader("Collected data (first 20 rows)")
    st.dataframe(raw.head(20))

# ==============================================================
# MODULE 2: DATA CLEANING
# ==============================================================
elif idx == 1:
    st.header("Module 2: Data Cleaning")
    st.write("The collected data is checked for missing values, duplicate records, "
             "incorrect information and inconsistent file extensions.")

    # ---- Before and after ----
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records before cleaning", len(raw))
    c2.metric("Records after cleaning", len(df))
    c3.metric("Exact duplicate rows removed", len(raw) - len(df))
    c4.metric("Records with missing values", int(df["Missing_Value_Flag"].sum()))

    # ---- 1. Missing values ----
    st.subheader("1. Missing values")
    check_cols = [c for c in ["Device_Type", "File_Name", "File_Size_KB", "File_Path",
                              "Created_Time", "Modified_Time", "Accessed_Time"]
                  if c in raw.columns]
    st.write("Missing values per column (records are flagged, not deleted):")
    st.dataframe(raw[check_cols].isnull().sum().rename("Missing count"))
    st.write("Records with a missing value:")
    st.dataframe(df[df["Missing_Value_Flag"]][
        ["Record_ID", "File_Name", "Device_Type", "File_Size_KB", "Created_Time"]])

    # ---- 2. Duplicate records ----
    st.subheader("2. Duplicate records")
    st.write(f"Exact duplicate rows found and removed: {len(raw) - len(df)}")
    st.caption("Files with the same content but different names are found "
               "later, in Module 4, using hash values.")

    # ---- 3. Incorrect information ----
    st.subheader("3. Incorrect information")
    time_cols = ["Created_Time", "Modified_Time", "Accessed_Time"]
    neg_size = int((df["File_Size_KB"] < 0).sum())
    future = int((df[time_cols] > pd.Timestamp.now()).any(axis=1).sum())
    i1, i2 = st.columns(2)
    i1.metric("Negative file sizes", neg_size)
    i2.metric("Timestamps in the future", future)

    # ---- 4. Inconsistent file extensions ----
    st.subheader("4. Inconsistent file extensions")
    known = "jpg|jpeg|png|mp4|pdf|txt|docx|xlsx|zip"
    double_ext = df["File_Name"].str.contains(
        rf"\.(?:{known})\.(?:{known})$", case=False, regex=True, na=False)
    st.write(f"Files with a double extension (like data01.xlsx.xlsx): {int(double_ext.sum())}")
    st.dataframe(df[double_ext][["Record_ID", "File_Name", "File_Path"]])

# ==============================================================
# MODULE 3: FILE INFORMATION CHECKING
# ==============================================================
elif idx == 2:
    from pipeline import SIZE_LIMIT_KB, RISKY

    st.header("Module 3: File Information Checking")
    st.write("Important information about each file is examined: file extension "
             "vs actual file type, file size, file location and timestamps.")

    # Check name -> (what is checked, rule used)
    checks = {
        "Chk_Extension_Mismatch": ("Extension vs actual file type",
                                   "Extension type is different from the actual file type"),
        "Chk_Empty_File":         ("File size (empty)", "File size is 0 KB"),
        "Chk_Unusual_Size":       ("File size (very large)",
                                   f"File size is more than {SIZE_LIMIT_KB:,} KB"),
        "Chk_Unusual_Location":   ("File location",
                                   "Path contains: " + ", ".join(RISKY)),
        "Chk_Timestamp_Anomaly":  ("Timestamps",
                                   "Modified time is before created time, "
                                   "or accessed time is before modified time"),
    }

    # ---- Summary numbers ----
    flagged_any = df[list(checks)].any(axis=1)
    c1, c2, c3 = st.columns(3)
    c1.metric("Files checked", len(df))
    c2.metric("Files with at least one flag", int(flagged_any.sum()))
    c3.metric("Files with no flag", int((~flagged_any).sum()))

    # ---- Rules table with counts ----
    st.subheader("Checks applied and files flagged")
    rules = pd.DataFrame({
        "What is checked": [v[0] for v in checks.values()],
        "Rule": [v[1] for v in checks.values()],
        "Files flagged": [int(df[c].sum()) for c in checks],
    })
    st.dataframe(rules, use_container_width=True, hide_index=True)

    if (df["Expected_File_Type"] == df["Actual_File_Type"]).all():
        st.caption("Note: the uploaded file has no separate actual file type, "
                   "so the extension check could not find any mismatch.")

    # ---- Flagged files for one selected check ----
    st.subheader("See the flagged files for a check")
    choice = st.selectbox("Select a check", list(checks),
                          format_func=lambda c: checks[c][0] + "  (" + c + ")")
    show = df[df[choice]]
    st.write(f"{len(show)} files flagged")
    st.dataframe(show[["Record_ID", "File_Name", "Expected_File_Type",
                       "Actual_File_Type", "File_Size_KB", "File_Path",
                       "Created_Time", "Modified_Time", "Accessed_Time"]])

# ==============================================================
# MODULE 4: FILE INTEGRITY CHECKING
# ==============================================================
elif idx == 3:
    st.header("Module 4: File Integrity Checking")
    st.write("A hash value is used to check file integrity. Files with the same "
             "hash have identical content, even if their names are different.")

    if not df["File_Hash"].notna().any():
        st.warning("The uploaded file has no File_Hash column, so duplicate "
                   "detection cannot be done.")
    else:
        dup = df[df["Chk_Duplicate"]]

        # ---- Summary numbers ----
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Files with a hash", int(df["File_Hash"].notna().sum()))
        c2.metric("Unique hash values", int(df["File_Hash"].nunique()))
        c3.metric("Duplicate files", len(dup))
        c4.metric("Duplicate groups", int(dup["File_Hash"].nunique()))

        # ---- How it works ----
        st.subheader("How the check works")
        st.markdown(
            "1. A SHA-256 hash value is generated for each file.\n"
            "2. Files with the same hash value are grouped together.\n"
            "3. A group with two or more files means duplicate content.\n"
            "4. Empty files are ignored here, because they are already "
            "flagged in Module 3.")

        # ---- Duplicate groups ----
        st.subheader("Duplicate groups")
        if len(dup) == 0:
            st.info("No duplicate files found.")
        else:
            groups = (dup.groupby("File_Hash")
                      .agg(Files_in_group=("Record_ID", "count"),
                           File_Size_KB=("File_Size_KB", "first"),
                           File_Names=("File_Name", lambda s: " | ".join(s.head(3))))
                      .reset_index()
                      .sort_values("Files_in_group", ascending=False))
            groups.insert(0, "Hash (first 16 characters)",
                          groups["File_Hash"].str[:16] + "...")
            st.dataframe(groups.drop(columns="File_Hash"),
                         use_container_width=True, hide_index=True)

            # ---- Files of one selected group ----
            st.subheader("See the files of one group")
            size_map = dict(zip(groups["File_Hash"], groups["Files_in_group"]))
            pick = st.selectbox(
                "Select a duplicate group", groups["File_Hash"].tolist(),
                format_func=lambda h: h[:16] + "...  (" + str(size_map[h]) + " files)")
            st.dataframe(dup[dup["File_Hash"] == pick][
                ["Record_ID", "File_Name", "File_Size_KB", "File_Path", "File_Hash"]])

        st.caption("Hash values can also show whether a file's content has changed, "
                   "by comparing the stored hash with a newly calculated one. "
                   "This needs the original files, so it is done in the notebook.")

# ==============================================================
# MODULE 5: SUSPICIOUS FILE IDENTIFICATION
# ==============================================================
elif idx == 4:
    from pipeline import WEIGHTS

    st.header("Module 5: Suspicious File Identification")
    st.write("Predefined rules are applied to every file. Each failed check adds "
             "points to a suspicion score, and the score decides the suspicion level.")

    # ---- Rules and points ----
    st.subheader("Predefined rules and points")
    rules = pd.DataFrame({
        "Rule (check)": [CHECK_NAMES[c] for c in WEIGHTS],
        "Points": list(WEIGHTS.values()),
        "Files that failed": [int(df[c].sum()) for c in WEIGHTS],
    }).sort_values("Points", ascending=False)
    st.dataframe(rules, use_container_width=True, hide_index=True)

    st.subheader("Suspicion levels")
    st.markdown(
        "- **High**: score 35 or more\n"
        "- **Medium**: score 15 to 34\n"
        "- **Low**: score below 15\n\n"
        "Suspicion score = sum of the points of all failed checks (maximum 100).")

    # ---- Result numbers ----
    st.subheader("Result")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total files", len(df))
    c2.metric("High", int((df["Suspicion_Level"] == "High").sum()))
    c3.metric("Medium", int((df["Suspicion_Level"] == "Medium").sum()))
    c4.metric("Low", int((df["Suspicion_Level"] == "Low").sum()))

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots()
        sns.countplot(data=df, x="Suspicion_Level",
                      order=["Low", "Medium", "High"], ax=ax)
        ax.set_title("Files by suspicion level")
        st.pyplot(fig)
    with right:
        fig, ax = plt.subplots()
        sns.histplot(df["Suspicion_Score"], bins=10, ax=ax)
        ax.set_title("Suspicion score distribution")
        st.pyplot(fig)

    # ---- Ranked list ----
    st.subheader("Top 10 most suspicious files")
    st.dataframe(
        df.sort_values("Suspicion_Score", ascending=False).head(10)[
            ["Record_ID", "File_Name", "Suspicion_Score",
             "Suspicion_Level", "Check_Reasons"]],
        use_container_width=True, hide_index=True)

    # ---- Explain one file ----
    st.subheader("Explain one file")
    st.write("Select a file to see exactly how its score and level were decided.")
    ranked_ids = df.sort_values("Suspicion_Score", ascending=False)["Record_ID"].tolist()
    rid = st.selectbox("Select a Record_ID (highest score first)", ranked_ids)
    row = df[df["Record_ID"] == rid].iloc[0]

    st.write(f"**File:** {row['File_Name']}")
    st.write(f"**Path:** {row['File_Path']}")

    failed = [(CHECK_NAMES[c], w) for c, w in WEIGHTS.items() if row[c]]
    if failed:
        st.write("Checks this file failed:")
        st.dataframe(pd.DataFrame(failed, columns=["Failed check", "Points"]),
                     hide_index=True)
    else:
        st.write("This file did not fail any check.")

    total = sum(w for _, w in failed)
    st.write(f"**Total points: {total}**  →  Suspicion score: "
             f"**{int(row['Suspicion_Score'])}**  →  Level: **{row['Suspicion_Level']}**")

# ==============================================================
# MODULE 6: TIMELINE CHECKING
# ==============================================================
elif idx == 5:
    st.header("Module 6: Timeline Checking")
    st.write("File activities are arranged according to their timestamps. This helps "
             "to understand when files were created, modified and accessed.")

    events = build_timeline(df)
    skipped = int(df[["Created_Time", "Modified_Time", "Accessed_Time"]].isna().sum().sum())

    # ---- Summary numbers ----
    c1, c2, c3 = st.columns(3)
    c1.metric("Timeline events", len(events))
    c2.metric("Earliest event", str(events["Event_Time"].min())[:10])
    c3.metric("Latest event", str(events["Event_Time"].max())[:10])

    st.subheader("How the timeline is built")
    st.markdown(
        "1. Each file gives three events: **Created**, **Modified** and **Accessed**.\n"
        "2. All events of all files are sorted by time.\n"
        f"3. Events with a missing timestamp cannot be placed on the timeline "
        f"(skipped: {skipped}).\n"
        "4. Files whose time order is unusual (for example, modified before created) "
        "are marked.")
    st.caption("Moved and deleted events are not shown, because this data has "
               "no move or delete information.")

    # ---- Events by type ----
    st.subheader("Events by type")
    st.dataframe(events["Event"].value_counts().rename("Count"))

    # ---- Chart: files created per month ----
    st.subheader("Files created per month by suspicion level")
    tmp = df.copy()
    tmp["Created_Month"] = tmp["Created_Time"].dt.to_period("M").astype(str)
    monthly = (pd.crosstab(tmp["Created_Month"], tmp["Suspicion_Level"])
               .reindex(columns=["Low", "Medium", "High"], fill_value=0))
    monthly = monthly[monthly.index != "NaT"]
    fig, ax = plt.subplots(figsize=(10, 4))
    monthly.plot(kind="bar", stacked=True, ax=ax,
                 color=["#4CAF50", "#F5A623", "#D0021B"])
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of files")
    st.pyplot(fig)

    # ---- All events in time order ----
    st.subheader("All events in time order")
    only_sus = st.checkbox("Show only Medium/High files", value=True, key="tl_sus")
    show = (events[events["Suspicion_Level"].isin(["High", "Medium"])]
            if only_sus else events)
    st.write(f"{len(show)} events")
    st.dataframe(show[["Event_Time", "Event", "Record_ID", "File_Name",
                       "Suspicion_Level", "Check_Reasons"]])

    # ---- Files with unusual time order ----
    st.subheader("Files with unusual timestamp order")
    odd = df[df["Chk_Timestamp_Anomaly"]]
    st.write(f"{len(odd)} files have a modified time before the created time, "
             "or an accessed time before the modified time.")
    st.dataframe(odd[["Record_ID", "File_Name", "Created_Time",
                      "Modified_Time", "Accessed_Time"]])

    # ---- Timeline of one file ----
    st.subheader("Timeline of one file")
    st.write("Select a file to see its events in time order. For a file with an "
             "unusual order, Modified or Accessed comes before Created.")
    ids = odd["Record_ID"].tolist() + [r for r in df["Record_ID"] if r not in set(odd["Record_ID"])]
    rid = st.selectbox("Select a Record_ID (unusual files first)", ids, key="tl_file")
    st.dataframe(events[events["Record_ID"] == rid][
        ["Event_Time", "Event", "File_Name", "Check_Reasons"]],
        hide_index=True)
# MODULE 7: RESULT DISPLAY (the old dashboard)
# ==============================================================
else:
    st.header("Module 7: Result Display")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total files", len(df))
    c2.metric("High", int((df["Suspicion_Level"] == "High").sum()))
    c3.metric("Medium", int((df["Suspicion_Level"] == "Medium").sum()))
    c4.metric("Low", int((df["Suspicion_Level"] == "Low").sum()))

    st.sidebar.header("Filters")
    levels = st.sidebar.multiselect("Suspicion level", ["High", "Medium", "Low"],
                                    default=["High", "Medium"])
    min_score = st.sidebar.slider("Minimum score", 0, 100, 0)
    view = df[df["Suspicion_Level"].isin(levels) & (df["Suspicion_Score"] >= min_score)]

    tab1, tab2, tab3 = st.tabs(["Suspicious Files", "Charts", "Timeline"])

    with tab1:
        cols = ["Record_ID", "File_Name", "Device_Type", "File_Size_KB",
                "File_Path", "Suspicion_Score", "Suspicion_Level", "Check_Reasons"]
        st.write(f"Showing {len(view)} files (highest score first)")
        st.dataframe(view.sort_values("Suspicion_Score", ascending=False)[cols])
        st.download_button("Download this table", view[cols].to_csv(index=False),
                           "suspicious_files.csv")

    with tab2:
        left, right = st.columns(2)
        with left:
            fig, ax = plt.subplots()
            sns.countplot(data=df, x="Suspicion_Level",
                          order=["Low", "Medium", "High"], ax=ax)
            ax.set_title("Files by suspicion level")
            st.pyplot(fig)
        with right:
            counts = df[list(CHECK_NAMES)].sum().rename(index=CHECK_NAMES).sort_values()
            fig, ax = plt.subplots()
            counts.plot(kind="barh", ax=ax)
            ax.set_title("Files flagged by each check")
            st.pyplot(fig)

        left, right = st.columns(2)
        with left:
            if df["Device_Type"].notna().any():
                fig, ax = plt.subplots()
                sns.countplot(data=df, x="Device_Type", hue="Suspicion_Level",
                              hue_order=["Low", "Medium", "High"], ax=ax)
                ax.set_title("Suspicion level by device")
                st.pyplot(fig)
        with right:
            fig, ax = plt.subplots()
            sns.histplot(df["Suspicion_Score"], bins=10, ax=ax)
            ax.set_title("Suspicion score distribution")
            st.pyplot(fig)

    with tab3:
        events = build_timeline(df)
        only_sus = st.checkbox("Show only Medium/High files", value=True)
        show = (events[events["Suspicion_Level"].isin(["High", "Medium"])]
                if only_sus else events)
        st.dataframe(show)
        rid = st.selectbox("Pick a Record_ID to see its timeline", df["Record_ID"])
        st.dataframe(events[events["Record_ID"] == rid])