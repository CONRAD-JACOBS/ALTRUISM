#!/usr/bin/python3
import pandas as pd
import json
from pathlib import Path
import statistics as stats

ROOT = Path(__file__).resolve().parents[1]
INFILE = ROOT / "ANALYSIS" / "1_assembled.csv"
OUTFILE = ROOT / "ANALYSIS" / "2_simplified.csv"
AUDITFILE = ROOT / "ANALYSIS" / "2_simplified_session_audit.csv"
DATA_DIR = ROOT / "DATA"

CAPTCHA_STAGES = ["captcha_pre", "captcha_post"]
QUESTIONNAIRE_STAGES = ["q_pre_captcha", "q_pre_idaq", "q_pre_2050", "q_post_gators", "q_post_specific"]

RESPONSES_COL = "questionnaire_json"   # <-- adjust if needed

# Session-level corrections are keyed by exp_sid, which is the stable identity
# across CSV rows and copied voice-session files. Raw files are left untouched.
SESSION_CORRECTIONS = {
    "846efc72e21e4402b3c5638b28a00e7c": {
        "exclude": True,
        "analysis_note": "Canceled robot-failure session; incomplete at q_pre_2050 and duplicated P101.",
    },
    "3e0a216249f6428a879be37b1eb5f75f": {
        "participant_number": 96,
        "analysis_note": "CSV contains stale participant_number 95 on post rows/voice metadata; source file and pre rows are P096.",
    },
    "b09679120fe74756b97eb9314a75dec4": {
        "voice_metrics_valid": False,
        "analysis_note": "Copied dialogue is only a pre-session interruption, so conversation metrics are treated as missing.",
    },
    "a0f2fb5dabb146c2830289f564f52723": {
        "voice_exp_sid": "b136c7590f164eb38f2cda10df8f9e63",
        "analysis_note": "Duplicate P093 CSV copies exist and voice metadata has stale exp_sid b136c759; exact duplicate raw rows are collapsed and voice metrics are matched by explicit alias.",
    },
}

VOICE_EXP_SID_ALIASES = {
    correction["voice_exp_sid"]: exp_sid
    for exp_sid, correction in SESSION_CORRECTIONS.items()
    if "voice_exp_sid" in correction
}

VOICE_METRIC_COLS = [
    "total_words",
    "mean_words_per_turn",
    "word_rate_wps",
    "robot_total_words",
    "robot_mean_words_per_turn",
    "robot_word_rate_wps",
    "mean_latency_sec",
]

def to_dt(s):
    return pd.to_datetime(s, errors="coerce")

def extract_numeric_mean(responses_json):
    """
    Given a JSON string of responses, return mean of numeric values.
    Non-numeric entries are ignored.
    """
    if not isinstance(responses_json, str) or responses_json.strip() == "":
        return None
    try:
        d = json.loads(responses_json)
    except Exception:
        return None

    nums = []
    for v in d.values():
        try:
            nums.append(float(v))
        except Exception:
            pass

    return float(sum(nums) / len(nums)) if nums else None
    
def summarize_captcha_stage(df, stage_id):
    d = df[df["stage_id"] == stage_id].copy()
    if d.empty:
        return pd.DataFrame(columns=["exp_sid"])

    def agg_one(g):
        exp_sid = g["exp_sid"].iloc[0]

        # ---- event-aware subsets ----
        g_enter = g[g["event"].fillna("") == "enter"]
        g_submit = g[g["event"].fillna("") == "submit"]

        # Attempts: count submit events (robust even if captcha_index is weird/missing)
        attempts = int(len(g_submit))

        # Completions: max total_correct_so_far (prefer submit rows; fallback to overall)
        if g_submit["total_correct_so_far"].notna().any():
            completions = int(g_submit["total_correct_so_far"].max())
        elif g["total_correct_so_far"].notna().any():
            completions = int(g["total_correct_so_far"].max())
        else:
            completions = 0

        # Goal: constant per stage (take first non-null)
        goal = int(g["goal_correct"].dropna().iloc[0]) if g["goal_correct"].notna().any() else None

        # Mean RT: average rt_sec on submit rows only
        if g_submit["rt_sec"].notna().any():
            mean_rt = float(g_submit["rt_sec"].mean())
        else:
            mean_rt = None  # or 0.0 if you prefer

        # Total time:
        # - if submits exist: last submit - first display (submit rows)
        # - else if "enter" exists: 0.0 (entered but did nothing)
        # - else: None (no evidence the stage occurred)
        if not g_submit.empty:
            t0 = g_submit["timestamp_display_dt"].min()
            t1 = g_submit["timestamp_submit_dt"].max()
            total_time = (t1 - t0).total_seconds() if pd.notna(t0) and pd.notna(t1) else None
        elif not g_enter.empty:
            total_time = 0.0
        else:
            total_time = None

        return pd.Series({
            "exp_sid": exp_sid,
            f"{stage_id}_attempts": attempts,
            f"{stage_id}_completions": completions,
            f"{stage_id}_goal": goal,
            f"{stage_id}_total_time": total_time,
            f"{stage_id}_mean_rt": mean_rt,
        })

    rows = [agg_one(g) for _, g in d.groupby("exp_sid", sort=False)]
    if not rows:
        return pd.DataFrame(columns=["exp_sid"])
    return pd.DataFrame(rows).reset_index(drop=True)

def summarize_questionnaire_stage(df, stage_id):
    d = df[df["stage_id"] == stage_id].copy()
    if d.empty:
        return pd.DataFrame(columns=["exp_sid"])

    # --- find questionnaire JSON column ---
    resp_col = None
    for c in ["responses_json", "questionnaire_json", "responses"]:
        if c in d.columns:
            resp_col = c
            break
    if resp_col is None:
        raise KeyError("No questionnaire response JSON column found")

    def agg_one(g):
        exp_sid = g["exp_sid"].iloc[0]

        # --- timing ---
        t0 = g["timestamp_display_dt"].min()
        t1 = g["timestamp_submit_dt"].max()
        total_time = (t1 - t0).total_seconds() if pd.notna(t0) and pd.notna(t1) else None

        # --- load responses ---
        raw = g[resp_col].dropna().iloc[0]
        responses_raw = json.loads(raw)
        responses = {k: pd.to_numeric(v, errors="coerce") for k, v in responses_raw.items()}


        comp_1 = None
        comp_2 = None
        extra_values = {}

        # ---------- q_pre_captcha ----------
        if stage_id == "q_pre_captcha":
            comp_1_name = "fun"
            comp_2_name = "difficulty"

            fun_keys = [
                "general_captcha_liking",
                "captcha_task_fun",
                "captcha_task_enjoyment",
            ]
            fun_vals = [float(responses[k]) for k in fun_keys if k in responses]
            comp_1 = stats.mean(fun_vals) if fun_vals else None
            comp_2 = float(responses.get("captcha_task_difficulty")) if "captcha_task_difficulty" in responses else None
            extra_values[f"{stage_id}_meaningfulness"] = (
                float(responses.get("captcha_task_meaningfulness"))
                if "captcha_task_meaningfulness" in responses
                else None
            )

        # ---------- q_pre_idaq ----------
        elif stage_id == "q_pre_idaq":
            comp_1_name = None
            comp_2_name = None
            comp_2 = None

            idaq_keys = [
                "idaq_3",
                "idaq_4",
                "idaq_7",
                "idaq_9",
                "idaq_11",
                "idaq_12",
                "idaq_13",
                "idaq_14",
                "idaq_17",
                "idaq_20",
                "idaq_21",
                "idaq_22",
                "idaq_23",
                "idaq_26",  
                "idaq_29"    
            ]
         
            idaq_vals = [float(responses[k]) for k in idaq_keys if k in responses]

            comp_1 = stats.mean(idaq_vals) if idaq_vals else None

        
         # ---------- q_pre_2050 ----------
        elif stage_id == "q_pre_2050":
            comp_1_name = "mean_futurism_score"
            comp_2_name = None
            comp_2 = None

            keys_2050 = ["2050_art", 
                        "2050_other_worlds", 
                        "2050_rubbish_collectors", 
                        "2050_flight_attendants",
                        "2050_experiments",
                        "2050_school",
                        "2050_hiking",
                        "2050_nursing_home",
                        "2050_rights",
                        "2050_neurosurgery",
                        "2050_crime", 
                        "2050_sports",
                        "2050_mannequins",
                        "2050_police",
                        "2050_shelters"
                        ]

            keys_2050_vals = [float(responses[k]) for k in keys_2050 if k in responses]            

            comp_1 = stats.mean(keys_2050_vals) if keys_2050_vals else None

        # ---------- q_post_gators ----------
        elif stage_id == "q_post_gators":
            comp_1_name = "pos"
            comp_2_name = "neg"

            pos_keys = [
                "gators_1","gators_2","gators_3","gators_4","gators_5",
                "gators_11","gators_12","gators_13","gators_14","gators_15",
            ]
            neg_keys = [
                "gators_6","gators_7","gators_8","gators_9","gators_10",
                "gators_16","gators_17","gators_18","gators_19","gators_20",
            ]

            pos_vals = [float(responses[k]) for k in pos_keys if k in responses]
            neg_vals = [float(responses[k]) for k in neg_keys if k in responses]

            comp_1 = stats.mean(pos_vals) if pos_vals else None
            comp_2 = stats.mean(neg_vals) if neg_vals else None

        # ---------- q_post_specific ----------
        elif stage_id == "q_post_specific":
            comp_1_name = "mentacy_belief_scale"
            comp_2_name = "likeability"

            belief = responses.get("mentacy_belief")
            conf = responses.get("belief_confidence")

            if belief == 0:
                sign = -1
            elif belief == 1:
                sign = 1
            else:
                sign = 0

            comp_1 = sign * (float(conf) - 1) if conf is not None else None

            comp_2 = float(responses.get("robot_likeability")) if "robot_likeability" in responses else None
            extra_values[f"{stage_id}_robot_empathy"] = (
                float(responses.get("robot_empathy"))
                if "robot_empathy" in responses
                else None
            )

        else:
            raise ValueError(f"Unknown questionnaire stage: {stage_id}")

        if comp_2_name == None: 
            comp_1_col = stage_id if comp_1_name is None else f"{stage_id}_{comp_1_name}"

            row = {
                "exp_sid": exp_sid,
                f"{stage_id}_total_time": total_time,
                comp_1_col: comp_1,
                }
            row.update(extra_values)
            return pd.Series(row)     

        else: 
            row = {
                "exp_sid": exp_sid,
                f"{stage_id}_total_time": total_time,
                f"{stage_id}_{comp_1_name}": comp_1,
                f"{stage_id}_{comp_2_name}": comp_2,
                }
            row.update(extra_values)
            return pd.Series(row)

    rows = [agg_one(g) for _, g in d.groupby("exp_sid", sort=False)]
    if not rows:
        return pd.DataFrame(columns=["exp_sid"])
    return pd.DataFrame(rows).reset_index(drop=True)


def load_watchdog_totals(data_dir):
    rows = []
    for path in sorted(Path(data_dir).glob("*_voice_session_metadata.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        exp_sid = str(payload.get("exp_sid") or "").strip()
        if not exp_sid:
            continue
        exp_sid = VOICE_EXP_SID_ALIASES.get(exp_sid, exp_sid)

        try:
            watchdog_total = int(payload.get("watchdog_total", 0) or 0)
        except Exception:
            watchdog_total = 0

        rows.append({
            "exp_sid": exp_sid,
            "watchdog_total": watchdog_total,
        })

    if not rows:
        return pd.DataFrame(columns=["exp_sid", "watchdog_total"])

    watchdog_df = pd.DataFrame(rows)
    watchdog_df = watchdog_df.drop_duplicates(subset=["exp_sid"], keep="last")
    return watchdog_df


def load_session_language_metrics(data_dir):
    rows = []
    for path in sorted(Path(data_dir).glob("*_voice_session_metadata.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        exp_sid = str(payload.get("exp_sid") or "").strip()
        if not exp_sid:
            continue
        exp_sid = VOICE_EXP_SID_ALIASES.get(exp_sid, exp_sid)

        rows.append({
            "exp_sid": exp_sid,
            "total_words": pd.to_numeric(payload.get("total_words"), errors="coerce"),
            "mean_words_per_turn": pd.to_numeric(payload.get("mean_words_per_turn"), errors="coerce"),
            "word_rate_wps": pd.to_numeric(payload.get("word_rate_wps"), errors="coerce"),
            "robot_total_words": pd.to_numeric(payload.get("robot_total_words"), errors="coerce"),
            "robot_mean_words_per_turn": pd.to_numeric(payload.get("robot_mean_words_per_turn"), errors="coerce"),
            "robot_word_rate_wps": pd.to_numeric(payload.get("robot_word_rate_wps"), errors="coerce"),
            "mean_latency_sec": pd.to_numeric(payload.get("mean_latency_sec"), errors="coerce"),
        })

    if not rows:
        return pd.DataFrame(
            columns=[
                "exp_sid",
                "total_words",
                "mean_words_per_turn",
                "word_rate_wps",
                "robot_total_words",
                "robot_mean_words_per_turn",
                "robot_word_rate_wps",
                "mean_latency_sec",
            ]
        )

    metrics_df = pd.DataFrame(rows)
    metrics_df = metrics_df.drop_duplicates(subset=["exp_sid"], keep="last")
    return metrics_df

def _first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None

def _line_count(path):
    if path is None or not path.exists():
        return None
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            return sum(1 for _ in f)
    except Exception:
        return None

def _voice_artifact_info(exp_sid, participant_number):
    if participant_number is None or pd.isna(participant_number):
        return {}

    pnum = int(participant_number)
    exp_short = str(exp_sid)[:8]
    correction = SESSION_CORRECTIONS.get(exp_sid, {})
    alias_short = str(correction.get("voice_exp_sid", ""))[:8]
    candidate_shorts = [exp_short]
    if alias_short and alias_short not in candidate_shorts:
        candidate_shorts.append(alias_short)

    meta_path = _first_existing([
        DATA_DIR / f"P{pnum:03d}_{short}_voice_session_metadata.json"
        for short in candidate_shorts
    ])
    dialogue_path = _first_existing([
        DATA_DIR / f"P{pnum:03d}_{short}_session_dialogue.txt"
        for short in candidate_shorts
    ])

    info = {
        "voice_metadata_file": meta_path.name if meta_path else "",
        "dialogue_file": dialogue_path.name if dialogue_path else "",
        "dialogue_line_count": _line_count(dialogue_path),
        "voice_metadata_exp_sid": "",
        "voice_metadata_participant_number": "",
        "voice_copied_at": "",
        "voice_source_session_dir": "",
    }

    if meta_path is not None:
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            info.update({
                "voice_metadata_exp_sid": payload.get("exp_sid", ""),
                "voice_metadata_participant_number": payload.get("participant_number", ""),
                "voice_copied_at": payload.get("copied_at", ""),
                "voice_source_session_dir": Path(str(payload.get("source_session_dir", ""))).name,
            })
        except Exception:
            pass

    return info

def _stage_timestamp_gap(g, before_stage, before_col, after_stage, after_col):
    before = g.loc[g["stage_id"] == before_stage, before_col]
    after = g.loc[g["stage_id"] == after_stage, after_col]
    if before.empty or after.empty:
        return None

    before_ts = pd.to_datetime(before, errors="coerce").max()
    after_ts = pd.to_datetime(after, errors="coerce").min()
    if pd.isna(before_ts) or pd.isna(after_ts):
        return None
    return (after_ts - before_ts).total_seconds()

def write_session_audit(df, merged):
    rows = []
    merged_sids = set(merged["exp_sid"])
    for exp_sid, g in df.groupby("exp_sid", sort=False):
        correction = SESSION_CORRECTIONS.get(exp_sid, {})
        participant_number = (
            int(merged.loc[merged["exp_sid"] == exp_sid, "participant_number"].iloc[0])
            if exp_sid in merged_sids
            else (
                int(g["participant_number"].dropna().mode().iloc[0])
                if g["participant_number"].notna().any()
                else None
            )
        )
        voice_info = _voice_artifact_info(exp_sid, participant_number)
        conversation_gap_sec = _stage_timestamp_gap(
            g, "q_pre_2050", "timestamp_submit", "q_post_gators", "timestamp_display"
        )
        copied_at = pd.to_datetime(voice_info.get("voice_copied_at", ""), errors="coerce")
        q_post_display = pd.to_datetime(
            g.loc[g["stage_id"] == "q_post_gators", "timestamp_display"], errors="coerce"
        ).min()
        voice_to_post_gap_sec = (
            (q_post_display - copied_at).total_seconds()
            if pd.notna(copied_at) and pd.notna(q_post_display)
            else None
        )

        warnings = []
        if conversation_gap_sec is not None and conversation_gap_sec < 120:
            warnings.append("short_pre_to_post_gap")
        if voice_info.get("voice_metadata_exp_sid") and voice_info["voice_metadata_exp_sid"] != exp_sid:
            alias = VOICE_EXP_SID_ALIASES.get(voice_info["voice_metadata_exp_sid"])
            if alias == exp_sid:
                warnings.append("voice_exp_sid_alias_used")
            else:
                warnings.append("voice_exp_sid_mismatch")
        if voice_info.get("voice_metadata_participant_number") not in ("", participant_number):
            warnings.append("voice_participant_number_mismatch")
        if not voice_info.get("dialogue_file"):
            warnings.append("missing_dialogue_file")
        elif voice_info.get("dialogue_line_count") is not None and voice_info["dialogue_line_count"] < 5:
            warnings.append("very_short_dialogue")

        rows.append({
            "exp_sid": exp_sid,
            "participant_numbers_in_rows": "|".join(
                sorted(str(int(x)) for x in g["participant_number"].dropna().unique())
            ),
            "analysis_participant_number": participant_number,
            "n_rows": len(g),
            "stage_ids": "|".join(sorted(g["stage_id"].dropna().astype(str).unique())),
            "conversation_gap_sec": conversation_gap_sec,
            "voice_to_post_gap_sec": voice_to_post_gap_sec,
            **voice_info,
            "excluded_from_simplified": bool(correction.get("exclude", False)),
            "voice_metrics_valid": bool(correction.get("voice_metrics_valid", True)),
            "audit_warnings": "|".join(warnings),
            "analysis_note": correction.get("analysis_note", ""),
        })

    audit = pd.DataFrame(rows)
    audit.to_csv(AUDITFILE, index=False)
    print(f"Wrote session audit -> {AUDITFILE}")

def main():
    df = pd.read_csv(INFILE, dtype=str)

    print("UNIQUE stage_id values:")
    print(sorted(df["stage_id"].dropna().astype(str).unique().tolist()))

    print("\nCOUNTS by stage_id:")
    print(df["stage_id"].value_counts(dropna=False).head(30))


    # --- numeric coercion ---
    num_cols = [
        "participant_number", "age", "captcha_index", "correct",
        "total_correct_so_far", "goal_correct", "rt_sec"
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    excluded_sids = {
        exp_sid for exp_sid, correction in SESSION_CORRECTIONS.items()
        if correction.get("exclude", False)
    }
    if excluded_sids:
        before = len(df)
        df_model = df[~df["exp_sid"].isin(excluded_sids)].copy()
        print(f"Excluded {before - len(df_model)} raw rows from known canceled sessions.")
    else:
        df_model = df.copy()

    before_dedup = len(df_model)
    df_model = df_model.drop_duplicates().copy()
    if len(df_model) != before_dedup:
        print(f"Collapsed {before_dedup - len(df_model)} exact duplicate raw rows.")

    # --- timestamps ---
    df_model["timestamp_display_dt"] = to_dt(df_model.get("timestamp_display"))
    df_model["timestamp_submit_dt"] = to_dt(df_model.get("timestamp_submit"))

    # --- demographics ---
    def demo_one(g):
        exp_sid = g["exp_sid"].iloc[0]
        correction = SESSION_CORRECTIONS.get(exp_sid, {})
        if "participant_number" in correction:
            participant_number = int(correction["participant_number"])
        elif g["participant_number"].notna().any():
            participant_number = int(g["participant_number"].dropna().mode().iloc[0])
        else:
            participant_number = None

        return pd.Series({
            "exp_sid": exp_sid,
            "participant_number": participant_number,
            "age": int(g["age"].dropna().iloc[0]) if g["age"].notna().any() else None,
            "gender": g["gender"].dropna().iloc[0] if g["gender"].notna().any() else None,
            "analysis_note": correction.get("analysis_note", ""),
             })

    demo_rows = [demo_one(g) for _, g in df_model.groupby("exp_sid", sort=False)]
    merged = pd.DataFrame(demo_rows).reset_index(drop=True)

    # --- captcha stages ---
    for stage in CAPTCHA_STAGES:
        merged = merged.merge(
            summarize_captcha_stage(df_model, stage),
            on="exp_sid",
            how="left"
        )

    # --- questionnaire stages ---
    for stage in QUESTIONNAIRE_STAGES:
        merged = merged.merge(
            summarize_questionnaire_stage(df_model, stage),
            on="exp_sid",
            how="left"
        )

    watchdog_df = load_watchdog_totals(DATA_DIR)
    merged = merged.merge(watchdog_df, on="exp_sid", how="left")
    merged["watchdog_total"] = pd.to_numeric(merged["watchdog_total"], errors="coerce").fillna(0).astype(int)

    language_df = load_session_language_metrics(DATA_DIR)
    merged = merged.merge(language_df, on="exp_sid", how="left")
    merged["voice_metrics_valid"] = merged["exp_sid"].map(
        lambda exp_sid: SESSION_CORRECTIONS.get(exp_sid, {}).get("voice_metrics_valid", True)
    )
    merged.loc[~merged["voice_metrics_valid"], VOICE_METRIC_COLS] = pd.NA

    OUTFILE.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTFILE, index=False)
    write_session_audit(df, merged)
    print(f"Wrote {len(merged)} rows -> {OUTFILE}")

if __name__ == "__main__":
    main()
