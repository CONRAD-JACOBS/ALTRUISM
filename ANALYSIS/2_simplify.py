#!/usr/bin/python3
import pandas as pd
import json
from pathlib import Path
import statistics as stats

ROOT = Path(__file__).resolve().parents[1]
INFILE = ROOT / "ANALYSIS" / "1_assembled.csv"
OUTFILE = ROOT / "ANALYSIS" / "2_simplified.csv"

CAPTCHA_STAGES = ["captcha_pre", "captcha_post"]
QUESTIONNAIRE_STAGES = ["q_pre_captcha", "q_pre_idaq", "q_pre_2050", "q_post_gators", "q_post_specific"]

RESPONSES_COL = "questionnaire_json"   # <-- adjust if needed

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

        # ---------- q_pre_captcha ----------
        if stage_id == "q_pre_captcha":
            comp_1_name = "fun"
            comp_2_name = "difficulty"

            fun_keys = [
                "captcha_task_meaningfulness",
                "general_captcha_liking",
                "captcha_task_fun",
                "captcha_task_enjoyment",
            ]
            fun_vals = [float(responses[k]) for k in fun_keys if k in responses]
            comp_1 = stats.mean(fun_vals) if fun_vals else None
            comp_2 = float(responses.get("captcha_task_difficulty")) if "captcha_task_difficulty" in responses else None

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
            comp_2_name = "empathy"

            belief = responses.get("mentacy_belief")
            conf = responses.get("belief_confidence")

            if belief == 0:
                sign = -1
            elif belief == 1:
                sign = 1
            else:
                sign = 0

            comp_1 = sign * (float(conf) - 1) if conf is not None else None

            emp_keys = [
                "conversation_interestingness",
                "robot_empathy",
                "robot_friendliness",
                "robot_likeability",
            ]
            emp_vals = [float(responses[k]) for k in emp_keys if k in responses]
            comp_2 = stats.mean(emp_vals) if emp_vals else None

        else:
            raise ValueError(f"Unknown questionnaire stage: {stage_id}")

        if comp_2_name == None: 
            comp_1_col = stage_id if comp_1_name is None else f"{stage_id}_{comp_1_name}"

            return pd.Series({
                "exp_sid": exp_sid,
                f"{stage_id}_total_time": total_time,
                comp_1_col: comp_1,
                })     

        else: 
            return pd.Series({
                "exp_sid": exp_sid,
                f"{stage_id}_total_time": total_time,
                f"{stage_id}_{comp_1_name}": comp_1,
                f"{stage_id}_{comp_2_name}": comp_2,
                })

    rows = [agg_one(g) for _, g in d.groupby("exp_sid", sort=False)]
    if not rows:
        return pd.DataFrame(columns=["exp_sid"])
    return pd.DataFrame(rows).reset_index(drop=True)

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

    # --- timestamps ---
    df["timestamp_display_dt"] = to_dt(df.get("timestamp_display"))
    df["timestamp_submit_dt"] = to_dt(df.get("timestamp_submit"))

    # --- demographics ---
    def demo_one(g):
        return pd.Series({
            "exp_sid": g["exp_sid"].iloc[0],
            "participant_number": int(g["participant_number"].dropna().iloc[0]) if g["participant_number"].notna().any() else None,
            "age": int(g["age"].dropna().iloc[0]) if g["age"].notna().any() else None,
            "gender": g["gender"].dropna().iloc[0] if g["gender"].notna().any() else None,
             })

    demo_rows = [demo_one(g) for _, g in df.groupby("exp_sid", sort=False)]
    merged = pd.DataFrame(demo_rows).reset_index(drop=True)

    # --- captcha stages ---
    for stage in CAPTCHA_STAGES:
        merged = merged.merge(
            summarize_captcha_stage(df, stage),
            on="exp_sid",
            how="left"
        )

    # --- questionnaire stages ---
    for stage in QUESTIONNAIRE_STAGES:
        merged = merged.merge(
            summarize_questionnaire_stage(df, stage),
            on="exp_sid",
            how="left"
        )

    OUTFILE.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTFILE, index=False)
    print(f"Wrote {len(merged)} rows -> {OUTFILE}")

if __name__ == "__main__":
    main()
