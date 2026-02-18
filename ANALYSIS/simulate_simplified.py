#!/usr/bin/python3
import argparse
import uuid
import numpy as np
import pandas as pd


# -----------------------
# helpers
# -----------------------
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def rtruncnorm_int(rng, mean, sd, lo, hi):
    # simple rejection sampling for small N is fine; for large N this is still OK
    while True:
        x = int(round(rng.normal(mean, sd)))
        if lo <= x <= hi:
            return x

def rlikert_cont(rng, mu, sd, lo, hi):
    x = rng.normal(mu, sd)
    return float(np.clip(x, lo, hi))

def rlognorm_bounded(rng, median, sigma, lo, hi):
    # lognormal with given median; clamp to bounds
    # median = exp(mu) => mu = log(median)
    mu = np.log(max(1e-9, median))
    x = rng.lognormal(mean=mu, sigma=sigma)
    return float(np.clip(x, lo, hi))

def p_logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


# -----------------------
# simulation core
# -----------------------
def simulate_one(rng, participant_number):
    exp_sid = uuid.uuid4().hex

    # demographics
    age = rtruncnorm_int(rng, mean=28, sd=10, lo=17, hi=70)
    gender = rng.choice(["male", "female", "other"], p=[0.45, 0.45, 0.10])

    # latent traits
    ability = rng.normal(0, 1)
    speed = rng.normal(0, 1)
    engagement = rng.normal(0, 1)
    anthro_trait = rng.normal(0, 1)  # drives IDAQ / some attitudes
    empathy_trait = rng.normal(0, 1)

    # -----------------------
    # captcha_pre
    # -----------------------
    captcha_pre_goal = 10
    lam_extra = 7.0  # tweak: 5 -> ~15 attempts avg, 7 -> ~17 attempts avg, 8 -> ~18 avg
    max_attempts = 35 # truncate extremes

    p_correct_pre = p_logistic(0.8 * ability - 0.3)

    # number of trials needed to achieve r successes (>= r always)
    # numpy uses: negative_binomial(n, p) = number of failures before n successes
    failures = rng.negative_binomial(n=captcha_pre_goal, p=p_correct_pre)
    pre_attempts = int(captcha_pre_goal + failures)
    pre_attempts = int(np.clip(pre_attempts, captcha_pre_goal, max_attempts))

    pre_completions = captcha_pre_goal

    pre_mean_rt = float(np.clip(
        rng.lognormal(mean=np.log(5.0), sigma=0.35) * np.exp(-0.15 * speed),
        2.0, 12
    ))
    pre_total_time = float(np.clip(pre_attempts * pre_mean_rt + rng.normal(0.5, 0.4), 0.0, 300.0))


    # -----------------------
    # q_pre (captcha feelings)
    # -----------------------
    q_pre_total_time = rlognorm_bounded(rng, median=8.0, sigma=0.45, lo=8.0, hi=180.0)

    # difficulty: higher when ability lower / RT higher
    q_pre_difficulty = rlikert_cont(
        rng,
        mu=2.8 + 0.7 * (-ability) + 0.10 * (pre_mean_rt - 5.0),
        sd=0.9,
        lo=1.0, hi=7.0
    )

    # fun: higher when engagement/empathy higher, lower when difficulty higher
    q_pre_fun = rlikert_cont(
        rng,
        mu=3.2 + 0.6 * empathy_trait + 0.3 * engagement - 0.5 * (q_pre_difficulty - 3.0),
        sd=0.9,
        lo=1.0, hi=7.0
    )

    # -----------------------
    # q_pre_idaq
    # -----------------------
    q_pre_idaq_total_time = rlognorm_bounded(rng, median=45.0, sigma=0.65, lo=10.0, hi=900.0)

    # two composites on 1-10
    q_pre_idaq = rlikert_cont(
        rng,
        mu=5.0 + 1.2 * anthro_trait + 0.4 * empathy_trait,
        sd=1.5,
        lo=1.0, hi=10.0
    )

    # -----------------------
    # q_pre_2050
    # -----------------------
    q_pre_2050_total_time = rlognorm_bounded(rng, median=18.0, sigma=0.55, lo=6.0, hi=240.0)

    q_pre_2050_mean_futurism_score = rlikert_cont(
        rng,
        mu=3.1 + 0.4 * empathy_trait + 0.25 * anthro_trait,
        sd=0.7,
        lo=1.0, hi=10.0
    )

    # -----------------------
    # q_post_gators
    # -----------------------
    q_post_gators_total_time = rlognorm_bounded(rng, median=22.0, sigma=0.55, lo=8.0, hi=300.0)

    q_post_gators_pos = rlikert_cont(
        rng,
        mu=3.0 + 0.55 * empathy_trait + 0.35 * anthro_trait,
        sd=0.75,
        lo=1.0, hi=6.0
    )
    q_post_gators_neg = rlikert_cont(
        rng,
        mu=3.0 - 0.45 * empathy_trait - 0.20 * anthro_trait,
        sd=0.75,
        lo=1.0, hi=6.0
    )

    # -----------------------
    # q_post_specific
    # -----------------------
    q_post_specific_total_time = rlognorm_bounded(rng, median=8.0, sigma=0.50, lo=3.0, hi=180.0)

    # mentacy belief scale in [-6, 6], integer
    # tends to be higher with anthro/empathy
    mentacy_lat = 0.9 * anthro_trait + 0.3 * empathy_trait + rng.normal(0, 0.8)
    mentacy_scaled = int(np.clip(round(mentacy_lat * 2.0), -6, 6))
    q_post_specific_mentacy_belief_scale = mentacy_scaled

    q_post_specific_empathy = rlikert_cont(
        rng,
        mu=3.5 + 0.60 * empathy_trait + 0.25 * q_post_gators_pos - 0.15 * q_post_gators_neg,
        sd=0.7,
        lo=1.0, hi=7.0
    )

    # -----------------------
    # captcha_post
    # -----------------------
    captcha_post_goal = 1000

    # attempts: allow true 0; driven by engagement, fun/value, difficulty
    post_lambda = np.exp(1.0 + 0.55 * engagement + 0.20 * empathy_trait + 0.15 * (q_pre_fun - 3.0) - 0.20 * (q_pre_difficulty - 3.0))
    post_attempts = int(np.clip(rng.poisson(lam=post_lambda), 0, 100))

    p_correct_post = p_logistic(0.65 * ability - 0.2)  # a little harder/ noisier
    post_completions = int(rng.binomial(post_attempts, p_correct_post)) if post_attempts > 0 else 0

    post_mean_rt = float(np.clip(
        rng.lognormal(mean=np.log(5.8), sigma=0.45) * np.exp(-0.12 * speed) * (1.0 + 0.05 * (q_pre_difficulty - 3.0)),
        2.0, 25.0
    ))
    post_total_time = float(np.clip(post_attempts * post_mean_rt + rng.normal(1.0, 0.8), 0.0, 3600.0))
    if post_attempts == 0:
        post_total_time = 0.0
        post_mean_rt = np.nan  # your simplify currently might write blank/NaN here

    return {
        "exp_sid": exp_sid,
        "participant_number": int(participant_number),
        "age": int(age),
        "gender": gender,

        "captcha_pre_attempts": int(pre_attempts),
        "captcha_pre_completions": int(pre_completions),
        "captcha_pre_goal": int(captcha_pre_goal),
        "captcha_pre_total_time": float(pre_total_time),
        "captcha_pre_mean_rt": float(pre_mean_rt) if pre_attempts > 0 else np.nan,

        "captcha_post_attempts": int(post_attempts),
        "captcha_post_completions": int(post_completions),
        "captcha_post_goal": int(captcha_post_goal),
        "captcha_post_total_time": float(post_total_time),
        "captcha_post_mean_rt": float(post_mean_rt),

        "q_pre_total_time": float(q_pre_total_time),
        "q_pre_fun": float(q_pre_fun),
        "q_pre_difficulty": float(q_pre_difficulty),

        "q_pre_idaq_total_time": float(q_pre_idaq_total_time),
        "q_pre_idaq": float(q_pre_idaq),

        "q_pre_2050_total_time": float(q_pre_2050_total_time),
        "q_pre_2050_mean_futurism_score": float(q_pre_2050_mean_futurism_score),

        "q_post_gators_total_time": float(q_post_gators_total_time),
        "q_post_gators_pos": float(q_post_gators_pos),
        "q_post_gators_neg": float(q_post_gators_neg),

        "q_post_specific_total_time": float(q_post_specific_total_time),
        "q_post_specific_mentacy_belief_scale": float(q_post_specific_mentacy_belief_scale),
        "q_post_specific_empathy": float(q_post_specific_empathy),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=str, default="ANALYSIS/simulated_simplified.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    rows = [simulate_one(rng, i + 1) for i in range(args.n)]
    df = pd.DataFrame(rows)

    # enforce exact column order matching your current simplified.csv
    cols = [
        "exp_sid","participant_number","age","gender",
        "captcha_pre_attempts","captcha_pre_completions","captcha_pre_goal","captcha_pre_total_time","captcha_pre_mean_rt",
        "captcha_post_attempts","captcha_post_completions","captcha_post_goal","captcha_post_total_time","captcha_post_mean_rt",
        "q_pre_total_time","q_pre_fun","q_pre_difficulty",
        "q_pre_idaq_total_time","q_pre_idaq",
        "q_pre_2050_total_time","q_pre_2050_mean_futurism_score",
        "q_post_gators_total_time","q_post_gators_pos","q_post_gators_neg",
        "q_post_specific_total_time","q_post_specific_mentacy_belief_scale","q_post_specific_empathy",
    ]
    df = df[cols]

    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows -> {args.out}")


if __name__ == "__main__":
    main()
