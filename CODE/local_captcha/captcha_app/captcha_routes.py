import csv, json, uuid, random
from datetime import datetime
from pathlib import Path
from flask import jsonify, redirect, request, render_template, send_from_directory, abort, make_response

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

def list_images(folder: Path):
    if not folder.exists():
        return []
    return sorted([p.name for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS])

def load_cfg(config_path: Path):
    def parse_nonnegative_int(value, default):
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return int(default)
        return parsed if parsed >= 0 else int(default)

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if "goal_correct" not in cfg:
        raise RuntimeError("Config missing required key: goal_correct")
    cfg["goal_correct"] = int(cfg["goal_correct"])
    cfg["grid_n"] = int(cfg.get("grid_n", 3))
    cfg["stop_when_targets_left"] = int(cfg.get("stop_when_targets_left", 0))
    cfg["start_total_correct"] = int(cfg.get("start_total_correct", 0))
    cfg["targets_per_captcha_min"] = int(cfg.get("targets_per_captcha_min", 3))
    cfg["targets_per_captcha_max"] = int(cfg.get("targets_per_captcha_max", 5))
    if cfg["targets_per_captcha_min"] > cfg["targets_per_captcha_max"]:
        cfg["targets_per_captcha_min"], cfg["targets_per_captcha_max"] = cfg["targets_per_captcha_max"], cfg["targets_per_captcha_min"]
    feedback_mode = str(cfg.get("captcha_feedback_mode", "original")).strip().lower()
    allowed_feedback_modes = {"original", "word_flash", "symbol_flash", "popup"}
    if feedback_mode not in allowed_feedback_modes:
        feedback_mode = "original"
    cfg["captcha_feedback_mode"] = feedback_mode
    cfg["captcha_feedback_right_ms"] = parse_nonnegative_int(cfg.get("captcha_feedback_right_ms", 1000), 1000)
    cfg["captcha_feedback_wrong_ms"] = parse_nonnegative_int(cfg.get("captcha_feedback_wrong_ms", 1000), 1000)
    return cfg

def register_captcha_routes(app, *, stage_id, targets_dir, distractors_dir, config_path,
                           start_mode="instructions", next_url="/done",
                           get_or_create_exp=None, EXP_SESSIONS=None):

    """
    Registers a CAPTCHA stage into an existing Flask app.

    Routes created:
      /stage/<stage_id>                 (chooser or task entry depending on start_mode)
      /stage/<stage_id>/captcha         (task page)
      /api/<stage_id>/state
      /api/<stage_id>/next
      /api/<stage_id>/submit
      /api/<stage_id>/quit              (marks inactive)
      /api/<stage_id>/finish            (returns next_url)
      /img/<stage_id>/<kind>/<filename> (serves images)
    """
    targets_dir = Path(targets_dir).resolve()
    distractors_dir = Path(distractors_dir).resolve()

    CFG = load_cfg(Path(config_path).resolve())
    GRID_N = CFG["grid_n"]
    GRID_TILES = GRID_N * GRID_N

    STAGE_STATE = {}  # exp_sid -> stage dict (captcha pools etc.)

    def append_stage_row(exp_sid, exp, sess, *, event, timestamp_display="", timestamp_submit="", rt_sec="",
                         captcha_index=0, correct="", shown_files=None, shown_labels=None,
                         selected_indices=None, selected_files=None, questionnaire_json=""):
        with open(exp["csv_path"], "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                exp_sid,
                stage_id,
                sess["participant_number"],
                sess["age"],
                sess["gender"],
                timestamp_display,
                timestamp_submit,
                rt_sec,
                captcha_index,
                correct,
                sess["prompt_text"],
                "" if shown_files is None else json.dumps(shown_files, ensure_ascii=False),
                "" if shown_labels is None else json.dumps(shown_labels, ensure_ascii=False),
                "" if selected_indices is None else json.dumps(selected_indices),
                "" if selected_files is None else json.dumps(selected_files, ensure_ascii=False),
                len(sess["targets_pool"]),
                sess["total_seen"],
                sess["total_correct"],
                sess["goal_correct"],
                event,
                "" if questionnaire_json == "" else json.dumps(questionnaire_json, ensure_ascii=False)
            ])

    def get_exp():
        exp_sid = request.cookies.get("exp_session")
        if (not exp_sid) or (EXP_SESSIONS is None) or (exp_sid not in EXP_SESSIONS):
            abort(400, "Missing experiment session. Go back to start.")
        return exp_sid, EXP_SESSIONS[exp_sid]

    def get_stage_state(exp_sid, exp):
        # create stage state once per exp session
        if exp_sid not in STAGE_STATE:
            # initialise pools etc. (NO file creation here)
            targets = list_images(Path(targets_dir))
            distractors = list_images(Path(distractors_dir))
            if not targets:
                raise RuntimeError(f"No images in {targets_dir}")
            if not distractors:
                raise RuntimeError(f"No images in {distractors_dir}")

            random.shuffle(targets)
            random.shuffle(distractors)

            STAGE_STATE[exp_sid] = {
                "participant_number": exp["participant_number"],
                "age": exp["age"],
                "gender": exp["gender"],
                "prompt_text": CFG.get("prompt_text", ""),
                # master lists / pools
                "all_targets": targets[:],      # keep master list for looping
                "targets_pool": targets[:],     # pop from here
                "targets_used": [],             # for debugging (optional)
                "distractors_pool": distractors,
                # progress
                "total_seen": 0,
                "total_correct": CFG.get("start_total_correct", 0),
                "goal_correct": CFG["goal_correct"],
                "captcha_index": 0,
                "active": True,
                "last_tiles": None,
            }


        return STAGE_STATE[exp_sid]

    def get_ctx():
        exp_sid, exp = get_exp()
        sess = get_stage_state(exp_sid, exp)
        return exp_sid, exp, sess

    def sample_distractors(sess, n):
        chosen = []
        for _ in range(n):
            if not sess["distractors_pool"]:
                sess["distractors_pool"] = list_images(distractors_dir)
                random.shuffle(sess["distractors_pool"])
            chosen.append(sess["distractors_pool"].pop())
        return chosen

    def make_captcha(sess):
        if sess["total_correct"] >= sess["goal_correct"]:
            return None

        if not sess["targets_pool"]:
            sess["targets_pool"] = sess["all_targets"][:]
            random.shuffle(sess["targets_pool"])

        if len(sess["targets_pool"]) <= CFG["stop_when_targets_left"]:
            return None

        k = random.randint(CFG["targets_per_captcha_min"], CFG["targets_per_captcha_max"])
        k = min(k, GRID_TILES)
        k = min(k, len(sess["targets_pool"]))

        target_names = [sess["targets_pool"].pop() for _ in range(k)]
        sess["targets_used"].extend(target_names)

        d = GRID_TILES - k
        distractor_names = sample_distractors(sess, d)

        tiles = [{"kind": "target", "filename": n} for n in target_names] + \
                [{"kind": "distractor", "filename": n} for n in distractor_names]
        random.shuffle(tiles)

        sess["captcha_index"] += 1
        sess["last_tiles"] = tiles
        return tiles

    # ---------- Page routes ----------

    @app.route(f"/stage/{stage_id}", endpoint=f"{stage_id}_stage_entry")
    def stage_entry():
        exp_sid = request.cookies.get("exp_session")

        if (not exp_sid) or (EXP_SESSIONS is None) or (exp_sid not in EXP_SESSIONS):
            if get_or_create_exp is None:
                abort(400, "Experiment not initialized. Go back to start.")          

            exp_sid = request.cookies.get("exp_session")
            if (not exp_sid) or (EXP_SESSIONS is None) or (exp_sid not in EXP_SESSIONS):
                return redirect("/")

            resp = make_response(render_template(f"{stage_id}_instructions.html",
                                                next_url=f"/stage/{stage_id}/captcha"))
            resp.set_cookie("exp_session", exp_sid, samesite="Lax")
            return resp

        return render_template(f"{stage_id}_instructions.html", next_url=f"/stage/{stage_id}/captcha")

    @app.route(f"/stage/{stage_id}/captcha", endpoint=f"{stage_id}_stage_captcha_page")
    def stage_captcha_page():
        exp_sid, exp = get_exp()
        sess = get_stage_state(exp_sid, exp)

        # WRITE ONE ROW ON FIRST ENTRY
        if not sess.get("entered"):
            sess["entered"] = True
            sess["stage_enter_ts"] = datetime.now().isoformat(timespec="milliseconds")
            append_stage_row(
                exp_sid,
                exp,
                sess,
                event="enter",
                timestamp_display=sess["stage_enter_ts"],
                captcha_index=0,
            )

        return render_template(
            "index.html",
            prompt_text=sess["prompt_text"],
            grid_n=GRID_N,
            stage_id=stage_id,
            captcha_feedback_mode=CFG["captcha_feedback_mode"],
            captcha_feedback_right_ms=CFG["captcha_feedback_right_ms"],
            captcha_feedback_wrong_ms=CFG["captcha_feedback_wrong_ms"],
        )




    # ---------- API routes ----------

    @app.route(f"/api/{stage_id}/state", endpoint=f"{stage_id}_api_state")
    def api_state():
        _, _, sess = get_ctx()
        remaining = max(0, sess["goal_correct"] - sess["total_correct"])
        return jsonify({
            "prompt_text": sess["prompt_text"],
            "grid_n": GRID_N,
            "total_correct": sess["total_correct"],
            "goal_correct": sess["goal_correct"],
            "recaptchas_remaining": remaining,
            "captcha_index": sess["captcha_index"],
            "active": sess["active"]
        })

    @app.route(f"/api/{stage_id}/next", endpoint=f"{stage_id}_api_next")
    def api_next():
        _, _, sess = get_ctx()
        if not sess["active"]:
            return jsonify({"done": True})

        tiles = make_captcha(sess)
        if tiles is None:
            sess["active"] = False
            return jsonify({"done": True})
        
        sess["last_display_ts"] = datetime.now().isoformat(timespec="milliseconds")

        remaining = max(0, sess["goal_correct"] - sess["total_correct"])
        return jsonify({
            "done": False,
            "tiles": tiles,
            "total_correct": sess["total_correct"],
            "goal_correct": sess["goal_correct"],
            "recaptchas_remaining": remaining,
            "captcha_index": sess["captcha_index"]
        })

    @app.route(f"/api/{stage_id}/submit", methods=["POST"], endpoint=f"{stage_id}_api_submit")
    def api_submit():
        exp_sid, exp, sess = get_ctx()

        if (not sess["active"]) or (not sess.get("last_tiles")):
            return jsonify({"error": "No active captcha."}), 400

        payload = request.get_json(force=True) or {}
        selected = payload.get("selected_indices", [])
        if (not isinstance(selected, list)) or any((not isinstance(i, int)) for i in selected):
            return jsonify({"error": "selected_indices must be a list of ints"}), 400

        tiles = sess["last_tiles"]
        target_idxs = {i for i, t in enumerate(tiles) if t["kind"] == "target"}
        selected_set = set(selected)
        correct = (selected_set == target_idxs)

        sess["total_seen"] += 1
        if correct:
            sess["total_correct"] += 1

        timestamp_submit = datetime.now().isoformat(timespec="milliseconds")
        timestamp_display = sess.get("last_display_ts")
        if not timestamp_display:
            # fallback so you don't crash if something weird happens
            timestamp_display = datetime.now().isoformat(timespec="milliseconds")
        display_dt = datetime.fromisoformat(timestamp_display)
        submit_dt = datetime.fromisoformat(timestamp_submit)
        rt_sec = (submit_dt - display_dt).total_seconds()

        shown_files = [t["filename"] for t in tiles]
        shown_labels = [("T" if t["kind"] == "target" else "D") for t in tiles]
        selected_files = [tiles[i]["filename"] for i in sorted(selected_set) if 0 <= i < len(tiles)]

        # Write to the master per-participant CSV (owned by EXP_SESSIONS)
        csv_path = exp.get("csv_path")
        if not csv_path:
            abort(500, "Experiment session missing csv_path")

        append_stage_row(
            exp_sid,
            exp,
            sess,
            event="submit",
            timestamp_display=timestamp_display,
            timestamp_submit=timestamp_submit,
            rt_sec=rt_sec,
            captcha_index=sess["captcha_index"],
            correct=int(correct),
            shown_files=shown_files,
            shown_labels=shown_labels,
            selected_indices=sorted(selected_set),
            selected_files=selected_files,
        )

        sess["last_tiles"] = None

        remaining = max(0, sess["goal_correct"] - sess["total_correct"])
        done = (remaining <= 0)

        return jsonify({
            "correct": bool(correct),
            "total_correct": sess["total_correct"],
            "goal_correct": sess["goal_correct"],
            "recaptchas_remaining": remaining,
            "done": bool(done)
        })


    @app.route(f"/api/{stage_id}/quit", methods=["POST"], endpoint=f"{stage_id}_api_quit")
    def api_quit():
        exp_sid, exp, sess = get_ctx()
        if not sess.get("quit_logged"):
            timestamp_submit = datetime.now().isoformat(timespec="milliseconds")
            timestamp_display = sess.get("last_display_ts") or sess.get("stage_enter_ts") or timestamp_submit
            display_dt = datetime.fromisoformat(timestamp_display)
            submit_dt = datetime.fromisoformat(timestamp_submit)
            rt_sec = (submit_dt - display_dt).total_seconds()

            tiles = sess.get("last_tiles") or []
            shown_files = [t["filename"] for t in tiles] if tiles else []
            shown_labels = [("T" if t["kind"] == "target" else "D") for t in tiles] if tiles else []

            append_stage_row(
                exp_sid,
                exp,
                sess,
                event="escape",
                timestamp_display=timestamp_display,
                timestamp_submit=timestamp_submit,
                rt_sec=rt_sec,
                captcha_index=sess.get("captcha_index", 0),
                correct="",
                shown_files=shown_files,
                shown_labels=shown_labels,
                selected_indices=[],
                selected_files=[],
                questionnaire_json={
                    "status": "incomplete",
                    "reason": "escape_key",
                    "had_active_captcha": bool(tiles),
                },
            )
            sess["quit_logged"] = True
        sess["active"] = False
        return jsonify({"ok": True, "csv_path": exp["csv_path"]})


    @app.route(f"/api/{stage_id}/finish", methods=["POST"], endpoint=f"{stage_id}_api_finish")
    def api_finish():
        exp_sid, exp, sess = get_ctx()
        if not sess.get("finish_logged"):
            timestamp_submit = datetime.now().isoformat(timespec="milliseconds")
            timestamp_display = sess.get("last_display_ts") or sess.get("stage_enter_ts") or timestamp_submit
            display_dt = datetime.fromisoformat(timestamp_display)
            submit_dt = datetime.fromisoformat(timestamp_submit)
            rt_sec = (submit_dt - display_dt).total_seconds()

            append_stage_row(
                exp_sid,
                exp,
                sess,
                event="finish",
                timestamp_display=timestamp_display,
                timestamp_submit=timestamp_submit,
                rt_sec=rt_sec,
                captcha_index=sess.get("captcha_index", 0),
                correct="",
                shown_files=[],
                shown_labels=[],
                selected_indices=[],
                selected_files=[],
                questionnaire_json={
                    "status": "complete",
                    "reason": "stage_finish",
                },
            )
            sess["finish_logged"] = True
        sess["active"] = False
        return jsonify({"ok": True, "next_url": next_url})


    @app.route(f"/img/{stage_id}/<kind>/<filename>", endpoint=f"{stage_id}_serve_image")
    def serve_image(kind, filename):
        directory = targets_dir if kind == "target" else distractors_dir if kind == "distractor" else None
        if directory is None:
            abort(404)
        p = directory / filename
        if (not p.exists()) or (not p.is_file()) or (p.suffix.lower() not in SUPPORTED_EXTS):
            abort(404)
        return send_from_directory(directory, filename)
