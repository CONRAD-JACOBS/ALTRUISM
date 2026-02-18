# captcha_core/captcha_app/server.py
import csv, json, uuid, random, webbrowser
from datetime import datetime
from pathlib import Path
from flask import Flask, jsonify, request, render_template, send_from_directory, abort, make_response

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

def list_images(folder: Path):
    if not folder.exists():
        return []
    return sorted([p.name for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS])

def load_cfg(config_path: Path):
    cfg = {
        "prompt_text": "Click on the images containing bicycles.",
        "per_captcha_targets": 4,
        "grid_n": 3,
        "stop_when_targets_left": 0,
        "start_total_correct": 0
    }
    if config_path and Path(config_path).exists():
        try:
            cfg.update(json.loads(Path(config_path).read_text(encoding="utf-8")))
        except Exception:
            pass
    cfg["grid_n"] = int(cfg.get("grid_n", 3))
    cfg["per_captcha_targets"] = int(cfg.get("per_captcha_targets", 4))
    cfg["stop_when_targets_left"] = int(cfg.get("stop_when_targets_left", 0))
    cfg["start_total_correct"] = int(cfg.get("start_total_correct", 0))
    return cfg

def create_app(*, targets_dir, distractors_dir, config_path, results_dir, participant_number, auto_open=True):
    targets_dir = Path(targets_dir).resolve()
    distractors_dir = Path(distractors_dir).resolve()
    results_dir = Path(results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    CFG = load_cfg(Path(config_path).resolve())
    GRID_N = CFG["grid_n"]
    GRID_TILES = GRID_N * GRID_N

    app = Flask(__name__, template_folder=str(Path(__file__).parent.parent / "templates"),
                         static_folder=str(Path(__file__).parent.parent / "static"))

    SESSIONS = {}

    def new_session():
        session_id = uuid.uuid4().hex
        targets = list_images(targets_dir)
        distractors = list_images(distractors_dir)
        if not targets:
            raise RuntimeError("No images in {}".format(targets_dir))
        if not distractors:
            raise RuntimeError("No images in {}".format(distractors_dir))

        random.shuffle(targets)
        random.shuffle(distractors)

        started = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = results_dir / "captcha_results_{}_{}.csv".format(started, session_id[:8])

        SESSIONS[session_id] = {
            "participant": participant_number,
            "created_at": started,
            "prompt_text": CFG["prompt_text"],
            "targets_pool": targets,
            "distractors_pool": distractors,
            "targets_used": [],
            "total_seen": 0,
            "total_correct": CFG["start_total_correct"],
            "csv_path": str(csv_path),
            "captcha_index": 0,
            "active": True,
            "last_tiles": None
        }

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "participant", "session_id", "timestamp", "captcha_index",
                "correct", "prompt",
                "shown_filenames", "shown_labels_T_or_D",
                "selected_indices", "selected_filenames",
                "targets_remaining_after",
                "total_seen_so_far",
                "total_correct_so_far"
            ])
        return session_id

    def get_session():
        sid = request.cookies.get("captcha_session")
        if not sid or sid not in SESSIONS:
            sid = new_session()
        return sid, SESSIONS[sid]

    def sample_distractors(sess, n):
        chosen = []
        for _ in range(n):
            if not sess["distractors_pool"]:
                sess["distractors_pool"] = list_images(distractors_dir)
                random.shuffle(sess["distractors_pool"])
            chosen.append(sess["distractors_pool"].pop())
        return chosen

    def make_captcha(sess):
        if len(sess["targets_pool"]) <= CFG["stop_when_targets_left"]:
            return None
        k = min(CFG["per_captcha_targets"], len(sess["targets_pool"]))
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

    @app.route("/")
    def index():
        sid, sess = get_session()
        resp = make_response(render_template("index.html",
                                             prompt_text=sess["prompt_text"],
                                             grid_n=GRID_N))
        resp.set_cookie("captcha_session", sid, samesite="Lax")
        return resp

    @app.route("/api/state")
    def api_state():
        sid, sess = get_session()
        return jsonify({
            "prompt_text": sess["prompt_text"],
            "grid_n": GRID_N,
            "total_correct": sess["total_correct"],
            "targets_remaining": len(sess["targets_pool"]),
            "captcha_index": sess["captcha_index"],
            "active": sess["active"]
        })

    @app.route("/api/next")
    def api_next():
        sid, sess = get_session()
        if not sess["active"]:
            return jsonify({"done": True})

        tiles = make_captcha(sess)
        if tiles is None:
            sess["active"] = False
            return jsonify({"done": True})

        return jsonify({
            "done": False,
            "tiles": tiles,
            "total_correct": sess["total_correct"],
            "targets_remaining": len(sess["targets_pool"]),
            "captcha_index": sess["captcha_index"]
        })

    @app.route("/api/submit", methods=["POST"])
    def api_submit():
        sid, sess = get_session()
        if not sess["active"] or not sess.get("last_tiles"):
            return jsonify({"error": "No active captcha."}), 400

        payload = request.get_json(force=True) or {}
        selected = payload.get("selected_indices", [])
        if not isinstance(selected, list) or any((not isinstance(i, int)) for i in selected):
            return jsonify({"error": "selected_indices must be a list of ints"}), 400

        tiles = sess["last_tiles"]
        target_idxs = {i for i, t in enumerate(tiles) if t["kind"] == "target"}
        selected_set = set(selected)
        correct = (selected_set == target_idxs)

        sess["total_seen"] += 1
        if correct:
            sess["total_correct"] += 1

        ts = datetime.now().isoformat(timespec="seconds")
        shown_files = [t["filename"] for t in tiles]
        shown_labels = [("T" if t["kind"] == "target" else "D") for t in tiles]
        selected_files = [tiles[i]["filename"] for i in sorted(selected_set) if 0 <= i < len(tiles)]

        with open(sess["csv_path"], "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                sid, ts, sess["captcha_index"],
                int(correct), sess["prompt_text"],
                json.dumps(shown_files, ensure_ascii=False),
                json.dumps(shown_labels, ensure_ascii=False),
                json.dumps(sorted(selected_set)),
                json.dumps(selected_files, ensure_ascii=False),
                len(sess["targets_pool"]),
                sess["total_seen"],
                sess["total_correct"]
            ])

        sess["last_tiles"] = None

        return jsonify({
            "correct": bool(correct),
            "total_correct": sess["total_correct"],
            "targets_remaining": len(sess["targets_pool"])
        })

    @app.route("/api/quit", methods=["POST"])
    def api_quit():
        sid, sess = get_session()
        sess["active"] = False
        return jsonify({"ok": True, "csv_path": sess["csv_path"]})

    @app.route("/img/<kind>/<filename>")
    def serve_image(kind, filename):
        if kind == "target":
            directory = targets_dir
        elif kind == "distractor":
            directory = distractors_dir
        else:
            abort(404)

        p = directory / filename
        if (not p.exists()) or (not p.is_file()) or (p.suffix.lower() not in SUPPORTED_EXTS):
            abort(404)
        return send_from_directory(directory, filename)

    # convenience: open browser when server starts (only if desired)
    app._auto_open = bool(auto_open)
    return app
