import os
import csv
import json
import random
import time
import tkinter as tk
from tkinter import ttk
from tkinter import simpledialog

STATE_PATH = "cb_state.json"
DATA_DIR = "local_questions/data"





def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_state(path):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                s = json.load(f)
            if "swap_binary" in s and isinstance(s["swap_binary"], bool):
                return s
        except Exception:
            pass
    return {"swap_binary": random.choice([True, False])}


def save_state(path, state):
    with open(path, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)


class QuestionnaireApp(tk.Tk):
    def __init__(self, questions, block_id="block", data_dir="data", state_path="cb_state.json"):
        super(QuestionnaireApp, self).__init__()

        self.block_id = block_id
        self.data_dir = data_dir
        self.state_path = state_path

        self.title("Questionnaire")
        self.geometry("900x520")
        self.configure(padx=18, pady=18)

        self.questions = questions
        self.i = 0

        # Ask for participant ID (optional)
        self.withdraw()  # hide while asking
        pid = simpledialog.askstring("Participant", "Participant ID (optional):", parent=self)
        self.deiconify()

        self.participant_id = (pid.strip() if pid else "") or time.strftime("P_%Y%m%d_%H%M%S")

        ensure_dir(DATA_DIR)
        self.csv_path = os.path.join(DATA_DIR, "{}.csv".format(self.participant_id))

        self.state = load_state(STATE_PATH)

        self.current_response_value = None
        self.current_highlighted = False

        # ---- UI ----
        header = ttk.Frame(self)
        header.pack(fill="x", pady=(0, 10))

        self.progress_var = tk.StringVar(master=self)
        self.progress_lbl = ttk.Label(header, textvariable=self.progress_var, font=("Helvetica", 12))
        self.progress_lbl.pack(side="left")

        self.pid_lbl = ttk.Label(header, text="Participant: {}".format(self.participant_id), font=("Helvetica", 12))
        self.pid_lbl.pack(side="right")

        self.q_text = tk.StringVar(master=self)
        self.q_lbl = ttk.Label(
            self,
            textvariable=self.q_text,
            wraplength=860,
            font=("Helvetica", 18),
            justify="center",
            anchor="center"
            )

        # Increase the first number to push the question down from he top.
        # Increase the second number to add space below it (pushing the response area down)
        self.q_lbl.pack(fill="x", pady=(300, 1))

        self.response_frame = ttk.Frame(self)
        self.response_frame.pack(fill="both", expand=True)

        footer = ttk.Frame(self)
        footer.pack(fill="x", pady=(10, 0))

        self.hint_var = tk.StringVar(master=self)
        self.hint_var.set("Use ← / → to select. Press Enter to confirm.")
        self.hint_lbl = ttk.Label(footer, textvariable=self.hint_var, font=("Helvetica", 11))
        self.hint_lbl.pack(side="left")

        self.status_var = tk.StringVar(master=self)
        self.status_lbl = ttk.Label(footer, textvariable=self.status_var, font=("Helvetica", 11))
        self.status_lbl.pack(side="right")

        self.bind("<Left>", self.on_left)
        self.bind("<Right>", self.on_right)
        self.bind("<Return>", self.on_enter)
        self.bind("<KP_Enter>", self.on_enter)
        self.bind("<Escape>", lambda e: self.destroy())

        self._init_csv()


        self.render_question()

    def _init_csv(self):
        write_header = not os.path.exists(self.csv_path)
        if write_header:
            with open(self.csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    self.participant_id,
                    self.block_id,
                    "q_index",
                    "question",
                    "rtype",
                    "neg_label",
                    "pos_label",
                    "max_value",
                    "swap_binary",
                    "left_label",
                    "right_label",
                    "response_value",
                    "response_label"
                ])

    def clear_response_frame(self):
        for child in self.response_frame.winfo_children():
            child.destroy()

    def render_question(self):
        if self.i >= len(self.questions):
            self.finish()
            return

        q = self.questions[self.i]
        self.current_response_value = None
        self.current_highlighted = False

        self.progress_var.set("Question {}/{}".format(self.i + 1, len(self.questions)))
        self.q_text.set(q["q"])
        self.status_var.set("")

        self.clear_response_frame()

        if q["rtype"] == "binary":
            self._render_binary(q)
        elif q["rtype"] == "scale":
            self._render_scale(q)
        else:
            raise ValueError("Unknown rtype: {}".format(q["rtype"]))

    # ----------------------------
    # BINARY
    # ----------------------------
    def _render_binary(self, q):
        swap = self.state.get("swap_binary", False)

        if not swap:
            left_label, right_label = q["neg"], q["pos"]
            left_code, right_code = 0, 1
        else:
            left_label, right_label = q["pos"], q["neg"]
            left_code, right_code = 1, 0

        self.binary_left_label = left_label
        self.binary_right_label = right_label
        self.binary_left_code = left_code
        self.binary_right_code = right_code

        container = ttk.Frame(self.response_frame)
        container.pack(expand=True)

        self.bin_left = ttk.Label(container, text=left_label, font=("Helvetica", 22), padding=18, anchor="center")
        self.bin_left.grid(row=0, column=0, padx=40, pady=30, sticky="nsew")

        self.bin_right = ttk.Label(container, text=right_label, font=("Helvetica", 22), padding=18, anchor="center")
        self.bin_right.grid(row=0, column=1, padx=40, pady=30, sticky="nsew")

        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=1)

        self._set_binary_highlight(None)

    def _set_binary_highlight(self, which):
        # which: None, "left", "right"
        def style(lbl, active):
            # ttk themes can ignore bg/fg; so we use bold as reliable fallback
            base = ("Helvetica", 25)
            response_size = base[1] + 10
            lbl.configure(font=("Helvetica", response_size) + ("bold",) if active else base)

        style(self.bin_left, which == "left")
        style(self.bin_right, which == "right")

    # ----------------------------
    # SCALE
    # ----------------------------
    def _on_canvas_configure(self, event=None):
        # Debounce so we don't redraw dozens of times during resize/layout
        if getattr(self, "_scale_redraw_scheduled", False):
            return
        self._scale_redraw_scheduled = True
        self.after_idle(self._do_scale_redraw)

    def _do_scale_redraw(self):
        self._scale_redraw_scheduled = False
        # Only redraw if we're currently on a scale question and canvas exists
        if hasattr(self, "canvas") and self.canvas.winfo_exists():
            self._draw_scale()


    def _render_scale(self, q):
        maxv = int(q["max"])
        if maxv < 2:
            raise ValueError("Scale questions need max >= 2. Got: {}".format(maxv))

        self.scale_max = maxv
        self.scale_center = (maxv + 1) // 2
        self.scale_pos = self.scale_center

        self.after(0, self._draw_scale)

        self.scale_neg_text = q["neg"]
        self.scale_pos_text = q["pos"]


        mid = ttk.Frame(self.response_frame)
        mid.pack(fill="both", expand=True, pady=(10, 0))

        self.canvas = tk.Canvas(mid, height=180, highlightthickness=0)
        self.canvas.pack(fill="x", expand=True, padx=20, pady=10)

        self.show_neutral = (maxv % 2 == 1)
        self.neutral_value = self.scale_center if self.show_neutral else None

        # Neutral label can vary by question; default to "Neutral"
        self.scale_neutral_text = q.get("neutral_lbl", "Neutral")


        # Redraw whenever the canvas gets its real size (prevents the “jump”)
        self._scale_redraw_scheduled = False
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # Also schedule one draw after idle, in case configure has already fired
        self.after_idle(self._draw_scale)

        self._draw_scale()

        bottom = ttk.Frame(self.response_frame)
        bottom.pack(fill="x", pady=(0, 10))

        # IMPORTANT: give StringVar an explicit master
        self.scale_value_var = tk.StringVar(master=self)
        self.scale_value_var.set("Selection: (none)")
        ttk.Label(bottom, textvariable=self.scale_value_var, font=("Helvetica", 12)).pack(side="left")

    def _draw_scale(self):
        c = self.canvas
        c.delete("all")

        # Use true canvas size (no max(600, ...) fallback)
        w = c.winfo_width()
        if w <= 2:
            return  # canvas not ready yet; Configure/after_idle will redraw

        pad = 40
        y_line = 90
        x0, x1 = pad, w - pad

        # Line
        c.create_line(x0, y_line, x1, y_line, width=3)

        n = self.scale_max
        xs = []
        for v in range(1, n + 1):
            t = (v - 1) / float(n - 1)
            x = x0 + t * (x1 - x0)
            xs.append(x)

            c.create_line(x, y_line - 12, x, y_line + 12, width=2)

            show_num = (n <= 11) or (v in [1, self.scale_center, n]) or (v % 2 == 1 and n <= 17)
            if show_num:
                c.create_text(x, y_line + 30, text=str(v), font=("Helvetica", 10))

        # Labels (all on the same level)
        y_label = y_line - 32  # same level for neg/neutral/pos

        # extremes aligned with endpoints
        c.create_text(x0, y_label, text=self.scale_neg_text, font=("Helvetica", 11), anchor="w")
        c.create_text(x1, y_label, text=self.scale_pos_text, font=("Helvetica", 11), anchor="e")

        # neutral label above midpoint when odd
        if self.show_neutral and self.neutral_value is not None:
            x_neu = xs[self.neutral_value - 1]
            c.create_text(x_neu, y_label, text=self.scale_neutral_text, font=("Helvetica", 11))


        # selection marker only after movement
        if self.current_highlighted:
            x_sel = xs[self.scale_pos - 1]
            r = 10
            c.create_oval(x_sel - r, y_line - r, x_sel + r, y_line + r, width=3)


    # ----------------------------
    # KEYS
    # ----------------------------
    def on_left(self, event=None):
        q = self.questions[self.i]
        if q["rtype"] == "binary":
            self.current_highlighted = True
            self.current_response_value = self.binary_left_code
            self._set_binary_highlight("left")
            self.status_var.set("")
        else:
            if not self.current_highlighted:
                self.current_highlighted = True
                self.scale_pos = self.scale_center
                if self.scale_pos > 1:
                    self.scale_pos -= 1
            else:
                if self.scale_pos > 1:
                    self.scale_pos -= 1
            self.current_response_value = self.scale_pos
            self.scale_value_var.set("Selection: {}".format(self.scale_pos))
            self._draw_scale()

    def on_right(self, event=None):
        q = self.questions[self.i]
        if q["rtype"] == "binary":
            self.current_highlighted = True
            self.current_response_value = self.binary_right_code
            self._set_binary_highlight("right")
            self.status_var.set("")
        else:
            if not self.current_highlighted:
                self.current_highlighted = True
                self.scale_pos = self.scale_center
                if self.scale_pos < self.scale_max:
                    self.scale_pos += 1
            else:
                if self.scale_pos < self.scale_max:
                    self.scale_pos += 1
            self.current_response_value = self.scale_pos
            self.scale_value_var.set("Selection: {}".format(self.scale_pos))
            self._draw_scale()

    def on_enter(self, event=None):
        if self.i >= len(self.questions):
            return
        if self.current_response_value is None:
            self.status_var.set("Please make a selection first.")
            return

        q = self.questions[self.i]
        self.write_row(q)

        if q["rtype"] == "binary":
            self.state["swap_binary"] = not self.state.get("swap_binary", False)
            save_state(STATE_PATH, self.state)

        self.i += 1
        self.render_question()

    def write_row(self, q):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        swap = self.state.get("swap_binary", False)

        if q["rtype"] == "binary":
            left_label = self.binary_left_label
            right_label = self.binary_right_label
            response_label = left_label if self.current_response_value == self.binary_left_code else right_label
        else:
            left_label = q["neg"]
            right_label = q["pos"]
            response_label = str(self.current_response_value)

        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                self.participant_id,
                self.block_id,
                ts,
                self.i,
                q["q"],
                q["rtype"],
                q["neg"],
                q["pos"],
                q["max"],
                int(bool(swap)),
                left_label,
                right_label,
                self.current_response_value,
                response_label
            ])

    def finish(self):
        self.clear_response_frame()
        self.q_text.set("All done — thank you!")
        self.progress_var.set("Finished")
        self.hint_var.set("Responses saved to: {}".format(self.csv_path))
        self.status_var.set("")
        ttk.Button(self.response_frame, text="Exit", command=self.destroy).pack(pady=40)


def run_questionnaire(questions, block_id="block", data_dir="data", state_path="cb_state.json"):
    app = QuestionnaireApp(questions, block_id=block_id, data_dir=data_dir, state_path=state_path)
    app.mainloop()

