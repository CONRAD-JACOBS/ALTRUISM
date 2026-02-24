

let tiles = [];
let selected = new Set();

const gridEl = document.getElementById("grid");
const submitBtn = document.getElementById("submit");
const feedbackEl = document.getElementById("feedback");
const feedbackFlashEl = document.getElementById("feedback-flash");
const completedEl = document.getElementById("completed");
const promptEl = document.getElementById("prompt");
const hintEl = document.getElementById("hint");
const STAGE = (promptEl?.dataset?.stage || "").trim();
const FEEDBACK_MODE = (promptEl?.dataset?.feedbackMode || "original").trim().toLowerCase();
const FEEDBACK_RIGHT_MS = Math.max(0, Number(promptEl?.dataset?.feedbackRightMs || 1000));
const FEEDBACK_WRONG_MS = Math.max(0, Number(promptEl?.dataset?.feedbackWrongMs || 1000));
const API = STAGE ? `/api/${STAGE}` : `/api`;
const IMG = STAGE ? `/img/${STAGE}` : `/img`;
const COMPLETED_LABEL =
  STAGE === "captcha_post"
    ? "You have boosted Zeek's score by"
    : "Completed in this Session";
let inFeedbackTransition = false;

if (completedEl && STAGE === "captcha_post") {
  completedEl.textContent = `${COMPLETED_LABEL}: --`;
}

if (hintEl) {
  if (STAGE === "captcha_pre") {
    hintEl.textContent = "";
    hintEl.style.display = "none";
  } else if (STAGE === "captcha_post") {
    hintEl.textContent = "Press ESC to quit";
    hintEl.classList.add("hint--prominent");
  }
}

/*
function setStats(totalCorrect, targetsRemaining) {
  completedEl.textContent = `Completed in this Session: ${totalCorrect}`;
  remainingEl.textContent = `Remaining: ${targetsRemaining}`;
}
*/
	
function setStats(totalCorrect) {
  if (STAGE === "captcha_post") {
    completedEl.innerHTML = `${COMPLETED_LABEL}: <span class="score-count">${totalCorrect}</span>`;
    return;
  }
  completedEl.textContent = `${COMPLETED_LABEL}: ${totalCorrect}`;
}

function clearFeedback() {
  feedbackEl.textContent = "";
}

function hideWordFlash() {
  if (!feedbackFlashEl) return;
  feedbackFlashEl.classList.add("feedback-flash--hidden");
  feedbackFlashEl.classList.remove("feedback-flash--right", "feedback-flash--wrong");
  feedbackFlashEl.textContent = "";
  gridEl.style.visibility = "";
}

function showWordFlash(correct) {
  if (!feedbackFlashEl) return;
  gridEl.style.visibility = "hidden";
  feedbackFlashEl.textContent = correct ? "RIGHT" : "WRONG";
  feedbackFlashEl.classList.remove("feedback-flash--hidden", "feedback-flash--right", "feedback-flash--wrong");
  feedbackFlashEl.classList.add(correct ? "feedback-flash--right" : "feedback-flash--wrong");
}

function renderGrid(tilesData) {
  tiles = tilesData;
  selected.clear();
  gridEl.innerHTML = "";

  for (let i = 0; i < tiles.length; i++) {
    const t = tiles[i];
    const div = document.createElement("div");
    div.className = "tile";
    div.dataset.idx = i;

    const img = document.createElement("img");
   img.src = `${IMG}/${t.kind}/${encodeURIComponent(t.filename)}`;
   img.alt = "tile";

    div.appendChild(img);

    div.addEventListener("click", () => {
      const idx = Number(div.dataset.idx);
      if (selected.has(idx)) {
        selected.delete(idx);
        div.classList.remove("selected");
      } else {
        selected.add(idx);
        div.classList.add("selected");
      }
    });

    gridEl.appendChild(div);
  }
}

async function nextCaptcha() {
  hideWordFlash();
  clearFeedback();
  submitBtn.disabled = true;
  inFeedbackTransition = false;

  const r = await fetch(`${API}/next`);
  const data = await r.json();

  if (data.done) {
    feedbackEl.textContent = "Finished. Continuing…";
    submitBtn.disabled = true;

    try {
      const r2 = await fetch(`${API}/finish`, { method: "POST" });
      const out = await r2.json();

      if (out && out.next_url) {
        window.location.href = out.next_url;
        return;
      }

      // fallback if server didn't provide next_url
      feedbackEl.textContent = "Finished. Please notify the researcher.";
    } catch (e) {
      feedbackEl.textContent = "Finished. Please notify the researcher.";
    }

    return;
  }

  renderGrid(data.tiles);
  //setStats(data.total_correct, data.targets_remaining);
  setStats(data.total_correct);
  submitBtn.disabled = false;
}

async function submitCaptcha() {
  if (inFeedbackTransition) return;
  submitBtn.disabled = true;

  const sel = Array.from(selected.values());
  const r = await fetch(`${API}/submit`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ selected_indices: sel })
  });

  const data = await r.json();

  if (data.error) {
    feedbackEl.textContent = data.error;
    submitBtn.disabled = false;
    return;
  }

  inFeedbackTransition = true;
  const isCorrect = Boolean(data.correct);
  const feedbackDelayMs = isCorrect ? FEEDBACK_RIGHT_MS : FEEDBACK_WRONG_MS;
  if (FEEDBACK_MODE === "word_flash") {
    showWordFlash(isCorrect);
  } else {
    feedbackEl.textContent = data.correct ? "RIGHT" : "WRONG";
  }
  //setStats(data.total_correct, data.targets_remaining);
  setStats(data.total_correct);
  // feedback display cadence before loading the next captcha
  setTimeout(() => nextCaptcha(), feedbackDelayMs);
}

submitBtn.addEventListener("click", submitCaptcha);

if (STAGE === "captcha_post") {
  document.addEventListener("keydown", async (e) => {
    if (e.key === "Escape") {
      await fetch(`${API}/quit`, { method: "POST" });
      const r2 = await fetch(`${API}/finish`, { method: "POST" });
      const out = await r2.json();
      if (out && out.next_url) window.location.href = out.next_url;
    }
  });
}

// Start
function setPrompt(robotName) {
  const base = "Click on all the bicycles to add a point";
  if (robotName && robotName.length > 0) {
    promptEl.textContent = `${base} to ${robotName}'s total.`;
  } else {
    promptEl.textContent = "Click on all the bicycles.";
  }
}

(async function init() {
  try {
    const resp = await fetch(`${API}/state`);
    const st = await resp.json();
    setStats(st.total_correct);   // optional, but nice
    await nextCaptcha();
  } catch (err) {
    console.error("Init failed:", err);
    feedbackEl.textContent = "Error loading task. Please notify the researcher.";
    submitBtn.disabled = true;
  }
})();
