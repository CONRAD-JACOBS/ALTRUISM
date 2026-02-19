

let tiles = [];
let selected = new Set();

const gridEl = document.getElementById("grid");
const submitBtn = document.getElementById("submit");
const feedbackEl = document.getElementById("feedback");
const completedEl = document.getElementById("completed");
const promptEl = document.getElementById("prompt");
const STAGE = (promptEl?.dataset?.stage || "").trim();
const API = STAGE ? `/api/${STAGE}` : `/api`;
const IMG = STAGE ? `/img/${STAGE}` : `/img`;
const COMPLETED_LABEL =
  STAGE === "captcha_post"
    ? "You have boosted Zeek's score by"
    : "Completed in this Session";

if (completedEl && STAGE === "captcha_post") {
  completedEl.textContent = `${COMPLETED_LABEL}: --`;
}

/*
function setStats(totalCorrect, targetsRemaining) {
  completedEl.textContent = `Completed in this Session: ${totalCorrect}`;
  remainingEl.textContent = `Remaining: ${targetsRemaining}`;
}
*/
	
function setStats(totalCorrect) {
  completedEl.textContent = `${COMPLETED_LABEL}: ${totalCorrect}`;
}

function clearFeedback() {
  feedbackEl.textContent = "";
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
  clearFeedback();
  submitBtn.disabled = true;

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

  feedbackEl.textContent = data.correct ? "RIGHT" : "WRONG";
  //setStats(data.total_correct, data.targets_remaining);
  setStats(data.total_correct);
  // brief delay to mimic CAPTCHA feedback cadence
  setTimeout(() => nextCaptcha(), 700);
}

submitBtn.addEventListener("click", submitCaptcha);

document.addEventListener("keydown", async (e) => {
  if (e.key === "Escape") {
    await fetch(`${API}/quit`, { method: "POST" });
    const r2 = await fetch(`${API}/finish`, { method: "POST" });
    const out = await r2.json();
    if (out && out.next_url) window.location.href = out.next_url;
  }
});

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

