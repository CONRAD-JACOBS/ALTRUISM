

let tiles = [];
let selected = new Set();
let requiredClicksPerTile = [];
let clickProgressPerTile = [];

const gridEl = document.getElementById("grid");
const submitBtn = document.getElementById("submit");
const feedbackEl = document.getElementById("feedback");
const feedbackFlashEl = document.getElementById("feedback-flash");
const completedEl = document.getElementById("completed");
const ZekeScoreEl = document.getElementById("Zeke-score");
const captchaCardEl = document.getElementById("captcha-card");
const promptEl = document.getElementById("prompt");
const hintEl = document.getElementById("hint");
const saveExitBtn = document.getElementById("save-exit-trigger");
const stageAdvanceBtn = document.getElementById("stage-advance-trigger");
const STAGE = (promptEl?.dataset?.stage || "").trim();
const FEEDBACK_MODE = (promptEl?.dataset?.feedbackMode || "original").trim().toLowerCase();
const FEEDBACK_WRONG_MS = Math.max(0, Number(promptEl?.dataset?.feedbackWrongMs || 1000));
const STAGE_ENTER_TS = (promptEl?.dataset?.stageEnterTs || "").trim();
const STAGE_TIMEOUT_SEC = Math.max(0, Number(promptEl?.dataset?.stageTimeoutSec || 0));
const API = STAGE ? `/api/${STAGE}` : `/api`;
const IMG = STAGE ? `/img/${STAGE}` : `/img`;
const SHOULD_AUTO_SCROLL_BOTTOM = STAGE === "captcha_pre" || STAGE === "captcha_post";
const COMPLETED_LABEL =
  STAGE === "captcha_post"
    ? "You have boosted Zeke's total by"
    : "Completed in this Session";
let inFeedbackTransition = false;
let stageGoal = null;
let preStageLocked = false;
let bottomScrollSettled = false;
let isStageFinishing = false;
let stageTimeoutHandle = null;

if (completedEl && STAGE === "captcha_post") {
  completedEl.textContent = `${COMPLETED_LABEL}: --`;
}

if (hintEl) {
  if (STAGE === "captcha_pre") {
    hintEl.textContent = "";
    hintEl.style.display = "none";
  } else if (STAGE === "captcha_post") {
    hintEl.textContent = "";
    hintEl.style.display = "none";
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
    if (ZekeScoreEl) {
      const baseScore = Number(ZekeScoreEl.dataset.baseScore || 0);
      ZekeScoreEl.textContent = String(baseScore + totalCorrect);
    }
    return;
  }
  if (STAGE === "captcha_pre") {
    completedEl.innerHTML = `${COMPLETED_LABEL}: <span class="score-count">${totalCorrect}</span>`;
    return;
  }
  completedEl.textContent = `${COMPLETED_LABEL}: ${totalCorrect}`;
}

function updatePreStageLock(totalCorrect, goalCorrect) {
  if (STAGE !== "captcha_pre") return false;
  const resolvedGoal = Number(goalCorrect);
  if (!Number.isFinite(resolvedGoal) || resolvedGoal <= 0) return false;

  const shouldLock = Number(totalCorrect) >= resolvedGoal;
  preStageLocked = shouldLock;

  if (stageAdvanceBtn) {
    stageAdvanceBtn.disabled = !shouldLock;
    stageAdvanceBtn.classList.toggle("save-exit-trigger--active", shouldLock);
  }
  if (captchaCardEl) {
    captchaCardEl.classList.toggle("card--inactive", shouldLock);
  }

  return shouldLock;
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

function showWordFlashWrong() {
  if (!feedbackFlashEl) return;
  gridEl.style.visibility = "hidden";
  feedbackFlashEl.textContent = "WRONG";
  feedbackFlashEl.classList.remove("feedback-flash--hidden", "feedback-flash--right", "feedback-flash--wrong");
  feedbackFlashEl.classList.add("feedback-flash--wrong");
}

function scrollToPageBottom() {
  window.scrollTo({
    top: Math.max(document.body.scrollHeight, document.documentElement.scrollHeight),
    behavior: "auto"
  });
}

function settleInitialBottomScroll() {
  if (!SHOULD_AUTO_SCROLL_BOTTOM || bottomScrollSettled) return;

  let attempts = 0;
  let previousHeight = -1;
  let stableFrames = 0;

  const tick = () => {
    const currentHeight = Math.max(
      document.body.scrollHeight,
      document.documentElement.scrollHeight
    );

    scrollToPageBottom();

    if (currentHeight === previousHeight) {
      stableFrames += 1;
    } else {
      stableFrames = 0;
      previousHeight = currentHeight;
    }

    attempts += 1;
    if (stableFrames >= 2 || attempts >= 24) {
      bottomScrollSettled = true;
      scrollToPageBottom();
      return;
    }

    window.requestAnimationFrame(tick);
  };

  window.requestAnimationFrame(tick);
}

function renderGrid(tilesData, trickclickRequiredClicks) {
  tiles = tilesData;
  selected.clear();
  requiredClicksPerTile = Array.isArray(trickclickRequiredClicks)
    ? trickclickRequiredClicks.map((v) => Math.max(1, Number(v) || 1))
    : [];
  clickProgressPerTile = new Array(tiles.length).fill(0);
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
        // Asymmetric trickclicks: deselection is always single-click.
        selected.delete(idx);
        div.classList.remove("selected");
        clickProgressPerTile[idx] = 0;
        return;
      }

      const required = Math.max(1, Number(requiredClicksPerTile[idx]) || 1);
      clickProgressPerTile[idx] = (clickProgressPerTile[idx] || 0) + 1;
      if (clickProgressPerTile[idx] < required) {
        return;
      }
      clickProgressPerTile[idx] = 0;
      selected.add(idx);
      div.classList.add("selected");
    });

    gridEl.appendChild(div);
  }

  settleInitialBottomScroll();
}

function disableStageInteraction() {
  submitBtn.disabled = true;
  if (saveExitBtn) saveExitBtn.disabled = true;
  if (stageAdvanceBtn) stageAdvanceBtn.disabled = true;
  if (captchaCardEl) {
    captchaCardEl.classList.add("card--inactive");
  }
}

async function finishStage(reason, fallbackMessage) {
  if (isStageFinishing) return;
  isStageFinishing = true;
  inFeedbackTransition = true;
  disableStageInteraction();

  try {
    const r = await fetch(`${API}/finish`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ reason })
    });
    const out = await r.json();

    if (out && out.next_url) {
      window.location.href = out.next_url;
      return;
    }
  } catch (e) {
    console.error("Stage finish failed:", e);
  }

  feedbackEl.textContent = fallbackMessage;
}

function getStageTimeoutRemainingMs() {
  if (STAGE !== "captcha_post" || STAGE_TIMEOUT_SEC <= 0 || !STAGE_ENTER_TS) {
    return null;
  }

  const enteredAtMs = Date.parse(STAGE_ENTER_TS);
  if (Number.isNaN(enteredAtMs)) {
    return null;
  }

  const deadlineMs = enteredAtMs + (STAGE_TIMEOUT_SEC * 1000);
  return Math.max(0, deadlineMs - Date.now());
}

function scheduleStageTimeout() {
  const remainingMs = getStageTimeoutRemainingMs();
  if (remainingMs === null) return;

  if (stageTimeoutHandle !== null) {
    window.clearTimeout(stageTimeoutHandle);
  }

  stageTimeoutHandle = window.setTimeout(() => {
    feedbackEl.textContent = "Time is up. Continuing…";
    finishStage("stage_timeout", "Time is up. Please notify the researcher.");
  }, remainingMs);
}

async function nextCaptcha() {
  if (preStageLocked || isStageFinishing) return;
  hideWordFlash();
  clearFeedback();
  submitBtn.disabled = true;
  inFeedbackTransition = false;

  const r = await fetch(`${API}/next`);
  const data = await r.json();

  if (data.done) {
    feedbackEl.textContent = data.timed_out ? "Time is up. Continuing…" : "Finished. Continuing…";
    await finishStage(
      data.timed_out ? "stage_timeout" : "stage_finish",
      data.timed_out
        ? "Time is up. Please notify the researcher."
        : "Finished. Please notify the researcher."
    );
    return;
  }

  renderGrid(data.tiles, data.trickclick_required_clicks);
  //setStats(data.total_correct, data.targets_remaining);
  setStats(data.total_correct);
  if (!updatePreStageLock(data.total_correct, data.goal_correct)) {
    submitBtn.disabled = false;
  }
}

async function submitCaptcha() {
  if (inFeedbackTransition || preStageLocked || isStageFinishing) return;
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

  if (data.done) {
    if (STAGE === "captcha_pre" && !data.timed_out) {
      setStats(data.total_correct);
      updatePreStageLock(data.total_correct, data.goal_correct);
      hideWordFlash();
      clearFeedback();
      return;
    }

    feedbackEl.textContent = data.timed_out ? "Time is up. Continuing…" : "Finished. Continuing…";
    await finishStage(
      data.timed_out ? "stage_timeout" : "stage_finish",
      data.timed_out
        ? "Time is up. Please notify the researcher."
        : "Finished. Please notify the researcher."
    );
    return;
  }

  inFeedbackTransition = true;
  const isCorrect = Boolean(data.correct);
  //setStats(data.total_correct, data.targets_remaining);
  setStats(data.total_correct);
  const lockedAfterSubmit = updatePreStageLock(data.total_correct, data.goal_correct);

  if (STAGE === "captcha_pre" && lockedAfterSubmit) {
    hideWordFlash();
    clearFeedback();
    return;
  }

  if (isCorrect) {
    hideWordFlash();
    clearFeedback();
    nextCaptcha();
    return;
  }

  if (FEEDBACK_MODE === "word_flash") {
    showWordFlashWrong();
  } else {
    feedbackEl.textContent = "WRONG";
  }
  // feedback display cadence before loading the next captcha
  setTimeout(() => nextCaptcha(), FEEDBACK_WRONG_MS);
}

submitBtn.addEventListener("click", submitCaptcha);

if (STAGE === "captcha_post" && saveExitBtn) {
  saveExitBtn.addEventListener("click", async () => {
    if (isStageFinishing) return;
    await fetch(`${API}/quit`, { method: "POST" });
    await finishStage("stage_finish", "Finished. Please notify the researcher.");
  });
}

if (STAGE === "captcha_pre" && stageAdvanceBtn) {
  stageAdvanceBtn.addEventListener("click", async () => {
    if (stageAdvanceBtn.disabled || isStageFinishing) return;
    await finishStage("stage_finish", "Finished. Please notify the researcher.");
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
    settleInitialBottomScroll();
    scheduleStageTimeout();
    const resp = await fetch(`${API}/state`);
    const st = await resp.json();
    stageGoal = Number(st.goal_correct);
    setStats(st.total_correct);   // optional, but nice
    if (st.timed_out) {
      feedbackEl.textContent = "Time is up. Continuing…";
      await finishStage("stage_timeout", "Time is up. Please notify the researcher.");
      return;
    }
    if (updatePreStageLock(st.total_correct, stageGoal)) {
      return;
    }
    await nextCaptcha();
  } catch (err) {
    console.error("Init failed:", err);
    feedbackEl.textContent = "Error loading task. Please notify the researcher.";
    submitBtn.disabled = true;
  }
})();

if (SHOULD_AUTO_SCROLL_BOTTOM) {
  window.addEventListener("load", settleInitialBottomScroll, { once: true });
}
