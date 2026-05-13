/**
 * BankSecure – Fraud Intelligence Dashboard
 * Handles form submission, result rendering, stats, and recent table.
 */

document.addEventListener("DOMContentLoaded", () => {
  // ── Elements ────────────────────────────────────────────────────────────
  const form       = document.getElementById("transaction-form");
  const submitBtn  = document.getElementById("submit-btn");
  const btnLabel   = document.getElementById("btn-label");
  const btnSpinner = document.getElementById("btn-spinner");

  // Stats
  const statTotal      = document.getElementById("stat-total");
  const statCritical   = document.getElementById("stat-critical");
  const statSuspicious = document.getElementById("stat-suspicious");
  const statSafe       = document.getElementById("stat-safe");
  const statAvgProb    = document.getElementById("stat-avg-prob");

  // Result card
  const resultCard     = document.getElementById("result-card");
  const resultIdle     = document.getElementById("result-idle");
  const resultLive     = document.getElementById("result-live");
  const riskBadge      = document.getElementById("risk-badge");
  const resultActionTxt= document.getElementById("result-action-text");
  const resultTime     = document.getElementById("result-time");
  const gaugeBar       = document.getElementById("gauge-bar");
  const gaugePct       = document.getElementById("gauge-pct");
  const rgScore        = document.getElementById("rg-score");
  const rgPred         = document.getElementById("rg-pred");
  const rgAction       = document.getElementById("rg-action");

  // Recent table
  const tbody      = document.getElementById("recent-tbody");
  const limitSel   = document.getElementById("limit-sel");
  const btnRefresh = document.getElementById("btn-refresh");

  // Demo buttons
  const btnSafe  = document.getElementById("btn-safe");
  const btnRisky = document.getElementById("btn-risky");

  // ── Demo Data ────────────────────────────────────────────────────────────
  // A "safe" transaction: near-zero PCA components, small amount
  const SAFE_VALUES = {
    V1: 1.19, V2: 0.26, V3: 0.17, V4: 0.45, V5: -0.18,
    V6: -0.36, V7: 0.10, V8: -0.04, V9: 0.11, V10: -0.10,
    V11: 0.21, V12: 0.03, V13: -0.06, V14: 0.08, V15: 0.07,
    V16: -0.14, V17: -0.06, V18: 0.01, V19: 0.06, V20: 0.00,
    V21: -0.03, V22: 0.02, V23: -0.01, V24: 0.02, V25: 0.02,
    V26: 0.04, V27: 0.00, V28: 0.01, Amount: 45.50,
  };
  // A "risky" transaction: extreme PCA values, large amount
  const RISKY_VALUES = {
    V1: -3.04, V2: 2.94, V3: -4.81, V4: 3.41, V5: -2.56,
    V6: -2.43, V7: -3.38, V8: 1.10, V9: -2.01, V10: -4.76,
    V11: 3.31, V12: -5.04, V13: -0.04, V14: -8.01, V15: 0.51,
    V16: -1.43, V17: -10.06, V18: -2.59, V19: 0.31, V20: 0.64,
    V21: 0.74, V22: -0.16, V23: 0.55, V24: 0.15, V25: -0.64,
    V26: 0.30, V27: 0.20, V28: 0.10, Amount: 1998.00,
  };

  function fillForm(values) {
    Object.entries(values).forEach(([key, val]) => {
      const el = document.getElementById(key);
      if (el) el.value = val;
    });
  }

  btnSafe.addEventListener("click", () => fillForm(SAFE_VALUES));
  btnRisky.addEventListener("click", () => fillForm(RISKY_VALUES));

  // ── Stats ────────────────────────────────────────────────────────────────
  async function loadStats() {
    try {
      const res = await fetch("/stats");
      if (!res.ok) return;
      const d = await res.json();
      statTotal.textContent      = d.total.toLocaleString();
      statCritical.textContent   = d.critical.toLocaleString();
      statSuspicious.textContent = d.suspicious.toLocaleString();
      statSafe.textContent       = d.safe.toLocaleString();
      statAvgProb.textContent    = (d.avg_probability * 100).toFixed(1) + "%";
    } catch (_) {}
  }

  // ── Recent Table ─────────────────────────────────────────────────────────
  function riskClass(level) {
    if (level === "Critical")   return "critical";
    if (level === "Suspicious") return "suspicious";
    return "low";
  }

  function formatTs(ts) {
    try {
      const d = new Date(ts + "Z"); // treat as UTC
      return d.toLocaleString(undefined, { month: "short", day: "numeric",
        hour: "2-digit", minute: "2-digit", second: "2-digit" });
    } catch (_) { return ts; }
  }

  async function loadRecent(animate = false) {
    const limit = limitSel.value;
    try {
      const res = await fetch(`/recent?limit=${limit}`);
      if (!res.ok) return;
      const rows = await res.json();
      tbody.innerHTML = "";
      if (rows.length === 0) {
        tbody.innerHTML = '<tr class="no-data"><td colspan="5">No transactions yet.</td></tr>';
        return;
      }
      rows.forEach((row, idx) => {
        const rc = riskClass(row.risk_level);
        const tr = document.createElement("tr");
        if (animate && idx === 0) tr.classList.add("new-row");
        tr.innerHTML = `
          <td>${formatTs(row.timestamp)}</td>
          <td>$${Number(row.amount).toFixed(2)}</td>
          <td><span class="badge-risk ${rc}">${row.risk_level.toUpperCase()}</span></td>
          <td>${row.action}</td>
          <td>${(row.probability * 100).toFixed(2)}%</td>
        `;
        tbody.appendChild(tr);
      });
    } catch (_) {}
  }

  limitSel.addEventListener("change", () => loadRecent());
  btnRefresh.addEventListener("click", () => { loadRecent(); loadStats(); });

  // ── Show Result ───────────────────────────────────────────────────────────
  function showResult(data) {
    const prob    = data.fraud_probability;
    const pct     = (prob * 100).toFixed(2);
    const level   = data.risk_level;    // "Low", "Suspicious", "Critical"
    const rc      = riskClass(level);
    const isfraud = data.predicted_class === 1;

    // Card state
    resultCard.dataset.state = rc;
    resultIdle.classList.add("hidden");
    resultLive.classList.remove("hidden");

    // Badge
    riskBadge.textContent = level.toUpperCase();
    riskBadge.className = "risk-badge " + rc;

    // Action text + time
    resultActionTxt.textContent = data.action;
    resultTime.textContent = "Analyzed at " + new Date().toLocaleTimeString();

    // Gauge – animate with slight delay for visual flair
    gaugeBar.className = "gauge-bar " + (rc === "low" ? "" : rc);
    setTimeout(() => {
      gaugeBar.style.width = pct + "%";
      gaugeBar.setAttribute("aria-valuenow", pct);
    }, 60);
    gaugePct.textContent = pct + "%";

    // Detail cells
    rgScore.textContent  = prob.toFixed(6);
    rgPred.textContent   = isfraud ? "⚠ Fraud" : "✓ Legitimate";
    rgAction.textContent = data.action;
  }

  // ── Form Submit ──────────────────────────────────────────────────────────
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    setLoading(true);

    const formData = new FormData(form);
    const payload  = {};
    formData.forEach((v, k) => { payload[k] = v; });

    try {
      const res  = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) { alert(data.error || "Prediction failed."); return; }
      showResult(data);
      loadRecent(true);
      loadStats();
    } catch (err) {
      console.error(err);
      alert("An error occurred while processing the prediction.");
    } finally {
      setLoading(false);
    }
  });

  function setLoading(on) {
    submitBtn.disabled = on;
    btnLabel.classList.toggle("hidden", on);
    btnSpinner.classList.toggle("hidden", !on);
  }

  // ── Init ──────────────────────────────────────────────────────────────────
  loadStats();
  loadRecent();
});
