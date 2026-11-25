const TOKEN_KEY = "creovate_gp_token";
const USER_KEY = "creovate_gp_user";
const USERS_DB_KEY = "creovate_gp_users_db";
const HISTORY_KEY = "creovate_gp_history";
const RULES_ACK_KEY = "creovate_rules_ack";
const THEME_KEY = "creovate_theme";

const state = {
  token: localStorage.getItem(TOKEN_KEY) || null,
  user: JSON.parse(localStorage.getItem(USER_KEY) || "null"),
  bundle: null,
  activeView: "home",
  activeAboutKey: "ABSTRACT",
};

const ABOUT_SECTIONS = {
  ABSTRACT: {
    title: "Abstract",
    body: "Direct Digital Lending has emerged as a pivotal mechanism for accelerating financial inclusion, particularly for Economically Weaker Sections (EWS) and Backward Class (BC) beneficiaries. This project implements an AI-driven Beneficiary Credit Scoring System integrated with an Income Verification Layer.\n\nAn Artificial Neural Network (ANN) analyzes non-linear relationships between repayment behavior, business activity metrics, financial stability, and socio-economic indicators, while rule-based income verification flags discrepancies against declared loan amounts and document validity.",
  },
  INTRODUCTION: {
    title: "1. Introduction",
    body: "Financial inclusion is a cornerstone of national development. Public-sector welfare lending often lags behind commercial fintech due to subjective manual assessment and the Income Verification Paradox.\n\nCreovate shifts evaluation from purely asset-based heuristics to cash-flow and repayment-behavior scoring with transparent verification checks.",
  },
  "LITERATURE SURVEY": {
    title: "2. Literature Survey",
    body: "We reviewed classical statistical scoring (Logistic Regression), tree ensembles (Random Forest, Gradient Boosting, XGBoost, LightGBM), Support Vector Machines, and Deep Learning neural networks alongside SHAP-based explainability for auditable credit decisions.",
  },
  "PROBLEM STATEMENT": {
    title: "3. Problem Statement",
    body: "Channel partners face a scalability vs. risk-quality dilemma. Manual verification introduces Type I and Type II errors. We formulate beneficiary creditworthiness prediction using standardized financial signals paired with an explicit income-to-loan verification check.",
  },
  METHODOLOGY: {
    title: "4. Methodology",
    body: "Following CRISP-DM adapted for welfare lending:\n• Median imputation for missing numeric values and StandardScaler normalization.\n• 4-layer ANN with ReLU activations, Batch Normalization, and Dropout regularization.\n• Income Verification Layer computing Loan-to-Income, Loan-to-Property, and Document Validity flags.",
  },
  "DATASET AND FEATURES": {
    title: "5. Dataset & Features",
    body: "The model uses 20,000 beneficiary records (beneficiary_credit_20k.csv) and 8 core numeric features: annual_income, joint_income, property_value, business_income, loan_amount, repayment_score, beneficiary_count, and age.\n\nSensitive attributes such as name, gender, and region are excluded from the ANN feature vector to uphold fairness.",
  },
  "RESULTS AND DISCUSSIONS": {
    title: "6. Results & Discussions",
    body: "The deployed ANN achieved ~91.5% accuracy and 0.94 ROC-AUC in comparative evaluation, outperforming Logistic Regression, SVM, Random Forest, and boosting baselines on complex non-linear interactions.",
  },
  CONCLUSION: {
    title: "7. Conclusion",
    body: "Combining deep learning credit scoring with explicit income verification and per-feature impact breakdowns improves decision consistency, transparency, and auditability for concessional digital lending prototypes.",
  },
  "FUTURE SCOPE": {
    title: "8. Future Scope",
    body: "• Account Aggregator API integration for consented bank statement verification\n• Automated OCR document authentication\n• Geo-spatial and seasonal cash-flow analytics\n• Federated learning for privacy-preserving model updates",
  },
  REFERENCES: {
    title: "9. References",
    body: "• Kumar, A., & Bhattacharya, D. S. (2021). Credit Score Prediction System using Deep Learning.\n• Bhilare, M., Wadajkar, V., & Kale, K. (2024). Precision Credit Scoring through Advanced ML.\n• West, D. (2000). Neural network credit scoring models.\n• Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions (SHAP).",
  },
};

const MODELS_DATA = [
  { name: "ANN — Artificial Neural Network", badge: "Primary Engine", badgeClass: "badge-success", score: "91.5% Acc • 0.94 AUC", desc: "4-layer deep neural network with BatchNorm & Dropout. Captures non-linear interactions across all 8 financial & demographic signals." },
  { name: "XGBoost", badge: "Runner-Up", badgeClass: "badge-info", score: "89.8% Acc", desc: "Regularized gradient boosted trees with fast tabular performance; close to ANN on structured features." },
  { name: "LightGBM", badge: "High Speed", badgeClass: "badge-info", score: "89.1% Acc", desc: "Leaf-wise histogram boosting optimized for large-scale tabular datasets." },
  { name: "Gradient Boosting", badge: "Ensemble", badgeClass: "badge-info", score: "87.9% Acc", desc: "Sequential tree ensemble that iteratively corrects residual prediction errors." },
  { name: "Random Forest", badge: "Baseline Ensemble", badgeClass: "badge-warn", score: "86.4% Acc", desc: "Bagged decision tree ensemble offering solid robustness against outliers." },
  { name: "SVM (Support Vector Machine)", badge: "Margin Classifier", badgeClass: "badge-warn", score: "82.7% Acc", desc: "Finds separating hyperplanes; less effective on high-variance income interactions." },
  { name: "Logistic Regression", badge: "Linear Baseline", badgeClass: "badge-warn", score: "79.5% Acc", desc: "Interpretable linear baseline useful for sanity checks but limited on non-linear risk." },
];

const TEAM_DATA = [
  { name: "Tharun", role: "Model Training & Full-Stack App", detail: "ANN architecture, model export, and interactive web application." },
  { name: "Vinay", role: "Model Training & Research Paper", detail: "Comparative benchmarking and technical manuscript authoring." },
  { name: "Ashish", role: "Research Paper, Data & Training", detail: "Dataset curation, feature analysis, and model evaluation." },
  { name: "Yashwanth", role: "Report & Data Preprocessing", detail: "Missing-value handling, scaling pipelines, and documentation." },
];

const DEMO_USERS = [
  { id: 1, name: "Demo Beneficiary", email: "demo@creovate.ai", passwordHash: null, passwordPlain: "Creovate@2026", role: "Beneficiary" },
  { id: 2, name: "Lending Officer", email: "officer@creovate.ai", passwordHash: null, passwordPlain: "Officer@2026", role: "Lending Officer" },
];

// ---------------------------------------------------------------------------
// Client-Side ANN Inference Engine (Exact weights from ann_credit_score.h5)
// ---------------------------------------------------------------------------
function forwardPass(vec, bundle) {
  const { scaler_mean, scaler_scale, weights } = bundle;
  const scaled = vec.map((v, i) => (v - scaler_mean[i]) / scaler_scale[i]);

  function denseRelu(input, kernel, bias) {
    const outDim = bias.length;
    const out = new Array(outDim);
    for (let j = 0; j < outDim; j++) {
      let sum = bias[j];
      for (let i = 0; i < input.length; i++) {
        sum += input[i] * kernel[i][j];
      }
      out[j] = sum > 0 ? sum : 0;
    }
    return out;
  }

  function batchNorm(input, gamma, beta, mean, variance, eps = 1e-3) {
    const out = new Array(input.length);
    for (let i = 0; i < input.length; i++) {
      out[i] = (gamma[i] * (input[i] - mean[i])) / Math.sqrt(variance[i] + eps) + beta[i];
    }
    return out;
  }

  let h = denseRelu(scaled, weights.w0, weights.b0);
  h = batchNorm(h, weights.bn0_g, weights.bn0_b, weights.bn0_m, weights.bn0_v);
  h = denseRelu(h, weights.w1, weights.b1);
  h = batchNorm(h, weights.bn1_g, weights.bn1_b, weights.bn1_m, weights.bn1_v);
  h = denseRelu(h, weights.w2, weights.b2);

  let raw = weights.b3[0];
  for (let i = 0; i < h.length; i++) {
    raw += h[i] * weights.w3[i][0];
  }
  return Math.max(300, Math.min(900, raw));
}

function predictCreditScore(payload, bundle) {
  const features = bundle.features;
  const vec = features.map((f) => Number(payload[f]));
  const score = Number(forwardPass(vec, bundle).toFixed(2));
  const baseline = features.map((f) => Number(bundle.medians[f]));

  const impacts = features.map((feat, idx) => {
    const perturbed = vec.slice();
    perturbed[idx] = baseline[idx];
    const pScore = forwardPass(perturbed, bundle);
    return {
      feature: feat,
      impact: Number((score - pScore).toFixed(2)),
      value: vec[idx],
      median: baseline[idx],
    };
  });
  impacts.sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact));

  const threshold = bundle.eligibility_threshold;
  const eligible = score >= threshold;
  let risk_tier = "Needs Review";
  if (score >= 750) risk_tier = "Excellent";
  else if (score >= 680) risk_tier = "Good";
  else if (eligible) risk_tier = "Fair";

  const totalIncome = payload.annual_income + payload.joint_income + payload.business_income;
  const lti = Number((payload.loan_amount / Math.max(totalIncome, 1)).toFixed(2));
  const ltp = Number((payload.loan_amount / Math.max(payload.property_value, 1)).toFixed(2));

  const flags = [];
  if (payload.document_validity !== "Valid") {
    flags.push(`Document status is '${payload.document_validity}' — manual review required.`);
  }
  if (payload.id_verified !== "Yes") {
    flags.push("Identity verification is incomplete.");
  }
  if (lti > 5.0) {
    flags.push(`High loan-to-income ratio (${lti}x combined annual income).`);
  }
  if (totalIncome <= 0) {
    flags.push("Combined declared income is zero.");
  }

  return {
    predicted_score: score,
    eligible,
    eligibility_threshold: threshold,
    risk_tier,
    impacts,
    income_verification: {
      status: flags.length === 0 ? "Verified" : "Flagged for Review",
      total_declared_income: Number(totalIncome.toFixed(2)),
      loan_to_income_ratio: lti,
      loan_to_property_ratio: ltp,
      document_validity: payload.document_validity,
      id_verified: payload.id_verified,
      flags,
    },
  };
}

// ---------------------------------------------------------------------------
// Utilities & Local Auth
// ---------------------------------------------------------------------------
function showToast(msg) {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.classList.remove("hidden");
  clearTimeout(showToast._timer);
  showToast._timer = setTimeout(() => el.classList.add("hidden"), 3200);
}

function formatINR(val) {
  return "₹" + Number(val || 0).toLocaleString("en-IN", { maximumFractionDigits: 0 });
}

function escapeHtml(str) {
  return String(str).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

async function sha256(text) {
  const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return Array.from(new Uint8Array(buf))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

function getUsersDB() {
  return JSON.parse(localStorage.getItem(USERS_DB_KEY) || "[]");
}

function saveUsersDB(users) {
  localStorage.setItem(USERS_DB_KEY, JSON.stringify(users));
}

function getHistoryStore() {
  return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
}

function saveHistoryItem(item) {
  const items = getHistoryStore();
  items.unshift(item);
  localStorage.setItem(HISTORY_KEY, JSON.stringify(items.slice(0, 25)));
}

// ---------------------------------------------------------------------------
// Theme & Navigation
// ---------------------------------------------------------------------------
function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem(THEME_KEY, theme);
  document.getElementById("theme-icon").innerHTML = theme === "dark" ? "&#9788;" : "&#9790;";
}

function navigate(viewName) {
  const valid = ["home", "about", "models", "rules", "team", "login", "predictor"];
  const target = valid.includes(viewName) ? viewName : "home";
  state.activeView = target;

  document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
  document.getElementById(`view-${target}`).classList.add("active");

  document.querySelectorAll(".nav-link").forEach((a) => {
    a.classList.toggle("active", a.dataset.nav === target);
  });

  document.getElementById("nav-links").classList.remove("open");
  document.getElementById("mobile-menu-btn").setAttribute("aria-expanded", "false");

  if (window.location.hash !== `#${target}`) {
    history.replaceState(null, "", `#${target}`);
  }
  if (target === "predictor") {
    syncPredictorAuthView();
  }
  window.scrollTo({ top: 0, behavior: "smooth" });
}

// ---------------------------------------------------------------------------
// Auth State
// ---------------------------------------------------------------------------
function setSession(user) {
  const token = "gp_" + Math.random().toString(36).slice(2) + Date.now();
  state.token = token;
  state.user = user;
  localStorage.setItem(TOKEN_KEY, token);
  localStorage.setItem(USER_KEY, JSON.stringify(user));
  updateAuthUI();
}

function logout(notify = true) {
  state.token = null;
  state.user = null;
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
  updateAuthUI();
  if (state.activeView === "predictor") syncPredictorAuthView();
  if (notify) showToast("Signed out successfully.");
}

function updateAuthUI() {
  const slot = document.getElementById("auth-nav-slot");
  const lockDot = document.getElementById("predictor-lock-dot");

  if (state.user && state.token) {
    lockDot.classList.add("unlocked");
    lockDot.title = "Predictor unlocked";
    slot.innerHTML = `
      <div class="user-pill">
        <span>${escapeHtml(state.user.name)}</span>
        <span class="user-role">${escapeHtml(state.user.role)}</span>
        <button type="button" class="btn-link" id="logout-btn">Logout</button>
      </div>
    `;
    document.getElementById("logout-btn").addEventListener("click", () => logout(true));
  } else {
    lockDot.classList.remove("unlocked");
    lockDot.title = "Sign in required";
    slot.innerHTML = `<a href="#login" class="btn btn-primary btn-sm" data-nav="login">Sign In</a>`;
  }
}

function syncPredictorAuthView() {
  const gate = document.getElementById("predictor-auth-gate");
  const workspace = document.getElementById("predictor-workspace");
  if (state.token && state.user) {
    gate.classList.add("hidden");
    workspace.classList.remove("hidden");
    loadHistory();
  } else {
    gate.classList.remove("hidden");
    workspace.classList.add("hidden");
  }
}

// ---------------------------------------------------------------------------
// Static Content Renderers
// ---------------------------------------------------------------------------
function renderAbout() {
  const navList = document.getElementById("about-nav-list");
  navList.innerHTML = "";
  Object.keys(ABOUT_SECTIONS).forEach((key) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `about-nav-btn ${state.activeAboutKey === key ? "active" : ""}`;
    btn.textContent = ABOUT_SECTIONS[key].title;
    btn.addEventListener("click", () => {
      state.activeAboutKey = key;
      renderAbout();
    });
    navList.appendChild(btn);
  });

  const sec = ABOUT_SECTIONS[state.activeAboutKey];
  document.getElementById("about-content-panel").innerHTML = `
    <h2>${escapeHtml(sec.title)}</h2>
    <p>${escapeHtml(sec.body)}</p>
  `;
}

function renderModels() {
  document.getElementById("models-grid").innerHTML = MODELS_DATA.map(
    (m) => `
    <article class="card">
      <div class="result-top">
        <span class="badge ${m.badgeClass}">${escapeHtml(m.badge)}</span>
        <span class="muted"><strong>${escapeHtml(m.score)}</strong></span>
      </div>
      <h3 style="margin: 6px 0 8px;">${escapeHtml(m.name)}</h3>
      <p class="muted" style="margin: 0;">${escapeHtml(m.desc)}</p>
    </article>
  `
  ).join("");
}

function renderTeam() {
  document.getElementById("team-grid").innerHTML = TEAM_DATA.map(
    (t) => `
    <article class="card feature-card">
      <div class="brand-mark" style="margin-bottom: 14px;">${escapeHtml(t.name[0])}</div>
      <h3>${escapeHtml(t.name)}</h3>
      <p style="color: var(--primary); font-weight: 600; font-size: 0.84rem; margin-bottom: 8px;">${escapeHtml(t.role)}</p>
      <p>${escapeHtml(t.detail)}</p>
    </article>
  `
  ).join("");
}

function syncRulesAckUI() {
  const ack = localStorage.getItem(RULES_ACK_KEY) === "true";
  const btn = document.getElementById("acknowledge-rules-btn");
  const txt = document.getElementById("rules-status-text");
  if (ack) {
    btn.textContent = "Rules Acknowledged ✓";
    btn.classList.remove("btn-primary");
    btn.classList.add("btn-outline");
    txt.textContent = "You have acknowledged the Creovate educational, privacy, and verification rules.";
  } else {
    btn.textContent = "Acknowledge Rules";
    btn.classList.add("btn-primary");
    btn.classList.remove("btn-outline");
  }
}

// ---------------------------------------------------------------------------
// Load Model Bundle & Render Results
// ---------------------------------------------------------------------------
async function loadBundle() {
  try {
    const res = await fetch("./model_bundle.json");
    if (!res.ok) throw new Error("Failed to load model_bundle.json");
    const bundle = await res.json();
    state.bundle = bundle;

    document.getElementById("hero-model-status").textContent = `ANN Model Ready • ${bundle.dataset_rows.toLocaleString()} Records`;
    document.getElementById("home-threshold").textContent = bundle.eligibility_threshold.toFixed(1);
    document.getElementById("pred-threshold-label").textContent = bundle.eligibility_threshold.toFixed(2);

    applyFeatureValues(bundle.medians);

    const presetBar = document.getElementById("preset-bar");
    presetBar.innerHTML = "";
    bundle.presets.forEach((p) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "btn btn-outline btn-sm";
      btn.textContent = p.label;
      btn.title = p.description;
      btn.addEventListener("click", () => {
        document.getElementById("p-name").value = p.applicant_name;
        document.getElementById("p-region").value = p.region;
        document.getElementById("p-occupation").value = p.occupation;
        document.getElementById("p-id-verified").value = p.id_verified;
        document.getElementById("p-doc-validity").value = p.document_validity;
        applyFeatureValues(p.features);
        showToast(`Loaded "${p.label}" preset`);
      });
      presetBar.appendChild(btn);
    });
  } catch (err) {
    console.error("Failed to load model bundle:", err);
    showToast("Error loading ANN model bundle.");
  }
}

function applyFeatureValues(values) {
  Object.entries(values).forEach(([key, val]) => {
    const el = document.getElementById(`f-${key}`);
    if (el) el.value = Math.round(Number(val));
  });
}

function renderPredictionResult(data, applicantName) {
  document.getElementById("result-placeholder").classList.add("hidden");
  document.getElementById("result-content").classList.remove("hidden");

  const score = data.predicted_score;
  document.getElementById("res-score").textContent = score.toFixed(1);
  document.getElementById("res-applicant-label").textContent = applicantName;

  const tierBadge = document.getElementById("res-tier-badge");
  tierBadge.textContent = data.risk_tier;
  tierBadge.className = "badge " + (score >= 700 ? "badge-success" : score >= data.eligibility_threshold ? "badge-info" : "badge-warn");

  const pct = Math.max(0, Math.min(1, (score - 300) / 600));
  const arc = document.getElementById("gauge-arc");
  arc.style.strokeDashoffset = (283 * (1 - pct)).toFixed(1);
  arc.style.stroke = score >= 700 ? "var(--primary)" : score >= data.eligibility_threshold ? "var(--accent)" : "var(--warn)";

  const eligBanner = document.getElementById("res-eligibility-banner");
  if (data.eligible) {
    eligBanner.className = "eligibility-banner ok";
    eligBanner.textContent = `Eligible — Score (${score.toFixed(1)}) meets threshold (${data.eligibility_threshold.toFixed(1)}).`;
  } else {
    eligBanner.className = "eligibility-banner no";
    eligBanner.textContent = `Needs Review — Score (${score.toFixed(1)}) is below threshold (${data.eligibility_threshold.toFixed(1)}).`;
  }

  const verif = data.income_verification;
  const vBadge = document.getElementById("res-verif-badge");
  vBadge.textContent = verif.status;
  vBadge.className = "badge " + (verif.status === "Verified" ? "badge-success" : "badge-warn");

  document.getElementById("res-total-income").textContent = formatINR(verif.total_declared_income);
  document.getElementById("res-lti").textContent = `${verif.loan_to_income_ratio}x`;
  document.getElementById("res-ltp").textContent = `${verif.loan_to_property_ratio}x`;

  const flagsEl = document.getElementById("res-verif-flags");
  flagsEl.innerHTML = verif.flags.length
    ? verif.flags.map((f) => `<li>${escapeHtml(f)}</li>`).join("")
    : `<li style="color: var(--primary);">All income &amp; document checks passed.</li>`;

  const maxImpact = Math.max(10, ...data.impacts.map((i) => Math.abs(i.impact)));
  document.getElementById("res-impacts-list").innerHTML = data.impacts
    .map((item) => {
      const width = Math.min(100, (Math.abs(item.impact) / maxImpact) * 100);
      const pos = item.impact >= 0;
      const label = item.feature.replace(/_/g, " ");
      return `
        <div class="impact-row">
          <span style="text-transform: capitalize;">${escapeHtml(label)}</span>
          <div class="bar-track"><div class="bar-fill ${pos ? "" : "warn"}" style="width: ${width.toFixed(0)}%"></div></div>
          <span class="impact-val ${pos ? "pos" : "neg"}">${pos ? "+" : ""}${item.impact.toFixed(1)}</span>
        </div>
      `;
    })
    .join("");
}

function loadHistory() {
  if (!state.user) return;
  const container = document.getElementById("history-container");
  const rows = getHistoryStore().filter((r) => r.email === state.user.email).slice(0, 10);
  if (!rows.length) {
    container.innerHTML = `<p class="muted">No assessments recorded yet in this account.</p>`;
    return;
  }
  container.innerHTML = `
    <table class="history-table">
      <thead>
        <tr><th>Applicant</th><th>Score</th><th>Tier</th><th>Docs</th></tr>
      </thead>
      <tbody>
        ${rows
          .map(
            (r) => `
          <tr>
            <td>${escapeHtml(r.applicant_name)}</td>
            <td><strong>${Number(r.predicted_score).toFixed(1)}</strong></td>
            <td>${escapeHtml(r.risk_tier)}</td>
            <td>${escapeHtml(r.document_validity)}</td>
          </tr>
        `
          )
          .join("")}
      </tbody>
    </table>
  `;
}

// ---------------------------------------------------------------------------
// Event Wiring
// ---------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  applyTheme(localStorage.getItem(THEME_KEY) || "dark");
  updateAuthUI();
  renderAbout();
  renderModels();
  renderTeam();
  syncRulesAckUI();
  loadBundle();

  document.body.addEventListener("click", (e) => {
    const link = e.target.closest("[data-nav]");
    if (link) {
      e.preventDefault();
      navigate(link.dataset.nav);
    }
  });

  document.getElementById("theme-toggle").addEventListener("click", () => {
    const next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
    applyTheme(next);
  });

  document.getElementById("mobile-menu-btn").addEventListener("click", () => {
    const nav = document.getElementById("nav-links");
    const open = nav.classList.toggle("open");
    document.getElementById("mobile-menu-btn").setAttribute("aria-expanded", String(open));
  });

  document.getElementById("acknowledge-rules-btn").addEventListener("click", () => {
    localStorage.setItem(RULES_ACK_KEY, "true");
    syncRulesAckUI();
    showToast("Website rules acknowledged.");
  });

  const tabSignIn = document.getElementById("tab-signin");
  const tabRegister = document.getElementById("tab-register");
  const signInForm = document.getElementById("signin-form");
  const registerForm = document.getElementById("register-form");
  const authError = document.getElementById("auth-error");

  function switchAuthTab(mode) {
    authError.classList.add("hidden");
    const isSignIn = mode === "signin";
    tabSignIn.classList.toggle("active", isSignIn);
    tabRegister.classList.toggle("active", !isSignIn);
    signInForm.classList.toggle("hidden", !isSignIn);
    registerForm.classList.toggle("hidden", isSignIn);
    document.getElementById("login-title").textContent = isSignIn ? "Welcome back" : "Create your account";
    document.getElementById("auth-subtext").textContent = isSignIn
      ? "Sign in to access the live ANN Credit Score Predictor and assessment history."
      : "Register to save and audit your beneficiary credit score assessments.";
  }

  tabSignIn.addEventListener("click", () => switchAuthTab("signin"));
  tabRegister.addEventListener("click", () => switchAuthTab("register"));

  document.querySelectorAll(".toggle-pw").forEach((btn) => {
    btn.addEventListener("click", () => {
      const input = document.getElementById(btn.dataset.target);
      const isPw = input.type === "password";
      input.type = isPw ? "text" : "password";
      btn.textContent = isPw ? "Hide" : "Show";
    });
  });

  async function performLogin(email, password) {
    authError.classList.add("hidden");
    const cleanEmail = email.trim().toLowerCase();
    const demoMatch = DEMO_USERS.find((u) => u.email === cleanEmail && u.passwordPlain === password);
    if (demoMatch) {
      setSession({ id: demoMatch.id, name: demoMatch.name, email: demoMatch.email, role: demoMatch.role });
      showToast(`Signed in as ${demoMatch.name}`);
      navigate("predictor");
      return;
    }

    const pwHash = await sha256(password);
    const found = getUsersDB().find((u) => u.email === cleanEmail && u.passwordHash === pwHash);
    if (!found) {
      authError.textContent = "Invalid email or password.";
      authError.classList.remove("hidden");
      return;
    }
    setSession({ id: found.id, name: found.name, email: found.email, role: found.role });
    showToast(`Signed in as ${found.name}`);
    navigate("predictor");
  }

  signInForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const email = document.getElementById("signin-email").value.trim();
    const pw = document.getElementById("signin-password").value;
    if (!email || !pw) {
      authError.textContent = "Please enter both email and password.";
      authError.classList.remove("hidden");
      return;
    }
    performLogin(email, pw);
  });

  registerForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    authError.classList.add("hidden");
    const name = document.getElementById("reg-name").value.trim();
    const email = document.getElementById("reg-email").value.trim().toLowerCase();
    const role = document.getElementById("reg-role").value;
    const password = document.getElementById("reg-password").value;

    if (name.length < 2 || !email.includes("@") || password.length < 8) {
      authError.textContent = "Enter a valid name, email, and password (at least 8 characters).";
      authError.classList.remove("hidden");
      return;
    }

    const users = getUsersDB();
    if (users.some((u) => u.email === email) || DEMO_USERS.some((u) => u.email === email)) {
      authError.textContent = "An account with this email already exists.";
      authError.classList.remove("hidden");
      return;
    }

    const passwordHash = await sha256(password);
    const newUser = { id: Date.now(), name, email, role, passwordHash };
    users.push(newUser);
    saveUsersDB(users);
    setSession({ id: newUser.id, name, email, role });
    showToast("Account created! Welcome to Creovate.");
    navigate("predictor");
  });

  document.getElementById("demo-beneficiary-btn").addEventListener("click", () => {
    performLogin("demo@creovate.ai", "Creovate@2026");
  });
  document.getElementById("demo-officer-btn").addEventListener("click", () => {
    performLogin("officer@creovate.ai", "Officer@2026");
  });
  document.getElementById("gate-demo-btn").addEventListener("click", () => {
    performLogin("demo@creovate.ai", "Creovate@2026");
  });

  const fileInput = document.getElementById("p-image");
  const previewBox = document.getElementById("image-preview-box");
  const previewImg = document.getElementById("image-preview-img");
  fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      previewImg.src = reader.result;
      previewBox.classList.remove("hidden");
    };
    reader.readAsDataURL(file);
  });
  document.getElementById("clear-image-btn").addEventListener("click", () => {
    fileInput.value = "";
    previewBox.classList.add("hidden");
  });

  document.getElementById("reset-medians-btn").addEventListener("click", () => {
    if (state.bundle) {
      applyFeatureValues(state.bundle.medians);
      showToast("Reset features to dataset medians.");
    }
  });

  document.getElementById("refresh-history-btn").addEventListener("click", loadHistory);

  document.getElementById("predict-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const errBox = document.getElementById("predict-error");
    errBox.classList.add("hidden");

    if (!state.bundle) {
      errBox.textContent = "Model bundle is still loading. Please try again.";
      errBox.classList.remove("hidden");
      return;
    }

    const payload = {
      applicant_name: document.getElementById("p-name").value.trim() || "Applicant",
      gender: document.getElementById("p-gender").value,
      region: document.getElementById("p-region").value.trim() || "Hyderabad",
      occupation: document.getElementById("p-occupation").value.trim() || "Self-Employed",
      id_verified: document.getElementById("p-id-verified").value,
      document_validity: document.getElementById("p-doc-validity").value,
      annual_income: Number(document.getElementById("f-annual_income").value),
      joint_income: Number(document.getElementById("f-joint_income").value),
      business_income: Number(document.getElementById("f-business_income").value),
      property_value: Number(document.getElementById("f-property_value").value),
      loan_amount: Number(document.getElementById("f-loan_amount").value),
      repayment_score: Number(document.getElementById("f-repayment_score").value),
      beneficiary_count: Number(document.getElementById("f-beneficiary_count").value),
      age: Number(document.getElementById("f-age").value),
    };

    if (payload.age < 18 || payload.age > 100 || payload.repayment_score < 0 || payload.repayment_score > 1000) {
      errBox.textContent = "Please enter valid ranges (Age: 18–100, Repayment Score: 0–1000).";
      errBox.classList.remove("hidden");
      return;
    }

    const result = predictCreditScore(payload, state.bundle);
    renderPredictionResult(result, payload.applicant_name);
    saveHistoryItem({
      email: state.user.email,
      applicant_name: payload.applicant_name,
      predicted_score: result.predicted_score,
      risk_tier: result.risk_tier,
      document_validity: payload.document_validity,
      created_at: Date.now(),
    });
    loadHistory();
    showToast("Credit score predicted!");
  });

  const initialHash = window.location.hash.replace("#", "") || "home";
  navigate(initialHash);
});
