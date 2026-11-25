const API_BASE = "";
const TOKEN_KEY = "creovate_token";
const USER_KEY = "creovate_user";
const RULES_ACK_KEY = "creovate_rules_ack";
const THEME_KEY = "creovate_theme";

const state = {
  token: localStorage.getItem(TOKEN_KEY) || null,
  user: JSON.parse(localStorage.getItem(USER_KEY) || "null"),
  metadata: null,
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

// ---------------------------------------------------------------------------
// Utilities
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

async function apiFetch(path, options = {}) {
  const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
  if (state.token) headers["Authorization"] = `Bearer ${state.token}`;
  const res = await fetch(`${API_BASE}${path}`, { ...options, headers });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    if (res.status === 401 && state.token) {
      logout(false);
    }
    throw new Error(data.detail || `Request failed (${res.status})`);
  }
  return data;
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
function setSession(token, user) {
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

function escapeHtml(str) {
  return String(str).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
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
// Metadata, Presets & Prediction
// ---------------------------------------------------------------------------
async function loadMetadata() {
  try {
    const meta = await apiFetch("/api/metadata");
    state.metadata = meta;
    document.getElementById("hero-model-status").textContent = `ANN Model Ready • ${meta.dataset_rows.toLocaleString()} Records`;
    document.getElementById("home-threshold").textContent = meta.eligibility_threshold.toFixed(1);
    document.getElementById("pred-threshold-label").textContent = meta.eligibility_threshold.toFixed(2);

    applyFeatureValues(meta.medians);

    const presetBar = document.getElementById("preset-bar");
    presetBar.innerHTML = "";
    meta.presets.forEach((p) => {
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
    console.error("Failed to load metadata:", err);
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

  // Gauge arc (length ~283)
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

async function loadHistory() {
  if (!state.token) return;
  const container = document.getElementById("history-container");
  try {
    const data = await apiFetch("/api/predictions/history");
    if (!data.history.length) {
      container.innerHTML = `<p class="muted">No assessments recorded yet in this account.</p>`;
      return;
    }
    container.innerHTML = `
      <table class="history-table">
        <thead>
          <tr><th>Applicant</th><th>Score</th><th>Tier</th><th>Docs</th></tr>
        </thead>
        <tbody>
          ${data.history
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
  } catch (err) {
    container.innerHTML = `<p class="muted">Could not load history.</p>`;
  }
}

// ---------------------------------------------------------------------------
// Event Wiring
// ---------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  applyTheme(localStorage.getItem(THEME_KEY) || "light");
  updateAuthUI();
  renderAbout();
  renderModels();
  renderTeam();
  syncRulesAckUI();
  loadMetadata();

  // Global navigation links
  document.body.addEventListener("click", (e) => {
    const link = e.target.closest("[data-nav]");
    if (link) {
      e.preventDefault();
      navigate(link.dataset.nav);
    }
  });

  // Theme & mobile menu
  document.getElementById("theme-toggle").addEventListener("click", () => {
    const next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
    applyTheme(next);
  });

  document.getElementById("mobile-menu-btn").addEventListener("click", () => {
    const nav = document.getElementById("nav-links");
    const open = nav.classList.toggle("open");
    document.getElementById("mobile-menu-btn").setAttribute("aria-expanded", String(open));
  });

  // Rules acknowledgement
  document.getElementById("acknowledge-rules-btn").addEventListener("click", () => {
    localStorage.setItem(RULES_ACK_KEY, "true");
    syncRulesAckUI();
    showToast("Website rules acknowledged.");
  });

  // Auth tabs
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
    try {
      const data = await apiFetch("/api/auth/login", {
        method: "POST",
        body: JSON.stringify({ email, password }),
      });
      setSession(data.token, data.user);
      showToast(`Signed in as ${data.user.name}`);
      navigate("predictor");
    } catch (err) {
      authError.textContent = err.message;
      authError.classList.remove("hidden");
    }
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
    const email = document.getElementById("reg-email").value.trim();
    const role = document.getElementById("reg-role").value;
    const password = document.getElementById("reg-password").value;

    if (name.length < 2 || !email.includes("@") || password.length < 8) {
      authError.textContent = "Enter a valid name, email, and password (at least 8 characters).";
      authError.classList.remove("hidden");
      return;
    }

    try {
      const data = await apiFetch("/api/auth/register", {
        method: "POST",
        body: JSON.stringify({ name, email, password, role }),
      });
      setSession(data.token, data.user);
      showToast("Account created! Welcome to Creovate.");
      navigate("predictor");
    } catch (err) {
      authError.textContent = err.message;
      authError.classList.remove("hidden");
    }
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

  // Image preview (local-only per privacy rules)
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
    if (state.metadata) {
      applyFeatureValues(state.metadata.medians);
      showToast("Reset features to dataset medians.");
    }
  });

  document.getElementById("refresh-history-btn").addEventListener("click", loadHistory);

  // Predict Form Submit
  document.getElementById("predict-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const errBox = document.getElementById("predict-error");
    errBox.classList.add("hidden");

    const submitBtn = document.getElementById("predict-submit-btn");
    submitBtn.disabled = true;
    submitBtn.textContent = "Running ANN Inference...";

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

    try {
      const result = await apiFetch("/api/predict", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      renderPredictionResult(result, payload.applicant_name);
      loadHistory();
      showToast("Credit score predicted!");
    } catch (err) {
      errBox.textContent = err.message;
      errBox.classList.remove("hidden");
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = "Run ANN Credit Score Prediction";
    }
  });

  // Initial route from hash
  const initialHash = window.location.hash.replace("#", "") || "home";
  navigate(initialHash);
});
