// Configure Backend API Endpoint
// We will update the production URL once the Render service is created.
const RENDER_API_URL = "https://oral-cancer-backend-asls.onrender.com"; 
const API_BASE_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? 'http://localhost:8000'
    : RENDER_API_URL;

// State management
let state = {
    apiOnline: false,
    predictionResult: null,
    userInputs: null,
    comparisonChart: null
};

// Initial setup
document.addEventListener("DOMContentLoaded", () => {
    initNavigation();
    checkApiHealth();
    initCharts();
});

// 1. Navigation handling
function initNavigation() {
    const navItems = document.querySelectorAll(".nav-item");
    navItems.forEach(item => {
        item.addEventListener("click", () => {
            const targetPage = item.getAttribute("data-target");
            navigateToPage(targetPage);
        });
    });
}

function navigateToPage(pageId) {
    // Hide all sections
    const sections = document.querySelectorAll(".page-section");
    sections.forEach(sec => sec.classList.remove("active"));
    
    // Deactivate all nav buttons
    const navItems = document.querySelectorAll(".nav-item");
    navItems.forEach(item => item.classList.remove("active"));
    
    // Show target section
    const targetSection = document.getElementById(`page-${pageId}`);
    if (targetSection) {
        targetSection.classList.add("active");
    }
    
    // Highlight nav button
    const activeNav = document.querySelector(`.nav-item[data-target="${pageId}"]`);
    if (activeNav) {
        activeNav.classList.add("active");
    }
    
    // Update top bar title
    const topTitle = document.getElementById("topbar-title");
    if (topTitle) {
        topTitle.textContent = pageId.charAt(0).toUpperCase() + pageId.slice(1) + " Section";
        if (pageId === "home") topTitle.textContent = "Dashboard Overview";
        if (pageId === "results") topTitle.textContent = "Assessment Report";
        if (pageId === "performance") topTitle.textContent = "Model Evaluations";
    }
}

// 2. Health check
async function checkApiHealth() {
    const dot = document.getElementById("api-status-dot");
    const text = document.getElementById("api-status-text");
    
    try {
        const response = await fetch(`${API_BASE_URL}/health`, { signal: AbortSignal.timeout(5000) });
        if (response.ok) {
            state.apiOnline = true;
            dot.className = "status-dot online";
            text.textContent = "API Live (Secure Backend)";
            loadPerformanceMetadata();
        } else {
            throw new Error("API health check failed");
        }
    } catch (e) {
        state.apiOnline = false;
        dot.className = "status-dot offline";
        text.textContent = "Backend Offline (Mock Mode enabled)";
        console.warn("Backend API is currently offline. Running in frontend mock fallback mode.", e);
        // Load fallback static performance data
        renderFallbackPerformance();
    }
}

// 3. Form Submission
async function handleFormSubmit(event) {
    event.preventDefault();
    
    const form = event.target;
    const formData = new FormData(form);
    const payload = {};
    
    formData.forEach((value, key) => {
        payload[key] = value;
    });
    
    // Convert Age to integer
    payload.Age = parseInt(payload.Age, 10);
    
    // Show loading state
    const submitBtn = form.querySelector("button[type='submit']");
    const originalText = submitBtn.textContent;
    submitBtn.disabled = true;
    submitBtn.textContent = "Running ML Inference... 🧠";
    
    try {
        let result;
        
        if (state.apiOnline) {
            // Live request
            const response = await fetch(`${API_BASE_URL}/predict`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            if (!response.ok) throw new Error("Inference failed");
            result = await response.json();
        } else {
            // Mock offline fallback
            result = getMockPrediction(payload);
        }
        
        // Save to state
        state.predictionResult = result;
        state.userInputs = payload;
        
        // Render results and redirect
        renderPredictionResult(result, payload);
        
        // Enable Results tab
        const navResBtn = document.getElementById("nav-results");
        navResBtn.removeAttribute("disabled");
        
        navigateToPage("results");
        
    } catch (err) {
        alert(`Prediction Error: ${err.message}. Enabling mock mode to run prediction.`);
        // Run mock fallback
        const result = getMockPrediction(payload);
        state.predictionResult = result;
        state.userInputs = payload;
        renderPredictionResult(result, payload);
        const navResBtn = document.getElementById("nav-results");
        navResBtn.removeAttribute("disabled");
        navigateToPage("results");
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = originalText;
    }
}

// 4. Render Results Page
function renderPredictionResult(result, inputs) {
    const riskCat = result.risk_category;
    const probPct = result.probability_pct;
    
    // Select HTML elements
    const card = document.getElementById("result-status-card");
    const catText = document.getElementById("result-risk-category");
    const probText = document.getElementById("result-probability-pct");
    const progressFill = document.getElementById("result-progress-fill");
    const recText = document.getElementById("recommendation-text");
    
    // Set text
    catText.textContent = riskCat.toUpperCase();
    probText.textContent = `${probPct.toFixed(1)}%`;
    
    // Style progress bar and card border
    card.className = "result-card"; // reset
    if (riskCat === "Low Risk") {
        card.classList.add("low-risk");
        progressFill.style.backgroundColor = "var(--color-emerald)";
        recText.textContent = "Maintain healthy oral hygiene standards (brushing, flossing), avoid tobacco/arecanut exposure, eat fresh fruits/vegetables, and consult a dentist for routine annual oral health checkups.";
    } else if (riskCat === "Moderate Risk") {
        card.classList.add("mod-risk");
        progressFill.style.backgroundColor = "var(--color-amber)";
        recText.textContent = "A moderate risk score indicates lifestyle or clinical attributes that warrant attention. We suggest discussing your risk exposures (tobacco, alcohol, poor hygiene) with a dentist or medical practitioner to formulate a preventative plan.";
    } else {
        card.classList.add("high-risk");
        progressFill.style.backgroundColor = "var(--color-red)";
        recText.textContent = "A high-risk score indicates significant risk factor matches (e.g. tobacco use, areca nut chewing, visible clinical symptoms). We strongly advise consulting a dentist, oral pathologist, or physician immediately for a comprehensive physical screening and examination.";
    }
    
    // Animate progress bar fill
    setTimeout(() => {
        progressFill.style.width = `${probPct}%`;
    }, 100);
    
    // Handle Symptom Alerts
    const symptomAlertBox = document.getElementById("symptoms-alert-box");
    const symptomListText = document.getElementById("symptoms-list-text");
    
    const activeSymptoms = [];
    if (inputs.Oral_Lesions === "Yes") activeSymptoms.push("Oral Lesions / Ulcers");
    if (inputs.Unexplained_Bleeding === "Yes") activeSymptoms.push("Mouth Bleeding");
    if (inputs.White_or_Red_Patches_in_Mouth === "Yes") activeSymptoms.push("Persistent Red/White Patches");
    if (inputs.Difficulty_Swallowing === "Yes") activeSymptoms.push("Difficulty Swallowing (Dysphagia)");
    
    if (activeSymptoms.length > 0) {
        symptomAlertBox.classList.remove("hidden");
        symptomListText.innerHTML = `You reported experiencing: <strong>${activeSymptoms.join(", ")}</strong>. <br>Persistent clinical warning signs require direct clinical evaluation by a medical professional or oral pathologist, independent of the model's computed risk score.`;
    } else {
        symptomAlertBox.classList.add("hidden");
    }
    
    // Render explanations (Counterfactual factor attributions)
    const container = document.getElementById("contributors-container");
    container.innerHTML = ""; // clear
    
    if (result.explanations && result.explanations.length > 0) {
        result.explanations.forEach(item => {
            const div = document.createElement("div");
            div.className = "contrib-item";
            div.innerHTML = `
                <div class="contrib-header">
                    <span class="c-lbl">${item.label}</span>
                    <span class="c-weight">+${item.contribution_pct.toFixed(1)}%</span>
                </div>
                <div class="contrib-text">${item.explanation_text}</div>
            `;
            container.appendChild(div);
        });
    } else {
        const p = document.createElement("p");
        p.textContent = "No significant local risk factors identified. Your profile matches healthy reference baselines.";
        p.style.color = "var(--text-secondary)";
        p.style.fontSize = "0.9rem";
        container.appendChild(p);
    }
}

// 5. Model Performance and Fallbacks
async function loadPerformanceMetadata() {
    try {
        // Load metadata
        const metaRes = await fetch(`${API_BASE_URL}/metadata`);
        const meta = await metaRes.json();
        
        document.getElementById("selected-model-name").textContent = meta.model_name;
        document.getElementById("meta-accuracy").textContent = `${(meta.metrics.accuracy * 100).toFixed(1)}%`;
        document.getElementById("meta-recall").textContent = `${(meta.metrics.recall * 100).toFixed(1)}%`;
        document.getElementById("meta-f1").textContent = `${(meta.metrics.f1_score * 100).toFixed(1)}%`;
        document.getElementById("meta-auc").textContent = meta.metrics.roc_auc.toFixed(3);
        
        // Load comparison
        const compRes = await fetch(`${API_BASE_URL}/comparison`);
        const comp = await compRes.json();
        renderComparisonTable(comp.models);
        updateComparisonChart(comp.models);
        
    } catch (e) {
        console.error("Failed to load live metadata, falling back to static presentation", e);
        renderFallbackPerformance();
    }
}

function renderFallbackPerformance() {
    const fallbackModels = [
        { Model: "Decision Tree (Selected)", Test_Accuracy: 0.4964, Test_Recall: 0.5993, Test_F1_Score: 0.5427, Test_ROC_AUC: 0.4987 },
        { Model: "Random Forest", Test_Accuracy: 0.4970, Test_Recall: 0.4818, Test_F1_Score: 0.4886, Test_ROC_AUC: 0.4934 },
        { Model: "Logistic Regression", Test_Accuracy: 0.4962, Test_Recall: 0.4734, Test_F1_Score: 0.4838, Test_ROC_AUC: 0.4937 },
        { Model: "XGBoost", Test_Accuracy: 0.4943, Test_Recall: 0.4746, Test_F1_Score: 0.4835, Test_ROC_AUC: 0.4923 }
    ];
    renderComparisonTable(fallbackModels);
    updateComparisonChart(fallbackModels);
}

function renderComparisonTable(models) {
    const tbody = document.getElementById("comparison-table-body");
    tbody.innerHTML = ""; // clear
    
    models.forEach(m => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td><strong>${m.Model || m.model_name}</strong></td>
            <td>${((m.Test_Accuracy || m.accuracy || m.Test_Accuracy) * 100).toFixed(2)}%</td>
            <td>${((m.Test_Recall || m.recall || m.Test_Recall) * 100).toFixed(2)}%</td>
            <td>${((m.Test_F1_Score || m.f1_score || m.Test_F1_Score) * 100).toFixed(2)}%</td>
            <td>${(m.Test_ROC_AUC || m.roc_auc || m.Test_ROC_AUC).toFixed(4)}</td>
        `;
        tbody.appendChild(tr);
    });
}

// 6. ChartJS Visualizations
function initCharts() {
    const ctx = document.getElementById("chart-comparison").getContext("2d");
    
    state.comparisonChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Recall (Sensitivity)',
                    backgroundColor: '#10b981',
                    borderColor: '#10b981',
                    borderWidth: 1,
                    data: []
                },
                {
                    label: 'F1-Score',
                    backgroundColor: '#0d9488',
                    borderColor: '#0d9488',
                    borderWidth: 1,
                    data: []
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#f8fafc' }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#94a3b8' }
                },
                y: {
                    min: 0,
                    max: 100,
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#94a3b8', callback: function(value) { return value + "%" } }
                }
            }
        }
    });
}

function updateComparisonChart(models) {
    if (!state.comparisonChart) return;
    
    const labels = [];
    const recalls = [];
    const f1s = [];
    
    models.forEach(m => {
        labels.push(m.Model || m.model_name);
        recalls.push(((m.Test_Recall || m.recall) * 100).toFixed(1));
        f1s.push(((m.Test_F1_Score || m.f1_score) * 100).toFixed(1));
    });
    
    state.comparisonChart.data.labels = labels;
    state.comparisonChart.data.datasets[0].data = recalls;
    state.comparisonChart.data.datasets[1].data = f1s;
    state.comparisonChart.update();
}

// 7. Mock predictions for offline mode
function getMockPrediction(payload) {
    // Generate a semi-realistic probability from inputs to simulate ML response
    let score = 0.35; // base probability
    
    // Add weights for risk exposures
    if (payload.Tobacco_Use === "Current") score += 0.20;
    if (payload.Tobacco_Use === "Former") score += 0.08;
    
    if (payload.Alcohol_Consumption === "Current") score += 0.12;
    if (payload.Betel_Quid_Use === "Current") score += 0.18;
    
    if (payload.Poor_Oral_Hygiene === "Poor") score += 0.08;
    if (payload.Diet_Fruits_Vegetables_Intake === "Low") score += 0.05;
    
    if (payload.Family_History_of_Cancer === "Yes") score += 0.07;
    if (payload.HPV_Infection === "Yes") score += 0.09;
    if (payload.Compromised_Immune_System === "Compromised") score += 0.08;
    
    // Symptoms add minor score weight but alert directly
    if (payload.Oral_Lesions === "Yes") score += 0.08;
    if (payload.Unexplained_Bleeding === "Yes") score += 0.08;
    if (payload.White_or_Red_Patches_in_Mouth === "Yes") score += 0.09;
    if (payload.Difficulty_Swallowing === "Yes") score += 0.07;
    
    // Factor age
    if (payload.Age > 50) score += 0.05;
    
    // cap probability
    score = Math.max(0.1, Math.min(0.95, score));
    
    // Define category
    let cat = "Low Risk";
    if (score >= DEFAULT_LOW_THRESHOLD && score < DEFAULT_HIGH_THRESHOLD) {
        cat = "Moderate Risk";
    } else if (score >= DEFAULT_HIGH_THRESHOLD) {
        cat = "High Risk";
    }
    
    // Generate explanations
    const explanations = [];
    
    if (payload.Tobacco_Use === "Current") {
        explanations.push({
            label: "Tobacco Use (Current)",
            explanation_text: "Current tobacco exposure is the single largest clinical risk factor for oral malignancies, increasing estimated risk score by +20.0%.",
            contribution_pct: 20.0
        });
    }
    if (payload.Betel_Quid_Use === "Current") {
        explanations.push({
            label: "Betel Quid / Areca Nut (Current)",
            explanation_text: "Chewing areca nut or betel preparations is a primary class-1 carcinogen, contributing +18.0% to the estimated risk.",
            contribution_pct: 18.0
        });
    }
    if (payload.Alcohol_Consumption === "Current") {
        explanations.push({
            label: "Alcohol Consumption (Current)",
            explanation_text: "Alcohol intake increases mucosal permeability, acting synergistically with other carcinogens (+12.0%).",
            contribution_pct: 12.0
        });
    }
    if (payload.HPV_Infection === "Yes") {
        explanations.push({
            label: "HPV Infection (Yes)",
            explanation_text: "Human Papillomavirus exposure is associated with high rates of base-of-tongue and tonsil cells (+9.0%).",
            contribution_pct: 9.0
        });
    }
    if (payload.Oral_Lesions === "Yes") {
        explanations.push({
            label: "Oral Lesions Presence (Yes)",
            explanation_text: "Visible mouth sores or chronic ulcers are primary symptoms, increasing estimated risk index (+8.0%).",
            contribution_pct: 8.0
        });
    }
    if (payload.White_or_Red_Patches_in_Mouth === "Yes") {
        explanations.push({
            label: "Persistent Red/White Patches (Yes)",
            explanation_text: "Persistent patches represent clinical leukoplakia/erythroplakia and require pathologist biopsy screening (+9.0%).",
            contribution_pct: 9.0
        });
    }
    
    return {
        probability: score,
        probability_pct: score * 100,
        risk_category: cat,
        explanations: explanations.sort((a,b) => b.contribution_pct - a.contribution_pct).slice(0, 3)
    };
}
