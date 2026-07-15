/* =========================================================================
   EEG DEMENTIA AI — APPLICATION LOGIC
   Talks to a Flask backend at POST /predict via the Fetch API.

   Table of Contents:
     1.  Constants & Configuration
     2.  DOM Element Cache
     3.  Application State
     4.  Initialization
     5.  Navigation (mobile menu)
     6.  File Upload Handling
     7.  Input Validation
     8.  Predict Flow
     9.  Backend Request
     10. Result Rendering
     11. Probability Table
     12. Progress Bars
     13. Chart.js Rendering
     14. Disease Information
     15. Loading State
     16. Notifications / Errors
     17. Predict Again / Clear / Reset
     18. Download Report
     19. Utilities
   ========================================================================= */

/* =========================================================================
   1. CONSTANTS & CONFIGURATION
   ========================================================================= */
const PREDICT_ENDPOINT = "/predict";
const REQUEST_TIMEOUT_MS = 30000;
// Allowed file types matched with backend configuration
const ALLOWED_EXTENSIONS = ["csv", "edf", "mat", "txt", "npy", "set"];

const MODEL_LABELS = {
    binary: "Binary Classification",
    "three-class": "Three-Class Classification"
};

const CLASS_LABELS = {
    AD: "Alzheimer's Disease",
    FTD: "Frontotemporal Dementia",
    HC: "Healthy Control"
};

// Local clinical reference data used to enrich the disease information card.
// The backend only needs to return a short "description" string; everything
// else here is presentation content mapped from the prediction label.
const DISEASE_INFO = {
    "Alzheimer's Disease": {
        overview:
            "Alzheimer's disease is a progressive neurodegenerative disorder " +
            "that gradually impairs memory, reasoning, and daily functioning.",
        symptoms:
            "Memory loss, disorientation, difficulty with language, and " +
            "impaired judgment.",
        importance:
            "Early identification allows for timely intervention, treatment " +
            "planning, and support for the patient and caregivers.",
        recommendation: "Consult a neurologist for further clinical evaluation.",
        risk: "High"
    },
    "Frontotemporal Dementia": {
        overview:
            "Frontotemporal dementia primarily affects the frontal and " +
            "temporal lobes, impacting behavior, personality, and language.",
        symptoms:
            "Behavioral changes, impulsivity, apathy, and language " +
            "difficulties.",
        importance:
            "Distinguishing FTD from other dementias is critical, as " +
            "management and prognosis differ significantly.",
        recommendation: "Further neurological assessment is recommended.",
        risk: "High"
    },
    "Healthy Control": {
        overview:
            "No dementia-related pattern was detected in the analyzed EEG " +
            "signal.",
        symptoms: "No significant cognitive or behavioral symptoms indicated.",
        importance:
            "Routine screening supports early detection should patterns " +
            "change over time.",
        recommendation:
            "No significant abnormalities detected. Continue regular " +
            "monitoring if clinically advised.",
        risk: "Low"
    }
};

/* =========================================================================
   2. DOM ELEMENT CACHE
   ========================================================================= */
const elements = {};

function cacheElements() {
    elements.navToggle = document.getElementById("navToggle");
    elements.navMenu = document.getElementById("navMenu");

    elements.modelSelect = document.getElementById("modelSelect");
    elements.eegFile = document.getElementById("eegFile");
    elements.uploadArea = document.getElementById("uploadArea");
    elements.uploadedFileName = document.getElementById("uploadedFileName");

    elements.predictBtn = document.getElementById("predictBtn");
    elements.loading = document.getElementById("loading");
    elements.resultCard = document.getElementById("resultCard");

    elements.fileName = document.getElementById("fileName");
    elements.selectedModel = document.getElementById("selectedModel");
    elements.prediction = document.getElementById("predictionResult");
    elements.confidence = document.getElementById("confidence");
    elements.predictionDateTime = document.getElementById("predictionDateTime");
    elements.status = document.getElementById("status");
    elements.riskLevel = document.getElementById("riskLevel");

    elements.probabilityTableBody = document.getElementById("probabilityTableBody");
    elements.diseaseDescription = document.getElementById("diseaseDescription");
    elements.recommendationText = document.getElementById("recommendationText");
    elements.predictionChart = document.getElementById("predictionChart");

    elements.adBar = document.getElementById("adBar");
    elements.ftdBar = document.getElementById("ftdBar");
    elements.hcBar = document.getElementById("hcBar");
    elements.adBarPercent = document.getElementById("adBarPercent");
    elements.ftdBarPercent = document.getElementById("ftdBarPercent");
    elements.hcBarPercent = document.getElementById("hcBarPercent");

    elements.predictAgain = document.getElementById("predictAgain");
    elements.clearBtn = document.getElementById("clearBtn");
    elements.downloadBtn = document.getElementById("downloadBtn");
}

/* =========================================================================
   3. APPLICATION STATE
   ========================================================================= */
const state = {
    selectedFile: null,
    isSubmitting: false,
    chartInstance: null,
    lastResult: null // cached for "Predict Again" and "Download Report"
};

/* =========================================================================
   4. INITIALIZATION
   ========================================================================= */
document.addEventListener("DOMContentLoaded", initializeApp);

function initializeApp() {
    cacheElements();
    initializeNav();
    initializeUploadArea();
    initializeFormControls();
    initializeActionButtons();
}

/* =========================================================================
   5. NAVIGATION (MOBILE MENU)
   ========================================================================= */
function initializeNav() {
    if (!elements.navToggle || !elements.navMenu) return;

    elements.navToggle.addEventListener("click", () => {
        const isOpen = elements.navMenu.classList.toggle("is-open");
        elements.navToggle.setAttribute("aria-expanded", String(isOpen));
    });

    elements.navMenu.querySelectorAll(".nav-link").forEach((link) => {
        link.addEventListener("click", () => {
            elements.navMenu.classList.remove("is-open");
            elements.navToggle.setAttribute("aria-expanded", "false");
        });
    });
}

/* =========================================================================
   6. FILE UPLOAD HANDLING
   ========================================================================= */
function initializeUploadArea() {
    const { uploadArea, eegFile } = elements;
    if (!uploadArea || !eegFile) return;

    // Clicking or activating the drop zone opens the file dialog.
    uploadArea.addEventListener("click", () => eegFile.click());
    uploadArea.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            eegFile.click();
        }
    });

    eegFile.addEventListener("change", () => {
        const file = eegFile.files && eegFile.files[0];
        handleFileSelect(file);
    });

    // Drag & drop behaviour.
    ["dragenter", "dragover"].forEach((eventName) => {
        uploadArea.addEventListener(eventName, (event) => {
            event.preventDefault();
            event.stopPropagation();
            uploadArea.classList.add("drag-active");
        });
    });

    ["dragleave", "drop"].forEach((eventName) => {
        uploadArea.addEventListener(eventName, (event) => {
            event.preventDefault();
            event.stopPropagation();
            uploadArea.classList.remove("drag-active");
        });
    });

    uploadArea.addEventListener("drop", (event) => {
        const file = event.dataTransfer.files && event.dataTransfer.files[0];
        if (file) {
            // Keep the native file input in sync so a subsequent form
            // submission (if any) reflects the dropped file too.
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);
            eegFile.files = dataTransfer.files;
        }
        handleFileSelect(file);
    });
}

function handleFileSelect(file) {
    if (!file) return;

    if (!isValidFileType(file)) {
        state.selectedFile = null;
        elements.uploadedFileName.textContent = "";
        showNotification(
            `Unsupported file type ".${getFileExtension(file.name)}". ` +
                `Please upload one of: ${ALLOWED_EXTENSIONS.map((ext) => "." + ext).join(", ")}.`,
            "error"
        );
        return;
    }

    state.selectedFile = file;
    elements.uploadedFileName.textContent = `Selected file: ${file.name}`;
}

function isValidFileType(file) {
    const extension = getFileExtension(file.name);
    return ALLOWED_EXTENSIONS.includes(extension);
}

function getFileExtension(filename) {
    return filename.split(".").pop().toLowerCase();
}

/* =========================================================================
   7. INPUT VALIDATION
   ========================================================================= */
function initializeFormControls() {
    elements.predictBtn.addEventListener("click", onPredictClick);
}

function validateInputs() {
    if (!state.selectedFile) {
        showNotification("Please select an EEG file before running a prediction.", "warning");
        return false;
    }

    if (!isValidFileType(state.selectedFile)) {
        showNotification("The selected file type is not supported.", "error");
        return false;
    }

    if (!elements.modelSelect.value) {
        showNotification("Please select a prediction model.", "warning");
        return false;
    }

    return true;
}

/* =========================================================================
   8. PREDICT FLOW
   ========================================================================= */
async function onPredictClick() {
    if (state.isSubmitting) return; // Prevent duplicate/simultaneous requests
    if (!validateInputs()) return;

    const file = state.selectedFile;
    const model = elements.modelSelect.value;

    try {
        state.isSubmitting = true;
        setControlsDisabled(true);
        resetResults();
        showLoading();

        const data = await sendPredictionRequest(file, model);

        hideLoading();
        displayResults(data, model, file.name);
    } catch (error) {
        hideLoading();
        showNotification(getFriendlyErrorMessage(error), "error");
    } finally {
        state.isSubmitting = false;
        setControlsDisabled(false);
    }
}

function setControlsDisabled(disabled) {
    elements.predictBtn.disabled = disabled;
    elements.predictBtn.classList.toggle("is-loading", disabled);
    elements.eegFile.disabled = disabled;
    elements.modelSelect.disabled = disabled;
    elements.uploadArea.setAttribute("aria-disabled", String(disabled));
}

/* =========================================================================
   9. BACKEND REQUEST
   ========================================================================= */
async function sendPredictionRequest(file, model) {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("model", model);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

    let response;
    try {
        response = await fetch(PREDICT_ENDPOINT, {
            method: "POST",
            body: formData,
            signal: controller.signal
        });
    } catch (networkError) {
        if (networkError.name === "AbortError") {
            throw new Error("TIMEOUT");
        }
        throw new Error("NETWORK_ERROR");
    } finally {
        clearTimeout(timeoutId);
    }

    if (!response.ok) {
        // Try to extract a server-provided message; fall back to status text.
        let serverMessage = "";
        try {
            const errorBody = await response.json();
            serverMessage = errorBody && (errorBody.message || errorBody.error);
        } catch (_) {
            // Response body wasn't JSON — ignore and use the generic message.
        }
        throw new Error(`HTTP_${response.status}:${serverMessage || response.statusText}`);
    }

    let data;
    try {
        data = await response.json();
    } catch (_) {
        throw new Error("INVALID_JSON");
    }

    if (!data || typeof data !== "object" || Object.keys(data).length === 0) {
        throw new Error("EMPTY_RESPONSE");
    }

    return data;
}

function getFriendlyErrorMessage(error) {
    const msg = error && error.message ? error.message : "";

    if (msg === "TIMEOUT") {
        return "The request timed out. Please check your connection and try again.";
    }
    if (msg === "NETWORK_ERROR") {
        return "Unable to reach the prediction server. Please make sure it is running and try again.";
    }
    if (msg === "INVALID_JSON") {
        return "The server returned an unreadable response. Please try again.";
    }
    if (msg === "EMPTY_RESPONSE") {
        return "The server returned an empty response. Please try again.";
    }
    if (msg.startsWith("HTTP_400")) {
        return "The request was invalid. Please check your file and model selection.";
    }
    if (msg.startsWith("HTTP_404")) {
        return "The prediction service could not be found. Please contact support.";
    }
    if (msg.startsWith("HTTP_500")) {
        return "The server encountered an error while processing your EEG file.";
    }
    if (msg.startsWith("HTTP_")) {
        const [, detail] = msg.split(":");
        return detail ? `Request failed: ${detail}` : "The prediction request failed. Please try again.";
    }

    return "Something went wrong while running the prediction. Please try again.";
}

/* =========================================================================
   10. RESULT RENDERING
   ========================================================================= */
function displayResults(data, modelValue, filename) {
    state.lastResult = { data, modelValue, filename, timestamp: new Date() };

    const modelType = modelValue === "binary" ? "binary" : "three-class";
    const probabilities = normalizeProbabilities(data.probabilities, modelType);

    elements.fileName.textContent = filename;
    elements.selectedModel.textContent = MODEL_LABELS[modelValue] || data.model || "—";
    elements.prediction.textContent = data.prediction || "—";
    elements.confidence.textContent = formatPercentage(data.confidence);
    elements.predictionDateTime.textContent = state.lastResult.timestamp.toLocaleString();

    renderStatusBadge(data.status);
    renderRiskBadge(data.prediction);
    updateProbabilityTable(probabilities, modelType);
    updateProgressBars(probabilities, modelType);
    renderChart(probabilities, modelType);
    renderDiseaseInformation(data.prediction, data.description);

    elements.resultCard.hidden = false;
    elements.resultCard.classList.remove("fade-in");
    // Force reflow so the animation replays on every new prediction.
    void elements.resultCard.offsetWidth;
    elements.resultCard.classList.add("slide-up");

    elements.resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
}

function normalizeProbabilities(probabilities, modelType) {
    const source = probabilities || {};
    const normalized = {
        AD: Number(source.AD) || 0,
        HC: Number(source.HC) || 0
    };
    if (modelType === "three-class") {
        normalized.FTD = Number(source.FTD) || 0;
    }
    return normalized;
}

function renderStatusBadge(status) {
    const label = status || "Success";
    const badgeClass = classForStatus(label);
    elements.status.innerHTML = `<span class="badge ${badgeClass}">${escapeHtml(label)}</span>`;
}

function classForStatus(status) {
    const normalized = String(status).toLowerCase();
    if (normalized === "error") return "badge-ftd";
    if (normalized === "warning") return "badge-ad";
    return "badge-healthy";
}

function renderRiskBadge(predictionLabel) {
    const info = DISEASE_INFO[predictionLabel];
    const risk = info ? info.risk : "Unknown";
    const riskClass =
        risk === "High" ? "badge-ftd" : risk === "Low" ? "badge-healthy" : "badge-ad";
    elements.riskLevel.innerHTML = `<span class="badge ${riskClass}">${escapeHtml(risk)}</span>`;
}

/* =========================================================================
   11. PROBABILITY TABLE
   ========================================================================= */
function updateProbabilityTable(probabilities, modelType) {
    const classOrder = modelType === "binary" ? ["AD", "HC"] : ["AD", "FTD", "HC"];

    // Clear any previous rows before rendering the new result.
    elements.probabilityTableBody.innerHTML = "";

    classOrder.forEach((classKey) => {
        const value = probabilities[classKey] || 0;
        const row = document.createElement("tr");
        row.innerHTML = `
            <td>${CLASS_LABELS[classKey]}</td>
            <td>${(value / 100).toFixed(3)}</td>
            <td>${formatPercentage(value)}</td>
        `;
        elements.probabilityTableBody.appendChild(row);
    });
}

/* =========================================================================
   12. PROGRESS BARS
   ========================================================================= */
function updateProgressBars(probabilities, modelType) {
    setBarValue(elements.adBar, elements.adBarPercent, probabilities.AD);
    setBarValue(elements.hcBar, elements.hcBarPercent, probabilities.HC);

    const ftdRow = elements.ftdBar.closest(".progress-item");
    if (modelType === "binary") {
        ftdRow.hidden = true;
    } else {
        ftdRow.hidden = false;
        setBarValue(elements.ftdBar, elements.ftdBarPercent, probabilities.FTD);
    }
}

function setBarValue(barEl, labelEl, value) {
    const percentage = clampPercentage(value);
    // Reset to 0 first so the width transition animates on every update.
    barEl.style.width = "0%";
    barEl.setAttribute("aria-valuenow", "0");

    requestAnimationFrame(() => {
        barEl.style.width = `${percentage}%`;
        barEl.setAttribute("aria-valuenow", String(percentage));
    });

    labelEl.textContent = formatPercentage(value);
}

function clampPercentage(value) {
    const num = Number(value) || 0;
    return Math.min(100, Math.max(0, num));
}

/* =========================================================================
   13. CHART.JS RENDERING
   ========================================================================= */
function renderChart(probabilities, modelType) {
    const classOrder = modelType === "binary" ? ["AD", "HC"] : ["AD", "FTD", "HC"];
    const labels = classOrder.map((key) => key);
    const values = classOrder.map((key) => Number(probabilities[key]) || 0);
    const colors = classOrder.map((key) => chartColorForClass(key));

    // Destroy any previous chart instance to avoid canvas reuse errors.
    if (state.chartInstance) {
        state.chartInstance.destroy();
        state.chartInstance = null;
    }

    const ctx = elements.predictionChart.getContext("2d");
    state.chartInstance = new Chart(ctx, {
        type: "bar",
        data: {
            labels,
            datasets: [
                {
                    label: "Probability (%)",
                    data: values,
                    backgroundColor: colors,
                    borderRadius: 8,
                    maxBarThickness: 64
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (context) => `${context.parsed.y.toFixed(1)}%`
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        color: "#D7CCC8",
                        callback: (value) => `${value}%`
                    },
                    grid: { color: "rgba(255,255,255,0.08)" }
                },
                x: {
                    ticks: { color: "#D7CCC8" },
                    grid: { display: false }
                }
            }
        }
    });
}

function chartColorForClass(classKey) {
    switch (classKey) {
        case "AD":
            return "#F2A65A";
        case "FTD":
            return "#E8615B";
        case "HC":
            return "#6FCF97";
        default:
            return "#E0B15C";
    }
}

/* =========================================================================
   14. DISEASE INFORMATION
   ========================================================================= */
function renderDiseaseInformation(predictionLabel, backendDescription) {
    const info = DISEASE_INFO[predictionLabel];
    const overview = backendDescription || (info ? info.overview : "No description available.");

    if (!info) {
        elements.diseaseDescription.innerHTML = `
            <h4>${escapeHtml(predictionLabel || "Unknown")}</h4>
            <p>${escapeHtml(overview)}</p>
        `;
        elements.recommendationText.textContent =
            "Please consult a qualified clinician to interpret this result.";
        return;
    }

    elements.diseaseDescription.innerHTML = `
        <h4>${escapeHtml(predictionLabel)}</h4>
        <p>${escapeHtml(overview)}</p>
        <p><strong>Common Symptoms:</strong> ${escapeHtml(info.symptoms)}</p>
        <p><strong>Clinical Importance:</strong> ${escapeHtml(info.importance)}</p>
        <div class="disease-note"><strong>Recommended Action:</strong> ${escapeHtml(info.recommendation)}</div>
    `;

    elements.recommendationText.textContent = info.recommendation;
}

/* =========================================================================
   15. LOADING STATE
   ========================================================================= */
function showLoading() {
    elements.loading.hidden = false;
    elements.loading.classList.add("fade-in");
}

function hideLoading() {
    elements.loading.hidden = true;
    elements.loading.classList.remove("fade-in");
}

/* =========================================================================
   16. NOTIFICATIONS / ERRORS
   ========================================================================= */
let notificationTimeoutId = null;

/**
 * Shows a lightweight, self-contained toast notification.
 * Uses inline styles so it works independently of any CSS class names,
 * while still matching the app's dark brown / gold visual language.
 */
function showNotification(message, type = "error") {
    let toast = document.getElementById("appToast");

    if (!toast) {
        toast = document.createElement("div");
        toast.id = "appToast";
        toast.setAttribute("role", "alert");
        toast.setAttribute("aria-live", "assertive");
        Object.assign(toast.style, {
            position: "fixed",
            bottom: "24px",
            left: "50%",
            transform: "translateX(-50%)",
            maxWidth: "90vw",
            padding: "0.9rem 1.4rem",
            borderRadius: "999px",
            fontFamily: "'Outfit', sans-serif",
            fontWeight: "600",
            fontSize: "0.9rem",
            color: "#2C1810",
            boxShadow: "0 12px 30px rgba(0,0,0,0.45)",
            zIndex: "3000",
            transition: "opacity 0.3s ease, transform 0.3s ease"
        });
        document.body.appendChild(toast);
    }

    const palette = {
        error: "#E8615B",
        warning: "#F2A65A",
        success: "#6FCF97"
    };
    toast.style.background = palette[type] || palette.error;
    toast.textContent = message;
    toast.style.opacity = "1";

    clearTimeout(notificationTimeoutId);
    notificationTimeoutId = setTimeout(() => {
        toast.style.opacity = "0";
    }, 4500);
}

/* =========================================================================
   17. PREDICT AGAIN / CLEAR / RESET
   ========================================================================= */
function initializeActionButtons() {
    elements.predictAgain.addEventListener("click", handlePredictAgain);
    elements.clearBtn.addEventListener("click", clearApplication);
    elements.downloadBtn.addEventListener("click", downloadReport);
}

function handlePredictAgain() {
    // Keep the selected file and model, just clear the displayed results.
    resetResults();
    elements.resultCard.hidden = true;
}

/**
 * Clears rendered results (table, chart, progress bars) without touching
 * the currently selected file or model. Used before every new prediction
 * and by "Predict Again".
 */
function resetResults() {
    elements.probabilityTableBody.innerHTML = "";
    elements.diseaseDescription.innerHTML = "";
    elements.recommendationText.textContent = "";

    [elements.adBar, elements.ftdBar, elements.hcBar].forEach((bar) => {
        bar.style.width = "0%";
        bar.setAttribute("aria-valuenow", "0");
    });
    [elements.adBarPercent, elements.ftdBarPercent, elements.hcBarPercent].forEach((label) => {
        label.textContent = "0%";
    });
    elements.ftdBar.closest(".progress-item").hidden = false;

    if (state.chartInstance) {
        state.chartInstance.destroy();
        state.chartInstance = null;
    }

    elements.fileName.textContent = "—";
    elements.selectedModel.textContent = "—";
    elements.prediction.textContent = "—";
    elements.confidence.textContent = "—";
    elements.predictionDateTime.textContent = "—";
    elements.status.textContent = "—";
    elements.riskLevel.textContent = "—";
}

/**
 * Fully resets the application to its initial state: clears the file,
 * model selection, and all rendered results.
 */
function clearApplication() {
    state.selectedFile = null;
    state.lastResult = null;

    elements.eegFile.value = "";
    elements.uploadedFileName.textContent = "";
    elements.modelSelect.value = "";

    hideLoading();
    elements.resultCard.hidden = true;
    resetResults();
}

/* =========================================================================
   18. DOWNLOAD REPORT
   ========================================================================= */
function downloadReport() {
    if (!state.lastResult) {
        showNotification("Run a prediction before downloading a report.", "warning");
        return;
    }

    const { data, modelValue, filename, timestamp } = state.lastResult;
    const modelType = modelValue === "binary" ? "binary" : "three-class";
    const probabilities = normalizeProbabilities(data.probabilities, modelType);

    const { jsPDF } = window.jspdf;
    const doc = new jsPDF({ unit: "pt", format: "a4" });
    const pageWidth = doc.internal.pageSize.getWidth();
    const marginX = 48;
    let y = 0;

    // Brand colors (matching the site's theme)
    const bgDark = [44, 24, 16];
    const gold = [224, 177, 92];
    const cream = [239, 235, 233];
    const muted = [140, 120, 112];
    const healthy = [111, 207, 151];
    const ad = [242, 166, 90];
    const ftd = [232, 97, 91];

    // --- Header banner ---
    doc.setFillColor(...bgDark);
    doc.rect(0, 0, pageWidth, 90, "F");
    doc.setTextColor(...gold);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(20);
    doc.text("EEG Dementia AI — Prediction Report", marginX, 40);
    doc.setTextColor(...cream);
    doc.setFont("helvetica", "normal");
    doc.setFontSize(10);
    doc.text(`Generated: ${timestamp.toLocaleString()}`, marginX, 62);

    y = 130;

    // --- Summary box ---
    doc.setDrawColor(...muted);
    doc.setFillColor(250, 248, 246);
    doc.roundedRect(marginX, y, pageWidth - marginX * 2, 110, 8, 8, "FD");

    const col1 = marginX + 20;
    const col2 = marginX + (pageWidth - marginX * 2) / 2 + 10;
    doc.setTextColor(...muted);
    doc.setFontSize(9);
    doc.text("UPLOADED FILE", col1, y + 24);
    doc.text("SELECTED MODEL", col2, y + 24);
    doc.text("PREDICTION RESULT", col1, y + 64);
    doc.text("CONFIDENCE SCORE", col2, y + 64);

    doc.setTextColor(20, 20, 20);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(11);
    doc.text(String(filename), col1, y + 40, { maxWidth: (pageWidth - marginX * 2) / 2 - 30 });
    doc.text(MODEL_LABELS[modelValue] || data.model || "—", col2, y + 40);

    doc.setTextColor(...gold);
    doc.setFontSize(14);
    doc.text(data.prediction || "—", col1, y + 84);
    doc.text(formatPercentage(data.confidence), col2, y + 84);

    y += 140;

    // --- Probability breakdown ---
    doc.setTextColor(20, 20, 20);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(13);
    doc.text("Probability Breakdown", marginX, y);
    y += 20;

    const barColors = { AD: ad, FTD: ftd, HC: healthy };
    const barWidth = pageWidth - marginX * 2 - 90;

    Object.keys(probabilities).forEach((key) => {
        const pct = Number(probabilities[key]) || 0;
        doc.setFont("helvetica", "normal");
        doc.setFontSize(10);
        doc.setTextColor(60, 60, 60);
        doc.text(CLASS_LABELS[key] || key, marginX, y);

        doc.setFillColor(230, 226, 222);
        doc.roundedRect(marginX, y + 6, barWidth, 10, 5, 5, "F");

        const fillColor = barColors[key] || gold;
        doc.setFillColor(...fillColor);
        doc.roundedRect(marginX, y + 6, barWidth * (pct / 100), 10, 5, 5, "F");

        doc.setTextColor(20, 20, 20);
        doc.text(formatPercentage(pct), marginX + barWidth + 12, y + 15);

        y += 30;
    });

    y += 20;

    // --- Description / recommendation ---
    doc.setFont("helvetica", "normal");
    doc.setFontSize(10);
    const descMaxWidth = pageWidth - marginX * 2 - 20;
    const descLines = doc.splitTextToSize(data.description || "Not provided.", descMaxWidth);
    const descLineHeight = 13;
    const descBlockHeight = Math.max(60, 32 + descLines.length * descLineHeight);

    doc.setDrawColor(...gold);
    doc.setLineWidth(2);
    doc.line(marginX, y, marginX, y + descBlockHeight);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(11);
    doc.setTextColor(...gold);
    doc.text(data.prediction || "Result", marginX + 14, y + 14);
    doc.setFont("helvetica", "normal");
    doc.setFontSize(10);
    doc.setTextColor(60, 60, 60);
    doc.text(descLines, marginX + 14, y + 32);

    y += descBlockHeight;

    // --- Prediction Analytics chart ---
    if (elements.predictionChart && state.chartInstance) {
        y += 30;

        // Start a new page if there isn't enough room left for the chart
        if (y + 220 > 780) {
            doc.addPage();
            y = 48;
        }

        doc.setFont("helvetica", "bold");
        doc.setFontSize(13);
        doc.setTextColor(20, 20, 20);
        doc.text("Prediction Analytics", marginX, y);
        y += 16;

        const chartImage = elements.predictionChart.toDataURL("image/png", 1.0);
        const chartWidth = pageWidth - marginX * 2;
        const chartHeight = chartWidth * 0.5;
        doc.addImage(chartImage, "PNG", marginX, y, chartWidth, chartHeight);
        y += chartHeight + 20;
    }

    // --- Footer ---
    doc.setFontSize(8);
    doc.setTextColor(...muted);
    doc.text(
        "This report is generated for informational purposes only and is not a substitute for professional medical diagnosis.",
        marginX,
        800,
        { maxWidth: pageWidth - marginX * 2 }
    );

    doc.save(`eeg-prediction-report-${Date.now()}.pdf`);
}

/* =========================================================================
   19. UTILITIES
   ========================================================================= */
function formatPercentage(value) {
    const num = Number(value);
    if (Number.isNaN(num)) return "—";
    return `${num.toFixed(1)}%`;
}

function escapeHtml(value) {
    const div = document.createElement("div");
    div.textContent = String(value ?? "");
    return div.innerHTML;
}