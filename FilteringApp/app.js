/**
 * Data Filtering Platform – Frontend Logic
 *
 * Handles:
 * 1. File upload with drag-and-drop + validation via API
 * 2. Shopping cart dataset selection with date filter toggles
 * 3. Form submission → ZIP download
 */

(function () {
    "use strict";

    // ── State ────────────────────────────────────────────────────────
    const state = {
        file: null,
        fileBytes: null,
        validation: null,
        selectedDatasets: new Set(),
        dateFilters: {},     // { datasetId: { start: int, end: int } }
    };

    // ── DOM refs ─────────────────────────────────────────────────────
    const $uploadZone = document.getElementById("uploadZone");
    const $fileInput = document.getElementById("partsFileInput");
    const $uploadContent = document.getElementById("uploadContent");
    const $uploadResult = document.getElementById("uploadResult");
    const $uploadFileName = document.getElementById("uploadFileName");
    const $uploadMeta = document.getElementById("uploadMeta");
    const $uploadClear = document.getElementById("uploadClear");
    const $validationErrors = document.getElementById("validationErrors");
    const $cartEmpty = document.getElementById("cartEmpty");
    const $cartContent = document.getElementById("cartContent");
    const $cartFileInfo = document.getElementById("cartFileInfo");
    const $cartCount = document.getElementById("cartCount");
    const $cartList = document.getElementById("cartList");
    const $cartDateSection = document.getElementById("cartDateSection");
    const $cartDateList = document.getElementById("cartDateList");
    const $submitBtn = document.getElementById("submitBtn");
    const $submitProgress = document.getElementById("submitProgress");

    // ══════════════════════════════════════════════════════════════════
    // FILE UPLOAD
    // ══════════════════════════════════════════════════════════════════

    $uploadZone.addEventListener("click", () => $fileInput.click());
    $uploadZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        $uploadZone.classList.add("upload-zone--drag");
    });
    $uploadZone.addEventListener("dragleave", () => {
        $uploadZone.classList.remove("upload-zone--drag");
    });
    $uploadZone.addEventListener("drop", (e) => {
        e.preventDefault();
        $uploadZone.classList.remove("upload-zone--drag");
        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });
    $fileInput.addEventListener("change", () => {
        if ($fileInput.files.length) handleFile($fileInput.files[0]);
    });
    $uploadClear.addEventListener("click", (e) => {
        e.stopPropagation();
        clearUpload();
    });

    async function handleFile(file) {
        const ext = file.name.split(".").pop().toLowerCase();
        if (!["csv", "xlsx", "xls"].includes(ext)) {
            showValidationError("Invalid file type. Please upload a .csv or .xlsx file.");
            return;
        }

        state.file = file;
        state.fileBytes = await file.arrayBuffer();

        // Show file info
        $uploadContent.hidden = true;
        $uploadResult.hidden = false;
        $uploadFileName.textContent = file.name;
        $uploadMeta.textContent = `${(file.size / 1024).toFixed(1)} KB`;
        $validationErrors.hidden = true;

        // Validate via API
        try {
            const formData = new FormData();
            formData.append("parts_file", file);
            const res = await fetch("/api/v1/validate", { method: "POST", body: formData });
            const data = await res.json();

            if (!res.ok) {
                const msg = data.detail?.message || data.detail || "Validation failed";
                const detail = data.detail?.detail || "";
                showValidationError(`${msg}${detail ? ": " + detail : ""}`);
                state.validation = null;
                updateSubmitState();
                return;
            }

            state.validation = data;

            if (!data.is_valid) {
                showValidationError(data.errors.join("\n"));
                updateSubmitState();
                return;
            }

            // Update UI
            let meta = `${data.row_count} rows · ${data.unique_parts} unique parts`;
            if (data.detected_mode === "part_orp") {
                meta += ` · ${data.unique_orps} unique ORP codes`;
            }
            meta += ` · Mode: ${data.detected_mode}`;
            $uploadMeta.textContent = meta;
            $cartFileInfo.textContent = `${file.name} (${data.unique_parts} parts)`;

            if (data.warnings.length) {
                $uploadMeta.textContent += " ⚠";
                $uploadMeta.title = data.warnings.join("\n");
            }

            // Disable incompatible datasets
            updateDatasetCompatibility(data.detected_mode);
            updateSubmitState();
        } catch (err) {
            showValidationError("Failed to connect to server for validation.");
        }
    }

    function clearUpload() {
        state.file = null;
        state.fileBytes = null;
        state.validation = null;
        $fileInput.value = "";
        $uploadContent.hidden = false;
        $uploadResult.hidden = true;
        $validationErrors.hidden = true;
        $cartFileInfo.textContent = "—";
        updateSubmitState();
    }

    function showValidationError(msg) {
        $validationErrors.hidden = false;
        $validationErrors.textContent = msg;
    }

    function updateDatasetCompatibility(detectedMode) {
        document.querySelectorAll(".dataset-item").forEach((item) => {
            const keyMode = item.dataset.keyMode;
            const checkbox = item.querySelector(".ds-checkbox");
            let incompatible = false;

            if (keyMode === "part_orp" && detectedMode === "only_part_number") {
                incompatible = true;
            }

            if (incompatible) {
                item.style.opacity = "0.5";
                checkbox.disabled = true;
                if (checkbox.checked) {
                    checkbox.checked = false;
                    state.selectedDatasets.delete(checkbox.value);
                    item.classList.remove("dataset-item--selected");
                }
            } else {
                item.style.opacity = "1";
                checkbox.disabled = false;
            }
        });
        updateCart();
    }

    // ══════════════════════════════════════════════════════════════════
    // SHOPPING CART
    // ══════════════════════════════════════════════════════════════════

    document.querySelectorAll(".ds-checkbox").forEach((cb) => {
        cb.addEventListener("change", function () {
            const item = this.closest(".dataset-item");
            const dsId = this.value;

            if (this.checked) {
                state.selectedDatasets.add(dsId);
                item.classList.add("dataset-item--selected");
                // Show date panel if available
                const datePanel = item.querySelector(".date-filter-panel");
                if (datePanel) datePanel.hidden = false;
            } else {
                state.selectedDatasets.delete(dsId);
                item.classList.remove("dataset-item--selected");
                // Hide and reset date panel
                const datePanel = item.querySelector(".date-filter-panel");
                if (datePanel) {
                    datePanel.hidden = true;
                    const toggle = datePanel.querySelector(".date-toggle");
                    if (toggle) toggle.checked = false;
                    const fields = datePanel.querySelector(".date-filter-panel__fields");
                    if (fields) fields.hidden = true;
                }
                delete state.dateFilters[dsId];
            }
            updateCart();
            updateSubmitState();
        });
    });

    // Date filter toggles
    document.querySelectorAll(".date-toggle").forEach((toggle) => {
        toggle.addEventListener("change", function () {
            const dsId = this.dataset.ds;
            const fields = this.closest(".date-filter-panel").querySelector(".date-filter-panel__fields");
            fields.hidden = !this.checked;
            if (!this.checked) {
                delete state.dateFilters[dsId];
            }
            updateCart();
        });
    });

    // Date input changes
    document.querySelectorAll(".date-start, .date-end").forEach((input) => {
        input.addEventListener("change", function () {
            const dsId = this.dataset.ds;
            const panel = this.closest(".date-filter-panel__fields");
            const start = parseInt(panel.querySelector(".date-start").value) || null;
            const end = parseInt(panel.querySelector(".date-end").value) || null;
            if (start && end) {
                state.dateFilters[dsId] = { start, end };
            } else {
                delete state.dateFilters[dsId];
            }
            updateCart();
        });
    });

    function updateCart() {
        const count = state.selectedDatasets.size;
        $cartCount.textContent = count;

        if (count === 0) {
            $cartEmpty.hidden = false;
            $cartContent.hidden = true;
            return;
        }

        $cartEmpty.hidden = true;
        $cartContent.hidden = false;

        // Build list
        $cartList.innerHTML = "";
        state.selectedDatasets.forEach((dsId) => {
            const item = document.querySelector(`.dataset-item[data-id="${dsId}"]`);
            const name = item ? item.querySelector(".dataset-item__name").textContent : dsId;
            const li = document.createElement("li");
            li.innerHTML = `<span>${name}</span><button class="cart-list__remove" data-ds="${dsId}" title="Remove">&times;</button>`;
            $cartList.appendChild(li);
        });

        // Remove buttons in cart
        $cartList.querySelectorAll(".cart-list__remove").forEach((btn) => {
            btn.addEventListener("click", function () {
                const dsId = this.dataset.ds;
                const cb = document.querySelector(`.ds-checkbox[value="${dsId}"]`);
                if (cb) {
                    cb.checked = false;
                    cb.dispatchEvent(new Event("change"));
                }
            });
        });

        // Date filters summary
        const dateKeys = Object.keys(state.dateFilters);
        if (dateKeys.length > 0) {
            $cartDateSection.hidden = false;
            $cartDateList.innerHTML = "";
            dateKeys.forEach((dsId) => {
                const df = state.dateFilters[dsId];
                const li = document.createElement("li");
                li.textContent = `${dsId}: ${df.start} → ${df.end}`;
                $cartDateList.appendChild(li);
            });
        } else {
            $cartDateSection.hidden = true;
        }
    }

    function updateSubmitState() {
        const hasFile = state.validation && state.validation.is_valid;
        const hasSelection = state.selectedDatasets.size > 0;
        $submitBtn.disabled = !(hasFile && hasSelection);
    }

    // ══════════════════════════════════════════════════════════════════
    // SUBMIT
    // ══════════════════════════════════════════════════════════════════

    $submitBtn.addEventListener("click", async () => {
        if ($submitBtn.disabled) return;

        $submitBtn.disabled = true;
        $submitProgress.hidden = false;

        try {
            const formData = new FormData();
            formData.append("parts_file", state.file);
            formData.append("selected_files", JSON.stringify([...state.selectedDatasets]));
            formData.append("date_filters", JSON.stringify(state.dateFilters));

            const res = await fetch("/api/v1/filter", { method: "POST", body: formData });

            if (!res.ok) {
                const errData = await res.json();
                const msg = errData.detail?.message || errData.detail || "Filtering failed.";
                alert("Error: " + msg);
                return;
            }

            // Download ZIP
            const blob = await res.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = res.headers.get("content-disposition")
                ? res.headers.get("content-disposition").split("filename=")[1]?.replace(/"/g, "")
                : "data_delivery.zip";
            document.body.appendChild(a);
            a.click();
            a.remove();
            URL.revokeObjectURL(url);
        } catch (err) {
            alert("Network error: " + err.message);
        } finally {
            $submitBtn.disabled = false;
            $submitProgress.hidden = true;
            updateSubmitState();
        }
    });
})();
