const form = document.getElementById("diagnose-form");
const imageInput = document.getElementById("image");
const preview = document.getElementById("preview");
const submitBtn = document.getElementById("submit-btn");
const statusLine = document.getElementById("status-line");
const result = document.getElementById("result");

imageInput.addEventListener("change", () => {
  const file = imageInput.files && imageInput.files[0];
  if (!file) {
    preview.classList.add("hidden");
    preview.removeAttribute("src");
    return;
  }
  preview.src = URL.createObjectURL(file);
  preview.classList.remove("hidden");
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = imageInput.files && imageInput.files[0];
  if (!file) {
    setStatus("Choose a leaf photo first.");
    return;
  }

  const data = new FormData();
  data.append("image", file);
  data.append("language", document.getElementById("language").value);

  submitBtn.disabled = true;
  result.classList.add("hidden");
  result.innerHTML = "";
  setStatus("Looking at the leaf… this can take a little while.");

  try {
    const response = await fetch("/api/diagnose", {
      method: "POST",
      body: data,
    });
    const payload = await response.json();
    renderResult(payload);
    setStatus("");
  } catch (error) {
    setStatus("Could not reach CropSense. Check that the server is running.");
  } finally {
    submitBtn.disabled = false;
  }
});

function setStatus(message) {
  if (!message) {
    statusLine.hidden = true;
    statusLine.textContent = "";
    return;
  }
  statusLine.hidden = false;
  statusLine.textContent = message;
}

function renderResult(payload) {
  const top = (payload.predictions && payload.predictions[0]) || null;
  const confidence = top ? Math.round(top.confidence * 100) : 0;
  const badgeClass =
    payload.status === "ok" ? "" : payload.status === "low_confidence" ? "warn" : "error";
  const badgeLabel =
    payload.status === "ok"
      ? "Likely diagnosis"
      : payload.status === "low_confidence"
        ? "Need a clearer photo"
        : "Something went wrong";

  const alternatives = (payload.predictions || []).slice(1);
  const altHtml = alternatives
    .map(
      (item) =>
        `<li>${escapeHtml(item.label)} · ${Math.round(item.confidence * 100)}%</li>`
    )
    .join("");

  result.innerHTML = `
    <span class="badge ${badgeClass}">${badgeLabel}</span>
    <h2>${top && payload.status === "ok" ? escapeHtml(top.label) : "No confirmed diagnosis"}</h2>
    ${
      top
        ? `<p>Top guess confidence: <strong>${confidence}%</strong></p>
           <div class="bar" aria-hidden="true"><span style="width:${confidence}%"></span></div>`
        : ""
    }
    ${
      altHtml
        ? `<p><strong>Other possibilities</strong></p><ul class="alt">${altHtml}</ul>`
        : ""
    }
    <p class="advice">${escapeHtml(payload.advice || "")}</p>
  `;
  result.classList.remove("hidden");
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}
