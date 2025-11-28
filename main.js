// Attach event listeners and handle API interactions for the dashboard

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("transaction-form");
  const resultCard = document.getElementById("result-card");
  const resultLevel = document.getElementById("result-level");
  const resultAction = document.getElementById("result-action");
  const fraudProgress = document.getElementById("fraud-progress");
  const recentTableBody = document.querySelector("#recent-table tbody");

  // Function to refresh recent decisions
  async function loadRecent() {
    try {
      const response = await fetch("/recent");
      if (!response.ok) throw new Error("Failed to fetch recent decisions");
      const data = await response.json();
      recentTableBody.innerHTML = "";
      data.forEach((row) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>${new Date(row.timestamp).toLocaleString()}</td>
          <td>${row.amount.toFixed(2)}</td>
          <td>${row.risk_level}</td>
          <td>${row.action}</td>
          <td>${(row.probability * 100).toFixed(2)}%</td>
        `;
        recentTableBody.appendChild(tr);
      });
    } catch (err) {
      console.error(err);
    }
  }

  // Initial load of recent decisions
  loadRecent();

  // Handle form submission
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    // Build payload
    const formData = new FormData(form);
    const payload = {};
    formData.forEach((value, key) => {
      payload[key] = value;
    });

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok) {
        alert(data.error || "Prediction failed");
        return;
      }

      // Update result card
      const probPercent = (data.fraud_probability * 100).toFixed(2);
      resultLevel.textContent = `Risk Level: ${data.risk_level}`;
      resultAction.textContent = `Recommended Action: ${data.action}`;
      // Update progress bar
      fraudProgress.style.width = `${probPercent}%`;
      fraudProgress.textContent = `${probPercent}%`;
      // Update card background based on risk level
      resultCard.classList.remove("bg-success", "bg-warning", "bg-danger");
      if (data.risk_level === "Safe") {
        resultCard.classList.add("bg-success");
      } else if (data.risk_level === "Suspicious") {
        resultCard.classList.add("bg-warning");
      } else if (data.risk_level === "Critical") {
        resultCard.classList.add("bg-danger");
      }
      // Refresh recent decisions
      loadRecent();
    } catch (err) {
      console.error(err);
      alert("An error occurred while processing the prediction.");
    }
  });
});