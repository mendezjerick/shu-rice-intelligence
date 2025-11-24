const sections = ["home", "check-price", "forecast"];
let forecastChart = null;

const demoApi = {
  async get_price(dateStr, location) {
    const today = new Date();
    const date = dateStr || today.toISOString().slice(0, 10);
    return {
      date,
      location: location || "Sample Region",
      price: parseFloat((48 + Math.random() * 6).toFixed(2)),
      status: "ok",
    };
  },
  async get_forecast(months) {
    const now = new Date();
    const historyDates = [];
    const historyValues = [];
    for (let i = 11; i >= 0; i -= 1) {
      const d = new Date(now.getFullYear(), now.getMonth() - i, 1);
      historyDates.push(formatMonth(d));
      historyValues.push(parseFloat((45 + i * 0.2).toFixed(2)));
    }

    const forecastDates = [];
    const forecastValues = [];
    const base = historyValues[historyValues.length - 1];
    for (let step = 1; step <= months; step += 1) {
      const d = new Date(now.getFullYear(), now.getMonth() + step, 1);
      forecastDates.push(formatMonth(d));
      forecastValues.push(parseFloat((base * (1 + 0.012 * step)).toFixed(2)));
    }

    const regions = ["Sample Region A", "Sample Region B", "Sample Region C"];
    const table = [];
    regions.forEach((region) => {
      forecastDates.forEach((date, idx) => {
        const forecastPrice = parseFloat((base * (1 + 0.012 * (idx + 1))).toFixed(2));
        table.push({
          region,
          current_date: historyDates[historyDates.length - 1],
          forecast_date: date,
          avg_price: base,
          forecast_price: forecastPrice,
          pct_change: parseFloat((((forecastPrice - base) / base) * 100).toFixed(2)),
          step: idx + 1,
        });
      });
    });

    const sampleRows = Array.from({ length: 6 }).map((_, idx) => ({
      admin1: regions[idx % regions.length],
      date: historyDates[idx],
      avg_price: historyValues[idx] || base,
    }));

    return {
      historical: { dates: historyDates, values: historyValues },
      forecast: { dates: forecastDates, values: forecastValues },
      table,
      train_head: sampleRows,
      holdout_head: sampleRows.slice().reverse(),
      metrics: { message: "Demo data shown because pywebview API is unavailable." },
      explanation: "This demo forecast runs only in the browser and will be replaced by your Python backend when packaged.",
    };
  },
};

const api = window.pywebview?.api || demoApi;

document.addEventListener("DOMContentLoaded", () => {
  initNavigation();
  setDefaultFormValues();

  document.getElementById("checkPriceBtn")?.addEventListener("click", handlePriceCheck);
  document.getElementById("runForecastBtn")?.addEventListener("click", handleForecast);

  showSection("home");
});

function initNavigation() {
  document.querySelectorAll("[data-target]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = btn.getAttribute("data-target");
      if (target) {
        showSection(target);
      }
    });
  });
}

function setDefaultFormValues() {
  const today = new Date();
  const dateInput = document.getElementById("priceDate");
  if (dateInput) {
    dateInput.value = today.toISOString().slice(0, 10);
  }

  const targetYear = document.getElementById("targetYear");
  if (targetYear) {
    const currentYear = today.getFullYear();
    targetYear.innerHTML = `<option value="">Auto</option>
      <option value="${currentYear}">${currentYear}</option>
      <option value="${currentYear + 1}">${currentYear + 1}</option>
      <option value="${currentYear + 2}">${currentYear + 2}</option>`;
  }
}

function showSection(id) {
  sections.forEach((name) => {
    const el = document.getElementById(name);
    if (el) {
      el.classList.toggle("active", name === id);
    }
  });
  window.scrollTo({ top: 0, behavior: "smooth" });
}

async function handlePriceCheck() {
  const date = document.getElementById("priceDate")?.value;
  const location = document.getElementById("locationSelect")?.value;
  setPriceStatus("Loading...");
  try {
    const result = await api.get_price(date, location);
    updatePriceResult(result);
  } catch (err) {
    updatePriceResult({
      date,
      location,
      price: null,
      status: "error",
      message: err?.message || "Something went wrong.",
    });
  }
}

function setPriceStatus(text) {
  const badge = document.getElementById("priceStatus");
  if (badge) {
    badge.textContent = text;
  }
}

function updatePriceResult(data) {
  const { date, location, price, status, message } = data || {};
  document.getElementById("priceDateValue").textContent = date || "--";
  document.getElementById("priceLocationValue").textContent = location || "--";
  document.getElementById("priceValue").textContent = price != null ? `${price.toFixed(2)} PHP/kg` : "--";
  setPriceStatus(status === "ok" ? "OK" : status === "no_data" ? "No data" : "Error");
  const title = document.getElementById("priceTitle");
  if (title) {
    if (status === "ok") {
      title.textContent = "Price found";
    } else if (status === "no_data") {
      title.textContent = message || "No data found for that selection.";
    } else {
      title.textContent = message || "Unable to fetch price.";
    }
  }
}

async function handleForecast() {
  const months = parseInt(document.getElementById("monthsAhead")?.value || "1", 10);
  const targetYearRaw = document.getElementById("targetYear")?.value;
  const targetMonthRaw = document.getElementById("targetMonth")?.value;
  const targetYear = targetYearRaw ? parseInt(targetYearRaw, 10) : null;
  const targetMonth = targetMonthRaw ? parseInt(targetMonthRaw, 10) : null;

  const btn = document.getElementById("runForecastBtn");
  if (btn) {
    btn.disabled = true;
    btn.textContent = "Running...";
  }

  try {
    const result = await api.get_forecast(months, targetYear, targetMonth);
    renderForecast(result);
  } catch (err) {
    renderForecast({
      historical: { dates: [], values: [] },
      forecast: { dates: [], values: [] },
      table: [],
      train_head: [],
      holdout_head: [],
      metrics: { error: err?.message || "Unable to run forecast." },
      explanation: "No forecast data available.",
    });
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.textContent = "Run Forecast";
    }
  }
}

function renderForecast(data) {
  if (!data) return;
  renderChart(data.historical, data.forecast);
  renderForecastTable(data.table || []);
  renderPreviewTable("trainTable", data.train_head || []);
  renderPreviewTable("holdoutTable", data.holdout_head || []);
  renderMetrics(data.metrics || {});
  renderExplanation(data.explanation || "");
}

function renderChart(historical, forecast) {
  const ctx = document.getElementById("forecastChart");
  if (!ctx) return;

  const labels = [...(historical?.dates || []), ...(forecast?.dates || [])];
  const histValues = [...(historical?.values || []), ...Array(forecast?.dates?.length || 0).fill(null)];
  const forecastValues = [
    ...Array(historical?.dates?.length || 0).fill(null),
    ...(forecast?.values || []),
  ];

  if (forecastChart) {
    forecastChart.destroy();
  }

  forecastChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Historical avg",
          data: histValues,
          borderColor: "#6bb7e0",
          backgroundColor: "rgba(107, 183, 224, 0.25)",
          tension: 0.22,
          spanGaps: true,
        },
        {
          label: "Forecast avg",
          data: forecastValues,
          borderColor: "#f2c649",
          backgroundColor: "rgba(242, 198, 73, 0.25)",
          borderDash: [6, 4],
          tension: 0.22,
          spanGaps: true,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "top",
          labels: { boxWidth: 14 },
        },
      },
      scales: {
        y: {
          title: { display: true, text: "PHP / kg" },
          grid: { color: "rgba(0,0,0,0.06)" },
        },
        x: {
          grid: { display: false },
        },
      },
    },
  });
}

function renderForecastTable(rows) {
  const tbody = document.querySelector("#forecastTable tbody");
  if (!tbody) return;
  tbody.innerHTML = "";
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="7">No forecast rows returned.</td></tr>';
    return;
  }

  rows.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.region ?? "--"}</td>
      <td>${row.current_date ?? "--"}</td>
      <td>${row.forecast_date ?? "--"}</td>
      <td>${formatNumber(row.avg_price)}</td>
      <td>${formatNumber(row.forecast_price)}</td>
      <td>${row.pct_change != null ? row.pct_change.toFixed(2) + "%" : "--"}</td>
      <td>${row.step ?? "--"}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderPreviewTable(tableId, rows) {
  const table = document.getElementById(tableId);
  if (!table) return;
  const thead = table.querySelector("thead");
  const tbody = table.querySelector("tbody");
  thead.innerHTML = "";
  tbody.innerHTML = "";

  if (!rows.length) {
    tbody.innerHTML = "<tr><td>No preview rows</td></tr>";
    return;
  }

  const headers = Object.keys(rows[0]);
  const headerRow = document.createElement("tr");
  headers.forEach((key) => {
    const th = document.createElement("th");
    th.textContent = key;
    headerRow.appendChild(th);
  });
  thead.appendChild(headerRow);

  rows.forEach((row) => {
    const tr = document.createElement("tr");
    headers.forEach((key) => {
      const td = document.createElement("td");
      td.textContent = row[key];
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
}

function renderMetrics(metrics) {
  const pre = document.getElementById("metricsBlock");
  if (!pre) return;
  pre.textContent = JSON.stringify(metrics, null, 2);
}

function renderExplanation(text) {
  const el = document.getElementById("explanationText");
  if (el) {
    el.textContent = text;
  }
}

function formatMonth(date) {
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}`;
}

function formatNumber(value) {
  return value != null ? Number(value).toFixed(2) : "--";
}
