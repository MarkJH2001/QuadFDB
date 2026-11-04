"use strict";

(() => {
  const metrics = ["detection_percent", "joint_isolation_percent", "FA_per_min", "delay_ms"];
  const severities = ["Mild", "Moderate", "Severe"];
  const masks = ["raw", "residual", "rf", "all"];
  const near = (a, b) => Math.abs(a - b) < 1e-9;
  const format = (value, precision) => Number(value).toFixed(precision);

  function numericCell(value, precision, withSD, best) {
    const cell = document.createElement("td");
    const content = best ? document.createElement("strong") : document.createElement("span");
    content.append(document.createTextNode(format(value.mean, precision)));
    if (withSD) {
      const sd = document.createElement("span");
      sd.className = "sd";
      sd.textContent = ` ± ${format(value.sample_sd, precision)}`;
      content.append(sd);
    }
    cell.append(content);
    return cell;
  }

  function renderResults(data, severity) {
    const rows = data.event_results.filter(row => row.severity === severity);
    const best = Object.fromEntries(metrics.map(key => [key,
      (key === "FA_per_min" || key === "delay_ms" ? Math.min : Math.max)(
        ...rows.map(row => row.metrics[key].mean))]));
    const body = document.getElementById("result-rows");
    const fragment = document.createDocumentFragment();
    rows.forEach(row => {
      const tr = document.createElement("tr");
      if (row.method === "RAGB") tr.className = "ragb";
      const name = document.createElement("th");
      name.scope = "row";
      name.textContent = row.method;
      tr.append(name);
      metrics.forEach(key => tr.append(numericCell(row.metrics[key], key === "FA_per_min" ? 2 : 1,
        key === "joint_isolation_percent", near(row.metrics[key].mean, best[key]))));
      fragment.append(tr);
    });
    body.replaceChildren(fragment);
    document.getElementById("cohort-note").textContent = `${data.fault_flights[severity]} fault flights · ${data.healthy_controls} healthy controls`;
    document.getElementById("result-caption").textContent = `${severity} fault diagnosis on real flights`;
    const panel = document.getElementById("result-panel");
    const activeID = `tab-${severity.toLowerCase()}`;
    panel.setAttribute("aria-labelledby", activeID);
    document.querySelectorAll("[data-severity]").forEach(button => {
      const active = button.dataset.severity === severity;
      button.setAttribute("aria-selected", String(active));
      button.tabIndex = active ? 0 : -1;
    });
  }

  function renderAblations(data) {
    const rows = data.controlled_results;
    const lookup = (head, mask, severity) => rows.find(row =>
      row.head === head && row.mask === mask && row.severity === severity).joint;
    const body = document.getElementById("ablation-rows");
    const fragment = document.createDocumentFragment();
    masks.forEach(mask => {
      const tr = document.createElement("tr");
      [mask === "residual" || mask === "all", mask === "rf" || mask === "all"].forEach((present, index) => {
        const cell = document.createElement(index === 0 ? "th" : "td");
        if (index === 0) cell.scope = "row";
        cell.textContent = present ? "✓" : "–";
        cell.setAttribute("aria-label", `${index === 0 ? "Residuals" : "Posteriors"} ${present ? "included" : "excluded"}`);
        tr.append(cell);
      });
      ["hgb", "mlp"].forEach(head => severities.forEach(severity => {
        const value = lookup(head, mask, severity);
        const best = Math.max(...masks.map(candidate => lookup(head, candidate, severity).mean));
        tr.append(numericCell(value, 1, true, near(value.mean, best)));
      }));
      fragment.append(tr);
    });
    body.replaceChildren(fragment);
  }

  const video = document.querySelector(".teaser video");
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches && video) {
    video.autoplay = false;
    video.pause();
  }

  fetch("assets/benchmark-results.json")
    .then(response => {
      if (!response.ok) throw new Error("Benchmark summary could not be loaded.");
      return response.json();
    })
    .then(data => {
      renderResults(data, "Mild");
      renderAblations(data);
      const buttons = [...document.querySelectorAll("[data-severity]")];
      buttons.forEach((button, index) => {
        button.addEventListener("click", () => renderResults(data, button.dataset.severity));
        button.addEventListener("keydown", event => {
          let next;
          if (event.key === "ArrowRight") next = (index + 1) % buttons.length;
          else if (event.key === "ArrowLeft") next = (index + buttons.length - 1) % buttons.length;
          else if (event.key === "Home") next = 0;
          else if (event.key === "End") next = buttons.length - 1;
          else return;
          event.preventDefault();
          renderResults(data, buttons[next].dataset.severity);
          buttons[next].focus();
        });
      });
    })
    .catch(error => {
      console.error(error.message);
      document.querySelectorAll("[data-severity]").forEach(button => { button.disabled = true; });
    });
})();
