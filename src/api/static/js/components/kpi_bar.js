/**
 * ExamGuard Vision — KPI Bar Component
 * Compact horizontal cards displaying session event and review counts.
 */

import { appState } from "../state.js";

export class KPIBarComponent {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.init();
  }

  init() {
    this.render();
    appState.subscribe((type) => {
      if (type === "EVENTS_UPDATED" || type === "EVENTS_RESET" || type === "EVENT_STATUS_CHANGED") {
        this.updateKPIs();
      }
    });
  }

  render() {
    this.container.innerHTML = `
      <div class="kpi-row" title="Mức cảnh báo dùng để ưu tiên giám thị xem xét, không phải kết luận gian lận.">
        <div class="kpi-chip kpi-total">
          <div class="kpi-label">TỔNG SỰ KIỆN</div>
          <div class="kpi-val" id="kpi-total">0</div>
        </div>
        <div class="kpi-chip kpi-awaiting">
          <div class="kpi-label">CHỜ DUYỆT</div>
          <div class="kpi-val" id="kpi-awaiting">0</div>
        </div>
        <div class="kpi-chip kpi-confirmed">
          <div class="kpi-label">ĐÃ XÁC NHẬN</div>
          <div class="kpi-val" id="kpi-confirmed">0</div>
        </div>
        <div class="kpi-chip kpi-dismissed">
          <div class="kpi-label">ĐÃ BỎ QUA</div>
          <div class="kpi-val" id="kpi-dismissed">0</div>
        </div>
      </div>
    `;
    this.updateKPIs();
  }

  updateKPIs() {
    const kpis = appState.getKPIs();
    const elTotal = document.getElementById("kpi-total");
    const elAwaiting = document.getElementById("kpi-awaiting");
    const elConfirmed = document.getElementById("kpi-confirmed");
    const elDismissed = document.getElementById("kpi-dismissed");

    if (elTotal) elTotal.innerText = kpis.total;
    if (elAwaiting) elAwaiting.innerText = kpis.awaiting;
    if (elConfirmed) elConfirmed.innerText = kpis.confirmed;
    if (elDismissed) elDismissed.innerText = kpis.dismissed;
  }
}
