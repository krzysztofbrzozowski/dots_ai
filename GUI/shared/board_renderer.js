"use strict";

export const PLAYER_1 = 1;
export const PLAYER_2 = -1;


function cssColor(variableName) {
  return getComputedStyle(document.documentElement)
    .getPropertyValue(variableName)
    .trim();
}


function formatCompactNumber(value) {
  if (Math.abs(value) >= 1000) return `${(value / 1000).toFixed(1)}k`;
  return String(Math.round(value));
}


function formatOverlayValue(overlay, rawQ, visits, totalVisits) {
  if (!visits) return "—";
  if (overlay === "value") {
    const meanValue = rawQ / visits;
    return `${meanValue >= 0 ? "+" : ""}${meanValue.toFixed(2)}`;
  }
  if (overlay === "raw-q") {
    return `${rawQ >= 0 ? "+" : ""}${formatCompactNumber(rawQ)}`;
  }
  if (overlay === "visits") return formatCompactNumber(visits);
  if (overlay === "policy") {
    const policy = totalVisits ? visits / totalVisits : 0;
    return `${Math.round(policy * 100)}%`;
  }
  return "";
}


export class DotsBoardRenderer {
  /* Render one analysis frame without knowing how that frame was loaded. */

  constructor(canvas, onCellSelected) {
    this.canvas = canvas;
    this.context = canvas.getContext("2d");
    this.onCellSelected = onCellSelected;
    this.frame = null;
    this.overlay = "value";
    this.selectedCell = null;
    this.layout = null;

    this.handleCanvasClick = this.handleCanvasClick.bind(this);
    this.resizeAndDraw = this.resizeAndDraw.bind(this);
    this.canvas.addEventListener("click", this.handleCanvasClick);

    this.resizeObserver = new ResizeObserver(this.resizeAndDraw);
    this.resizeObserver.observe(this.canvas.parentElement);
  }

  setFrame(frame) {
    this.frame = frame;
    this.canvas.classList.toggle("is-interactive", Boolean(frame));
    this.resizeAndDraw();
  }

  setOverlay(overlay) {
    this.overlay = overlay;
    this.draw();
  }

  setSelectedCell(cell) {
    this.selectedCell = cell ? [...cell] : null;
    this.draw();
  }

  resizeAndDraw() {
    const bounds = this.canvas.getBoundingClientRect();
    const width = Math.max(1, bounds.width);
    const height = Math.max(1, bounds.height);
    const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);

    this.canvas.width = Math.round(width * pixelRatio);
    this.canvas.height = Math.round(height * pixelRatio);
    this.context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);

    if (!this.frame) {
      this.layout = null;
      this.context.clearRect(0, 0, width, height);
      return;
    }

    const labelSpace = 38;
    const rightSpace = 20;
    // The selected-action and selected-cell rings extend beyond the heatmap
    // circle. Keep enough space below the final row so neither ring is clipped.
    const bottomSpace = 44;
    const availableWidth = Math.max(1, width - labelSpace - rightSpace);
    const availableHeight = Math.max(1, height - labelSpace - bottomSpace);
    const step = Math.min(
      availableWidth / Math.max(this.frame.cols - 1, 1),
      availableHeight / Math.max(this.frame.rows - 1, 1),
    );
    const gridWidth = step * Math.max(this.frame.cols - 1, 0);
    const gridHeight = step * Math.max(this.frame.rows - 1, 0);
    const originX = labelSpace + (availableWidth - gridWidth) / 2;
    const originY = labelSpace + (availableHeight - gridHeight) / 2;

    this.layout = { width, height, step, originX, originY };
    this.draw();
  }

  draw() {
    if (!this.frame || !this.layout) return;

    const { width, height } = this.layout;
    this.context.clearRect(0, 0, width, height);
    this.context.fillStyle = cssColor("--board-surface");
    this.context.fillRect(0, 0, width, height);

    const showAnalysisOverlay = this.overlay !== "none";
    if (showAnalysisOverlay) {
      this.drawTerritory();
      this.drawSearchOverlay();
    }

    // The lattice and placed dots form the base position. The "None" mode
    // deliberately stops here, leaving all search-specific marks hidden.
    this.drawGrid();
    this.drawDots();

    if (showAnalysisOverlay) {
      this.drawSelectionMarkers();
      this.drawSearchLabels();
    }
  }

  drawTerritory() {
    const { step, originX, originY } = this.layout;
    const territorySize = Math.max(11, step * 0.76);

    for (let row = 0; row < this.frame.rows; row += 1) {
      for (let col = 0; col < this.frame.cols; col += 1) {
        const owner = this.frame.territory[row][col];
        if (owner === 0) continue;

        const x = originX + col * step;
        const y = originY + row * step;
        this.context.fillStyle = cssColor(
          owner === PLAYER_1 ? "--territory-player-one" : "--territory-player-two",
        );
        this.context.beginPath();
        this.context.roundRect(
          x - territorySize / 2,
          y - territorySize / 2,
          territorySize,
          territorySize,
          Math.max(3, step * 0.12),
        );
        this.context.fill();
      }
    }
  }

  drawSearchOverlay() {
    const { step, originX, originY } = this.layout;
    const totalVisits = this.frame.visit_counts
      .flat()
      .reduce((sum, value) => sum + value, 0);
    const values = [];

    for (let row = 0; row < this.frame.rows; row += 1) {
      for (let col = 0; col < this.frame.cols; col += 1) {
        const visits = this.frame.visit_counts[row][col];
        if (!this.frame.legal_mask[row][col] || !visits) continue;
        const rawQ = this.frame.q_values[row][col];
        if (this.overlay === "value") values.push(Math.abs(rawQ / visits));
        if (this.overlay === "raw-q") values.push(Math.abs(rawQ));
        if (this.overlay === "visits") values.push(visits);
        if (this.overlay === "policy") {
          values.push(totalVisits ? visits / totalVisits : 0);
        }
      }
    }
    const maximumMagnitude = Math.max(...values, 1e-9);
    const markerRadius = Math.max(7, Math.min(step * 0.36, 22));

    for (let row = 0; row < this.frame.rows; row += 1) {
      for (let col = 0; col < this.frame.cols; col += 1) {
        if (!this.frame.legal_mask[row][col]) continue;

        const x = originX + col * step;
        const y = originY + row * step;
        const visits = this.frame.visit_counts[row][col];
        const rawQ = this.frame.q_values[row][col];

        if (!visits) {
          // A stored q value of zero is ambiguous. An empty ring makes it clear
          // that this legal action has not received a completed rollout.
          this.context.strokeStyle = cssColor("--unvisited-action");
          this.context.lineWidth = 1;
          this.context.beginPath();
          this.context.arc(x, y, Math.max(3, markerRadius * 0.28), 0, Math.PI * 2);
          this.context.stroke();
          continue;
        }

        let overlayValue = visits;
        if (this.overlay === "value") overlayValue = rawQ / visits;
        if (this.overlay === "raw-q") overlayValue = rawQ;
        if (this.overlay === "policy") {
          overlayValue = totalVisits ? visits / totalVisits : 0;
        }

        const intensity = Math.min(1, Math.abs(overlayValue) / maximumMagnitude);
        const isNegativeValue =
          (this.overlay === "value" || this.overlay === "raw-q") &&
          overlayValue < 0;
        this.context.save();
        this.context.globalAlpha = 0.14 + intensity * 0.42;
        this.context.fillStyle = cssColor(
          isNegativeValue ? "--negative-value" : "--positive-value",
        );
        this.context.beginPath();
        this.context.arc(x, y, markerRadius, 0, Math.PI * 2);
        this.context.fill();
        this.context.restore();

      }
    }
  }

  drawSearchLabels() {
    const { step, originX, originY } = this.layout;
    if (step < 36) return;

    const totalVisits = this.frame.visit_counts
      .flat()
      .reduce((sum, value) => sum + value, 0);
    const fontSize = Math.max(11, Math.min(14, step * 0.19));

    // Search labels are drawn after the grid and dots. This keeps the lattice
    // from crossing through the numbers and makes the value the top visual layer.
    this.context.save();
    this.context.font =
      `800 ${fontSize}px ui-monospace, SFMono-Regular, Menlo, monospace`;
    this.context.textAlign = "center";
    this.context.textBaseline = "middle";
    this.context.lineJoin = "round";
    this.context.strokeStyle = cssColor("--overlay-label-outline");
    // A two-pixel stroke leaves an even one-pixel outline around the glyph.
    // Using a half-pixel width here made the upper edge look heavier after
    // canvas anti-aliasing.
    this.context.lineWidth = 2;
    this.context.fillStyle = cssColor("--overlay-label");

    for (let row = 0; row < this.frame.rows; row += 1) {
      for (let col = 0; col < this.frame.cols; col += 1) {
        if (!this.frame.legal_mask[row][col]) continue;

        const visits = this.frame.visit_counts[row][col];
        if (!visits) continue;

        const label = formatOverlayValue(
          this.overlay,
          this.frame.q_values[row][col],
          visits,
          totalVisits,
        );
        const x = originX + col * step;
        const y = originY + row * step;

        // A narrow dark outline preserves contrast over both positive and
        // negative heatmap colors while keeping the white fill easy to scan.
        this.context.strokeText(label, x, y);
        this.context.fillText(label, x, y);
      }
    }
    this.context.restore();
  }

  drawGrid() {
    const { step, originX, originY } = this.layout;
    const { rows, cols } = this.frame;

    this.context.strokeStyle = cssColor("--grid");
    this.context.lineWidth = 1;
    this.context.beginPath();
    for (let col = 0; col < cols; col += 1) {
      const x = Math.round(originX + col * step) + 0.5;
      this.context.moveTo(x, originY);
      this.context.lineTo(x, originY + (rows - 1) * step);
    }
    for (let row = 0; row < rows; row += 1) {
      const y = Math.round(originY + row * step) + 0.5;
      this.context.moveTo(originX, y);
      this.context.lineTo(originX + (cols - 1) * step, y);
    }
    this.context.stroke();

    this.context.fillStyle = cssColor("--text-tertiary");
    this.context.font =
      "10px ui-monospace, SFMono-Regular, Menlo, monospace";
    this.context.textAlign = "center";
    this.context.textBaseline = "middle";
    const columnStride = step < 20 ? 2 : 1;
    for (let col = 0; col < cols; col += 1) {
      if (col % columnStride === 0) {
        this.context.fillText(String(col), originX + col * step, originY - 20);
      }
    }
    this.context.textAlign = "right";
    for (let row = 0; row < rows; row += 1) {
      this.context.fillText(String(row), originX - 16, originY + row * step);
    }
  }

  drawDots() {
    const { step, originX, originY } = this.layout;
    const dotRadius = Math.max(5, Math.min(10, step * 0.24));

    for (let row = 0; row < this.frame.rows; row += 1) {
      for (let col = 0; col < this.frame.cols; col += 1) {
        const player = this.frame.board[row][col];
        if (player === 0) continue;

        const x = originX + col * step;
        const y = originY + row * step;
        this.context.save();
        if (this.frame.territory[row][col] !== 0) {
          this.context.globalAlpha = 0.42;
        }
        this.context.fillStyle = cssColor(
          player === PLAYER_1 ? "--player-one" : "--player-two",
        );
        this.context.beginPath();
        this.context.arc(x, y, dotRadius, 0, Math.PI * 2);
        this.context.fill();
        this.context.strokeStyle = cssColor("--board-surface");
        this.context.lineWidth = 2;
        this.context.stroke();
        this.context.restore();
      }
    }
  }

  drawSelectionMarkers() {
    const { step, originX, originY } = this.layout;
    const markerRadius = Math.max(9, Math.min(19, step * 0.38));
    const [selectedRow, selectedCol] = this.frame.selected_action;
    const actionX = originX + selectedCol * step;
    const actionY = originY + selectedRow * step;

    this.context.strokeStyle = cssColor("--selected-action");
    this.context.lineWidth = 3;
    this.context.beginPath();
    this.context.arc(actionX, actionY, markerRadius + 3, 0, Math.PI * 2);
    this.context.stroke();

    if (!this.selectedCell) return;
    const [row, col] = this.selectedCell;
    const x = originX + col * step;
    const y = originY + row * step;
    this.context.save();
    this.context.setLineDash([4, 3]);
    this.context.strokeStyle = cssColor("--text-primary");
    this.context.lineWidth = 1.5;
    this.context.beginPath();
    this.context.arc(x, y, markerRadius + 8, 0, Math.PI * 2);
    this.context.stroke();
    this.context.restore();
  }

  handleCanvasClick(event) {
    if (!this.frame || !this.layout) return;

    const bounds = this.canvas.getBoundingClientRect();
    const pointerX = event.clientX - bounds.left;
    const pointerY = event.clientY - bounds.top;
    const { step, originX, originY } = this.layout;
    const col = Math.round((pointerX - originX) / step);
    const row = Math.round((pointerY - originY) / step);

    if (
      row < 0 ||
      row >= this.frame.rows ||
      col < 0 ||
      col >= this.frame.cols
    ) {
      return;
    }

    const cellX = originX + col * step;
    const cellY = originY + row * step;
    const pointerDistance = Math.hypot(pointerX - cellX, pointerY - cellY);
    if (pointerDistance > Math.max(13, step * 0.46)) return;

    this.onCellSelected([row, col]);
  }

  destroy() {
    this.resizeObserver.disconnect();
    this.canvas.removeEventListener("click", this.handleCanvasClick);
  }
}
