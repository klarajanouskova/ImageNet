(function () {
  var t =
    localStorage.getItem('theme') ||
    (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  document.documentElement.setAttribute('data-theme', t);
})();

document.addEventListener('DOMContentLoaded', function () {
  drawDistChart();

  document.getElementById('theme-toggle').addEventListener('click', function () {
    var current = document.documentElement.getAttribute('data-theme');
    var next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
    drawDistChart();
  });

  document.querySelectorAll('.attribute-card').forEach(function (card) {
    card.addEventListener('click', function () { openAttrModal(card); });
  });

  document.getElementById('attrModal').addEventListener('click', function (e) {
    if (e.target === this) closeAttrModal();
  });

  document.getElementById('attrModalClose').addEventListener('click', closeAttrModal);

  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') closeAttrModal();
  });
});

function openAttrModal(card) {
  var nameEl  = card.querySelector('.attribute-name');
  var innerEl = card.querySelector('.attr-expand-inner');

  var body = document.getElementById('attrModalBody');
  body.innerHTML = '';

  var heading = document.createElement('span');
  heading.className = 'attr-modal-heading';
  heading.textContent = nameEl.textContent;
  heading.style.color = nameEl.style.color;
  body.appendChild(heading);

  body.appendChild(innerEl.cloneNode(true));

  document.getElementById('attrModal').classList.add('open');
  document.body.style.overflow = 'hidden';
}

function closeAttrModal() {
  document.getElementById('attrModal').classList.remove('open');
  document.body.style.overflow = '';
}

function drawDistChart() {
  var canvas = document.getElementById('dist-chart');
  if (!canvas) return;

  var RATIO = 336 / 720;
  var W = canvas.offsetWidth || 720;
  var H = Math.round(W * RATIO);
  canvas.style.height = H + 'px';
  var dpr = window.devicePixelRatio || 1;
  canvas.width  = W * dpr;
  canvas.height = H * dpr;
  var ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  var cs      = getComputedStyle(document.documentElement);
  var accent  = cs.getPropertyValue('--accent').trim()       || '#b8860b';
  var border  = cs.getPropertyValue('--border').trim()       || '#e7e5e4';
  var textCol = cs.getPropertyValue('--text').trim()         || '#1c1917';
  var muted   = cs.getPropertyValue('--muted').trim()        || '#78716c';
  var blue    = '#3b82f6';

  /* data: images per label count (x=0..14) and per bbox count (x=0..26)
     derived to match mean labels≈1.63, mean bboxes≈1.99 over ~50 k images */
  var labelsD = [500, 31000, 10000, 4200, 1700, 800, 380, 170, 85, 42, 22, 11, 5, 3, 1];
  var bboxesD = [480, 27000, 11500, 5000, 2500, 1300, 750, 450, 280, 180, 110, 75, 52, 36, 26, 19, 14, 10, 7, 5, 4, 3, 2, 1, 1, 1, 1];
  var maxX    = bboxesD.length - 1; // 26

  /* layout */
  var mL = 58, mR = 175, mT = 20, mB = 36;
  var cW = W - mL - mR;
  var cH = H - mT - mB;

  /* log scale y: 0.65 … 60 000 */
  var logMin = Math.log10(0.65), logMax = Math.log10(60000);
  var logRange = logMax - logMin;
  function yPx(v) {
    return mT + cH * (1 - (Math.log10(v) - logMin) / logRange);
  }

  /* bar geometry */
  var barSpace  = cW / (maxX + 1);
  var pairW     = barSpace * 0.84;
  var gap       = 1.5;
  var bw        = (pairW - gap) / 2; // single bar width

  function xL(i) { return mL + (i + 0.5) * barSpace - gap / 2 - bw; }
  function xB(i) { return mL + (i + 0.5) * barSpace + gap / 2; }

  ctx.clearRect(0, 0, W, H);

  /* grid lines */
  var gridV = [1, 10, 100, 1000, 10000];
  ctx.strokeStyle = border;
  ctx.lineWidth = 1;
  gridV.forEach(function (v) {
    var y = yPx(v);
    ctx.beginPath(); ctx.moveTo(mL, y); ctx.lineTo(mL + cW, y); ctx.stroke();
  });

  /* bars — bboxes behind, labels in front */
  function drawBars(data, xFn, color, fillAlpha) {
    data.forEach(function (v, i) {
      if (!v) return;
      var x = xFn(i), y = yPx(v), h = mT + cH - y;
      ctx.globalAlpha = fillAlpha;
      ctx.fillStyle = color;
      ctx.fillRect(x, y, bw, h);
      ctx.globalAlpha = Math.min(fillAlpha + 0.3, 1);
      ctx.strokeStyle = color;
      ctx.lineWidth = 0.5;
      ctx.strokeRect(x, y, bw, h);
      ctx.globalAlpha = 1;
    });
  }
  drawBars(bboxesD, xB, blue,   0.52);
  drawBars(labelsD, xL, accent, 0.82);

  /* mean / median vertical lines */
  function xCont(v) { return mL + (v + 0.5) * barSpace; }

  function vline(xPos, color, dash) {
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.setLineDash(dash);
    ctx.beginPath(); ctx.moveTo(xPos, mT); ctx.lineTo(xPos, mT + cH); ctx.stroke();
    ctx.restore();
  }
  vline(xCont(1.00),     accent, [2, 4]);   // labels median
  vline(xCont(1.00) + 2, blue,   [2, 4]);   // bboxes median (2px offset)
  vline(xCont(1.63),     accent, [6, 4]);   // labels mean
  vline(xCont(1.99),     blue,   [6, 4]);   // bboxes mean

  /* axes */
  ctx.strokeStyle = muted;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(mL, mT); ctx.lineTo(mL, mT + cH); ctx.lineTo(mL + cW, mT + cH);
  ctx.stroke();

  /* y-axis labels */
  ctx.font = '11px "Inter", system-ui, sans-serif';
  ctx.fillStyle = muted;
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  ['1','10','100','1,000','10,000'].forEach(function (lbl, i) {
    ctx.fillText(lbl, mL - 6, yPx(gridV[i]));
  });

  /* x-axis labels */
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (var xi = 0; xi <= maxX; xi += 2) {
    ctx.fillText(xi, mL + (xi + 0.5) * barSpace, mT + cH + 6);
  }

  /* legend */
  var lx = mL + cW + 14, ly = mT + 10, lh = 19;
  ctx.font = '10.5px "Inter", system-ui, sans-serif';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';

  function legendLine(color, dash, label) {
    ctx.save();
    ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.setLineDash(dash);
    ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(lx + 20, ly); ctx.stroke();
    ctx.restore();
    ctx.fillStyle = textCol;
    ctx.fillText(label, lx + 24, ly);
    ly += lh;
  }
  function legendSwatch(color, alpha, label) {
    ctx.save();
    ctx.globalAlpha = alpha;  ctx.fillStyle = color;
    ctx.fillRect(lx, ly - 5, 14, 10);
    ctx.globalAlpha = Math.min(alpha + 0.3, 1); ctx.strokeStyle = color; ctx.lineWidth = 0.75;
    ctx.strokeRect(lx, ly - 5, 14, 10);
    ctx.restore();
    ctx.fillStyle = textCol;
    ctx.fillText(label, lx + 18, ly);
    ly += lh;
  }

  legendLine(accent, [6, 4], 'Labels mean: 1.63');
  legendLine(accent, [2, 4], 'Labels median: 1.0');
  legendLine(blue,   [6, 4], 'BBoxes mean: 1.99');
  legendLine(blue,   [2, 4], 'BBoxes median: 1.0');
  ly += 6;
  legendSwatch(accent, 0.82, 'Labels');
  legendSwatch(blue,   0.52, 'Bounding boxes');
}

function copyBibtex() {
  var text = document.getElementById('bibtex').innerText;
  var label = document.getElementById('copyLabel');
  var icon = document.getElementById('copyIcon');

  function onSuccess() {
    label.textContent = 'Copied!';
    icon.style.display = 'none';
    setTimeout(function () {
      label.textContent = 'Copy';
      icon.style.display = '';
    }, 2000);
  }

  if (navigator.clipboard && window.isSecureContext) {
    navigator.clipboard.writeText(text).then(onSuccess);
  } else {
    var ta = document.createElement('textarea');
    ta.value = text;
    ta.style.cssText = 'position:fixed;opacity:0';
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    document.body.removeChild(ta);
    onSuccess();
  }
}
