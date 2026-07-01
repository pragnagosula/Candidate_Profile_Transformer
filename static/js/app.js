/* ── DOM refs ─────────────────────────────────────────────────── */
const uploadSection   = document.getElementById('upload-section');
const progressSection = document.getElementById('progress-section');
const errorSection    = document.getElementById('error-section');
const resultsSection  = document.getElementById('results-section');
const form            = document.getElementById('upload-form');
const candidateInput  = document.getElementById('candidate-files');
const configInput     = document.getElementById('config-file');
const dropZone        = document.getElementById('drop-zone');
const fileListEl      = document.getElementById('file-list');
const transformBtn    = document.getElementById('transform-btn');
const progressMsg     = document.getElementById('progress-message');

/* ── State ────────────────────────────────────────────────────── */
let selectedFiles = [];   // files picked for candidate-files
let lastResult    = null; // last successful API response

/* ── File management ──────────────────────────────────────────── */

function addFiles(incoming) {
  const existing = new Set(selectedFiles.map(f => f.name + f.size));
  for (const f of incoming) {
    if (!existing.has(f.name + f.size)) {
      selectedFiles.push(f);
      existing.add(f.name + f.size);
    }
  }
  renderFileList();
}

function removeFile(idx) {
  selectedFiles.splice(idx, 1);
  renderFileList();
}

function renderFileList() {
  fileListEl.innerHTML = '';
  selectedFiles.forEach((f, i) => {
    const span = document.createElement('span');
    span.className = 'file-tag';
    span.innerHTML =
      `<span class="name">${esc(f.name)}</span>` +
      `<span class="remove" role="button" aria-label="Remove ${ esc(f.name) }" onclick="removeFile(${ i })">&#x2715;</span>`;
    fileListEl.appendChild(span);
  });
}

// Drop zone
dropZone.addEventListener('click', () => candidateInput.click());
dropZone.addEventListener('keydown', e => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); candidateInput.click(); }
});
dropZone.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', ()  => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  addFiles(Array.from(e.dataTransfer.files));
});
candidateInput.addEventListener('change', () => {
  addFiles(Array.from(candidateInput.files));
  candidateInput.value = '';  // allow same file to be re-added
});

// Config filename label
configInput.addEventListener('change', () => {
  document.getElementById('config-filename').textContent =
    configInput.files[0]?.name ?? 'No file chosen';
});

/* ── Form submission ──────────────────────────────────────────── */

const PROGRESS_STEPS = ['Uploading files…', 'Processing candidates…', 'Generating profiles…'];

form.addEventListener('submit', async e => {
  e.preventDefault();

  if (selectedFiles.length === 0) {
    setSection('error');
    document.getElementById('error-message').textContent =
      'Please select at least one candidate file (CSV, JSON, or PDF).';
    return;
  }

  setSection('progress');
  transformBtn.disabled = true;

  let stepIdx = 0;
  progressMsg.textContent = PROGRESS_STEPS[0];
  const ticker = setInterval(() => {
    stepIdx = Math.min(stepIdx + 1, PROGRESS_STEPS.length - 1);
    progressMsg.textContent = PROGRESS_STEPS[stepIdx];
  }, 1400);

  try {
    const body = new FormData();
    selectedFiles.forEach(f => body.append('candidate_files', f));
    const cfg = configInput.files[0];
    if (cfg) body.append('config_file', cfg);

    const resp = await fetch('/transform', { method: 'POST', body });
    clearInterval(ticker);

    if (!resp.ok) {
      const payload = await resp.json().catch(() => ({ detail: `HTTP ${ resp.status }` }));
      const msg = Array.isArray(payload.detail)
        ? payload.detail.map(d => d.msg ?? d).join('; ')
        : (payload.detail ?? `Server error ${ resp.status }`);
      throw new Error(msg);
    }

    lastResult = await resp.json();
    renderResults(lastResult);
    setSection('results');
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

  } catch (err) {
    clearInterval(ticker);
    setSection('error');
    document.getElementById('error-message').textContent = err.message;
  } finally {
    transformBtn.disabled = false;
  }
});

/* ── Section visibility ───────────────────────────────────────── */

function setSection(name) {
  progressSection.classList.add('hidden');
  errorSection.classList.add('hidden');
  resultsSection.classList.add('hidden');
  if (name === 'progress') progressSection.classList.remove('hidden');
  if (name === 'error')    errorSection.classList.remove('hidden');
  if (name === 'results')  resultsSection.classList.remove('hidden');
}

function resetToForm() {
  setSection('none');
  lastResult = null;
}

/* ── Results rendering ────────────────────────────────────────── */

function renderResults(data) {
  const root = resultsSection;
  root.innerHTML = '';

  // Back button
  div(root, 'actions-bar mt-1',
    `<button class="btn btn-secondary btn-sm" onclick="resetToForm()">&#8592; Transform Another</button>`
  );

  // Summary banner
  const summaryHtml = [
    summaryItem(data.total_inputs,      'Records parsed'),
    summaryItem(data.total_groups,      'Entity groups'),
    summaryItem(data.profiles.length,   'Profiles produced'),
    data.errors.length ? summaryItem(data.errors.length, 'Pipeline errors', true) : '',
  ].join('');
  div(root, 'summary-banner', summaryHtml);

  // Unsupported files warning
  if (data.unsupported_files && data.unsupported_files.length) {
    div(root, 'alert alert-warning',
      `<strong>Skipped:</strong> ${ data.unsupported_files.map(esc).join(', ') } — file type not supported.`
    );
  }

  // Pipeline-level errors
  if (data.errors.length) {
    const card = div(root, 'card error-card');
    card.innerHTML = `<h2>Pipeline Errors</h2>
      <ul style="padding-left:1.25rem;font-size:.875rem;color:#7f1d1d;display:flex;flex-direction:column;gap:.25rem">
        ${ data.errors.map(e => `<li>${ esc(e) }</li>`).join('') }
      </ul>`;
  }

  // Profiles
  if (data.profiles.length === 0) {
    div(root, 'empty-state', 'No candidate profiles were produced. Check any errors listed above.');
  } else {
    data.profiles.forEach((profile, idx) => {
      root.appendChild(buildProfileCard(profile, idx, data.field_map ?? {}));
    });
  }

  // Download bar
  const dl = div(root, 'download-bar mt-2');
  dl.innerHTML = '<h3>Download</h3>';
  const dlBtn = document.createElement('button');
  dlBtn.className = 'btn btn-secondary btn-sm';
  dlBtn.innerHTML = '&#x2B07; Download JSON';
  dlBtn.onclick = downloadJSON;
  dl.appendChild(dlBtn);
}

function summaryItem(val, label, isErr = false) {
  return `<div class="summary-item${ isErr ? ' err' : '' }">
    <strong>${ val }</strong><span>${ label }</span>
  </div>`;
}

/* ── Profile card ─────────────────────────────────────────────── */

function buildProfileCard(profile, idx, fieldMap) {
  const card = document.createElement('div');
  card.className = 'profile-card card';

  const f    = profile.fields        ?? {};
  const conf = profile.confidence;
  const val  = profile.validation;
  const prov = profile.provenance    ?? [];

  const scoreMap = buildScoreMap(conf, fieldMap);
  const provMap  = buildProvMap(prov, fieldMap);

  let html = '';

  /* Header */
  html += `<div class="profile-header">
    <h2>${ esc(f.full_name ?? `Candidate ${ idx + 1 }`) }</h2>
    ${ conf ? overallConfHtml(conf) : '' }
  </div>`;

  /* Validation */
  if (val && val.issues && val.issues.length) {
    html += validationHtml(val);
  }

  /* Basic Information */
  const basicFields = [
    { key: 'full_name',     label: 'Full Name'  },
    { key: 'email_address', label: 'Email'      },
    { key: 'phone_number',  label: 'Phone'      },
    { key: 'location',      label: 'Location'   },
  ].filter(r => f[r.key] != null && f[r.key] !== '');

  if (basicFields.length) {
    const rows = basicFields.map(r => `
      <tr>
        <td class="td-label">${ r.label }</td>
        <td class="td-value">${ esc(String(f[r.key])) }${ confBadgeHtml(scoreMap[r.key]) }</td>
        <td class="td-badge">${ sourceBadgeHtml(provMap[r.key]) }</td>
      </tr>`).join('');
    html += sectionHtml('Basic Information',
      `<table class="info-table"><tbody>${ rows }</tbody></table>`);
  }

  /* Professional Summary */
  if (f.professional_summary) {
    html += sectionHtml('Professional Summary',
      `<p class="summary-text">${ esc(f.professional_summary) }</p>`);
  }

  /* Skills */
  if (f.skills && f.skills.length) {
    const tags = f.skills.map(s => `<span class="skill-tag">${ esc(s) }</span>`).join('');
    html += sectionHtml('Skills',
      `<div class="skills-wrap">${ tags }${ confBadgeHtml(scoreMap['skills']) }</div>`);
  }

  /* Work Experience */
  if (f.work_experience && f.work_experience.length) {
    html += sectionHtml('Work Experience',
      `<div class="timeline">${ f.work_experience.map(expItemHtml).join('') }</div>`);
  }

  /* Education */
  if (f.education && f.education.length) {
    html += sectionHtml('Education',
      `<div class="timeline">${ f.education.map(eduItemHtml).join('') }</div>`);
  }

  /* Links */
  if (f.links && f.links.length) {
    const chips = f.links.map(l => `
      <a class="link-chip" href="${ esc(l.url) }" target="_blank" rel="noopener noreferrer">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
          <path d="M18 13v6a2 2 0 01-2 2H5a2 2 0 01-2-2V8a2 2 0 012-2h6M15 3h6m0 0v6m0-6L10 14"/>
        </svg>
        ${ esc(l.label ?? l.url) }
      </a>`).join('');
    html += sectionHtml('Links', `<div class="links-wrap">${ chips }</div>`);
  }

  /* Confidence Details */
  if (conf) {
    html += sectionHtml('Confidence Details', confDetailsHtml(conf));
  }

  /* Provenance (collapsible) */
  if (prov.length) {
    html += `
      <details>
        <summary>
          Provenance
          <span class="text-muted" style="font-weight:400;font-size:.78rem;margin-left:.5rem">${ prov.length } entries</span>
        </summary>
        <div class="details-body">
          ${ prov.length
              ? provTableHtml(prov)
              : '<p class="text-muted" style="font-size:.82rem">No provenance entries. Upload a config with <code>include_provenance: true</code>.</p>' }
        </div>
      </details>`;
  }

  /* Raw JSON (collapsible) */
  html += `
    <details>
      <summary>Raw JSON</summary>
      <div class="details-body">
        <pre class="raw-json-pre">${ esc(JSON.stringify(profile, null, 2)) }</pre>
      </div>
    </details>`;

  card.innerHTML = html;
  return card;
}

/* ── Sub-renderers ────────────────────────────────────────────── */

function overallConfHtml(conf) {
  const pct   = Math.round(conf.overall_score * 100);
  const color = pct >= 80 ? 'var(--green)' : pct >= 60 ? 'var(--amber)' : 'var(--red)';
  return `
    <div class="overall-conf">
      <span class="label">Confidence</span>
      <div class="conf-bar">
        <div class="conf-bar-fill" style="width:${ pct }%;background:${ color }"></div>
      </div>
      <span class="score" style="color:${ color }">${ pct }%</span>
    </div>`;
}

function confBadgeHtml(entry) {
  if (!entry) return '';
  const pct = Math.round(entry.score * 100);
  const cls = pct >= 80 ? 'conf-high' : pct >= 60 ? 'conf-mid' : 'conf-low';
  return `<span class="conf-badge ${ cls }" title="Confidence: ${ pct }%">${ pct }%</span>`;
}

function sourceBadgeHtml(entries) {
  if (!entries || !entries.length) return '';
  const sources = [...new Set(entries.map(e => e.source))];
  return `<span class="source-badge" title="From: ${ sources.join(', ') }">${ sources.join(', ') }</span>`;
}

function validationHtml(v) {
  const issues = v.issues ?? [];
  const rows = issues.map(i => {
    const cls  = i.severity === 'error' ? 'error' : 'warning';
    const icon = i.severity === 'error' ? '&#x2717;' : '&#x26A0;';
    return `<div class="issue ${ cls }"><em class="issue-icon">${ icon }</em>${ esc(i.message) }</div>`;
  }).join('');
  return `<div class="validation-block"><div class="issue-list">${ rows }</div></div>`;
}

function confDetailsHtml(conf) {
  const compPct = Math.round(conf.completeness     * 100);
  const agrPct  = Math.round(conf.source_agreement * 100);
  const scores  = conf.field_scores ?? [];

  const grid = `
    <div class="conf-detail-grid">
      <div class="conf-detail-item">
        <div class="val">${ compPct }%</div>
        <div class="lbl">Completeness</div>
      </div>
      <div class="conf-detail-item">
        <div class="val">${ agrPct }%</div>
        <div class="lbl">Source Agreement</div>
      </div>
    </div>`;

  if (!scores.length) return grid;

  const rows = scores.map(fs => {
    const srcs = (fs.contributing_sources ?? []).join(', ');
    return `
      <tr>
        <td>${ esc(fs.field_name) }</td>
        <td>${ confBadgeHtml(fs) }</td>
        <td style="color:var(--gray-500)">${ esc(srcs) }</td>
        <td style="color:var(--gray-400)">${ esc(fs.reason ?? '') }</td>
      </tr>`;
  }).join('');

  return `${ grid }
    <table class="scores-table">
      <thead><tr><th>Field</th><th>Score</th><th>Sources</th><th>Reason</th></tr></thead>
      <tbody>${ rows }</tbody>
    </table>`;
}

function provTableHtml(entries) {
  const rows = entries.map(e => `
    <tr>
      <td>${ esc(e.field_name) }</td>
      <td><span class="skill-tag" style="font-size:.72rem">${ esc(e.source) }</span></td>
      <td class="prov-original">${ esc(String(e.original_value  ?? '—')) }</td>
      <td>${ esc(String(e.normalized_value ?? '—')) }</td>
      <td style="color:var(--gray-400)">${ esc(e.extraction_method ?? '') }</td>
    </tr>`).join('');
  return `
    <div style="overflow-x:auto">
      <table class="prov-table">
        <thead><tr><th>Field</th><th>Source</th><th>Original</th><th>Normalised</th><th>Method</th></tr></thead>
        <tbody>${ rows }</tbody>
      </table>
    </div>`;
}

function expItemHtml(exp) {
  const org    = exp.company   ? esc(exp.company)   : '';
  const loc    = exp.location  ? ` &bull; ${ esc(exp.location) }` : '';
  const dates  = fmtDuration(exp.duration);
  const desc   = exp.description ? esc(exp.description) : '';
  return `
    <div class="tl-item">
      <div class="tl-role">${ esc(exp.title ?? '') }</div>
      ${ org ? `<div class="tl-org">${ org }${ loc }</div>` : '' }
      ${ dates ? `<div class="tl-dates">${ dates }</div>` : '' }
      ${ desc  ? `<div class="tl-desc">${ desc }</div>` : '' }
    </div>`;
}

function eduItemHtml(edu) {
  const degree  = [edu.degree, edu.field_of_study].filter(Boolean).map(esc).join(', ');
  const inst    = edu.institution ? esc(edu.institution) : '';
  const dates   = fmtDuration(edu.duration);
  const gpa     = edu.gpa ? `GPA: ${ esc(String(edu.gpa)) }` : '';
  return `
    <div class="tl-item">
      ${ degree ? `<div class="tl-role">${ degree }</div>` : '' }
      ${ inst   ? `<div class="tl-org">${ inst }</div>` : '' }
      ${ dates  ? `<div class="tl-dates">${ dates }</div>` : '' }
      ${ gpa    ? `<div class="tl-meta">${ gpa }</div>` : '' }
    </div>`;
}

function fmtDuration(d) {
  if (!d) return '';
  const start = d.start ?? '';
  const end   = d.is_current ? 'Present' : (d.end ?? '');
  return [start, end].filter(Boolean).join(' – ');
}

/* ── Lookup map builders ─────────────────────────────────────── */

function buildScoreMap(conf, fieldMap) {
  const map = {};
  if (!conf || !conf.field_scores) return map;
  conf.field_scores.forEach(fs => {
    // fieldMap: { output_name → source_field_name }
    const out = Object.keys(fieldMap).find(k => fieldMap[k] === fs.field_name);
    if (out) map[out] = fs;
  });
  return map;
}

function buildProvMap(provenance, fieldMap) {
  const map = {};
  provenance.forEach(entry => {
    const out = Object.keys(fieldMap).find(k => fieldMap[k] === entry.field_name)
              ?? entry.field_name;
    (map[out] ??= []).push(entry);
  });
  return map;
}

/* ── Download ────────────────────────────────────────────────── */

function downloadJSON() {
  if (!lastResult) return;
  const blob = new Blob(
    [JSON.stringify(lastResult.profiles, null, 2)],
    { type: 'application/json' }
  );
  const url = URL.createObjectURL(blob);
  const a   = Object.assign(document.createElement('a'), {
    href: url, download: 'candidate_profiles.json'
  });
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/* ── Helpers ─────────────────────────────────────────────────── */

function esc(str) {
  return String(str)
    .replace(/&/g,  '&amp;')
    .replace(/</g,  '&lt;')
    .replace(/>/g,  '&gt;')
    .replace(/"/g,  '&quot;')
    .replace(/'/g,  '&#39;');
}

function sectionHtml(title, body) {
  return `<div class="profile-section">
    <h3 class="section-heading">${ title }</h3>
    ${ body }
  </div>`;
}

function div(parent, className, innerHTML = '') {
  const el = document.createElement('div');
  if (className) el.className = className;
  if (innerHTML) el.innerHTML = innerHTML;
  parent.appendChild(el);
  return el;
}
