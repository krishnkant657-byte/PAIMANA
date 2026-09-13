/* PAIMANA — shared frontend runtime.
   API access, layout chrome, formatting helpers, and the provenance inspector. */

const API = {
  token: () => localStorage.getItem('paimana_token'),
  user: () => { try { return JSON.parse(localStorage.getItem('paimana_user') || 'null') || { username: 'analyst', full_name: 'Public Analyst', role: 'ADMIN' }; } catch { return { username: 'analyst', full_name: 'Public Analyst', role: 'ADMIN' }; } },

  async request(path, options = {}) {
    const headers = { 'Content-Type': 'application/json', ...(options.headers || {}) };
    const token = API.token();
    if (token) headers['Authorization'] = `Bearer ${token}`;

    const res = await fetch(path, { ...options, headers });
    let body = null;
    try { body = await res.json(); } catch { /* empty body */ }

    if (!res.ok) {
      const message = (body && body.detail) || `Request failed (${res.status})`;
      const error = new Error(message);
      error.status = res.status;
      error.body = body;
      throw error;
    }
    return body;
  },

  get: (p) => API.request(p),
  post: (p, data) => API.request(p, { method: 'POST', body: JSON.stringify(data) }),
  patch: (p, data) => API.request(p, { method: 'PATCH', body: JSON.stringify(data) }),
};

/* ---------- formatting ---------------------------------------------------- */
const UNKNOWN_HTML = '<span class="unknown-value">UNKNOWN</span>';

function num(v, decimals = 2) {
  if (v === null || v === undefined || Number.isNaN(v)) return UNKNOWN_HTML;
  return Number(v).toLocaleString('en-IN', {
    minimumFractionDigits: decimals, maximumFractionDigits: decimals,
  });
}
function int(v) {
  if (v === null || v === undefined) return UNKNOWN_HTML;
  return Number(v).toLocaleString('en-IN');
}
function crore(v, decimals = 0) {
  if (v === null || v === undefined) return UNKNOWN_HTML;
  return '₹' + Number(v).toLocaleString('en-IN', {
    minimumFractionDigits: decimals, maximumFractionDigits: decimals,
  }) + ' Cr';
}
function pct(v, decimals = 1) {
  if (v === null || v === undefined) return UNKNOWN_HTML;
  return Number(v).toFixed(decimals) + '%';
}
function months(v) {
  if (v === null || v === undefined) return UNKNOWN_HTML;
  return `${v} mo`;
}
function badge(level) {
  const l = level || 'UNKNOWN';
  return `<span class="badge badge-${l}">${l}</span>`;
}
function trendLabel(t) {
  const map = {
    DETERIORATING: '▲ Deteriorating',
    IMPROVING: '▼ Improving',
    STABLE: '— Stable',
    INSUFFICIENT_HISTORY: 'Insufficient history',
  };
  return `<span class="trend trend-${t}">${map[t] || t}</span>`;
}
function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}
function periodLabel(p) {
  if (!p) return 'UNKNOWN';
  const [y, m] = p.split('-');
  const names = ['January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'];
  return `${names[Number(m) - 1]} ${y}`;
}
function bar(value, cls = '') {
  const v = value === null || value === undefined ? 0 : Math.max(0, Math.min(100, value));
  return `<div class="bar ${cls}"><span style="width:${v}%"></span></div>`;
}
function qs(name) { return new URLSearchParams(location.search).get(name); }

/* ---------- layout chrome ------------------------------------------------- */
const NAV = [
  ['/', 'Home'], ['/projects', 'Projects'], ['/monitor', 'National Monitor'],
  ['/warnings', 'Early Warnings'], ['/interventions', 'Interventions'],
  ['/analytics', 'Analytics'], ['/simulator', 'Simulator'],
  ['/assistant', 'PAIMANA AI'], ['/data', 'Data Quality'], ['/reports', 'Reports'],
];

function renderChrome(active) {
  const user = API.user();
  const links = NAV.map(([href, label]) =>
    `<a href="${href}"${href === active ? ' class="active" aria-current="page"' : ''}>${label}</a>`
  ).join('');

  document.body.insertAdjacentHTML('afterbegin', `
    <a href="#main" class="skip-link">Skip to main content</a>
    <div class="utility">
      <div class="wrap">
        <div class="utility-left">
          <span style="font-weight:700;font-size:12.5px;letter-spacing:.04em;color:#fff">PAIMANA PLATFORM</span>
        </div>
        <div class="utility-right">
          <span class="hide-sm" style="font-size:12px;opacity:.9">Source: published PAIMANA Flash Reports</span>
          <span id="authSlot">${user
            ? `<span style="color:#fff;font-weight:600;font-size:12.5px;margin-right:6px">${esc(user.full_name || user.username)}</span> <button id="signOut" class="btn-signin-pill">Sign out</button>`
            : '<button id="signIn" class="btn-signin-pill">Sign in</button>'}</span>
        </div>
      </div>
    </div>

    <header class="identity">
      <div class="wrap">
        <div class="brand">
          <img class="brand-mark" src="/static/logo.png" alt="PAIMANA Logo"/>
          <div class="brand-text">
            <div class="org">Infrastructure Project Intelligence Platform</div>
            <div class="name">PAIMANA</div>
            <div class="expand">Project Assessment, Intelligence, Monitoring &amp; Analytics
              Network for Accelerated Infrastructure</div>
          </div>
        </div>
        <div class="identity-tools">
          <form class="searchbox" role="presentation" autocomplete="off" action="#" onsubmit="event.preventDefault();
            const el = document.getElementById('paimana_nav_search');
            if (el && el.value.trim()) location.href='/projects?q='+encodeURIComponent(el.value.trim());">
            <label for="paimana_nav_search" class="sr-only">Search projects</label>
            <input type="text" id="paimana_nav_search" name="paimana_no_autofill_${Date.now()}" placeholder="Search projects or codes" autocomplete="off" aria-autocomplete="none" autocapitalize="off" spellcheck="false"/>
            <button type="submit" aria-label="Search">⌕</button>
          </form>
        </div>
      </div>
    </header>

    <nav class="nav" aria-label="Primary">
      <div class="wrap">
        <button class="nav-toggle" aria-expanded="false" aria-controls="navLinks">☰ Menu</button>
        <div class="nav-links" id="navLinks">${links}</div>
      </div>
    </nav>
  `);

  document.body.insertAdjacentHTML('beforeend', `
    <div class="disclaimer-strip">
      <div class="wrap"><strong>Notice:</strong> PAIMANA is an independent analytical
      prototype built for Smart India Hackathon 2026. It is not an official Government of
      India portal and does not publish official statistics.</div>
    </div>
    <footer class="footer">
      <div class="wrap footer-cols">
        <div class="footer-brand">
          <img class="brand-mark" src="/static/logo.png" alt="PAIMANA Logo"/>
          <h4 style="margin-top:14px">PAIMANA</h4>
          <p style="font-size:13.4px;line-height:1.65">Evidence-grounded monitoring, risk
          intelligence and intervention tracking for public infrastructure projects.</p>
        </div>
        <div>
          <h4>Platform</h4>
          <ul>
            <li><a href="/monitor">National Monitor</a></li>
            <li><a href="/projects">Project Explorer</a></li>
            <li><a href="/warnings">Early Warnings</a></li>
            <li><a href="/interventions">Intervention Centre</a></li>
          </ul>
        </div>
        <div>
          <h4>Intelligence</h4>
          <ul>
            <li><a href="/analytics">Analytics Studio</a></li>
            <li><a href="/simulator">Scenario Lab</a></li>
            <li><a href="/assistant">PAIMANA AI Assistant</a></li>
            <li><a href="/reports">Reports</a></li>
          </ul>
        </div>
        <div>
          <h4>Transparency</h4>
          <ul>
            <li><a href="/data">Data Quality Centre</a></li>
            <li><a href="/data#sources">Source Documents</a></li>
          </ul>
        </div>
      </div>
      <div class="footer-bottom">
        <div class="wrap">
          <span>PAIMANA · Smart India Hackathon 2026 · Problem Statement PS26103</span>
          <span>Built on published Flash Report data</span>
        </div>
      </div>
    </footer>

    <div class="modal-backdrop" id="provModal" role="dialog" aria-modal="true"
         aria-labelledby="provTitle">
      <div class="modal">
        <div class="modal-head">
          <h3 id="provTitle">Data Provenance</h3>
          <button type="button" aria-label="Close" onclick="closeProvenance()">×</button>
        </div>
        <div class="modal-body" id="provBody"></div>
      </div>
    </div>
  `);

  const toggle = document.querySelector('.nav-toggle');
  toggle?.addEventListener('click', () => {
    const links = document.getElementById('navLinks');
    const open = links.classList.toggle('open');
    toggle.setAttribute('aria-expanded', String(open));
    toggle.innerHTML = open ? '✕ Close' : '☰ Menu';
  });

  document.getElementById('signOut')?.addEventListener('click', (e) => {
    e.preventDefault();
    localStorage.removeItem('paimana_token');
    localStorage.removeItem('paimana_user');
    location.reload();
  });
  document.getElementById('signIn')?.addEventListener('click', (e) => {
    e.preventDefault();
    openSignIn();
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeProvenance();
  });
}

/* ---------- provenance inspector ------------------------------------------ */
async function showProvenance(projectCode, field, period) {
  const modal = document.getElementById('provModal');
  const body = document.getElementById('provBody');
  modal.classList.add('open');
  body.innerHTML = '<div class="loading"><span class="spinner"></span> Tracing value to source…</div>';

  try {
    const url = `/api/projects/${encodeURIComponent(projectCode)}/provenance/${field}`
      + (period ? `?period=${period}` : '');
    const p = await API.get(url);

    const rows = [];
    const add = (k, v) => rows.push(`<dt>${k}</dt><dd>${v}</dd>`);

    add('Value', `<strong>${p.value ?? UNKNOWN_HTML}</strong>`);
    add('Origin', `<span class="badge badge-${p.origin === 'EXTRACTED' ? 'LOW' : 'INFO'}">${p.origin}</span>`);
    add('Reporting period', periodLabel(p.report_period));

    if (p.origin === 'EXTRACTED') {
      add('Source document', `<span class="mono">${esc(p.source_document || '—')}</span>`);
      add('Report', esc(p.source_label || '—'));
      add('Page', `<span class="mono">${p.source_page ?? '—'}</span>`);
      add('Source table', esc(p.source_table || '—'));
      add('Original field', esc(p.original_field || '—'));
      add('Raw value in source', `<span class="mono">${esc(p.raw_value ?? '—')}</span>`);
      add('Transformation', esc(p.transformation || '—'));
      add('Extraction confidence',
        p.extraction_confidence != null ? pct(p.extraction_confidence * 100) : UNKNOWN_HTML);
      add('Extraction date',
        p.extraction_date ? new Date(p.extraction_date).toLocaleString('en-IN') : '—');
      if (p.parse_warnings?.length) {
        add('Parse warnings', `<span class="mono">${esc(p.parse_warnings.join(', '))}</span>`);
      }
    } else {
      add('Derivation', `<span class="mono">${esc(p.derivation || '—')}</span>`);
      add('Computed from', esc(p.source_label || '—')
        + (p.source_page ? `, page ${p.source_page}` : ''));
      if (p.engine_version) add('Engine version', `<span class="mono">${esc(p.engine_version)}</span>`);
    }

    body.innerHTML = `
      <dl class="kv">${rows.join('')}</dl>
      <div class="notice" style="margin-top:16px">
        ${esc(p.note || 'This value is traceable to the source document and page shown above.')}
      </div>`;
  } catch (err) {
    body.innerHTML = `<div class="notice notice-warn">Could not load provenance: ${esc(err.message)}</div>`;
  }
}

function closeProvenance() {
  document.getElementById('provModal')?.classList.remove('open');
}

function provLink(value, projectCode, field, period) {
  const display = value === null || value === undefined ? 'UNKNOWN' : value;
  return `<button type="button" class="prov" onclick="showProvenance('${projectCode}','${field}','${period || ''}')"
    title="Show where this value came from">${display}</button>`;
}

/* ---------- sign-in ------------------------------------------------------- */
function openSignIn() {
  const modal = document.getElementById('provModal');
  document.getElementById('provTitle').textContent = 'Sign in';
  document.getElementById('provBody').innerHTML = `
    <p style="font-size:13.5px;margin-top:0">Sign in to raise interventions, update alert status,
    generate reports and upload Flash Reports.</p>
    <div class="field"><label for="u">Username</label><input id="u" autocomplete="username"/></div>
    <div class="field"><label for="p">Password</label>
      <input id="p" type="password" autocomplete="current-password"/></div>
    <div id="loginErr"></div>
    <button class="btn btn-primary" id="doLogin">Sign in</button>
    <div class="notice" style="margin-top:16px;font-size:12.5px">
      Demo accounts — <span class="mono">admin / paimana-admin</span>,
      <span class="mono">analyst / paimana-analyst</span>,
      <span class="mono">viewer / paimana-viewer</span>.
      Change these before any real deployment.
    </div>`;
  modal.classList.add('open');

  document.getElementById('doLogin').onclick = async () => {
    const btn = document.getElementById('doLogin');
    btn.disabled = true;
    try {
      const r = await API.post('/api/auth/login', {
        username: document.getElementById('u').value,
        password: document.getElementById('p').value,
      });
      localStorage.setItem('paimana_token', r.access_token);
      localStorage.setItem('paimana_user', JSON.stringify(r));
      location.reload();
    } catch (err) {
      document.getElementById('loginErr').innerHTML =
        `<div class="notice notice-warn">${esc(err.message)}</div>`;
      btn.disabled = false;
    }
  };
}

/* ---------- tiny sparkline / bar chart (no chart library needed) ---------- */
function sparkline(values, { width = 260, height = 60, color = '#D9581E' } = {}) {
  const pts = values.filter((v) => v !== null && v !== undefined);
  if (pts.length < 2) return '<div class="empty" style="padding:12px">Insufficient history</div>';
  const min = Math.min(...pts), max = Math.max(...pts);
  const span = max - min || 1;
  const step = width / (pts.length - 1);
  const coords = pts.map((v, i) => `${(i * step).toFixed(1)},${(height - ((v - min) / span) * (height - 8) - 4).toFixed(1)}`);
  return `<svg viewBox="0 0 ${width} ${height}" width="100%" height="${height}" role="img"
      aria-label="Trend line"><polyline fill="none" stroke="${color}" stroke-width="2"
      stroke-linejoin="round" points="${coords.join(' ')}"/>
      ${coords.map((c) => { const [x, y] = c.split(','); return `<circle cx="${x}" cy="${y}" r="2.5" fill="${color}"/>`; }).join('')}
    </svg>`;
}

function barChart(items, { valueKey = 'value', labelKey = 'label', max = null, unit = '' } = {}) {
  if (!items.length) return '<div class="empty">No data available</div>';
  const top = max ?? (Math.max(...items.map((i) => Number(i[valueKey]) || 0)) || 1);
  return `<div style="display:grid;gap:9px">${items.map((i) => {
    const v = Number(i[valueKey]) || 0;
    return `<div>
      <div style="display:flex;justify-content:space-between;font-size:12.5px;margin-bottom:3px">
        <span>${esc(i[labelKey])}</span>
        <span class="mono">${v.toLocaleString('en-IN', { maximumFractionDigits: 1 })}${unit}</span>
      </div>
      <div class="bar"><span style="width:${(v / top) * 100}%"></span></div>
    </div>`;
  }).join('')}</div>`;
}

function riskColour(level) {
  return { SEVERE: '#A31515', HIGH: '#B3261E', MODERATE: '#97650A', LOW: '#1E7A4B' }[level] || '#8A93A0';
}
