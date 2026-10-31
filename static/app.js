/* FaceVeda front end: login, patient onboarding (intake -> scan -> doctor), and the patient / doctor / admin screens. Vanilla JS, no build step. */
(() => {
const $ = (s, r = document) => r.querySelector(s);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const app = $('#app');
let me = null, SYSTEMS = {};
const ZONE_LABEL = { forehead: 'Forehead', left_cheek: 'Left cheek', right_cheek: 'Right cheek', nose: 'Nose', chin: 'Chin and jaw' };
const DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];

const ICON = {
  home: '<path d="M3 11l9-8 9 8M5 10v10h14V10"/>', scan: '<path d="M4 8V5a1 1 0 011-1h3M16 4h3a1 1 0 011 1v3M20 16v3a1 1 0 01-1 1h-3M8 20H5a1 1 0 01-1-1v-3"/><circle cx="12" cy="10" r="3"/><path d="M6.5 18a6 6 0 0111 0"/>',
  cal: '<rect x="3" y="5" width="18" height="16" rx="2"/><path d="M8 3v4M16 3v4M3 10h18"/>', doctor: '<path d="M12 3v6M9 6h6"/><circle cx="12" cy="14" r="3"/><path d="M5 21a7 7 0 0114 0"/>',
  form: '<rect x="5" y="3" width="14" height="18" rx="2"/><path d="M9 8h6M9 12h6M9 16h4"/>', clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
  users: '<circle cx="9" cy="8" r="3"/><path d="M3 20a6 6 0 0112 0M16 4a3 3 0 010 6M18 20a5 5 0 00-3-4.5"/>', plus: '<path d="M12 5v14M5 12h14"/>',
  out: '<path d="M10 4H5a1 1 0 00-1 1v14a1 1 0 001 1h5M15 8l4 4-4 4M19 12H9"/>', alert: '<path d="M12 9v4M12 17h.01"/><path d="M10.3 3.9L1.8 18a2 2 0 001.7 3h17a2 2 0 001.7-3L13.7 3.9a2 2 0 00-3.4 0z"/>',
  chart: '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>', mail: '<rect x="3" y="5" width="18" height="14" rx="2"/><path d="M3 7l9 6 9-6"/>',
  user: '<circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0116 0"/>', key: '<circle cx="8" cy="15" r="4"/><path d="M11 12l9-9M16 7l3 3"/>', eye: '<path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/>', check: '<path d="M5 12l5 5 9-10"/>', cam: '<path d="M4 8h3l2-3h6l2 3h3a1 1 0 011 1v10a1 1 0 01-1 1H4a1 1 0 01-1-1V9a1 1 0 011-1z"/><circle cx="12" cy="13" r="3.5"/>',
};
const icon = (n, c = '') => `<svg class="icon ${c}" viewBox="0 0 24 24" aria-hidden="true">${ICON[n] || ''}</svg>`;
const MARK = '<svg viewBox="0 0 32 32" aria-hidden="true"><circle cx="16" cy="16" r="9" fill="none" stroke="#325385" stroke-width="2.2"/><circle cx="13" cy="14" r="1.4" fill="#325385"/><circle cx="19" cy="14" r="1.4" fill="#325385"/><path d="M12.5 19.5q3.5 3 7 0" stroke="#ed7639" stroke-width="2" fill="none" stroke-linecap="round"/></svg>';

window.FV_MARK = MARK;      // fallback if static/logo-mark.png is missing
const markImg = () => `<img src="/static/logo-mark.png" alt="" onerror="this.outerHTML=window.FV_MARK">`;
const logoFull = () => `<img class="logo-full" src="/static/logo.png" alt="FaceVeda: face wellness" onerror="this.outerHTML='<span class=&quot;brand-mark&quot;>'+window.FV_MARK+'</span><div><h1>FaceVeda</h1><p class=&quot;muted&quot;>Face wellness platform</p></div>'">`;

async function api(path, opts = {}) {
  const r = await fetch('/api' + path, { method: opts.method || 'GET', headers: opts.body ? { 'Content-Type': 'application/json' } : {}, body: opts.body ? JSON.stringify(opts.body) : undefined, credentials: 'same-origin' });
  let data = null; try { data = await r.json(); } catch { /* empty body */ }
  if (!r.ok) { const d = data && data.detail; const e = new Error(typeof d === 'string' ? d : Array.isArray(d) ? d.map(x => x.msg).join('; ') : 'Request failed'); e.status = r.status; throw e; }
  return data;
}
function toast(msg, kind = '') {
  const t = document.createElement('div'); t.className = 'toast ' + kind; t.setAttribute('role', 'status'); t.innerHTML = `<span>${esc(msg)}</span><button class="toast-close" aria-label="Dismiss">×</button>`;
  $('#toasts').append(t); t.querySelector('button').onclick = () => t.remove(); setTimeout(() => t.remove(), 5000);
}
const D = ts => new Date(String(ts).replace(' ', 'T'));
const fmtDT = ts => D(ts).toLocaleString('en-IN', { weekday: 'short', day: 'numeric', month: 'short', hour: 'numeric', minute: '2-digit' });
const fmtDate = ts => D(ts).toLocaleDateString('en-IN', { weekday: 'short', day: 'numeric', month: 'short', year: 'numeric' });
const ymd = d => `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
const statusBadge = s => `<span class="badge ${s === 'completed' ? 'success' : s === 'cancelled' ? 'neutral' : s === 'missed' ? 'danger' : 'info'}">${esc(s)}</span>`;
const upcoming = a => a.status === 'booked' && (a.display_status || a.status) === 'booked';
const movedBadge = a => a.reschedule_count ? ` <span class="badge accent" title="Was ${esc(fmtDT(a.rescheduled_from))}${a.reschedule_reason ? '. Reason: ' + esc(a.reschedule_reason) : ''}">Moved by ${esc(a.rescheduled_by || '')}${a.reschedule_count > 1 ? ' ×' + a.reschedule_count : ''}</span>` : '';
const reschedBtn = (a, who) => upcoming(a) ? `<button class="btn btn-secondary btn-sm" data-resched="${a.id}" data-when="${esc(fmtDT(a.start_ts))}" data-who="${esc(who)}">Reschedule</button>` : '';
const tile = (label, value, note = '', cls = '') => `<div class="tile ${cls}"><div class="tile-label">${esc(label)}</div><div class="tile-value">${value}</div>${note ? `<div class="tile-note">${esc(note)}</div>` : ''}</div>`;
const alertBox = (kind, html, ic = 'info') => `<div class="alert alert-${kind}" role="${kind === 'error' ? 'alert' : 'note'}">${icon(ic)}<div>${html}</div></div>`;
const stateBox = (title, text) => `<div class="state"><div class="state-icon">${icon('info')}</div><h3>${esc(title)}</h3><p>${esc(text)}</p></div>`;
const infoTip = text => `<span class="info-dot" tabindex="0" role="img" aria-label="${esc(text)}" data-tip="${esc(text)}">i</span>`;
const initials = n => String(n).split(' ').map(w => w[0] || '').slice(0, 2).join('').toUpperCase();
const card = (title, body, head = '') => `<section class="card"><div class="card-head"><h2>${esc(title)}</h2>${head}</div>${body}</section>`;

/* ---------------- shell ---------------- */
const NAV = {
  patient: [['home', 'Home', 'home'], ['scan', 'Face scan', 'scan'], ['progress', 'My progress', 'chart'], ['appointments', 'Appointments', 'cal'], ['doctor', 'My doctor', 'doctor'], ['intake', 'My health form', 'form']],
  doctor: [['home', 'Dashboard', 'home'], ['appointments', 'Appointments', 'cal'], ['availability', 'Availability', 'clock'], ['/acne-lab/', 'Acne Lab (GAGS)', 'scan']],
  admin: [['home', 'Overview', 'home'], ['analytics', 'Analytics', 'chart'], ['calendar', 'Master calendar', 'cal'], ['/acne-lab/', 'Acne Lab (GAGS)', 'scan'], ['emails', 'Email log', 'mail'], ['add-doctor', 'Add a doctor', 'plus']],
};
function shell(nav, active, inner) {
  const links = nav ? NAV[me.role].map(([k, l, i]) => `<a class="nav-link" href="${k.startsWith('/') ? k : '#/' + k}" ${k === active ? 'aria-current="page"' : ''}>${icon(i)}<span class="label">${l}</span></a>`).join('') : '';
  app.innerHTML = `<div class="app-shell ${nav ? '' : 'is-collapsed'}" style="${nav ? '' : 'grid-template-columns:minmax(0,1fr);grid-template-areas:\'top\' \'main\''}">
    <header class="topbar"><a class="brand" href="#/home"><span class="brand-mark">${markImg()}</span><span>FaceVeda</span></a>${previewChip()}<span class="topbar-spacer"></span>
      <span class="role-chip">${esc(me.role)}</span>${userMenuHtml()}</header>
    ${nav ? `<nav class="sidebar" aria-label="Main"><div class="nav-section">${me.role === 'patient' ? 'My care' : me.role === 'doctor' ? 'My practice' : 'Administration'}</div>${links}</nav>` : ''}
    <main class="main" id="main" tabindex="-1">${viewAsBanner()}${nav && me.role === 'patient' ? scanBanner() : ''}${inner}</main></div>`;
  wireUserMenu();
}
const ROLE_LABEL = { admin: 'Platform admin', doctor: 'Doctor', patient: 'Patient' };
function previewChip() {
  if (!me.viewing_as) return '';
  return `<span class="preview-chip" title="You stay signed in as ${esc(me.viewing_as.admin_name)}. The preview is read-only and face photos are hidden.">${icon('eye', 'sm')} Previewing as ${ROLE_LABEL[me.role] || esc(me.role)}: ${esc(me.name)}<button class="preview-exit" type="button" data-um-exit>Exit</button></span>`;
}
function userMenuHtml() {
  const va = me.viewing_as; const mi = (k, ic, t) => `<button class="menu-item" role="menuitem" data-um="${k}">${icon(ic, 'sm')} ${t}</button>`;
  const switchable = me.role === 'admin' || !!va;      // only an admin can switch view (also while previewing, to jump straight to someone else)
  return `<div class="menu" id="usermenu"><button class="user-btn" id="umbtn" aria-haspopup="menu" aria-expanded="false"><span class="avatar">${esc(initials(me.name))}</span><span class="label">${esc(me.name)}</span><span aria-hidden="true">▾</span></button>
    <div class="menu-panel" id="umenu" role="menu" hidden><div class="menu-head"><strong>${esc(me.name)}</strong><div class="small muted">${esc(me.email)}</div><div class="small muted">${esc(ROLE_LABEL[me.role] || me.role)}</div>
    ${va ? `<div class="small muted" style="margin-top:6px">Previewing: signed in as ${esc(va.admin_name)}</div>` : ''}</div>
    ${mi('profile', 'user', 'Profile')}${va ? '' : mi('password', 'key', 'Change password')}${switchable ? mi('view', 'eye', 'Switch view') : ''}${va ? mi('exit', 'out', 'Back to admin view') : ''}${mi('logout', 'out', 'Log out')}</div></div>`;
}
function viewAsBanner() { return ''; }      // TAP-style: the chip in the top bar replaces a banner
function wireUserMenu() {
  const btn = $('#umbtn'), panel = $('#umenu'); if (!btn) return;
  btn.onclick = e => { e.stopPropagation(); panel.hidden = !panel.hidden; btn.setAttribute('aria-expanded', String(!panel.hidden)); };
  panel.onclick = e => { const b = e.target.closest('[data-um]'); if (!b) return; panel.hidden = true; userAction(b.dataset.um); };
  const x = $('[data-um-exit]'); if (x) x.onclick = () => userAction('exit');
}
async function userAction(k) {
  if (k === 'logout') { await api('/logout', { method: 'POST' }); me = null; location.hash = ''; boot(); }
  else if (k === 'exit') { try { await api('/view-as/exit', { method: 'POST' }); me = await api('/me'); location.hash = '#/home'; route(); } catch (e) { toast(e.message, 'error'); } }
  else if (k === 'profile') profileModal();
  else if (k === 'password') passwordModal();
  else if (k === 'view') viewAsModal();
}
async function profileModal() {
  let extra = '';
  try {
    const docs = await api('/doctors');
    if (me.role === 'doctor') { const d = docs.find(x => x.id === me.id); if (d) extra = `<dt>Specialization</dt><dd>${esc(d.specialization || '-')}</dd><dt>Focus areas</dt><dd>${esc(d.focus.map(f => SYSTEMS[f] || f).join(', ') || 'General')}</dd>`; }
    else if (me.role === 'patient') { const d = docs.find(x => x.id === me.onboarding.doctor_id); extra = `<dt>Current doctor</dt><dd>${d ? esc(d.name) : 'Not chosen yet'}</dd>`; }
  } catch { /* profile still works without the extras */ }
  modal('Profile', `<dl class="kv"><dt>Name</dt><dd>${esc(me.name)}</dd><dt>Email</dt><dd>${esc(me.email)}</dd><dt>Role</dt><dd style="text-transform:capitalize">${esc(me.role === 'admin' ? 'Platform admin' : me.role)}</dd>
    <dt>Member since</dt><dd>${esc(fmtDate(me.created.replace('T', ' ')))}</dd>${extra}</dl>`, '<button class="btn btn-secondary" data-close>Close</button>');
}
function passwordModal() {
  const m = modal('Change password', `<div class="stack"><div class="field"><label for="pw0">Current password</label><input class="input" id="pw0" type="password" autocomplete="current-password"></div>
    <div class="field"><label for="pw1">New password</label><input class="input" id="pw1" type="password" autocomplete="new-password"><span class="hint">At least 8 characters.</span></div>
    <div class="field"><label for="pw2">Repeat new password</label><input class="input" id="pw2" type="password" autocomplete="new-password"></div><div id="pwerr"></div></div>`,
    '<button class="btn btn-secondary" data-close>Cancel</button><button class="btn btn-primary" id="pwsave">Change password</button>');
  m.el.querySelector('#pwsave').onclick = async () => {
    const v = id => m.el.querySelector('#' + id).value, err = t => { m.el.querySelector('#pwerr').innerHTML = alertBox('error', esc(t), 'alert'); };
    if (v('pw1') !== v('pw2')) return err('The two new passwords do not match.');
    try { await api('/me/password', { method: 'PUT', body: { current: v('pw0'), new: v('pw1') } }); m.close(); toast('Password changed', 'success'); } catch (e) { err(e.message); }
  };
}
async function viewAsModal() {
  const cur = me.viewing_as ? { role: me.role, id: me.id } : null;
  const m = modal('Switch view', `<div class="stack"><p class="muted">See the app exactly as another kind of user would. You stay signed in as yourself: only what is shown changes. The preview is read-only, and face photos stay hidden.</p>
    <div class="field"><label for="vr">View as</label><select class="select" id="vr"><option value="doctor">Doctor</option><option value="patient">Patient</option></select></div>
    <div class="field" id="vpf"><label for="vp" id="vpl">Doctor <span class="req">*</span></label><select class="select" id="vp"><option value="">Select a doctor…</option></select><span class="field-error" id="vperr" role="alert"></span></div>
    <div class="hint" id="vst" role="status" aria-live="polite"></div></div>`, '<button class="btn btn-secondary" data-close>Cancel</button><button class="btn btn-primary" id="vapply">Switch view</button>');
  const q = id => m.el.querySelector('#' + id); let rows = [];
  const load = async () => {
    const role = q('vr').value; q('vpl').innerHTML = (role === 'doctor' ? 'Doctor' : 'Patient') + ' <span class="req">*</span>'; q('vperr').textContent = ''; q('vst').textContent = 'Loading…';
    q('vp').innerHTML = `<option value="">Select a ${role}…</option>`;
    try {
      rows = await api('/admin/users?role=' + role); q('vst').textContent = rows.length ? '' : `There are no ${role}s yet.`;
      q('vp').innerHTML += rows.map(r => `<option value="${r.id}">${esc(r.name)} — ${esc(r.detail || r.email)}</option>`).join('');
      if (cur && cur.role === role) q('vp').value = String(cur.id);
    } catch (e) { q('vst').textContent = `The list could not be loaded: ${e.message}`; }
  };
  q('vr').onchange = load; if (cur) q('vr').value = cur.role; await load();
  q('vapply').onclick = async () => {
    if (!q('vp').value) { q('vperr').textContent = `Select a ${q('vr').value}.`; return; }
    try { if (me.viewing_as) await api('/view-as/exit', { method: 'POST' }); await api('/admin/view-as', { method: 'POST', body: { user_id: +q('vp').value } }); m.close(); me = await api('/me'); location.hash = '#/home'; route(); }
    catch (er) { q('vperr').textContent = er.message; }
  };
}
async function rescheduleModal(id, when, who) {
  const m = modal('Reschedule appointment', `<div class="stack"><p class="muted">Currently <strong>${esc(when)}</strong>. ${who === 'doctor' ? 'The patient will be emailed about the new time.' : 'Your doctor will be told about the new time.'}</p>
    <div class="slot-days" id="rsdays"></div><div class="slot-times" id="rstimes"></div>
    <div class="field"><label for="rsr">${who === 'doctor' ? 'Reason (the patient will see it)' : 'Reason (optional)'}</label><input class="input" id="rsr" value="${who === 'doctor' ? 'Doctor unavailable' : ''}" maxlength="200"></div><div id="rserr"></div></div>`,
    '<button class="btn btn-secondary" data-close>Cancel</button><button class="btn btn-primary" id="rsgo" disabled>Reschedule</button>');
  let slots = {}, day = null, time = null; const q = x => m.el.querySelector(x);
  const draw = () => { q('#rsdays').innerHTML = Object.keys(slots).map(d => `<button class="slot-btn" data-d="${d}" aria-pressed="${d === day}">${esc(fmtDate(d + ' 00:00'))}</button>`).join('');
    q('#rstimes').innerHTML = (slots[day] || []).map(t => `<button class="slot-btn" data-t="${t}" aria-pressed="${t === time}">${t}</button>`).join(''); q('#rsgo').disabled = !time; };
  try { slots = await api(`/appointments/${id}/slots?days=28`); const ks = Object.keys(slots); day = ks[0] || null; if (!ks.length) q('#rsdays').innerHTML = '<p class="muted">No free slots in the next 4 weeks.</p>'; else draw(); }
  catch (e) { q('#rserr').innerHTML = alertBox('error', esc(e.message), 'alert'); }
  m.el.addEventListener('click', async e => {
    const d = e.target.closest('[data-d]'), t = e.target.closest('[data-t]');
    if (d) { day = d.dataset.d; time = null; draw(); } else if (t) { time = t.dataset.t; draw(); }
    else if (e.target.closest('#rsgo')) { try { await api(`/appointments/${id}/reschedule`, { method: 'POST', body: { start: `${day} ${time}`, reason: q('#rsr').value } }); m.close(); toast(who === 'doctor' ? 'Rescheduled: the patient has been emailed' : 'Appointment rescheduled', 'success'); route(); } catch (er) { q('#rserr').innerHTML = alertBox('error', esc(er.message), 'alert'); await rsRefresh(); } }
  });
  async function rsRefresh() { try { slots = await api(`/appointments/${id}/slots?days=28`); if (!slots[day]) day = Object.keys(slots)[0] || null; time = null; draw(); } catch { /* keep the old list */ } }
}
function scanBanner() {
  const st = me.scan_status; if (!st || !st.due) return '';
  return `<div class="page" style="margin-bottom:16px" id="scanbanner">${alertBox(st.status === 'overdue' ? 'error' : 'warn', `<strong>Your weekly face scan is ${st.status === 'overdue' ? 'overdue' : 'due'}.</strong> It has been ${st.days_since} days since your last scan. A scan each week lets you and your doctor see how your skin is changing. <a class="btn btn-primary btn-sm" style="margin-left:8px" href="#/scan">Scan now</a>`, 'clock')}</div>`;
}
const header = (title, sub = '', actions = '', crumb = '') => `<div class="page-header"><div>${crumb ? `<p class="breadcrumb">${crumb}</p>` : ''}<h1 class="page-title" tabindex="-1">${esc(title)}</h1>${sub ? `<p class="sub">${esc(sub)}</p>` : ''}</div><div class="page-actions">${actions}</div></div>`;

/* ---------------- login ---------------- */
function loginView(mode = 'login') {
  const reg = mode === 'register';
  app.innerHTML = `<div class="auth-page"><div class="auth-card"><div class="brand-lg">${logoFull()}</div>
    <form id="authf" class="stack" novalidate>${reg ? `<div class="field"><label for="n">Full name</label><input class="input" id="n" autocomplete="name" required></div>` : ''}
      <div class="field"><label for="e">Email</label><input class="input" id="e" type="email" autocomplete="username" required></div>
      <div class="field"><label for="p">Password</label><input class="input" id="p" type="password" autocomplete="${reg ? 'new-password' : 'current-password'}" required>${reg ? '<span class="hint">At least 8 characters.</span>' : ''}</div>
      <div id="aerr"></div><button class="btn btn-primary btn-block" type="submit">${reg ? 'Create patient account' : 'Sign in'}</button></form>
    <div class="auth-links"><button class="link-btn" id="swap">${reg ? 'I already have an account' : 'New patient? Create an account'}</button></div>
    <p class="muted small" style="margin-top:16px;text-align:center">Doctors and admins are added by the platform admin.</p></div></div>`;
  $('#swap').onclick = () => loginView(reg ? 'login' : 'register');
  $('#authf').onsubmit = async ev => {
    ev.preventDefault(); const email = $('#e').value.trim(), password = $('#p').value; $('#aerr').innerHTML = '';
    try { await api(reg ? '/register' : '/login', { method: 'POST', body: reg ? { name: $('#n').value.trim(), email, password } : { email, password } }); await boot(); }
    catch (e) { $('#aerr').innerHTML = alertBox('error', esc(e.message), 'alert'); }
  };
}

/* ---------------- shared pieces ---------------- */
function reliabilityHtml(f) {
  const r = f.reliability; if (!r) return '';
  return `<p class="small ${r.low ? '' : 'muted'}">${r.low ? '<span class="badge danger">Low reliability in this zone</span> ' : ''}In our test on expert-labelled photos, the scan found about ${Math.round(r.recall * 100)}% of the lesions in this zone, and ${Math.round(r.precision * 100)}% of the spots it flagged were real.${r.low ? ' Please interpret this zone with extra caution.' : ''}</p>`;
}
const PHOTO_VIEWS = [['front', 'Front'], ['left', 'Left side'], ['right', 'Right side']];
function photoGallery(scanId, views) {
  if (!views || !views.length) return '<p class="muted small">No photos are stored for this scan.</p>';
  if (me.viewing_as) return '<p class="muted small">Face photos are hidden in admin view.</p>';
  return `<div class="photo-gallery">${PHOTO_VIEWS.filter(([k]) => views.includes(k)).map(([k, l]) => `<figure><img loading="lazy" src="/api/scans/${scanId}/photo/${k}/annotated" data-sid="${scanId}" data-view="${k}" data-kind="annotated" alt="${l} photo with zones and spots marked">
    <figcaption>${l} · <button class="link-btn small" type="button" data-photo-toggle>show original</button></figcaption></figure>`).join('')}</div>`;
}
function simulatedHtml(rep) {
  const o = rep.simulated_outlook; if (!o) return '';
  const cls = { higher: 'progress', typical: 'neutral', lower: 'success' }, txt = { higher: 'Higher attention', typical: 'Typical', lower: 'Lower attention' };
  return `<details class="outlook"><summary><strong>Simulated future outlook (research prototype)</strong></summary><div class="stack" style="margin-top:12px">
    ${alertBox('warn', '<strong>This is a simulation, not a medical prediction.</strong> It was trained on a synthetic population generated from assumptions, because no real dataset links face patterns to later conditions. It shows how such a tool could work. Do not use it to decide anything about your health.', 'alert')}
    <ul class="legend">${o.tiers.map(t => `<li><span>${esc(t.label)}</span><span class="count"><span class="badge ${cls[t.tier]}">${txt[t.tier]}</span></span></li>`).join('')}</ul>
    <p class="muted small">Tiers compare you with the simulated population, using your health form, your scan and the assumed face-map link (${esc(o.scenario)} strength). They are not probabilities.</p></div></details>`;
}
const GAGS_TONE = { Clear: 'success', Mild: 'info', Moderate: 'progress', Severe: 'danger', 'Very Severe': 'danger' };
function gagsHtml(rep) {
  const g = rep.gags; if (!g) return '';
  const rows = Object.keys(ZONE_LABEL).map(z => { const d = g.zones[z] || { count: 0, max_grade: 0, factor: 1, score: 0, types: {} }; const t = Object.entries(d.types).map(([c, n]) => `${n} ${esc(c)}`).join(', ') || 'none';
    return `<tr><td class="cell-title">${ZONE_LABEL[z]}</td><td>${t}</td><td class="num">${d.max_grade}</td><td class="num">&times;${d.factor}</td><td class="num"><strong>${d.score}</strong></td></tr>`; }).join('');
  const types = (g.acne_types || []).map(t => `<li><strong>${esc(t.type)}</strong>: ${t.count} (${t.percent}%) <span class="muted small">${esc(t.description)}</span></li>`).join('') || '<li>No acne lesions found.</li>';
  return `<div class="finding"><h3>Acne grading (GAGS) <span class="badge ${GAGS_TONE[g.severity] || 'neutral'}">${esc(g.severity)}</span> <span class="muted small">score ${g.total_score} of ${g.max_score}</span></h3>
    ${g.severity_basis === 'lesion_count' ? alertBox('info', `Raised from <strong>${esc(g.gags_severity)}</strong> (zone score) to <strong>${esc(g.severity)}</strong> because of the number of lesions found.`) : ''}
    <p class="small"><strong>Acne types found</strong></p><ul class="clean">${types}</ul>
    <div class="table-wrap"><table class="table"><thead><tr><th>Zone</th><th>Lesions</th><th class="num">Worst grade</th><th class="num">Factor</th><th class="num">Score</th></tr></thead><tbody>${rows}</tbody></table></div>
    <p class="muted small" style="margin-top:8px">${esc(g.note)}</p></div>`;
}
function accuracyHtml(rep) {
  const a = rep.accuracy; if (!a) return '';
  return `<details class="outlook"><summary><strong>How accurate is this scan?</strong></summary><p class="small" style="margin-top:8px">Tested on ${a.photos} expert-labelled photos of different people (${esc(a.validation)}): the scan finds about ${Math.round(a.recall * 100)}% of real lesions, about ${Math.round(a.precision * 100)}% of the spots it flags are real,
    and it names the right most-affected zone in about ${Math.round(a.zone_match * 100)}% of photos (guessing would be about 20%). It is a screening aid, not a count of acne, and it works best in even light with the whole face in view.</p></details>`;
}
function outlookHtml(f) {
  if (!f.outlook || !f.outlook.length) return '';
  return `<details class="outlook"><summary><strong>If this is not addressed: what doctors watch for</strong></summary><div class="stack" style="margin-top:12px">
    <p class="muted small">These are real conditions of the body area this zone is traditionally linked to. They are listed so a doctor can check for them. The link from facial spots to them is traditional and not proven.</p>
    ${f.outlook.map(o => `<div><strong>${esc(o.system)}</strong> <span class="badge ${o.evidence === 'traditional' ? 'accent' : 'info'}">${o.evidence === 'traditional' ? 'Traditional link' : 'Traditional + partial modern support'}</span>${o.already_reported ? ' <span class="badge progress">You reported an issue here</span>' : ''}
      <ul class="clean">${o.watch_for.map(w => `<li>${esc(w)}</li>`).join('')}</ul>${o.risk_factors.length ? `<p class="small"><strong>Factors you reported that raise attention:</strong> ${o.risk_factors.map(esc).join('; ')}.</p>` : ''}</div>`).join('')}</div></details>`;
}
function skinOutlookHtml(rep) {
  const o = rep.skin_outlook; if (!o) return '';
  return `<div class="finding"><h3>If the spots themselves are not managed <span class="badge success">Well established</span></h3><ul class="clean">${o.watch_for.map(w => `<li>${esc(w)}</li>`).join('')}</ul></div>`;
}
function reportHtml(rep, previews = [], { canBook = false } = {}) {
  const bars = Object.keys(ZONE_LABEL).map(z => { const s = (rep.zone_share || {})[z] || 0; return `<div class="bar-row"><span>${ZONE_LABEL[z]}</span><div class="bar-track"><div class="bar-fill" style="--w:${Math.round(s * 100)}%"></div></div><span>${Math.round(s * 100)}%</span></div>`; }).join('');
  const finds = (rep.findings || []).map(f => `<article class="finding"><h3>${esc(f.label)} <span class="badge ${f.evidence === 'traditional' ? 'accent' : 'info'}">${f.evidence === 'traditional' ? 'Traditional indication' : 'Traditional + partial modern support'}</span>
      <span class="muted small">${f.spots} spots · ${Math.round(f.share * 100)}% of the density</span></h3>
      <div class="trad"><div><b>Traditional Chinese medicine:</b> ${esc(f.tcm)}</div><div><b>Ayurveda:</b> ${esc(f.ayurveda)}</div><div><b>Modern view:</b> ${esc(f.modern)}</div></div>
      <div class="note-box">${esc(f.personal_note)}</div>${reliabilityHtml(f)}${outlookHtml(f)}</article>`).join('');
  return `<div class="stack">
    <p><strong>${esc(rep.summary)}</strong></p>
    ${rep.suggest_appointment ? alertBox('info', `A doctor review is suggested.${canBook ? ' <a href="#/appointments">Book an appointment</a>' : ''}`) : ''}
    <div class="bars" aria-label="Spot density by zone">${bars}</div>
    ${finds || stateBox('No clear regional pattern', 'Too few spots were found, or they are spread evenly across the face.')}
    ${gagsHtml(rep)}
    ${skinOutlookHtml(rep)}
    ${simulatedHtml(rep)}
    ${accuracyHtml(rep)}
    ${previews.length ? `<div class="preview-grid">${previews.map(p => `<figure><img src="${p.image}" alt="${esc(p.view)} photo with face zones and spots marked"><figcaption>${esc(p.view)} view · ${p.spots} spots marked</figcaption></figure>`).join('')}</div>` : ''}</div>`;
}
function modal(title, bodyHtml, actionsHtml = '') {
  const el = document.createElement('div'); el.className = 'modal-backdrop'; el.innerHTML = `<div class="modal" role="dialog" aria-modal="true" aria-label="${esc(title)}"><div class="modal-head"><h2>${esc(title)}</h2><button class="icon-btn" aria-label="Close" data-close>×</button></div><div class="modal-body">${bodyHtml}</div><div class="modal-foot">${actionsHtml}</div></div>`;
  document.body.append(el); const close = () => el.remove(); el.addEventListener('click', e => { if (e.target === el || e.target.closest('[data-close]')) close(); }); return { el, close };
}
async function fileToDataUrl(file, max = 2000) {
  const bmp = await createImageBitmap(file); const k = Math.min(1, max / Math.max(bmp.width, bmp.height)); const c = document.createElement('canvas');
  c.width = Math.round(bmp.width * k); c.height = Math.round(bmp.height * k); c.getContext('2d').drawImage(bmp, 0, 0, c.width, c.height); return c.toDataURL('image/jpeg', 0.9);
}

/* ---------------- intake form ---------------- */
function intakeForm(initial, onSaved) {
  const v = initial || {}, L = v.lifestyle || {}, has = k => (v.conditions || []).includes(k);
  const sel = (id, opts, cur) => `<select class="select" id="${id}">${opts.map(o => `<option ${o === cur ? 'selected' : ''}>${o}</option>`).join('')}</select>`;
  const html = `<form id="intake" class="stack" novalidate>
    <div class="form-section">About you</div><div class="form-grid">
      <div class="field"><label for="h">Height (cm)</label><input class="input" id="h" type="number" min="50" max="250" value="${esc(v.height_cm ?? '')}"></div>
      <div class="field"><label for="w">Weight (kg)</label><input class="input" id="w" type="number" min="10" max="300" step="0.1" value="${esc(v.weight_kg ?? '')}"></div>
      <div class="field"><label for="a">Age</label><input class="input" id="a" type="number" min="1" max="120" value="${esc(v.age ?? '')}"></div>
      <div class="field"><label for="sx">Sex</label>${sel('sx', ['', 'Female', 'Male', 'Other'], v.sex || '')}</div></div>
    <div class="form-section">Health history: do you have, or have you had, any issue with:</div>
    <div class="cond-grid">${Object.entries(SYSTEMS).map(([k, l]) => `<label class="choice"><input type="checkbox" name="cond" value="${k}" ${has(k) ? 'checked' : ''}> ${esc(l)}</label>`).join('')}</div>
    <div class="form-grid"><div class="field span-2"><label for="cn">Tell us more about any of the above</label><textarea class="textarea" id="cn">${esc(v.condition_notes || '')}</textarea></div>
      <div class="field"><label for="md">Current medicines</label><input class="input" id="md" value="${esc(v.medications || '')}"></div>
      <div class="field"><label for="al">Allergies</label><input class="input" id="al" value="${esc(v.allergies || '')}"></div>
      <div class="field span-2"><label for="fh">Family history (diabetes, thyroid, PCOS, digestive conditions...)</label><input class="input" id="fh" value="${esc(v.family_history || '')}"></div></div>
    <div class="form-section">Lifestyle</div><div class="form-grid">
      <div class="field"><label for="sl">Sleep (hours a night)</label><input class="input" id="sl" type="number" min="0" max="16" step="0.5" value="${esc(L.sleep_hours ?? '')}"></div>
      <div class="field"><label for="dt">Diet</label>${sel('dt', ['', 'Vegetarian', 'Vegan', 'Eggetarian', 'Non-vegetarian', 'Mixed'], L.diet || '')}</div>
      <div class="field"><label for="ex">Exercise</label>${sel('ex', ['', 'None', '1-2 days a week', '3-5 days a week', 'Daily'], L.exercise || '')}</div>
      <div class="field"><label for="st">Stress level</label>${sel('st', ['', 'Low', 'Moderate', 'High'], L.stress || '')}</div>
      <div class="field"><label for="sm">Smoking</label>${sel('sm', ['', 'No', 'Occasionally', 'Regularly'], L.smoking || '')}</div>
      <div class="field"><label for="ac">Alcohol</label>${sel('ac', ['', 'No', 'Occasionally', 'Regularly'], L.alcohol || '')}</div>
      <div class="field span-2"><label for="sk">Skin and hair care (products, hats or helmets, anything you changed recently)</label><textarea class="textarea" id="sk">${esc(v.skin_notes || '')}</textarea></div></div>
    <div id="ierr"></div><div class="form-actions"><button class="btn btn-primary" type="submit">Save and continue</button></div></form>`;
  const mount = root => {
    $('#intake', root).onsubmit = async ev => {
      ev.preventDefault(); const num = id => { const x = $('#' + id, root).value; return x === '' ? null : Number(x); }; const txt = id => $('#' + id, root).value.trim();
      const body = { height_cm: num('h'), weight_kg: num('w'), age: num('a'), sex: txt('sx'), conditions: [...root.querySelectorAll('input[name=cond]:checked')].map(i => i.value), condition_notes: txt('cn'),
        medications: txt('md'), allergies: txt('al'), family_history: txt('fh'), skin_notes: txt('sk'), lifestyle: { sleep_hours: num('sl'), diet: txt('dt'), exercise: txt('ex'), stress: txt('st'), smoking: txt('sm'), alcohol: txt('ac') } };
      try { await api('/patient/intake', { method: 'PUT', body }); toast('Health form saved', 'success'); onSaved(); } catch (e) { $('#ierr', root).innerHTML = alertBox('error', esc(e.message), 'alert'); }
    };
  };
  return { html, mount };
}

/* ---------------- face scan ---------------- */
function scanPanel(onDone, { canBook = true, requireAll = false } = {}) {
  const html = `<div class="stack"><p class="muted">Use a clear, even-light photo with your whole face visible. ${requireAll ? 'Your first scan needs <strong>all three angles</strong>: front, left side and right side, so every part of your face is covered.' : 'A front photo is required; left and right side photos make the result more accurate.'}
    Your photos are stored securely and are visible only to you and the doctors you have been under. Administrators cannot see them.</p>
    <div class="photo-slots">${[['front', 'Front (required)'], ['left', requireAll ? 'Left side (required)' : 'Left side (recommended)'], ['right', requireAll ? 'Right side (required)' : 'Right side (recommended)']].map(([k, l]) => `<div class="slot" data-slot="${k}"><strong>${l}</strong><div class="thumb">No photo</div>
      <div class="row"><label class="btn btn-secondary btn-sm">Choose<input type="file" accept="image/*" hidden></label><button class="btn btn-ghost btn-sm" type="button" data-cam="${k}">${icon('cam', 'sm')} Camera</button></div></div>`).join('')}</div>
    <div id="serr"></div><div class="row"><button class="btn btn-primary" id="analyse" disabled>Analyse my face</button><span id="busy" class="scan-busy" hidden><span class="spinner"></span> Analysing, this takes a few seconds...</span></div><div id="sres"></div></div>`;
  const mount = root => {
    const photos = {}; const check = () => { $('#analyse', root).disabled = requireAll ? !(photos.front && photos.left && photos.right) : !photos.front; };
    const setPhoto = (k, url) => { photos[k] = url; const s = $(`[data-slot=${k}]`, root); s.classList.add('filled'); const t = s.querySelector('.thumb'); t.style.backgroundImage = `url(${url})`; t.textContent = ''; check(); };
    root.querySelectorAll('[data-slot]').forEach(s => s.querySelector('input').onchange = async e => { const f = e.target.files[0]; if (f) { try { setPhoto(s.dataset.slot, await fileToDataUrl(f)); } catch { toast('That file could not be opened as a photo', 'error'); } } });
    root.querySelectorAll('[data-cam]').forEach(b => b.onclick = async () => {
      let stream; try { stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } }); } catch { toast('Camera not available. Use Choose to upload a photo.', 'error'); return; }
      const m = modal('Take a photo', '<div class="cam"><video autoplay playsinline muted></video></div>', '<button class="btn btn-secondary" data-close>Cancel</button><button class="btn btn-primary" id="snap">Capture</button>');
      const vid = m.el.querySelector('video'); vid.srcObject = stream; const stop = () => stream.getTracks().forEach(t => t.stop()); m.el.addEventListener('click', e => { if (e.target === m.el || e.target.closest('[data-close]')) stop(); });
      m.el.querySelector('#snap').onclick = () => { const c = document.createElement('canvas'); c.width = vid.videoWidth; c.height = vid.videoHeight; c.getContext('2d').drawImage(vid, 0, 0); setPhoto(b.dataset.cam, c.toDataURL('image/jpeg', 0.9)); stop(); m.close(); };
    });
    $('#analyse', root).onclick = async () => {
      $('#serr', root).innerHTML = ''; $('#sres', root).innerHTML = ''; $('#analyse', root).disabled = true; $('#busy', root).hidden = false;
      try { const r = await api('/patient/scan', { method: 'POST', body: photos }); $('#sres', root).innerHTML = `<hr>${reportHtml(r.report, r.previews, { canBook })}`; me = await api('/me'); const b = $('#scanbanner'); if (b) b.remove(); if (onDone) onDone(r); }
      catch (e) { $('#serr', root).innerHTML = alertBox('error', esc(e.message), 'alert'); }
      finally { $('#busy', root).hidden = true; check(); }
    };
  };
  return { html, mount };
}

/* ---------------- doctor picker (first choice and switch) ---------------- */
async function doctorPicker(mode, onDone) {
  const rec = await api('/patient/recommended-doctors'); const cur = me.onboarding.doctor_id; const list = mode === 'switch' ? rec.filter(d => d.id !== cur) : rec;
  const html = `<div class="stack"><p class="muted">${mode === 'switch' ? 'Choose a different doctor. Your previous doctor keeps read access to your record, and the new doctor sees your full history, including earlier notes. Upcoming bookings with your previous doctor are cancelled.' : 'Doctors are ranked by how well their focus matches your scan and health form. You can pick any doctor.'}</p>
    <div class="picker"><ul class="picker-list" role="radiogroup" aria-label="Doctors">${list.map((d, i) => `<li class="picker-item ${i === 0 && d.recommended ? 'is-selected' : ''}"><input type="radio" name="doc" value="${d.id}" id="d${d.id}" ${i === 0 && d.recommended ? 'checked' : ''}>
      <label for="d${d.id}" style="flex:1;cursor:pointer"><strong>${esc(d.name)}</strong> ${d.recommended && !d.generalist ? '<span class="badge success">Recommended</span>' : d.generalist ? '<span class="badge neutral">General</span>' : ''}<br><span class="muted small">${esc(d.specialization)}</span>
      ${d.why.length ? `<br><span class="small">Matches your: ${d.why.map(esc).join(', ')}</span>` : ''}</label></li>`).join('')}</ul></div>
    ${mode === 'switch' ? '<div class="field"><label for="why">Why are you switching? (shared with both doctors)</label><input class="input" id="why" placeholder="e.g. new hormonal concern"></div>' : ''}
    <div id="derr"></div><div class="form-actions"><button class="btn btn-primary" id="pick">${mode === 'switch' ? 'Switch doctor' : 'Choose this doctor'}</button></div></div>`;
  const mount = root => {
    root.querySelectorAll('.picker-item').forEach(li => li.onclick = () => { li.querySelector('input').checked = true; root.querySelectorAll('.picker-item').forEach(x => x.classList.toggle('is-selected', x === li)); });
    $('#pick', root).onclick = async () => {
      const sel = root.querySelector('input[name=doc]:checked'); if (!sel) { $('#derr', root).innerHTML = alertBox('error', 'Please choose a doctor.', 'alert'); return; }
      try { await api(mode === 'switch' ? '/patient/switch-doctor' : '/patient/doctor', { method: 'POST', body: mode === 'switch' ? { doctor_id: +sel.value, reason: ($('#why', root) || {}).value || '' } : { doctor_id: +sel.value } }); toast(mode === 'switch' ? 'Doctor changed' : 'Doctor chosen', 'success'); onDone(); }
      catch (e) { $('#derr', root).innerHTML = alertBox('error', esc(e.message), 'alert'); }
    };
  };
  return { html, mount };
}


/* ---------------- progress charts (inline SVG, no library) ---------------- */
const PALETTE = ['#325385', '#ed7639', '#1f6b24', '#a82e22', '#6b3fa0'];
function lineChart(series, labels, label) {
  const W = 720, H = 250, m = { l: 44, r: 16, t: 14, b: 38 }, n = labels.length; const max = Math.max(1, ...series.flatMap(s => s.values)) * 1.15;
  const x = i => m.l + (n === 1 ? (W - m.l - m.r) / 2 : i * (W - m.l - m.r) / (n - 1)), y = v => H - m.b - v / max * (H - m.t - m.b);
  const grid = [0, 1, 2, 3, 4].map(k => { const v = max * k / 4, yy = y(v); return `<line x1="${m.l}" x2="${W - m.r}" y1="${yy}" y2="${yy}" stroke="#ececec"/><text x="${m.l - 8}" y="${yy + 4}" font-size="11" text-anchor="end" fill="#6b6b6b">${Math.round(v)}</text>`; }).join('');
  const xl = labels.map((l, i) => (n <= 8 || i % Math.ceil(n / 8) === 0 || i === n - 1) ? `<text x="${x(i)}" y="${H - 14}" font-size="11" text-anchor="middle" fill="#6b6b6b">${esc(l)}</text>` : '').join('');
  const lines = series.map(s => `<polyline fill="none" stroke="${s.color}" stroke-width="2.5" points="${s.values.map((v, i) => `${x(i)},${y(v)}`).join(' ')}"/>${s.values.map((v, i) => `<circle cx="${x(i)}" cy="${y(v)}" r="4" fill="${s.color}"><title>${esc(s.name)}: ${v} on ${esc(labels[i])}</title></circle>`).join('')}`).join('');
  return `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto" role="img" aria-label="${esc(label)}">${grid}${xl}${lines}</svg>`;
}
function progressHtml(pr, { patient = false } = {}) {
  if (!pr.n) return stateBox('No scans yet', 'Progress appears here after the first scan.');
  const S = pr.series, labels = S.map(p => D(p.date).toLocaleDateString('en-IN', { day: 'numeric', month: 'short' }));
  const statusBadgeS = { on_track: '<span class="badge success">On track</span>', due: '<span class="badge progress">Due now</span>', overdue: '<span class="badge danger">Overdue</span>' }[pr.status];
  const ch = pr.change_vs_first_pct, chTxt = ch === null ? '-' : (ch > 0 ? '+' : '') + ch + '%';
  const zoneSeries = Object.keys(ZONE_LABEL).map((z, i) => ({ name: ZONE_LABEL[z], color: PALETTE[i], values: S.map(p => (p.zones || {})[z] || 0) }));
  const rows = S.slice().reverse().map((p, i, arr) => { const prev = arr[i + 1]; const d = prev && prev.total ? Math.round(100 * (p.total - prev.total) / prev.total) : null;
    const top = Object.entries(p.share && Object.keys(p.share).length ? p.share : p.zones || {}).sort((a, b) => b[1] - a[1])[0];
    return `<tr><td class="cell-title">${esc(fmtDate(p.date))}</td><td class="num">${p.total}</td><td class="num">${d === null ? '-' : (d > 0 ? '+' : '') + d + '%'}</td><td>${top ? esc(ZONE_LABEL[top[0]] || top[0]) : '-'}</td></tr>`; }).join('');
  const wk = pr.adherence.weeks_with_scan;
  return `<div class="stack"><div class="grid cols-4">${tile('Scans taken', pr.n)}${tile('Last scan', esc(fmtDate(pr.series[S.length - 1].date)), pr.days_since + ' days ago', 'sm')}
      ${tile('Next scan due', esc(fmtDate(pr.next_due)), '', 'sm')}${tile('Change since first scan', chTxt, ch === null ? 'Need two scans to compare' : ch < 0 ? 'fewer spots found' : ch > 0 ? 'more spots found' : 'no change', ch > 0 ? 'warn' : '')}</div>
    <div class="row">${statusBadgeS} <span class="muted small">Weekly streak: <strong>${pr.streak_weeks}</strong> week${pr.streak_weeks === 1 ? '' : 's'}</span></div>
    ${card('Total spots found over time', S.length > 1 ? lineChart([{ name: 'Total spots', color: PALETTE[0], values: S.map(p => p.total) }], labels, 'Total spots found per scan') : `<p class="muted">One scan so far. ${patient ? 'Take another next week to start seeing a trend.' : ''}</p>${lineChart([{ name: 'Total spots', color: PALETTE[0], values: S.map(p => p.total) }], labels, 'Total spots found per scan')}`)}
    ${card('Spots by face zone', lineChart(zoneSeries, labels, 'Spots found per zone per scan') + `<ul class="legend" style="grid-template-columns:repeat(auto-fit,minmax(170px,1fr))">${zoneSeries.map(z => `<li><span class="swatch" style="background:${z.color}"></span>${esc(z.name)}</li>`).join('')}</ul>`)}
    ${card('Scanning habit', `<div class="bars"><div class="bar-row"><span>Weeks with a scan (last 8)</span><div class="bar-track"><div class="bar-fill" style="--w:${Math.round(100 * wk / 8)}%"></div></div><span>${wk}/8</span></div></div>`)}
    ${card('Every scan', `<div class="table-wrap"><table class="table"><thead><tr><th>Date</th><th class="num">Spots</th><th class="num">vs previous</th><th>Most affected zone</th></tr></thead><tbody>${rows}</tbody></table></div>`)}
    ${alertBox('info', 'Spot counts depend on lighting, distance and camera. For a fair comparison, scan in similar light, at a similar distance, without make-up. Treat changes as a guide and talk to your doctor about what they mean.')}</div>`;
}
async function pProgress() {
  const pr = await api('/patient/progress');
  const rem = card('Email reminders', `<label class="choice"><input type="checkbox" id="remtoggle" ${pr.email_reminders ? 'checked' : ''}> Email me a weekly scan reminder and a reminder before each appointment</label>
    <p class="muted small" style="margin-top:8px">Reminder emails never include your health findings. ${pr.email_live ? '' : 'Email sending is in <strong>test mode</strong> on this server: reminders are recorded but not actually sent.'}</p>`);
  return { title: 'My progress', sub: 'How your face scans change over time', actions: '<a class="btn btn-primary" href="#/scan">New face scan</a>', html: `<div class="stack">${progressHtml(pr, { patient: true })}${rem}</div>`,
    mount(r) { $('#remtoggle', r).onchange = async e => { try { await api('/patient/reminders', { method: 'PUT', body: { enabled: e.target.checked } }); toast(e.target.checked ? 'Reminders on' : 'Reminders off', 'success'); } catch (er) { toast(er.message, 'error'); } }; } };
}
async function aEmails() {
  const d = await api('/admin/emails'); const st = { sent: 'success', dry_run: 'info', failed: 'danger' };
  return { title: 'Email log', sub: d.live ? 'Live email sending is on' : 'Test mode: emails are recorded here but not sent', actions: '<button class="btn btn-secondary" id="runrem">Run reminders now</button>',
    html: card('Recent emails', d.rows.length ? `<div class="table-wrap"><table class="table"><thead><tr><th>When</th><th>To</th><th>Subject</th><th>Type</th><th>Status</th></tr></thead><tbody>${d.rows.map(e => `<tr><td>${esc(fmtDT(e.created.replace('T', ' ')))}</td><td>${esc(e.to_addr)}</td><td>${esc(e.subject)}</td><td>${esc(e.kind)}</td><td><span class="badge ${st[e.status] || 'neutral'}">${esc(e.status)}</span>${e.error ? `<div class="cell-sub">${esc(e.error)}</div>` : ''}</td></tr>`).join('')}</tbody></table></div>` : stateBox('No emails yet', 'Booking an appointment or running reminders will add rows here.')),
    mount(r) { $('#runrem').onclick = async () => { const x = await api('/admin/run-reminders', { method: 'POST' }); toast(`${x.processed.length} reminder(s) processed`, 'success'); route(); }; } };
}

/* ---------------- patient ---------------- */
async function wizard() {
  const ob = me.onboarding; const step = ob.needs_intake ? 0 : ob.needs_scan ? 1 : 2;
  const names = ['Health form', 'Face scan', 'Choose your doctor']; const steps = `<ol class="stepper">${names.map((n, i) => `<li class="${i < step ? 'done' : i > step ? 'todo' : ''}" ${i === step ? 'aria-current="step"' : ''}><span class="n">${i < step ? '✓' : i + 1}</span>${n}</li>`).join('')}</ol>`;
  const next = async () => { me = await api('/me'); route(); };
  let body, part;
  if (step === 0) { part = intakeForm(null, next); body = card('Tell us about yourself', part.html); }
  else if (step === 1) {
    part = scanPanel(() => { $('#contwrap').hidden = false; }, { canBook: false, requireAll: true });
    body = card('Your first face scan', part.html + '<div class="form-actions" id="contwrap" hidden><button class="btn btn-primary" id="cont">Continue: choose your doctor</button></div>', '<span class="badge info">Step 2 of 3</span>');
  }
  else { part = await doctorPicker('first', next); body = card('Choose your doctor', part.html); }
  shell(false, '', `<div class="page" style="max-width:920px">${header(`Welcome, ${me.name.split(' ')[0]}`, 'Three quick steps to set up your account.')}${steps}${body}</div>`);
  const root = $('#main'); part.mount(root);
  if (step === 1) $('#cont', root).onclick = next;
}
async function pHome() {
  const [appts, scans, docs] = await Promise.all([api('/patient/appointments'), api('/patient/scans'), api('/doctors')]); const doc = docs.find(d => d.id === me.onboarding.doctor_id);
  const next = appts.filter(a => a.status === 'booked').sort((a, b) => a.start_ts.localeCompare(b.start_ts))[0]; const last = scans[0]; const lastNote = appts.find(a => a.note);
  return { title: `Hello, ${me.name.split(' ')[0]}`, sub: 'Your face wellness overview', actions: '<a class="btn btn-primary" href="#/scan">New face scan</a><a class="btn btn-secondary" href="#/appointments">Book appointment</a>',
    html: `<div class="grid cols-4">${tile('Your doctor', esc(doc ? doc.name : '-'), doc ? doc.specialization : '', 'sm')}${tile('Next appointment', next ? esc(fmtDT(next.start_ts)) : 'None', next ? 'with ' + next.doctor : 'Book one when you like', 'sm')}${tile('Last scan', last ? esc(fmtDate(last.created)) : 'None', '', 'sm')}${tile('Spots found', last ? last.report.total_spots : '-', last ? 'in your last scan' : '')}</div>
    <div class="stack" style="margin-top:24px">${lastNote ? card('Latest note from your doctor', `<div class="note-box">${esc(lastNote.note)}</div><p class="muted small" style="margin-top:8px">${esc(lastNote.doctor)} · ${esc(fmtDate(lastNote.start_ts))}</p>`) : ''}
    ${last ? card('Your latest scan', reportHtml(last.report, [], { canBook: true }) + `<hr><h3 style="margin-bottom:12px">Photos from this scan</h3>${photoGallery(last.id, last.photos)}`) : card('No scan yet', stateBox('Take your first scan', 'It takes a minute.'))}</div>` };
}
async function pScan() {
  const part = scanPanel(null); const scans = await api('/patient/scans');
  return { title: 'Face scan', sub: 'Find where spots are concentrated on your face', html: card('New scan', part.html) + (scans.length ? `<div style="margin-top:24px">${card('Past scans', `<table class="table"><thead><tr><th>Date</th><th>Spots</th><th>Summary</th><th>Photos</th></tr></thead><tbody>${scans.map(s => `<tr><td>${esc(fmtDate(s.created))}</td><td>${s.report.total_spots}</td><td>${esc(s.report.summary)}</td><td>${s.photos.length ? `<button class="btn btn-secondary btn-sm" data-scan-photos="${s.id}" data-views="${s.photos.join(',')}" data-date="${esc(fmtDate(s.created))}">View photos</button>` : '<span class="muted">-</span>'}</td></tr>`).join('')}</tbody></table>`)}</div>` : ''), mount: r => part.mount(r) };
}
async function pAppointments() {
  const [slots, appts] = await Promise.all([api('/slots?days=28'), api('/patient/appointments')]); const days = Object.keys(slots);
  const rows = appts.map(a => `<tr><td class="cell-title">${esc(fmtDT(a.start_ts))}<div class="cell-sub">${esc(a.doctor)}</div></td><td>${statusBadge(a.display_status || a.status)}${movedBadge(a)}</td><td>${esc(a.reason || '')}</td><td>${a.note ? `<div class="note-box">${esc(a.note)}</div>` : '<span class="muted">-</span>'}</td>
    <td class="actions">${reschedBtn(a, 'patient')} ${upcoming(a) ? `<button class="btn btn-secondary btn-sm" data-cancel="${a.id}">Cancel</button>` : ''}</td></tr>`).join('');
  return { title: 'Appointments', sub: 'Book a free slot with your doctor', html: `<div class="stack">${card('Book a slot', days.length ? `<div class="slot-days" id="days">${days.map((d, i) => `<button class="slot-btn" data-day="${d}" aria-pressed="${i === 0}">${esc(fmtDate(d + ' 00:00'))}</button>`).join('')}</div><div class="slot-times" id="times"></div>
      <div class="field" style="margin-top:16px;max-width:520px"><label for="rs">What would you like to discuss?</label><input class="input" id="rs" placeholder="e.g. spots on my forehead"></div><div class="form-actions" style="justify-content:flex-start"><button class="btn btn-primary" id="book" disabled>Book selected slot</button></div><div id="berr"></div>` : stateBox('No free slots', 'Your doctor has no open slots in the next 4 weeks. Please check back later.'))}
    ${card('Your appointments', appts.length ? `<div class="table-wrap"><table class="table"><thead><tr><th>When</th><th>Status</th><th>Reason</th><th>Doctor's notes</th><th></th></tr></thead><tbody>${rows}</tbody></table></div>` : stateBox('No appointments yet', 'Booked appointments and your doctor\'s notes will appear here.'))}</div>`,
    mount(r) {
      let day = days[0], time = null; const draw = () => { $('#times', r) && ($('#times', r).innerHTML = (slots[day] || []).map(t => `<button class="slot-btn" data-t="${t}" aria-pressed="${t === time}">${t}</button>`).join('')); $('#book', r) && ($('#book', r).disabled = !time); }; draw();
      r.addEventListener('click', async e => {
        const d = e.target.closest('[data-day]'), t = e.target.closest('[data-t]'), c = e.target.closest('[data-cancel]');
        if (d) { day = d.dataset.day; time = null; r.querySelectorAll('[data-day]').forEach(x => x.setAttribute('aria-pressed', x === d)); draw(); }
        else if (t) { time = t.dataset.t; draw(); }
        else if (c) { if (confirm('Cancel this appointment?')) { await api('/appointments/' + c.dataset.cancel, { method: 'DELETE' }); toast('Appointment cancelled', 'success'); route(); } }
        else if (e.target.closest('#book')) { try { await api('/appointments', { method: 'POST', body: { start: `${day} ${time}`, reason: $('#rs', r).value } }); toast('Appointment booked', 'success'); route(); } catch (er) { $('#berr', r).innerHTML = alertBox('error', esc(er.message), 'alert'); route(); } }
      });
    } };
}
async function pDoctor() {
  const [docs, hist] = await Promise.all([api('/doctors'), api('/patient/doctors-history')]); const doc = docs.find(d => d.id === me.onboarding.doctor_id); const picker = await doctorPicker('switch', async () => { me = await api('/me'); route(); });
  return { title: 'My doctor', sub: 'Your current doctor and doctor history', html: `<div class="stack">${card('Current doctor', `<dl class="kv"><dt>Name</dt><dd>${esc(doc.name)}</dd><dt>Focus</dt><dd>${esc(doc.specialization)}</dd></dl>`)}
    ${card('Doctor history', `<table class="table"><thead><tr><th>Doctor</th><th>From</th><th>To</th><th>Reason</th></tr></thead><tbody>${hist.map(h => `<tr><td class="cell-title">${esc(h.doctor)}</td><td>${esc(fmtDate(h.start_ts))}</td><td>${h.end_ts ? esc(fmtDate(h.end_ts)) : '<span class="badge success">Current</span>'}</td><td>${esc(h.reason || '')}</td></tr>`).join('')}</tbody></table>`)}
    ${card('Switch to another doctor', picker.html)}</div>`, mount: r => picker.mount(r) };
}
async function pIntake() {
  const cur = await api('/patient/intake'); const part = intakeForm(cur, () => { location.hash = '#/home'; });
  return { title: 'My health form', sub: 'Keep this up to date: it shapes your insights and your doctor sees it', html: card('Health form', part.html), mount: r => part.mount(r) };
}

/* ---------------- doctor / admin shared record view ---------------- */
function scanSections(p) {
  if (p.scan_summary) return card('Face scans', `<p>${p.scan_summary.count} scan${p.scan_summary.count === 1 ? '' : 's'}${p.scan_summary.last ? `, the latest on ${esc(fmtDate(p.scan_summary.last))}` : ''}.</p>
    <div class="alert alert-info">${icon('info')}<div>Face photos and scan results are private to the patient and the doctors they have been under. Administrators cannot view them.</div></div>`);
  if (!p.scans || !p.scans.length) return card('Face scan', stateBox('No scan yet', ''));
  const photos = p.scans.map((s, i) => `<details class="outlook" style="margin-bottom:10px" ${i === 0 ? 'open' : ''}><summary><strong>${esc(fmtDate(s.created))}</strong> · ${s.report.total_spots} spots · ${s.photos.length} photo${s.photos.length === 1 ? '' : 's'}</summary><div style="margin-top:12px">${photoGallery(s.id, s.photos)}</div></details>`).join('');
  return card(`Latest face scan · ${fmtDate(p.scans[0].created)}`, reportHtml(p.scans[0].report)) + card('Face photos (every scan, newest first)', photos);
}
function recordHtml(p, { canNote = false } = {}) {
  const it = p.intake, L = (it && it.lifestyle) || {}; const conds = it ? (it.conditions || []).map(k => SYSTEMS[k] || k) : [];
  const kv = it ? `<dl class="kv"><dt>Height / weight</dt><dd>${it.height_cm || '-'} cm / ${it.weight_kg || '-'} kg ${it.bmi ? `(BMI ${it.bmi})` : ''}</dd><dt>Age / sex</dt><dd>${it.age || '-'} / ${esc(it.sex || '-')}</dd>
    <dt>Reported issues</dt><dd>${conds.length ? conds.map(esc).join(', ') : 'None reported'}</dd>${it.condition_notes ? `<dt>Details</dt><dd>${esc(it.condition_notes)}</dd>` : ''}<dt>Medicines</dt><dd>${esc(it.medications || '-')}</dd><dt>Allergies</dt><dd>${esc(it.allergies || '-')}</dd>
    <dt>Family history</dt><dd>${esc(it.family_history || '-')}</dd><dt>Lifestyle</dt><dd>${esc(Object.entries(L).filter(([, x]) => x !== '' && x !== null).map(([k, x]) => `${k.replace('_', ' ')}: ${x}`).join(' · ') || '-')}</dd></dl>` : stateBox('No health form yet', 'The patient has not filled it in.');
  const appts = (p.appointments || []).map(a => `<tr><td class="cell-title">${esc(fmtDT(a.start_ts))}<div class="cell-sub">${esc(a.doctor || '')}</div></td><td>${statusBadge(a.display_status || a.status)}${movedBadge(a)}</td><td>${esc(a.reason || '')}</td><td>${a.note ? `<div class="note-box">${esc(a.note)}</div>` : '<span class="muted">-</span>'}</td>
    <td class="actions">${canNote && a.doctor === me.name ? reschedBtn(a, 'doctor') : ''} ${canNote && a.doctor === me.name && a.status !== 'cancelled' ? `<button class="btn btn-secondary btn-sm" data-note="${a.id}" data-text="${esc(a.note || '')}">${a.note ? 'Edit note' : 'Add note'}</button>` : ''}</td></tr>`).join('');
  return `<div class="stack">${p.progress && p.progress.n ? card('Scan progress', progressHtml(p.progress)) : ''}${card('Health form', kv)}${scanSections(p)}
    ${card('Doctors', `<table class="table"><thead><tr><th>Doctor</th><th>From</th><th>To</th><th>Reason</th></tr></thead><tbody>${(p.assignments || []).map(h => `<tr><td class="cell-title">${esc(h.doctor)}</td><td>${esc(fmtDate(h.start_ts))}</td><td>${h.end_ts ? esc(fmtDate(h.end_ts)) : '<span class="badge success">Current</span>'}</td><td>${esc(h.reason || '')}</td></tr>`).join('')}</tbody></table>`)}
    ${card('Appointments and notes (all doctors)', appts ? `<div class="table-wrap"><table class="table"><thead><tr><th>When / doctor</th><th>Status</th><th>Reason</th><th>Note</th><th></th></tr></thead><tbody>${appts}</tbody></table></div>` : stateBox('No appointments', ''))}</div>`;
}
function noteHandlers(r, reload) {
  r.addEventListener('click', e => {
    const b = e.target.closest('[data-note]'); if (!b) return; const m = modal('Summary note for the patient', `<p class="muted small">The patient can read this note.</p><textarea class="textarea" id="nt" rows="6">${esc(b.dataset.text)}</textarea><div id="nerr"></div>`, '<button class="btn btn-secondary" data-close>Cancel</button><button class="btn btn-primary" id="savenote">Save note</button>');
    m.el.querySelector('#savenote').onclick = async () => { const s = m.el.querySelector('#nt').value.trim(); if (!s) return; try { await api(`/doctor/appointments/${b.dataset.note}/note`, { method: 'POST', body: { summary: s } }); m.close(); toast('Note saved', 'success'); reload(); } catch (er) { m.el.querySelector('#nerr').innerHTML = alertBox('error', esc(er.message), 'alert'); } };
  });
}

/* ---------------- doctor ---------------- */
async function dHome() {
  const s = await api('/doctor/summary'); const today = ymd(new Date()); const tCount = s.upcoming.filter(a => a.start_ts.startsWith(today)).length;
  const pr = s.patients.map(p => `<tr><td class="cell-title">${esc(p.name)}<div class="cell-sub">${esc(p.email)}</div></td><td>${p.intake ? '<span class="badge success">Done</span>' : '<span class="badge progress">Pending</span>'}</td>
    <td>${p.last_scan ? esc(p.last_scan.summary) + (p.last_scan.suggest_appointment ? ' <span class="badge accent">Review suggested</span>' : '') : '<span class="muted">No scan</span>'}</td><td>${p.next_appointment ? esc(fmtDT(p.next_appointment)) : '-'}</td><td class="actions"><a class="btn btn-secondary btn-sm" href="#/patient/${p.id}">Open record</a></td></tr>`).join('');
  const up = s.upcoming.map(a => `<tr><td class="cell-title">${esc(fmtDT(a.start_ts))}${movedBadge(a)}</td><td><a href="#/patient/${a.patient_id}">${esc(a.patient)}</a></td><td>${esc(a.reason || '')}</td><td class="actions">${reschedBtn(a, 'doctor')}</td></tr>`).join('');
  return { title: 'Dashboard', sub: 'Your patients and schedule', html: `<div class="grid cols-4">${tile('My patients', s.patients.length)}${tile('Upcoming appointments', s.upcoming.length)}${tile('Today', tCount)}${tile('Former patients', s.former_patients.length, 'read-only access')}</div>
    <div class="stack" style="margin-top:24px">${card('Upcoming appointments', up ? `<div class="table-wrap"><table class="table"><thead><tr><th>When</th><th>Patient</th><th>Reason</th><th></th></tr></thead><tbody>${up}</tbody></table></div>` : stateBox('Nothing scheduled', ''))}
    ${card('My patients', pr ? `<div class="table-wrap"><table class="table"><thead><tr><th>Patient</th><th>Health form</th><th>Latest scan</th><th>Next visit</th><th></th></tr></thead><tbody>${pr}</tbody></table></div>` : stateBox('No patients yet', 'Patients appear here once they choose you.'))}
    ${s.former_patients.length ? card('Former patients (moved to another doctor)', `<ul class="legend">${s.former_patients.map(f => `<li><a href="#/patient/${f.id}">${esc(f.name)}</a><span class="count muted small">left ${esc(fmtDate(f.left_on))}</span></li>`).join('')}</ul>`) : ''}</div>` };
}
async function dPatient(id) {
  const p = await api('/doctor/patients/' + id);
  return { title: p.name, sub: p.is_current_doctor ? 'Current patient' : 'Former patient: you have read-only access to this record', crumb: '<a href="#/home">Dashboard</a> / Patient', html: recordHtml(p, { canNote: true }), mount: r => noteHandlers(r, route) };
}
async function dAppointments() {
  const s = await api('/doctor/summary'); const row = a => `<tr><td class="cell-title">${esc(fmtDT(a.start_ts))}</td><td><a href="#/patient/${a.patient_id}">${esc(a.patient)}</a></td><td>${statusBadge(a.display_status || a.status)}${movedBadge(a)}</td><td>${esc(a.reason || '')}</td><td>${a.note ? `<div class="note-box">${esc(a.note)}</div>` : '-'}</td>
    <td class="actions">${reschedBtn(a, 'doctor')} <button class="btn btn-secondary btn-sm" data-note="${a.id}" data-text="${esc(a.note || '')}">${a.note ? 'Edit note' : 'Add note'}</button></td></tr>`;
  const tbl = rows => rows.length ? `<div class="table-wrap"><table class="table"><thead><tr><th>When</th><th>Patient</th><th>Status</th><th>Reason</th><th>Note</th><th></th></tr></thead><tbody>${rows.map(row).join('')}</tbody></table></div>` : stateBox('None', '');
  return { title: 'Appointments', sub: 'Upcoming and previous visits', html: `<div class="stack">${card('Upcoming', tbl(s.upcoming))}${card('Previous', tbl(s.previous))}</div>`, mount: r => noteHandlers(r, route) };
}
async function dAvailability() {
  const rows = await api('/doctor/availability'); const by = d => rows.filter(r => r.weekday === d);
  const win = (d, w) => `<div class="win" data-d="${d}"><input class="input" type="time" value="${w.start}" data-f="start" aria-label="Start"> to <input class="input" type="time" value="${w.end}" data-f="end" aria-label="End">
    <select class="select" data-f="slot" aria-label="Slot length">${[10, 15, 20, 30, 45, 60].map(m => `<option value="${m}" ${m === w.slot_minutes ? 'selected' : ''}>${m} min</option>`).join('')}</select><button class="btn btn-ghost btn-sm" data-rm type="button">Remove</button></div>`;
  return { title: 'Availability', sub: 'Your weekly chart: patients can only book inside these hours', html: card('Weekly hours', `<div class="stack">${DAYS.map((n, d) => `<div data-day="${d}"><strong>${n}</strong>${by(d).length ? '' : ' <span class="muted small">Not available</span>'}<div class="wins">${by(d).map(w => win(d, w)).join('')}</div><button class="btn btn-ghost btn-sm" data-add="${d}" type="button">${icon('plus', 'sm')} Add hours</button></div>`).join('<hr>')}</div>
    <p class="muted small" style="margin-top:16px">Changing your chart never cancels bookings that already exist.</p><div id="aerr"></div><div class="form-actions"><button class="btn btn-primary" id="savehrs">Save chart</button></div>`),
    mount(r) {
      r.addEventListener('click', async e => {
        if (e.target.closest('[data-rm]')) e.target.closest('.win').remove();
        const add = e.target.closest('[data-add]'); if (add) { const d = +add.dataset.add; const h = document.createElement('div'); h.innerHTML = win(d, { start: '10:00', end: '13:00', slot_minutes: 20 }); r.querySelector(`[data-day="${d}"] .wins`).append(h.firstElementChild); }
        if (e.target.closest('#savehrs')) { const out = [...r.querySelectorAll('.win')].map(w => ({ weekday: +w.dataset.d, start: w.querySelector('[data-f=start]').value, end: w.querySelector('[data-f=end]').value, slot_minutes: +w.querySelector('[data-f=slot]').value }));
          try { await api('/doctor/availability', { method: 'PUT', body: out }); toast('Availability saved', 'success'); route(); } catch (er) { $('#aerr', r).innerHTML = alertBox('error', esc(er.message), 'alert'); } }
      });
    } };
}

/* ---------------- admin ---------------- */
async function aHome() {
  const [st, docs] = await Promise.all([api('/admin/stats'), api('/admin/doctors')]);
  return { title: 'Overview', sub: 'Everything across the platform', html: `<div class="grid cols-4">${tile('Doctors', st.doctors)}${tile('Patients', st.patients, st.unassigned_patients + ' not yet assigned')}${tile('Face scans', st.scans)}${tile('Booked appointments', st.booked)}</div>
    <div style="margin-top:24px">${card('Doctors', `<div class="table-wrap"><table class="table"><thead><tr><th>Doctor</th><th>Focus</th><th class="num">Patients${infoTip('Patients who currently have this doctor.')}</th><th class="num">Upcoming${infoTip('Appointments already booked with this doctor from now onward.')}</th><th></th></tr></thead><tbody>${docs.map(d => `<tr><td class="cell-title">${esc(d.name)}<div class="cell-sub">${esc(d.email)}</div></td><td>${esc(d.specialization)}</td><td class="num">${d.patients}</td><td class="num">${d.upcoming}</td><td class="actions"><a class="btn btn-secondary btn-sm" href="#/adoctor/${d.id}">View patients</a></td></tr>`).join('')}</tbody></table></div>`)}</div>` };
}
const A_COL = { completed: '#2e7d32', upcoming: '#4674b9', missed: '#c0392b', cancelled: '#9aa5b1' };
function stackedBars(weeks) {
  const W = 760, H = 270, m = { l: 40, r: 12, t: 12, b: 40 }, n = weeks.length, bw = (W - m.l - m.r) / n; const max = Math.max(4, ...weeks.map(w => w.completed + w.upcoming + w.missed + w.cancelled)) * 1.1;
  const y = v => H - m.b - v / max * (H - m.t - m.b); const thisWeek = ymd((() => { const d = new Date(); d.setHours(0, 0, 0, 0); d.setDate(d.getDate() - ((d.getDay() + 6) % 7)); return d; })());
  const grid = [0, 1, 2, 3, 4].map(k => { const v = max * k / 4; return `<line x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}" stroke="#ececec"/><text x="${m.l - 6}" y="${y(v) + 4}" font-size="11" text-anchor="end" fill="#6b6b6b">${Math.round(v)}</text>`; }).join('');
  const bars = weeks.map((w, i) => { let acc = 0; const x = m.l + i * bw + bw * 0.14, ww = bw * 0.72; const segs = ['completed', 'upcoming', 'missed', 'cancelled'].map(k => { const v = w[k]; if (!v) return ''; const y0 = y(acc + v), h = y(acc) - y0; acc += v; return `<rect x="${x}" y="${y0}" width="${ww}" height="${h}" fill="${A_COL[k]}"><title>${k}: ${v} (week of ${w.week})</title></rect>`; }).join('');
    const cur = w.week === thisWeek; return segs + `<text x="${x + ww / 2}" y="${H - 22}" font-size="11" text-anchor="middle" fill="${cur ? '#325385' : '#6b6b6b'}" font-weight="${cur ? 700 : 400}">${esc(D(w.week + ' 00:00').toLocaleDateString('en-IN', { day: 'numeric', month: 'short' }))}</text>${cur ? `<text x="${x + ww / 2}" y="${H - 8}" font-size="10" text-anchor="middle" fill="#325385">this week</text>` : ''}`; }).join('');
  return `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto" role="img" aria-label="Appointments per week by outcome">${grid}${bars}</svg>`;
}
const pctOf = (v, t) => t ? Math.round(100 * v / t) : 0;
async function aAnalytics() {
  const d = await api('/admin/analytics'), A = d.appointments, R = d.recovery, rs = d.reschedules, sc = d.scanning;
  const att = A.attendance_rate === null ? '-' : Math.round(A.attendance_rate * 100) + '%'; const scored = R.improved + R.stable + R.worsened; const spt = d.totals.patients || 1;
  const legend = items => `<ul class="legend">${items.map(([c, l, v, p]) => `<li><span class="swatch ${c}"></span>${esc(l)}<span class="count">${v}${p !== undefined ? ` <span class="muted small">(${p}%)</span>` : ''}</span></li>`).join('')}</ul>`;
  const stack = parts => { const t = parts.reduce((a, [, v]) => a + v, 0) || 1; return `<div class="stackbar">${parts.map(([c, v]) => `<span class="${c}" style="--w:${100 * v / t}%" title="${v}"></span>`).join('')}</div>`; };
  const edgeLabels = ['Down 40% or more', 'Down 15 to 40%', 'About the same (within 15%)', 'Up 15 to 40%', 'Up 40% or more']; const hmax = Math.max(1, ...R.histogram.counts);
  const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']; const dmax = Math.max(1, ...d.by_weekday); const recBy = Object.fromEntries(R.by_doctor.map(x => [x.doctor_id, x]));
  const drows = d.doctors.map(x => { const r = recBy[x.id] || { improved: 0, stable: 0, worsened: 0 };
    return `<tr><td class="cell-title">${esc(x.name)}<div class="cell-sub">${esc(x.specialization)}</div></td><td class="num">${x.patients}</td><td class="num">${x.completed}</td><td class="num">${x.missed}</td><td class="num">${x.cancelled}</td><td class="num">${x.upcoming}</td>
      <td class="num">${x.attendance_rate === null ? '-' : Math.round(x.attendance_rate * 100) + '%'}</td><td style="min-width:110px"><div class="bar-track" title="${x.utilisation === null ? '' : Math.round(x.utilisation * 100) + '% of slots booked'}"><div class="bar-fill" style="--w:${Math.round(100 * (x.utilisation || 0))}%"></div></div></td>
      <td class="num">${x.rescheduled}</td><td style="min-width:120px">${stack([['c-green', r.improved], ['c-blue', r.stable], ['c-red', r.worsened]])}</td></tr>`; }).join('');
  return { title: 'Analytics', sub: 'Appointments, doctors and patient recovery across the platform',
    html: `<div class="stack"><div class="grid cols-4">${tile('Doctors', d.totals.doctors)}${tile('Patients', d.totals.patients, d.totals.scans + ' face scans in total')}${tile('Appointments', d.totals.appointments, 'all time, including cancelled')}${tile('Attendance rate', att, 'completed visits out of completed + missed')}</div>
    <div class="grid auto">${tile('Completed', A.completed)}${tile('Upcoming', A.upcoming)}${tile('Missed', A.missed, 'booked, time passed, not completed', A.missed ? 'warn' : '')}${tile('Cancelled', A.cancelled)}${tile('Rescheduled', A.rescheduled, 'times an appointment was moved')}</div>
    ${card('Appointments by week', stackedBars(d.weekly) + legend([['c-green', 'Completed', d.weekly.reduce((a, w) => a + w.completed, 0)], ['c-blue', 'Upcoming', d.weekly.reduce((a, w) => a + w.upcoming, 0)], ['c-red', 'Missed', d.weekly.reduce((a, w) => a + w.missed, 0)], ['c-gray', 'Cancelled', d.weekly.reduce((a, w) => a + w.cancelled, 0)]]).replace('class="legend"', 'class="legend" style="grid-template-columns:repeat(auto-fit,minmax(170px,1fr))"'), `<span class="muted small">last 8 weeks and next 4</span>`)}
    <div class="grid cols-2">
      ${card('Patient recovery', `<p class="muted small" style="margin-bottom:8px">Compares each patient's first scans with their latest scans (average spots found). ${infoTip('Improved = 15% or more fewer spots; worsened = 15% or more more spots; otherwise stable. Needs at least 2 scans.')}</p>
        <div class="tile-value" style="margin-bottom:8px">${R.rate === null ? '-' : Math.round(R.rate * 100) + '%'} <span class="muted small" style="font-weight:400">of patients improved</span></div>${stack([['c-green', R.improved], ['c-blue', R.stable], ['c-red', R.worsened]])}
        ${legend([['c-green', 'Improved', R.improved, pctOf(R.improved, scored)], ['c-blue', 'Stable', R.stable, pctOf(R.stable, scored)], ['c-red', 'Worsened', R.worsened, pctOf(R.worsened, scored)], ['c-gray', 'Not enough scans yet', R.insufficient]])}
        <div class="bars" style="margin-top:12px">${edgeLabels.map((l, i) => `<div class="bar-row"><span>${l}</span><div class="bar-track"><div class="bar-fill ${i < 2 ? '' : i === 2 ? '' : 'accent'}" style="--w:${100 * R.histogram.counts[i] / hmax}%"></div></div><span>${R.histogram.counts[i]}</span></div>`).join('')}</div>`)}
      <div class="stack">${card('Weekly scanning', `<p class="muted small" style="margin-bottom:8px">How up to date patients are with their weekly face scan.</p>${stack([['c-green', sc.scanned_last_7_days], ['c-amber', sc.due], ['c-red', sc.overdue], ['c-gray', sc.never]])}
        ${legend([['c-green', 'Scanned in the last 7 days', sc.scanned_last_7_days, pctOf(sc.scanned_last_7_days, spt)], ['c-amber', 'Due (7 to 9 days)', sc.due, pctOf(sc.due, spt)], ['c-red', 'Overdue (10+ days)', sc.overdue, pctOf(sc.overdue, spt)], ['c-gray', 'Never scanned', sc.never, pctOf(sc.never, spt)]])}`)}
      ${card('Busiest days', `<div class="bars">${days.map((l, i) => `<div class="bar-row"><span>${l}</span><div class="bar-track"><div class="bar-fill" style="--w:${100 * d.by_weekday[i] / dmax}%"></div></div><span>${d.by_weekday[i]}</span></div>`).join('')}</div>`)}</div></div>
    ${card('Doctors', `<div class="table-wrap"><table class="table"><thead><tr><th>Doctor</th><th class="num">Patients</th><th class="num">Completed</th><th class="num">Missed</th><th class="num">Cancelled</th><th class="num">Upcoming</th>
      <th class="num">Attendance${infoTip('Completed visits out of completed + missed. Cancelled visits are not counted.')}</th><th>Utilisation${infoTip('Share of the doctor’s weekly slots that are booked, over the last 8 and next 4 weeks.')}</th><th class="num">Moved${infoTip('Appointments that were rescheduled at least once.')}</th><th>Patient recovery${infoTip('Green improved, blue stable, red worsened.')}</th></tr></thead><tbody>${drows}</tbody></table></div>`)}
    ${card('Rescheduling', `<div class="grid cols-2" style="margin-bottom:12px">${tile('Moved by patients', rs.by_patient)}${tile('Moved by doctors', rs.by_doctor, 'for example doctor unavailable')}</div>${rs.top_reasons.length ? `<table class="table"><thead><tr><th>Reason</th><th>By</th><th class="num">Times</th></tr></thead><tbody>${rs.top_reasons.map(x => `<tr><td class="cell-title" style="text-transform:capitalize">${esc(x.reason)}</td><td>${esc(x.by)}</td><td class="num">${x.count}</td></tr>`).join('')}</tbody></table>` : '<p class="muted">No reasons recorded yet.</p>'}`)}
    <div class="alert alert-info">${icon('info')}<div>Recovery and scanning figures are aggregates. Individual scans, results and photos stay private to the patient and the doctors they have been under.</div></div></div>` };
}
async function aDoctor(id) {
  const [docs, pats] = await Promise.all([api('/admin/doctors'), api(`/admin/doctors/${id}/patients`)]); const d = docs.find(x => x.id === +id);
  return { title: d.name, sub: d.specialization, crumb: '<a href="#/home">Overview</a> / Doctor', html: card('Patients under this doctor', pats.length ? `<table class="table"><thead><tr><th>Patient</th><th>Health form</th><th class="num">Scans</th><th></th></tr></thead><tbody>${pats.map(p => `<tr><td class="cell-title">${esc(p.name)}<div class="cell-sub">${esc(p.email)}</div></td><td>${p.intake_done ? '<span class="badge success">Done</span>' : '<span class="badge progress">Pending</span>'}</td><td class="num">${p.scans}</td><td class="actions"><a class="btn btn-secondary btn-sm" href="#/apatient/${p.id}">Open record</a></td></tr>`).join('')}</tbody></table>` : stateBox('No patients', '')) };
}
async function aPatient(id) {
  const p = await api('/admin/patients/' + id); return { title: p.name, sub: p.email, crumb: '<a href="#/home">Overview</a> / Patient', html: recordHtml(p) };
}
let calWeek = null;
async function aCalendar() {
  const base = calWeek || (() => { const d = new Date(); d.setHours(0, 0, 0, 0); d.setDate(d.getDate() - ((d.getDay() + 6) % 7)); return d; })(); calWeek = base;
  const end = new Date(base); end.setDate(end.getDate() + 7); const ap = await api(`/admin/calendar?start=${ymd(base)} 00:00&end=${ymd(end)} 00:00`);
  const palette = ['#325385', '#ed7639', '#1f6b24', '#a82e22', '#8a5a00', '#6b3fa0', '#0f766e']; const col = id => palette[id % palette.length]; const todayS = ymd(new Date());
  const cols = Array.from({ length: 7 }, (_, i) => { const d = new Date(base); d.setDate(d.getDate() + i); const k = ymd(d); const list = ap.filter(a => a.start_ts.startsWith(k));
    return `<div class="day ${k === todayS ? 'today' : ''}"><h4>${d.toLocaleDateString('en-IN', { weekday: 'short', day: 'numeric', month: 'short' })}</h4>${list.map(a => `<div class="appt-chip" style="--c:${col(a.doctor_id)}"><b>${esc(a.start_ts.slice(11))} ${esc(a.doctor)}</b>${esc(a.patient)}${a.reason ? '<br>' + esc(a.reason) : ''} ${a.status === 'completed' ? '✓' : ''}</div>`).join('') || '<span class="muted small">Free</span>'}</div>`; }).join('');
  const docsInWeek = [...new Map(ap.map(a => [a.doctor_id, a.doctor])).entries()];
  return { title: 'Master calendar', sub: `Week of ${fmtDate(ymd(base) + ' 00:00')}: all doctors`, actions: '<button class="btn btn-secondary" data-w="-1">‹ Previous week</button><button class="btn btn-secondary" data-w="0">This week</button><button class="btn btn-secondary" data-w="1">Next week ›</button>',
    html: `<div class="week">${cols}</div>${docsInWeek.length ? `<ul class="legend">${docsInWeek.map(([id, n]) => `<li><span class="swatch" style="background:${col(id)}"></span>${esc(n)}</li>`).join('')}</ul>` : ''}`,
    mount(r) { document.querySelectorAll('[data-w]').forEach(b => b.onclick = () => { const w = +b.dataset.w; if (w === 0) calWeek = null; else { calWeek = new Date(base); calWeek.setDate(calWeek.getDate() + 7 * w); } route(); }); } };
}
async function aAddDoctor() {
  return { title: 'Add a doctor', sub: 'New doctors get a default Mon-Fri 10:00-13:00 chart they can change', html: card('Doctor details', `<form id="nd" class="stack" novalidate><div class="form-grid"><div class="field"><label for="dn">Full name</label><input class="input" id="dn" required></div><div class="field"><label for="de">Email</label><input class="input" id="de" type="email" required></div>
    <div class="field"><label for="dp">Temporary password</label><input class="input" id="dp" type="password" autocomplete="new-password" required><span class="hint">At least 8 characters. Ask the doctor to keep it private.</span></div><div class="field"><label for="ds">Specialization</label><input class="input" id="ds" placeholder="e.g. Ayurveda: digestive health"></div></div>
    <div class="form-section">Areas this doctor treats (used to recommend them to patients; leave empty for a general physician)</div><div class="cond-grid">${Object.entries(SYSTEMS).map(([k, l]) => `<label class="choice"><input type="checkbox" name="focus" value="${k}"> ${esc(l)}</label>`).join('')}</div>
    <div id="nderr"></div><div class="form-actions"><button class="btn btn-primary" type="submit">Create doctor</button></div></form>`),
    mount(r) { $('#nd', r).onsubmit = async ev => { ev.preventDefault(); try { await api('/admin/doctors', { method: 'POST', body: { name: $('#dn', r).value.trim(), email: $('#de', r).value.trim(), password: $('#dp', r).value, specialization: $('#ds', r).value.trim(), focus: [...r.querySelectorAll('input[name=focus]:checked')].map(i => i.value) } }); toast('Doctor created', 'success'); location.hash = '#/home'; } catch (e) { $('#nderr', r).innerHTML = alertBox('error', esc(e.message), 'alert'); } }; } };
}

/* ---------------- router ---------------- */
const PAGES = {
  patient: { home: pHome, scan: pScan, progress: pProgress, appointments: pAppointments, doctor: pDoctor, intake: pIntake },
  doctor: { home: dHome, appointments: dAppointments, availability: dAvailability, patient: dPatient },
  admin: { home: aHome, analytics: aAnalytics, calendar: aCalendar, emails: aEmails, 'add-doctor': aAddDoctor, adoctor: aDoctor, apatient: aPatient },
};
async function route() {
  if (!me) return loginView();
  if (me.role === 'patient' && (me.onboarding.needs_intake || me.onboarding.needs_scan || me.onboarding.needs_doctor)) return wizard();
  const [page = 'home', arg] = (location.hash.replace(/^#\//, '') || 'home').split('/'); const fn = PAGES[me.role][page] || PAGES[me.role].home;
  shell(true, NAV[me.role].some(n => n[0] === page) ? page : (page === 'patient' || page === 'adoctor' || page === 'apatient' ? 'home' : 'home'), `<div class="page"><div class="state"><span class="spinner"></span></div></div>`);
  try {
    const r = await fn(arg); const act = NAV[me.role].some(n => n[0] === page) ? page : 'home';
    shell(true, act, `<div class="page">${header(r.title, r.sub || '', r.actions || '', r.crumb || '')}${r.html}</div>`);
    if (r.mount) r.mount($('#main')); const h = $('.page-title'); if (h) h.focus({ preventScroll: true });
  } catch (e) { if (e.status === 401) { me = null; return loginView(); } shell(true, 'home', `<div class="page">${alertBox('error', esc(e.message), 'alert')}</div>`); }
}
async function boot() {
  try { me = await api('/me'); SYSTEMS = await api('/systems'); } catch { me = null; }
  if (me && me.role === 'patient' && me.scan_status && me.scan_status.due && !sessionStorage.getItem('dueShown')) { sessionStorage.setItem('dueShown', '1'); toast('Your weekly face scan is due', ''); }
  route();
}
document.addEventListener('click', e => {
  const p = $('#umenu'); if (p && !p.hidden && !e.target.closest('#usermenu')) { p.hidden = true; $('#umbtn').setAttribute('aria-expanded', 'false'); }
  const t = e.target.closest('[data-photo-toggle]');
  if (t) { const img = t.closest('figure').querySelector('img'), orig = img.dataset.kind === 'annotated'; img.dataset.kind = orig ? 'original' : 'annotated'; img.src = `/api/scans/${img.dataset.sid}/photo/${img.dataset.view}/${img.dataset.kind}`; t.textContent = orig ? 'show spots marked' : 'show original'; }
  const rs = e.target.closest('[data-resched]'); if (rs) rescheduleModal(rs.dataset.resched, rs.dataset.when, rs.dataset.who);
  const b = e.target.closest('[data-scan-photos]'); if (b) modal(`Photos · ${b.dataset.date}`, photoGallery(b.dataset.scanPhotos, b.dataset.views.split(',')), '<button class="btn btn-secondary" data-close>Close</button>').el.querySelector('.modal').classList.add('wide');
});
document.addEventListener('keydown', e => { if (e.key === 'Escape') { const p = $('#umenu'); if (p && !p.hidden) { p.hidden = true; $('#umbtn').focus(); } } });
addEventListener('hashchange', route);
boot();
})();
