// ── State ──────────────────────────────────────────────────────────────────────
const admin = {
  kbs:           [],
  users:         [],
  groups:        [],
  personalities: [],
  ollamaModels:  [],
  currentKBId:   null,
};

const MAX_UPLOAD_FILES = 5;
const MAX_UPLOAD_FILE_MB = 15;
const MAX_UPLOAD_FILE_BYTES = MAX_UPLOAD_FILE_MB * 1024 * 1024;

// ── Toast ──────────────────────────────────────────────────────────────────────
function toast(msg, type = 'success') {
  const icons = {
    success: `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>`,
    error:   `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>`,
  };
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.innerHTML = `<span class="toast-icon">${icons[type]||icons.success}</span>${msg}`;
  document.getElementById('toast-container').appendChild(el);
  setTimeout(() => el.remove(), 4000);
}

// ── Auth guard ─────────────────────────────────────────────────────────────────
function guardAdmin() {
  if (!Auth.token() || !Auth.isAdmin()) {
    window.location.href = 'index.html';
    return;
  }
  document.getElementById('admin-user-name').textContent = Auth.user().username;
  document.getElementById('logout-btn').addEventListener('click', () => {
    Auth.clear();
    window.location.href = 'index.html';
  });
}

// ── Navigation ─────────────────────────────────────────────────────────────────
function initNav() {
  document.querySelectorAll('.admin-nav-item').forEach(item => {
    item.addEventListener('click', () => {
      document.querySelectorAll('.admin-nav-item').forEach(i => i.classList.remove('active'));
      document.querySelectorAll('.admin-panel').forEach(p => p.classList.add('hidden'));
      item.classList.add('active');
      document.getElementById(`panel-${item.dataset.panel}`)?.classList.remove('hidden');
    });
  });
}

// ── Stats ──────────────────────────────────────────────────────────────────────
async function loadStats() {
  try {
    const s = await AdminAPI.getStats();
    document.getElementById('stat-users').textContent        = s.total_users;
    document.getElementById('stat-kbs').textContent          = s.total_kbs;
    document.getElementById('stat-active').textContent       = s.active_kbs;
    document.getElementById('stat-docs').textContent         = s.total_documents;
    document.getElementById('stat-groups').textContent       = s.total_groups;
    document.getElementById('stat-personalities').textContent= s.total_personalities;
  } catch { toast('Failed to load stats', 'error'); }
}

// ── Ollama Models ──────────────────────────────────────────────────────────────
async function refreshOllamaModels() {
  try {
    admin.ollamaModels = await AdminAPI.getOllamaModels();
    const sel = document.getElementById('kb-model');
    const current = sel.value;
    sel.innerHTML = '';
    (admin.ollamaModels.length ? admin.ollamaModels : ['llama3.2:3b', 'mistral', 'phi3']).forEach(m => {
      const opt = document.createElement('option');
      opt.value = m; opt.textContent = m;
      if (m === current) opt.selected = true;
      sel.appendChild(opt);
    });
    if (!sel.value && admin.ollamaModels.length) sel.value = admin.ollamaModels[0];
    toast(`${admin.ollamaModels.length} Ollama models loaded`);
  } catch { toast('Could not reach Ollama', 'error'); }
}

// ── Personalities ──────────────────────────────────────────────────────────────
async function loadPersonalities() {
  try {
    admin.personalities = await AdminAPI.getPersonalities();
    renderPersonalitiesTable();
    populatePersonalitySelectors();
  } catch { toast('Failed to load personalities', 'error'); }
}

function renderPersonalitiesTable() {
  const tbody = document.getElementById('personalities-tbody');
  tbody.innerHTML = '';
  admin.personalities.forEach(p => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>
        <div style="font-weight:500">${escapeHTML(p.name)}</div>
        <div style="font-size:.75rem;color:var(--text-3)">${escapeHTML(p.description||'')}</div>
      </td>
      <td><span class="badge badge-blue">${escapeHTML(p.tone)}</span></td>
      <td style="max-width:260px;font-size:.8rem;color:var(--text-2);overflow:hidden;text-overflow:ellipsis;white-space:nowrap">
        ${escapeHTML(p.system_prompt.substring(0,100))}…
      </td>
      <td><span class="badge ${p.is_preset ? 'badge-purple' : 'badge-green'}">${p.is_preset ? 'Preset' : 'Custom'}</span></td>
      <td>
        <div class="flex gap-2">
          <button class="btn btn-ghost btn-sm" onclick="openEditPersonalityModal(${p.id})">Edit</button>
          ${!p.is_preset ? `<button class="btn btn-danger btn-sm" onclick="confirmDeletePersonality(${p.id},'${escapeHTML(p.name)}')">Delete</button>` : ''}
        </div>
      </td>`;
    tbody.appendChild(tr);
  });
}

function populatePersonalitySelectors() {
  const sel = document.getElementById('kb-personality-id');
  if (!sel) return;
  const current = sel.value;
  sel.innerHTML = '<option value="">— None / use custom prompt —</option>';
  admin.personalities.forEach(p => {
    const opt = document.createElement('option');
    opt.value = p.id;
    opt.textContent = `${p.name} (${p.tone})`;
    if (String(p.id) === String(current)) opt.selected = true;
    sel.appendChild(opt);
  });
}

function onPersonalityChange() {
  const sel = document.getElementById('kb-personality-id');
  const preview = document.getElementById('personality-preview');
  const box     = document.getElementById('personality-preview-box');
  if (!sel.value) { box.classList.add('hidden'); return; }
  const p = admin.personalities.find(x => String(x.id) === sel.value);
  if (p) {
    preview.textContent = p.system_prompt;
    box.classList.remove('hidden');
  }
}

function openCreatePersonalityModal() {
  document.getElementById('personality-modal-title').textContent = 'Create Personality';
  document.getElementById('personality-form').reset();
  document.getElementById('personality-id-input').value = '';
  document.getElementById('personality-modal').classList.remove('hidden');
}

function openEditPersonalityModal(id) {
  const p = admin.personalities.find(x => x.id === id);
  if (!p) return;
  document.getElementById('personality-modal-title').textContent = 'Edit Personality';
  document.getElementById('personality-id-input').value = p.id;
  document.getElementById('personality-name').value        = p.name;
  document.getElementById('personality-tone').value        = p.tone;
  document.getElementById('personality-description').value = p.description || '';
  document.getElementById('personality-prompt').value      = p.system_prompt;
  document.getElementById('personality-modal').classList.remove('hidden');
}

async function savePersonality(e) {
  e.preventDefault();
  const id  = document.getElementById('personality-id-input').value;
  const btn = document.getElementById('save-personality-btn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>';
  const payload = {
    name:         document.getElementById('personality-name').value,
    tone:         document.getElementById('personality-tone').value,
    description:  document.getElementById('personality-description').value,
    system_prompt:document.getElementById('personality-prompt').value,
  };
  try {
    if (id) {
      await AdminAPI.updatePersonality(+id, payload);
      toast('Personality updated');
    } else {
      await AdminAPI.createPersonality(payload);
      toast('Personality created');
    }
    closeModal('personality-modal');
    await loadPersonalities();
    await loadStats();
  } catch (err) { toast(err.message, 'error'); }
  finally { btn.disabled = false; btn.textContent = 'Save'; }
}

async function confirmDeletePersonality(id, name) {
  if (!confirm(`Delete personality "${name}"?`)) return;
  try {
    await AdminAPI.deletePersonality(id);
    toast('Personality deleted');
    await loadPersonalities();
    await loadStats();
  } catch (err) { toast(err.message, 'error'); }
}

// ── Knowledge Bases ────────────────────────────────────────────────────────────
async function loadKBs() {
  try {
    admin.kbs = await AdminAPI.getKBs();
    renderKBTable();
  } catch { toast('Failed to load knowledge bases', 'error'); }
}

function renderKBTable() {
  const tbody = document.getElementById('kb-tbody');
  tbody.innerHTML = '';
  if (!admin.kbs.length) {
    tbody.innerHTML = `<tr><td colspan="7"><div class="empty-state"><h3>No knowledge bases</h3><p>Create your first KB to get started.</p></div></td></tr>`;
    return;
  }
  admin.kbs.forEach(kb => {
    const personality = admin.personalities.find(p => p.id === kb.personality_id);
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>
        <div style="font-weight:500">${escapeHTML(kb.name)}</div>
        <div style="font-size:.75rem;color:var(--text-3)">${escapeHTML(kb.description||'')}</div>
      </td>
      <td><span class="badge badge-purple">${escapeHTML(kb.department)}</span></td>
      <td><code style="font-size:.8rem;color:var(--accent)">${escapeHTML(kb.llm_model)}</code></td>
      <td>${personality ? `<span class="badge badge-blue">${escapeHTML(personality.name)}</span>` : '<span style="color:var(--text-3);font-size:.8rem">Custom</span>'}</td>
      <td>${kb.document_count}</td>
      <td><span class="badge ${kb.is_active ? 'badge-green' : 'badge-red'}">${kb.is_active ? 'Active' : 'Inactive'}</span></td>
      <td>
        <div class="flex gap-2">
          <button class="btn btn-ghost btn-sm" onclick="openDocsModal(${kb.id},'${escapeHTML(kb.name)}')">Docs</button>
          <button class="btn btn-ghost btn-sm" onclick="openEditKBModal(${kb.id})">Edit</button>
          <button class="btn btn-danger btn-sm" onclick="confirmDeleteKB(${kb.id},'${escapeHTML(kb.name)}')">Delete</button>
        </div>
      </td>`;
    tbody.appendChild(tr);
  });
}

// ── ( openCreateKBModal) ────────────────────────────────────────────
function openCreateKBModal() {
  document.getElementById('kb-modal-title').textContent = 'Create Knowledge Base';
  document.getElementById('kb-form').reset();
  document.getElementById('kb-id-input').value    = '';
  document.getElementById('kb-mmr-fetch-k').value = '16';
  document.getElementById('kb-mmr-lambda').value  = '0.7';
  document.getElementById('kb-active').checked    = true;
  populatePersonalitySelectors();
  populateKBModelDropdown();
  document.getElementById('kb-modal').classList.remove('hidden');
}

function populateKBModelDropdown() {
  const sel = document.getElementById('kb-model');
  const models = admin.ollamaModels.length
    ? admin.ollamaModels
    : ['llama3.2:3b', 'llama3.1', 'mistral', 'phi3', 'gemma2'];
  const current = sel.value || 'llama3.2:3b';
  sel.innerHTML = '';
  models.forEach(m => {
    const opt = document.createElement('option');
    opt.value = m; opt.textContent = m;
    if (m === current) opt.selected = true;
    sel.appendChild(opt);
  });
}

function openEditKBModal(id) {
  const kb = admin.kbs.find(k => k.id === id);
  if (!kb) return;
  document.getElementById('kb-modal-title').textContent     = 'Edit Knowledge Base';
  document.getElementById('kb-id-input').value              = kb.id;
  document.getElementById('kb-name').value                  = kb.name;
  document.getElementById('kb-department').value            = kb.department;
  document.getElementById('kb-description').value           = kb.description || '';
  document.getElementById('kb-temperature').value           = kb.temperature;
  document.getElementById('kb-max-tokens').value            = kb.max_tokens;
  document.getElementById('kb-top-k').value                 = kb.top_k_docs;
  document.getElementById('kb-mmr-fetch-k').value           = kb.mmr_fetch_k;
  document.getElementById('kb-mmr-lambda').value            = kb.mmr_lambda;
  document.getElementById('kb-system-prompt').value         = kb.system_prompt || '';
  document.getElementById('kb-active').checked              = kb.is_active;
  populatePersonalitySelectors();
  populateKBModelDropdown();
  document.getElementById('kb-model').value                 = kb.llm_model;
  document.getElementById('kb-embedding').value             = kb.embedding_model;
  document.getElementById('kb-personality-id').value        = kb.personality_id || '';
  onPersonalityChange();
  document.getElementById('kb-modal').classList.remove('hidden');
}

async function saveKB(e) {
  e.preventDefault();
  const id  = document.getElementById('kb-id-input').value;
  const btn = document.getElementById('save-kb-btn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Saving…';

  const payload = {
    name:            document.getElementById('kb-name').value,
    description:     document.getElementById('kb-description').value,
    department:      document.getElementById('kb-department').value,
    llm_model:       document.getElementById('kb-model').value,
    embedding_model: document.getElementById('kb-embedding').value,
    personality_id:  document.getElementById('kb-personality-id').value
                       ? +document.getElementById('kb-personality-id').value
                       : null,
    system_prompt:   document.getElementById('kb-system-prompt').value || null,
    temperature:     parseFloat(document.getElementById('kb-temperature').value),
    max_tokens:      parseInt(document.getElementById('kb-max-tokens').value),
    top_k_docs:      parseInt(document.getElementById('kb-top-k').value),
    mmr_fetch_k:     parseInt(document.getElementById('kb-mmr-fetch-k').value),
    mmr_lambda:      parseFloat(document.getElementById('kb-mmr-lambda').value),
    is_active:       document.getElementById('kb-active').checked,
  };

  try {
    if (id) {
      await AdminAPI.updateKB(+id, payload);
      toast('Knowledge base updated');
    } else {
      await AdminAPI.createKB(payload);
      toast('Knowledge base created');
    }
    closeModal('kb-modal');
    await loadKBs();
    await loadStats();
  } catch (err) {
    toast(err.message, 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Save';
  }
}

async function confirmDeleteKB(id, name) {
  if (!confirm(`Delete knowledge base "${name}"?\nAll documents and vector data will be permanently removed.`)) return;
  try {
    await AdminAPI.deleteKB(id);
    toast('Knowledge base deleted');
    await loadKBs();
    await loadStats();
  } catch (err) {
    toast(err.message, 'error');
  }
}

// ── Documents Modal ────────────────────────────────────────────────────────────
async function openDocsModal(kbId, kbName) {
  admin.currentKBId = kbId;
  document.getElementById('docs-modal-title').textContent = `Documents — ${kbName}`;
  document.getElementById('docs-modal').classList.remove('hidden');
  await loadDocuments();
}

async function loadDocuments() {
  const list = document.getElementById('docs-list');
  list.innerHTML = '<div class="spinner" style="margin:20px auto;display:block"></div>';
  try {
    const docs = await AdminAPI.getDocuments(admin.currentKBId);
    list.innerHTML = '';
    if (!docs.length) {
      list.innerHTML = `<div class="empty-state"><p>No documents uploaded yet.</p></div>`;
      return;
    }
    docs.forEach(doc => {
      const statusBadge = doc.status === 'ready'
        ? '<span class="badge badge-green">ready</span>'
        : doc.status === 'processing'
        ? '<span class="badge badge-blue">processing</span>'
        : '<span class="badge badge-red">error</span>';

      const el = document.createElement('div');
      el.className = 'doc-item';
      el.innerHTML = `
        <div class="doc-icon">
          <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24"
            fill="none" stroke="currentColor" stroke-width="2">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
            <polyline points="14 2 14 8 20 8"/>
          </svg>
        </div>
        <div class="doc-info">
          <div class="doc-name">${escapeHTML(doc.original_filename)}</div>
          <div class="doc-meta">
            ${formatSize(doc.file_size)} &middot; ${doc.chunk_count} chunks &middot; ${statusBadge}
            &middot; <span style="color:var(--text-3)">${formatDate(doc.uploaded_at)}</span>
          </div>
        </div>
        <button class="btn btn-danger btn-sm" onclick="deleteDocument(${doc.id})">Remove</button>`;
      list.appendChild(el);
    });
  } catch (err) {
    list.innerHTML = `<p style="color:var(--danger);padding:16px">${err.message}</p>`;
  }
}

async function deleteDocument(docId) {
  if (!confirm('Remove this document? Its chunks will be removed from the vector store.')) return;
  try {
    await AdminAPI.deleteDocument(docId);
    toast('Document removed');
    await loadDocuments();
    await loadKBs();
    await loadStats();
  } catch (err) {
    toast(err.message, 'error');
  }
}

// ── Upload state (module-level lock prevents duplicate concurrent uploads) ─────
let _uploadActive = false;

// ── initUploadZone — idempotent via data-attribute guard ──────────────────────
function initUploadZone() {
  const zone  = document.getElementById('upload-zone');
  const input = document.getElementById('file-input');
  if (!zone || !input) return;

  // Guard: if THIS exact input element already has a listener, skip re-init.
  // The data attribute is set on the INPUT (recreated each reset), not on zone,
  // so it correctly re-initializes after innerHTML replacement.
  if (input.dataset.initialized) return;
  input.dataset.initialized = '1';

  // Remove ALL existing listeners from zone by cloning it in place,
  // then re-attach fresh ones. This prevents listener accumulation
  // from repeated initUploadZone() calls on the same zone div.
  const freshZone = zone.cloneNode(true);
  zone.parentNode.replaceChild(freshZone, zone);
  const freshInput = freshZone.querySelector('input[type="file"]') || document.getElementById('file-input');
  if (freshInput) freshInput.dataset.initialized = '1';

  freshZone.addEventListener('click', (e) => {
    if (e.target.tagName === 'INPUT') return;   // don't double-trigger
    freshInput?.click();
  });
  freshZone.addEventListener('dragover',  (e) => { e.preventDefault(); freshZone.classList.add('dragover'); });
  freshZone.addEventListener('dragleave', ()  => freshZone.classList.remove('dragover'));
  freshZone.addEventListener('drop', (e) => {
    e.preventDefault();
    freshZone.classList.remove('dragover');
    if (e.dataTransfer.files?.length) handleUpload(e.dataTransfer.files);
  });
  freshInput?.addEventListener('change', () => {
    if (freshInput.files?.length) handleUpload(freshInput.files);
    freshInput.value = '';   // reset so same file can be re-selected
  });
}

// ── handleUpload — with lock to prevent double-submission ─────────────────────
function normalizeUploadFiles(fileList) {
  const files = [...(fileList || [])];
  if (!files.length) return [];
  if (files.length > MAX_UPLOAD_FILES) {
    throw new Error(`You can upload up to ${MAX_UPLOAD_FILES} files at once`);
  }
  for (const f of files) {
    if (f.size > MAX_UPLOAD_FILE_BYTES) {
      throw new Error(`"${f.name}" exceeds ${MAX_UPLOAD_FILE_MB} MB`);
    }
  }
  return files;
}

async function handleUpload(fileList) {
  if (_uploadActive || !fileList || !admin.currentKBId) return;
  let files = [];
  try {
    files = normalizeUploadFiles(fileList);
  } catch (err) {
    toast(err.message, 'error');
    return;
  }
  if (!files.length) return;

  _uploadActive = true;

  const zone = document.getElementById('upload-zone');
  const label = files.length === 1 ? files[0].name : `${files.length} files`;
  if (zone) {
    zone.innerHTML = `
      <div class="spinner" style="margin:0 auto;display:block"></div>
      <p style="margin-top:10px;color:var(--text-2)">Uploading &amp; queuing "${escapeHTML(label)}"…</p>
      <span>OCR runs automatically on images and scanned PDFs</span>`;
  }

  try {
    await AdminAPI.uploadDocuments(admin.currentKBId, files);
    toast(`${files.length} file${files.length > 1 ? 's' : ''} queued for indexing`);
    await loadDocuments();
    await loadKBs();
    await loadStats();
  } catch (err) {
    toast(err.message || 'Upload failed', 'error');
  } finally {
    _uploadActive = false;
    const z = document.getElementById('upload-zone');
    if (z) {
      z.innerHTML = `
        <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
        <p>Drop file here or click to browse</p>
        <span>PDF · DOCX · XLSX · PPTX · CSV · TXT · MD · HTML · JSON · PNG · JPG · TIFF… (up to 5 files, 15 MB each)</span>
        <input type="file" id="file-input" class="hidden" multiple accept=".pdf,.docx,.doc,.xlsx,.xls,.pptx,.ppt,.csv,.txt,.md,.html,.htm,.json,.png,.jpg,.jpeg,.tiff,.tif,.bmp,.gif,.webp">`;
      initUploadZone();
    }
  }
}


// ── Groups ─────────────────────────────────────────────────────────────────────
async function loadGroups() {
  try {
    admin.groups = await AdminAPI.getGroups();
    renderGroupsTable();
  } catch { toast('Failed to load groups', 'error'); }
}

function renderGroupsTable() {
  const tbody = document.getElementById('groups-tbody');
  tbody.innerHTML = '';

  if (!admin.groups.length) {
    tbody.innerHTML = `<tr><td colspan="5">
      <div class="empty-state">
        <h3>No groups yet</h3>
        <p>Create a group and assign knowledge base permissions to control user access.</p>
      </div></td></tr>`;
    return;
  }

  admin.groups.forEach(group => {
    const permBadges = group.kb_permissions
      .slice(0, 3)
      .map(p => {
        const kb = admin.kbs.find(k => k.id === p.kb_id);
        const color = p.permission === 'manage' ? 'badge-purple' : 'badge-blue';
        return `<span class="badge ${color}" title="${p.permission}">${escapeHTML(kb?.name || `KB#${p.kb_id}`)}</span>`;
      })
      .join(' ');
    const extra = group.kb_permissions.length > 3
      ? `<span style="font-size:.75rem;color:var(--text-3)">+${group.kb_permissions.length - 3} more</span>`
      : '';

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>
        <div style="font-weight:500">${escapeHTML(group.name)}</div>
        <div style="font-size:.75rem;color:var(--text-3)">${escapeHTML(group.description || '')}</div>
      </td>
      <td><span class="badge badge-green">${group.member_count} member${group.member_count !== 1 ? 's' : ''}</span></td>
      <td><div class="flex gap-2" style="flex-wrap:wrap;align-items:center">${permBadges}${extra}</div></td>
      <td style="font-size:.8rem;color:var(--text-3)">${formatDate(group.created_at)}</td>
      <td>
        <div class="flex gap-2">
          <button class="btn btn-ghost btn-sm" onclick="openEditGroupModal(${group.id})">Edit</button>
          <button class="btn btn-danger btn-sm" onclick="confirmDeleteGroup(${group.id},'${escapeHTML(group.name)}')">Delete</button>
        </div>
      </td>`;
    tbody.appendChild(tr);
  });
}

function openCreateGroupModal() {
  document.getElementById('group-modal-title').textContent = 'Create Group';
  document.getElementById('group-form').reset();
  document.getElementById('group-id-input').value = '';
  renderGroupKBPermissions([]);
  renderGroupMembersCheckboxes([]);
  document.getElementById('group-modal').classList.remove('hidden');
}

async function openEditGroupModal(id) {
  const group = admin.groups.find(g => g.id === id);
  if (!group) return;
  document.getElementById('group-modal-title').textContent = 'Edit Group';
  document.getElementById('group-id-input').value      = group.id;
  document.getElementById('group-name').value          = group.name;
  document.getElementById('group-description').value   = group.description || '';
  renderGroupKBPermissions(group.kb_permissions);

  // Load current members
  try {
    const memberIds = await AdminAPI.getGroupMembers(group.id);
    renderGroupMembersCheckboxes(memberIds);
  } catch {
    renderGroupMembersCheckboxes([]);
  }

  document.getElementById('group-modal').classList.remove('hidden');
}

function renderGroupKBPermissions(existingPerms = []) {
  const container = document.getElementById('group-kb-permissions');
  container.innerHTML = '';

  if (!admin.kbs.length) {
    container.innerHTML = '<p style="font-size:.82rem;color:var(--text-3)">No knowledge bases exist yet. Create one first.</p>';
    return;
  }

  admin.kbs.forEach(kb => {
    const existing = existingPerms.find(p => p.kb_id === kb.id);
    const row = document.createElement('div');
    row.style.cssText = 'display:flex;align-items:center;gap:12px;padding:10px 12px;background:var(--surface-2);border-radius:8px;border:1px solid var(--border-subtle)';
    row.innerHTML = `
      <label style="display:flex;align-items:center;gap:8px;flex:1;cursor:pointer;min-width:0">
        <input type="checkbox" class="kb-perm-check" data-kb-id="${kb.id}"
          ${existing ? 'checked' : ''}
          style="accent-color:var(--primary);width:14px;height:14px;flex-shrink:0">
        <span>
          <span style="font-weight:500;font-size:.875rem">${escapeHTML(kb.name)}</span>
          <span class="badge badge-purple" style="margin-left:6px">${escapeHTML(kb.department)}</span>
        </span>
      </label>
      <select class="select kb-perm-level" data-kb-id="${kb.id}"
        style="width:110px;padding:4px 8px;font-size:.8rem">
        <option value="read"   ${existing?.permission === 'read'   ? 'selected' : ''}>read</option>
        <option value="manage" ${existing?.permission === 'manage' ? 'selected' : ''}>manage</option>
      </select>
      <div style="font-size:.72rem;color:var(--text-3);min-width:120px">
        <div><strong>read</strong> — query only</div>
        <div><strong>manage</strong> — + upload/delete</div>
      </div>`;
    container.appendChild(row);
  });
}

function renderGroupMembersCheckboxes(selectedIds = []) {
  const container = document.getElementById('group-members-list');
  container.innerHTML = '';

  if (!admin.users.length) {
    container.innerHTML = '<p style="font-size:.82rem;color:var(--text-3)">No users found.</p>';
    return;
  }

  admin.users.forEach(user => {
    const label = document.createElement('label');
    label.style.cssText = 'display:flex;align-items:center;gap:10px;padding:8px;cursor:pointer;border-radius:8px;transition:background .15s';
    label.addEventListener('mouseenter', () => label.style.background = 'var(--surface-2)');
    label.addEventListener('mouseleave', () => label.style.background = 'transparent');
    label.innerHTML = `
      <input type="checkbox" class="member-check" value="${user.id}"
        ${selectedIds.includes(user.id) ? 'checked' : ''}
        style="accent-color:var(--primary);width:14px;height:14px;flex-shrink:0">
      <div class="user-avatar" style="width:28px;height:28px;font-size:.72rem;flex-shrink:0">
        ${(user.full_name || user.username)[0].toUpperCase()}
      </div>
      <div style="min-width:0">
        <div style="font-size:.875rem;font-weight:500">${escapeHTML(user.username)}</div>
        <div style="font-size:.75rem;color:var(--text-3)">${escapeHTML(user.email)}</div>
      </div>
      ${user.department ? `<span class="badge badge-blue" style="margin-left:auto">${escapeHTML(user.department)}</span>` : ''}`;
    container.appendChild(label);
  });
}

async function saveGroup(e) {
  e.preventDefault();
  const id  = document.getElementById('group-id-input').value;
  const btn = document.getElementById('save-group-btn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Saving…';

  const name        = document.getElementById('group-name').value;
  const description = document.getElementById('group-description').value;

  // Collect KB permissions (only checked ones)
  const permissions = [];
  document.querySelectorAll('.kb-perm-check:checked').forEach(chk => {
    const kbId = +chk.dataset.kbId;
    const level = document.querySelector(`.kb-perm-level[data-kb-id="${kbId}"]`)?.value || 'read';
    permissions.push({ kb_id: kbId, permission: level });
  });

  // Collect selected member IDs
  const memberIds = Array.from(
    document.querySelectorAll('.member-check:checked')
  ).map(c => +c.value);

  try {
    let groupId;
    if (id) {
      await AdminAPI.updateGroup(+id, { name, description });
      groupId = +id;
      toast('Group updated');
    } else {
      const created = await AdminAPI.createGroup({ name, description });
      groupId = created.id;
      toast('Group created');
    }

    // Save permissions and members in parallel
    await Promise.all([
      AdminAPI.setGroupPermissions(groupId, permissions),
      AdminAPI.setGroupMembers(groupId, memberIds),
    ]);

    closeModal('group-modal');
    await loadGroups();
    await loadStats();
  } catch (err) {
    toast(err.message, 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Save Group';
  }
}

async function confirmDeleteGroup(id, name) {
  if (!confirm(`Delete group "${name}"?\nUsers in this group will lose all associated KB access.`)) return;
  try {
    await AdminAPI.deleteGroup(id);
    toast('Group deleted');
    await loadGroups();
    await loadStats();
  } catch (err) {
    toast(err.message, 'error');
  }
}

// ── Users ──────────────────────────────────────────────────────────────────────
async function loadUsers() {
  try {
    admin.users = await AdminAPI.getUsers();
    renderUsersTable();
  } catch { toast('Failed to load users', 'error'); }
}

function renderUsersTable() {
  const tbody = document.getElementById('users-tbody');
  tbody.innerHTML = '';

  admin.users.forEach(user => {
    const userGroups = admin.groups.filter(g =>
      g.kb_permissions && user.group_ids?.includes(g.id)
    );
    const groupBadges = admin.groups
      .filter(g => user.group_ids?.includes(g.id))
      .slice(0, 3)
      .map(g => `<span class="badge badge-blue">${escapeHTML(g.name)}</span>`)
      .join(' ');

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>
        <div class="flex items-center gap-2">
          <div class="user-avatar" style="width:28px;height:28px;font-size:.75rem">
            ${(user.full_name || user.username)[0].toUpperCase()}
          </div>
          <div>
            <div style="font-weight:500">${escapeHTML(user.username)}</div>
            <div style="font-size:.75rem;color:var(--text-3)">${escapeHTML(user.email)}</div>
          </div>
        </div>
      </td>
      <td>${escapeHTML(user.full_name || '—')}</td>
      <td>${escapeHTML(user.department || '—')}</td>
      <td><span class="badge ${user.is_admin ? 'badge-purple' : 'badge-blue'}">${user.is_admin ? 'Admin' : 'User'}</span></td>
      <td><div class="flex gap-2" style="flex-wrap:wrap">${groupBadges || '<span style="color:var(--text-3);font-size:.8rem">No groups</span>'}</div></td>
      <td><span class="badge ${user.is_active ? 'badge-green' : 'badge-red'}">${user.is_active ? 'Active' : 'Inactive'}</span></td>
      <td>
        <div class="flex gap-2">
          <button class="btn btn-ghost btn-sm" onclick="openEditUserModal(${user.id})">Edit</button>
          ${!user.is_admin
            ? `<button class="btn btn-danger btn-sm" onclick="confirmDeleteUser(${user.id},'${escapeHTML(user.username)}')">Delete</button>`
            : ''}
        </div>
      </td>`;
    tbody.appendChild(tr);
  });
}

function openEditUserModal(id) {
  const user = admin.users.find(u => u.id === id);
  if (!user) return;

  document.getElementById('edit-user-id').value       = user.id;
  document.getElementById('edit-user-email').value    = user.email;
  document.getElementById('edit-user-fullname').value = user.full_name || '';
  document.getElementById('edit-user-dept').value     = user.department || '';
  document.getElementById('edit-user-admin').checked  = user.is_admin;
  document.getElementById('edit-user-active').checked = user.is_active;

  // Populate group checkboxes
  const container = document.getElementById('user-groups-list');
  container.innerHTML = '';

  if (!admin.groups.length) {
    container.innerHTML = '<p style="font-size:.82rem;color:var(--text-3)">No groups exist. Create groups first to assign access.</p>';
  } else {
    admin.groups.forEach(group => {
      const isMember = user.group_ids?.includes(group.id);
      const kbCount  = group.kb_permissions?.length || 0;
      const label    = document.createElement('label');
      label.style.cssText = 'display:flex;align-items:center;gap:10px;padding:8px;cursor:pointer;border-radius:8px;transition:background .15s';
      label.addEventListener('mouseenter', () => label.style.background = 'var(--surface-2)');
      label.addEventListener('mouseleave', () => label.style.background = 'transparent');
      label.innerHTML = `
        <input type="checkbox" class="user-group-check" value="${group.id}"
          ${isMember ? 'checked' : ''}
          style="accent-color:var(--primary);width:14px;height:14px;flex-shrink:0">
        <div style="flex:1;min-width:0">
          <div style="font-size:.875rem;font-weight:500">${escapeHTML(group.name)}</div>
          <div style="font-size:.75rem;color:var(--text-3)">${escapeHTML(group.description || '')} &middot; ${kbCount} KB permission${kbCount !== 1 ? 's' : ''}</div>
        </div>`;
      container.appendChild(label);
    });
  }

  document.getElementById('user-modal').classList.remove('hidden');
}

async function saveUser(e) {
  e.preventDefault();
  const id  = +document.getElementById('edit-user-id').value;
  const btn = document.getElementById('save-user-btn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Saving…';

  const group_ids = Array.from(
    document.querySelectorAll('.user-group-check:checked')
  ).map(c => +c.value);

  try {
    await AdminAPI.updateUser(id, {
      email:      document.getElementById('edit-user-email').value,
      full_name:  document.getElementById('edit-user-fullname').value,
      department: document.getElementById('edit-user-dept').value,
      is_admin:   document.getElementById('edit-user-admin').checked,
      is_active:  document.getElementById('edit-user-active').checked,
      group_ids,
    });
    toast('User updated');
    closeModal('user-modal');
    await loadUsers();
    await loadGroups(); // refresh member counts
  } catch (err) {
    toast(err.message, 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Save changes';
  }
}

async function confirmDeleteUser(id, name) {
  if (!confirm(`Delete user "${name}"? This cannot be undone.`)) return;
  try {
    await AdminAPI.deleteUser(id);
    toast('User deleted');
    await loadUsers();
    await loadStats();
  } catch (err) {
    toast(err.message, 'error');
  }
}

// ── Modal helpers ──────────────────────────────────────────────────────────────
function closeModal(id) {
  document.getElementById(id).classList.add('hidden');
}

// ── Formatters ─────────────────────────────────────────────────────────────────
function escapeHTML(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
function formatDate(iso) {
  return new Date(iso).toLocaleDateString([], { year: 'numeric', month: 'short', day: 'numeric' });
}
function formatSize(bytes) {
  if (!bytes)          return '0 B';
  if (bytes < 1024)    return bytes + ' B';
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / 1048576).toFixed(1) + ' MB';
}

// ── Boot ───────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  guardAdmin();
  initNav();

  // Form handlers
  document.getElementById('kb-form').addEventListener('submit', saveKB);
  document.getElementById('personality-form').addEventListener('submit', savePersonality);
  document.getElementById('group-form').addEventListener('submit', saveGroup);
  document.getElementById('user-form').addEventListener('submit', saveUser);

  // Close modals on backdrop click
  document.querySelectorAll('.modal-backdrop').forEach(backdrop => {
    backdrop.addEventListener('click', e => {
      if (e.target === backdrop) backdrop.classList.add('hidden');
    });
  });

  // Load all data in parallel
  await Promise.all([
    loadStats(),
    loadPersonalities(),
    loadKBs(),
    loadUsers(),
    loadGroups(),
  ]);

  // Load Ollama models in background (non-blocking)
  refreshOllamaModels().catch(() => {});

  // Init upload zone after modal is available
  document.getElementById('docs-modal').addEventListener('click', () => {}, { once: false });
  initUploadZone();
});
