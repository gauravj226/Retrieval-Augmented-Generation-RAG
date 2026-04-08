// â”€â”€ State â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
const state = {
  kbs: [],
  sessions: [],
  currentKB: null,
  currentSession: null,
  isLoading: false,
};

const MAX_UPLOAD_FILES = 5;
const MAX_UPLOAD_FILE_MB = 15;
const MAX_UPLOAD_FILE_BYTES = MAX_UPLOAD_FILE_MB * 1024 * 1024;
const FAST_MODE_STORAGE_KEY = 'rag_fast_mode';

// â”€â”€ DOM helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// Resolve message container across legacy and current markup.
const getInner = () =>
  document.getElementById('messages-inner') ||
  document.querySelector('.messages-inner') ||
  document.getElementById('messages-wrapper');

const getWrapper = () => document.getElementById('messages-wrapper');

function clearMessageBubbles() {
  const container = getInner();
  container?.querySelectorAll('.msg').forEach(el => el.remove());
}
function showWelcome() {
  document.getElementById('welcome-state')?.classList.remove('hidden');
}
function hideWelcome() {
  document.getElementById('welcome-state')?.classList.add('hidden');
}

// â”€â”€ Toast â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function toast(message, type = 'success') {
  const icon = type === 'success'
    ? `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>`
    : `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>`;
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.innerHTML = `<span class="toast-icon">${icon}</span>${message}`;
  document.getElementById('toast-container').appendChild(el);
  setTimeout(() => el.remove(), 4000);
}


// â”€â”€ XSS sanitizer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// Strips all script/event-handler content before any innerHTML assignment.
// Uses a temporary DOM element so the browser's own parser does the work.
function sanitize(html) {
  const tpl = document.createElement('template');
  tpl.innerHTML = html;
  const doc = tpl.content;

  // Remove dangerous elements
  const dangerous = ['script', 'iframe', 'object', 'embed', 'form',
                     'input', 'button', 'link', 'meta', 'style', 'base'];
  dangerous.forEach(tag => {
    doc.querySelectorAll(tag).forEach(el => el.remove());
  });

  // Strip event handler attributes (onclick, onerror, onload â€¦)
  doc.querySelectorAll('*').forEach(el => {
    [...el.attributes].forEach(attr => {
      if (/^on\w+/i.test(attr.name) ||
          (attr.name === 'href'  && /^javascript:/i.test(attr.value)) ||
          (attr.name === 'src'   && /^javascript:/i.test(attr.value)) ||
          (attr.name === 'action'&& /^javascript:/i.test(attr.value))) {
        el.removeAttribute(attr.name);
      }
    });
  });
  // Serialize sanitized fragment back to an HTML string.
  const wrapper = document.createElement('div');
  wrapper.appendChild(doc.cloneNode(true));
  return wrapper.innerHTML;
}
// â”€â”€ Markdown-lite renderer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function renderMarkdown(text) {
  const raw = text
    .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/^### (.*$)/gm, '<h3>$1</h3>')
    .replace(/^## (.*$)/gm, '<h2>$1</h2>')
    .replace(/^- (.*$)/gm, '<li>$1</li>')
    .replace(/(<li>.*<\/li>\n?)+/gs, m => `<ul>${m}</ul>`)
    .replace(/\n\n+/g, '</p><p>')
    .replace(/^(?!<[huplc])(.+)$/gm, '<p>$1</p>');
  return sanitize(raw);
}

// â”€â”€ Auth UI â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function initAuth() {
  const overlay   = document.getElementById('auth-overlay');
  const app       = document.getElementById('app');
  const tabBtns   = document.querySelectorAll('.auth-tab');
  const loginForm = document.getElementById('login-form');
  const regForm   = document.getElementById('register-form');
  const loginErr  = document.getElementById('login-error');
  const regErr    = document.getElementById('register-error');

  function showTab(tab) {
    tabBtns.forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
    loginForm.classList.toggle('hidden', tab !== 'login');
    regForm.classList.toggle('hidden', tab !== 'register');
  }
  tabBtns.forEach(b => b.addEventListener('click', () => showTab(b.dataset.tab)));

    loginForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    loginErr.classList.add('hidden');
    const btn = loginForm.querySelector('button[type=submit]');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span>';
    try {
      const data = await AuthAPI.login(
        document.getElementById('login-username').value,
        document.getElementById('login-password').value,
      );

      Auth.save(data.access_token, data.user || {});

      // Replace partial login payload with the canonical profile from /auth/me.
      const me = await AuthAPI.me();
      Auth.save(data.access_token, me);   // overwrite with complete profile

      overlay.classList.add('hidden');
      app.classList.remove('hidden');
      initApp();
    } catch (err) {
      loginErr.textContent = err.message;
      loginErr.classList.remove('hidden');
      Auth.clear();   // wipe partial data on failure
    } finally {
      btn.disabled = false;
      btn.textContent = 'Sign in';
    }
  });



  regForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    regErr.classList.add('hidden');
    const btn = regForm.querySelector('button[type=submit]');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span>';
    try {
      await AuthAPI.register({
        email:      document.getElementById('reg-email').value,
        username:   document.getElementById('reg-username').value,
        password:   document.getElementById('reg-password').value,
        full_name:  document.getElementById('reg-fullname').value,
        department: document.getElementById('reg-department').value,
      });
      toast('Account created! Please sign in.');
      showTab('login');
      regForm.reset();
    } catch (err) {
      regErr.textContent = err.message;
      regErr.classList.remove('hidden');
    } finally {
      btn.disabled = false;
      btn.textContent = 'Create account';
    }
  });

  if (Auth.token()) {
    overlay.classList.add('hidden');
    app.classList.remove('hidden');
    initApp();
  }
}

// â”€â”€ App Init â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
async function initApp() {
  const user = Auth.user();
  if (!user) return;

  document.getElementById('user-avatar').textContent =
    (user.full_name || user.username)[0].toUpperCase();
  document.getElementById('user-name').textContent = user.full_name || user.username;
  document.getElementById('user-dept').textContent = user.department || 'General';

  if (user.is_admin) {
    document.getElementById('admin-link')?.classList.remove('hidden');
  }

  document.getElementById('logout-btn').addEventListener('click', () => {
    Auth.clear();
    window.location.reload();
  });

  initFastModeToggle();
  await Promise.all([loadKBs(), loadSessions()]);
  setupInputBar();
}

function getFastModeEnabled() {
  return localStorage.getItem(FAST_MODE_STORAGE_KEY) === '1';
}

function initFastModeToggle() {
  const toggle = document.getElementById('fast-mode-toggle');
  if (!toggle || toggle.dataset.bound === 'true') return;

  toggle.dataset.bound = 'true';
  toggle.checked = getFastModeEnabled();
  toggle.addEventListener('change', () => {
    localStorage.setItem(FAST_MODE_STORAGE_KEY, toggle.checked ? '1' : '0');
  });
}

// â”€â”€ Knowledge Bases â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
async function loadKBs() {
  try {
    state.kbs = await ChatAPI.getKBs();
    renderKBSelector();
  } catch {
    toast('Failed to load knowledge bases', 'error');
  }
}

function renderKBSelector() {
  const sel = document.getElementById('kb-select');
  sel.innerHTML = '<option value="">â€” Select knowledge base â€”</option>';
  state.kbs.forEach(kb => {
    const opt = document.createElement('option');
    opt.value = kb.id;
    opt.textContent = `${kb.name} (${kb.department})`;
    sel.appendChild(opt);
  });
  sel.addEventListener('change', () => {
    const kb = state.kbs.find(k => k.id === +sel.value);
    selectKB(kb || null);
  });
}

function selectKB(kb) {
  state.currentKB      = kb;
  state.currentSession = null;

  // Close drawer when switching KBs
  closeKBDrawer();

  const toolbar = document.getElementById('chat-toolbar');

  if (kb) {
    toolbar?.classList.remove('hidden');
    document.getElementById('kb-dot').style.background = 'var(--accent)';
    document.getElementById('kb-name').textContent     = kb.name;
    document.getElementById('kb-dept').textContent     = kb.department;
    clearMessageBubbles();
    showWelcome();
    document.getElementById('chat-input').disabled = false;
    document.getElementById('send-btn').disabled   = false;
    // Check if this user can manage this KB
    checkAndShowManageButton(kb);
  } else {
    toolbar?.classList.add('hidden');
    document.getElementById('manage-kb-btn').classList.add('hidden');
    hideWelcome();
    document.getElementById('chat-input').disabled = true;
    document.getElementById('send-btn').disabled   = true;
  }
}


// â”€â”€ Sessions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
async function loadSessions() {
  try {
    state.sessions = await ChatAPI.getSessions();
    renderSessions();
  } catch { /* non-critical */ }
}

function renderSessions() {
  const list = document.getElementById('sessions-list');
  list.innerHTML = '';

  if (!state.sessions.length) {
    list.innerHTML = '<p style="padding:12px 16px;font-size:.8rem;color:var(--text-3)">No chats yet</p>';
    return;
  }

  state.sessions.forEach(session => {
    // â”€â”€ Use <div role="button"> as outer â€” never nest <button> in <button> â”€â”€
    const item = document.createElement('div');
    item.className   = `session-item ${state.currentSession?.id === session.id ? 'active' : ''}`;
    item.setAttribute('role',     'button');
    item.setAttribute('tabindex', '0');
    item.setAttribute('aria-label', session.title);

    item.innerHTML = `
      <span class="session-item-icon">
        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24"
          fill="none" stroke="currentColor" stroke-width="2">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
        </svg>
      </span>
      <span class="session-item-text">
        <span class="session-item-title truncate">${escapeHTML(session.title)}</span>
        <span class="session-item-time">${formatDate(session.created_at)}</span>
      </span>
      <button class="session-item-del"
        title="Delete chat"
        aria-label="Delete ${escapeHTML(session.title)}"
        data-id="${session.id}">
        <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24"
          fill="none" stroke="currentColor" stroke-width="2">
          <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
        </svg>
      </button>`;

    // Click outer div â†’ load session
    item.addEventListener('click', (e) => {
      if (e.target.closest('.session-item-del')) return;
      loadSession(session);
    });
    // Keyboard a11y for outer div
    item.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        loadSession(session);
      }
    });
    // Delete button
    item.querySelector('.session-item-del').addEventListener('click', (e) => {
      e.stopPropagation();
      deleteSession(session.id);
    });

    list.appendChild(item);
  });
}


async function loadSession(session) {
  const kb = state.kbs.find(k => k.id === session.kb_id);
  if (kb) {
    document.getElementById('kb-select').value = kb.id;
    selectKB(kb);                    // resets state.currentSession = null
  }
  state.currentSession = session;    // RESTORE after selectKB

  clearMessageBubbles();
  hideWelcome();
  renderSessions();

  try {
        const msgs = await ChatAPI.getMessages(session.id);
    msgs.forEach(msg => {
      let sources = [], trace = [], uiPayload = null;
      if (msg.sources) {
        try {
          const parsed = JSON.parse(msg.sources);
          if (Array.isArray(parsed)) {
            // Legacy format â€” plain array, no trace
            sources = parsed;
          } else if (parsed && typeof parsed === 'object') {
            // Current format â€” {sources: [], reasoning_trace: []}
            sources = Array.isArray(parsed.sources)         ? parsed.sources         : [];
            trace   = Array.isArray(parsed.reasoning_trace) ? parsed.reasoning_trace : [];
            uiPayload = parsed.ui_payload && typeof parsed.ui_payload === 'object' ? parsed.ui_payload : null;
          }
        } catch { /* ignore malformed stored data */ }
      }
      appendMessage(msg.role, msg.content, sources, trace, uiPayload);
    });


    const wrapper = getWrapper();
    if (wrapper) wrapper.scrollTop = wrapper.scrollHeight;
  } catch (err) {
    toast('Failed to load messages', 'error');
  }
}

async function deleteSession(id) {
  try {
    await ChatAPI.deleteSession(id);
    state.sessions = state.sessions.filter(s => s.id !== id);
    if (state.currentSession?.id === id) {
      state.currentSession = null;
      clearMessageBubbles();
      if (state.currentKB) showWelcome();
    }
    renderSessions();
    toast('Chat deleted');
  } catch (err) {
    toast(err.message, 'error');
  }
}

// â”€â”€ Input bar â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function setupInputBar() {
  const input   = document.getElementById('chat-input');
  const sendBtn = document.getElementById('send-btn');
  input.disabled   = true;
  sendBtn.disabled = true;

  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });
  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 160) + 'px';
  });
  sendBtn.addEventListener('click', sendMessage);
}

// â”€â”€ KB Management Drawer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
let chatDrawerOpen = false;

async function checkAndShowManageButton(kb) {
  const btn = document.getElementById('manage-kb-btn');
  if (!kb) { btn.classList.add('hidden'); return; }

  try {
    const perm = await ChatAPI.getKBPermission(kb.id);
    if (perm.can_manage) {
      btn.classList.remove('hidden');
    } else {
      btn.classList.add('hidden');
      closeKBDrawer();
    }
  } catch {
    btn.classList.add('hidden');
  }
}

function toggleKBDrawer() {
  chatDrawerOpen = !chatDrawerOpen;
  const drawer = document.getElementById('kb-manage-drawer');
  drawer.classList.toggle('hidden', !chatDrawerOpen);
  const btn = document.getElementById('manage-kb-btn');
  btn.classList.toggle('active', chatDrawerOpen);
  if (chatDrawerOpen) refreshChatDocs();
}

function closeKBDrawer() {
  chatDrawerOpen = false;
  document.getElementById('kb-manage-drawer').classList.add('hidden');
  document.getElementById('manage-kb-btn').classList.remove('active');
}

async function refreshChatDocs() {
  if (!state.currentKB) return;
  const list    = document.getElementById('chat-docs-list');
  const current = Auth.user();
  list.innerHTML = '<p style="color:var(--text-3);font-size:.82rem;padding:8px 0">Loadingâ€¦</p>';

  try {
    const docs = await ChatAPI.getDocuments(state.currentKB.id);
    list.innerHTML = '';

    if (!docs.length) {
      list.innerHTML = `
        <div style="text-align:center;padding:24px 0;color:var(--text-3)">
          <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24"
            fill="none" stroke="currentColor" stroke-width="1.2" style="opacity:.4;margin-bottom:8px">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
            <polyline points="14 2 14 8 20 8"/>
          </svg>
          <p style="font-size:.82rem;margin:0">No documents yet.<br>Upload one above.</p>
        </div>`;
      return;
    }

        docs.forEach(doc => {
      const statusColor = doc.status === 'ready'      ? 'badge-green'
                        : doc.status === 'processing' ? 'badge-blue'
                        : 'badge-red';

      // â”€â”€ Ownership check â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
      // Number() handles both string "3" and number 3 from the API.
      // If uploaded_by is null (legacy doc), treat as owned by no one
      // unless the user is admin.
      const currentId    = Number(current?.id);
      const uploadedById = doc.uploaded_by != null ? Number(doc.uploaded_by) : null;
      const isOwner      = current?.is_admin ||
                           (uploadedById !== null && uploadedById === currentId);

      const deleteBtn = isOwner
        ? `<button
             class="btn btn-danger btn-sm"
             style="flex-shrink:0"
             title="Delete document"
             onclick="deleteChatDoc(${doc.id}, '${escapeHTML(doc.original_filename)}')">
             <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12"
               viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
               <polyline points="3 6 5 6 21 6"/>
               <path d="M19 6l-1 14H6L5 6"/>
               <path d="M10 11v6"/><path d="M14 11v6"/>
             </svg>
           </button>`
        : `<span title="Uploaded by another user"
             style="flex-shrink:0;font-size:.72rem;color:var(--text-3);padding:0 6px;
                    cursor:default;user-select:none">ðŸ”’</span>`;

      const ownerNote = !isOwner
        ? `&middot; <span style="color:var(--text-3);font-style:italic">another user</span>`
        : '';

      const item = document.createElement('div');
      item.className = 'chat-doc-item';
      item.innerHTML = `
        <div class="chat-doc-icon">
          <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24"
            fill="none" stroke="currentColor" stroke-width="2">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
            <polyline points="14 2 14 8 20 8"/>
          </svg>
        </div>
        <div class="chat-doc-info">
          <div class="chat-doc-name" title="${escapeHTML(doc.original_filename)}">
            ${escapeHTML(doc.original_filename)}
          </div>
          <div class="chat-doc-meta">
            ${formatSize(doc.file_size)}
            &middot; ${doc.chunk_count} chunk${doc.chunk_count !== 1 ? 's' : ''}
            &middot; <span class="badge ${statusColor}"
              style="font-size:.68rem;padding:1px 6px">${doc.status}</span>
            ${ownerNote}
          </div>
        </div>
        ${deleteBtn}`;
      list.appendChild(item);
    });

  } catch (err) {
    list.innerHTML = `
      <p style="color:var(--danger);font-size:.82rem;padding:8px 0">
        âš ï¸ ${escapeHTML(err.message)}
      </p>`;
  }
}


async function deleteChatDoc(docId, filename) {
  if (!confirm(`Remove "${filename}" from this knowledge base?\n\nIts indexed chunks will be permanently deleted.`)) return;
  try {
    await ChatAPI.deleteDocument(docId);
    toast(`"${filename}" removed successfully`);
    await refreshChatDocs();
  } catch (err) {
    // Backend 403 means ownership check failed
    if (err.message.includes('only delete documents you uploaded')) {
      toast('You can only delete documents you uploaded yourself', 'error');
    } else {
      toast(err.message, 'error');
    }
  }
}


function initChatUploadZone() {
  const zone  = document.getElementById('chat-upload-zone');
  const input = document.getElementById('chat-file-input');
  if (!zone || !input || zone.dataset.bound === 'true') return;

  zone.dataset.bound = 'true';

  zone.addEventListener('click', (e) => {
    if (e.target.tagName === 'BUTTON') return;  // don't trigger on delete btns
    input.click();
  });
  zone.addEventListener('dragover',  (e) => { e.preventDefault(); zone.classList.add('dragover'); });
  zone.addEventListener('dragleave', ()  => zone.classList.remove('dragover'));
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('dragover');
    if (e.dataTransfer.files?.length) handleChatUpload(e.dataTransfer.files);
  });
  input.addEventListener('change', () => {
    if (input.files?.length) handleChatUpload(input.files);
    input.value = '';
  });
}

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

async function handleChatUpload(fileList) {
  if (!fileList || !state.currentKB) return;
  const zone  = document.getElementById('chat-upload-zone');
  const input = document.getElementById('chat-file-input');
  let files = [];
  try {
    files = normalizeUploadFiles(fileList);
  } catch (err) {
    toast(err.message, 'error');
    return;
  }
  if (!files.length) return;

  // Swap to loading state â€” keep same elements, just change content
  const originalHTML = zone.innerHTML;
  const label = files.length === 1 ? files[0].name : `${files.length} files`;
  zone.innerHTML = `
    <div class="spinner" style="margin:0 auto;display:block"></div>
    <p style="margin-top:10px;color:var(--text-2);font-size:.875rem">
      Uploading &amp; queuing "${escapeHTML(label)}"â€¦
    </p>
    <span>OCR will run automatically on images and scanned PDFs</span>`;

  try {
    await ChatAPI.uploadDocuments(state.currentKB.id, files);
    toast(`${files.length} file${files.length > 1 ? 's' : ''} queued for indexing`);
    refreshChatDocs();
  } catch (err) {
    toast(err.message, 'error');
  } finally {
    // Restore original HTML â€” dataset.bound stays true so no re-binding needed
    zone.innerHTML = originalHTML;
    if (input) input.value = '';
  }
}



async function sendMessage() {
  const input = document.getElementById('chat-input');
  const text  = input.value.trim();
  if (!text || !state.currentKB || state.isLoading) return;

  state.isLoading = true;
  input.value     = '';
  input.style.height = 'auto';
  document.getElementById('send-btn').disabled = true;

  hideWelcome();
  appendMessage('user', text);

  const typingId = 'typing-' + Date.now();
  const inner    = getInner();
  const wrapper  = getWrapper();

  inner?.insertAdjacentHTML('beforeend', `
    <div id="${typingId}" class="msg assistant">
      <div class="msg-avatar">AI</div>
      <div class="msg-body">
        <div class="typing-dots"><span></span><span></span><span></span></div>
      </div>
    </div>`);
  if (wrapper) wrapper.scrollTop = wrapper.scrollHeight;

  try {
    const res = await ChatAPI.sendMessage({
      message:    text,
      kb_id:      state.currentKB.id,
      session_id: state.currentSession?.id || null,
      fast_mode:  getFastModeEnabled(),
    });

    document.getElementById(typingId)?.remove();

    if (!state.currentSession) {
      state.currentSession = { id: res.session_id };
      await loadSessions();
    }

    // Normalize response fields (supports current + legacy payloads)
    const answer = (typeof res?.answer === 'string' && res.answer.trim())
      ? res.answer
      : (typeof res?.generation === 'string' && res.generation.trim())
        ? res.generation
        : 'No answer returned.';

    const sources = Array.isArray(res?.sources)
      ? res.sources
      : Array.isArray(res?.sources?.sources)
        ? res.sources.sources
        : [];

    const traceRaw = Array.isArray(res?.reasoning_trace)
      ? res.reasoning_trace
      : Array.isArray(res?.sources?.reasoning_trace)
        ? res.sources.reasoning_trace
        : [];

    const trace = traceRaw
      .map(t => typeof t === 'string' ? t : '')
      .filter(t => t.trim().length > 0);

    // Always render assistant message
    appendMessage('assistant', answer, sources, trace, (res && typeof res.ui_payload === 'object') ? res.ui_payload : null);

    

  } catch (err) {
    document.getElementById(typingId)?.remove();
    appendMessage('assistant', `âš ï¸ ${err.message}`);
    toast(err.message, 'error');
  } finally {
    state.isLoading = false;
    document.getElementById('send-btn').disabled = false;
    document.getElementById('chat-input').focus();
    if (wrapper) wrapper.scrollTop = wrapper.scrollHeight;
  }
}

// â”€â”€ Message renderer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function appendMessage(role, content, sources = [], trace = [], uiPayload = null) {
  const inner   = getInner();
  const wrapper = getWrapper();
  if (!inner || !wrapper) return;

  const isUser     = role === 'user';
  const user       = Auth.user();
  const avatarText = isUser
    ? (user?.full_name || user?.username || 'U')[0].toUpperCase()
    : 'AI';

  // â”€â”€ Reasoning trace â€” sanitize each step â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  const traceHTML = (!isUser && trace && trace.length > 1)
    ? `<details class="reasoning-trace">
        <summary>
          <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24"
            fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"/>
            <line x1="12" y1="8" x2="12" y2="12"/>
            <line x1="12" y1="16" x2="12.01" y2="16"/>
          </svg>
          Agent reasoning (${trace.length} steps)
        </summary>
        <ol class="trace-steps">
          ${trace.map(t => `<li>${sanitize(
            // Only allow bold/italic in trace text â€” no raw HTML from model
            escapeHTML(String(t ?? ''))
              .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
              .replace(/\*(.*?)\*/g, '<em>$1</em>')
          )}</li>`).join('')}
        </ol>
       </details>`
    : '';

  // â”€â”€ Sources â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  const pipelineLabel = {
  docling:  { text: 'ðŸ“Š Docling',  color: '#6366f1' },
  vlm:      { text: 'ðŸ‘ VLM',      color: '#f59e0b' },
  standard: { text: 'ðŸ“„ Text',     color: '#10b981' },
};

const sourcesHTML = (sources && sources.length)
  ? `<button class="sources-toggle" onclick="toggleSources(this)">â€¦${sources.length} source${sources.length > 1 ? 's' : ''}</button>
     <div class="sources-list hidden">
      ${sources.map(s => {
        const pl = pipelineLabel[s.pipeline] || pipelineLabel.standard;
        return `
          <div class="source-chip">
            <div class="source-chip-name">
              ${escapeHTML(s.source || 'Source')}
              ${s.page ? `<span style="color:var(--text-3)"> Â· p.${escapeHTML(s.page)}</span>` : ''}
              <span style="font-size:.68rem;padding:1px 6px;border-radius:4px;
                           background:${pl.color}22;color:${pl.color};margin-left:4px">
                ${pl.text}
              </span>
            </div>
            <div class="source-chip-text">${escapeHTML(s.content || '')}</div>
          </div>`;
      }).join('')}
     </div>`
  : '';

  const chartId = `chart-${Date.now()}-${Math.floor(Math.random() * 10000)}`;
  const chartHTML = (!isUser && uiPayload && uiPayload.type === 'chart')
    ? `<div class="msg-ui-card">
         <div class="msg-ui-title">${escapeHTML(uiPayload.title || 'Chart')}</div>
         <div class="msg-ui-canvas-wrap">
           <canvas id="${chartId}" height="220"></canvas>
         </div>
       </div>`
    : '';
         

  const el = document.createElement('div');
  el.className = `msg ${role}`;
  el.innerHTML = `
    <div class="msg-avatar">${escapeHTML(avatarText)}</div>
    <div class="msg-body">
      <div class="msg-bubble prose">${isUser
        ? escapeHTML(content)
        : sanitize(renderMarkdown(content))}</div>
      ${chartHTML}
      ${traceHTML}
      ${sourcesHTML}
      <div class="msg-time">${new Date().toLocaleTimeString([], {
        hour: '2-digit', minute: '2-digit'
      })}</div>
    </div>`;

  inner.appendChild(el);
  if (!isUser && uiPayload && uiPayload.type === 'chart') {
    renderGeneratedChart(chartId, uiPayload);
  }
  wrapper.scrollTop = wrapper.scrollHeight;
}


function toggleSources(btn) {
  const list = btn.nextElementSibling;
  list?.classList.toggle('hidden');
}

function renderGeneratedChart(canvasId, uiPayload) {
  if (typeof window.Chart === 'undefined') return;
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;

  const labels = Array.isArray(uiPayload.labels) ? uiPayload.labels : [];
  const datasets = Array.isArray(uiPayload.datasets) ? uiPayload.datasets : [];
  const palette = ['#7c3aed', '#06b6d4', '#10b981', '#f59e0b', '#ef4444', '#3b82f6'];
  const chartType = (uiPayload.chart_type === 'line') ? 'line' : 'bar';

  const mappedDatasets = datasets.map((ds, idx) => ({
    label: ds.label || `Series ${idx + 1}`,
    data: Array.isArray(ds.data) ? ds.data : [],
    backgroundColor: palette[idx % palette.length] + (chartType === 'bar' ? '99' : '00'),
    borderColor: palette[idx % palette.length],
    borderWidth: 2,
    tension: 0.25,
    fill: chartType !== 'line',
  }));

  new window.Chart(canvas, {
    type: chartType,
    data: { labels, datasets: mappedDatasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: { legend: { labels: { color: '#e4e4e7' } } },
      scales: {
        x: { ticks: { color: '#a1a1aa' }, grid: { color: 'rgba(161,161,170,0.15)' } },
        y: { ticks: { color: '#a1a1aa' }, grid: { color: 'rgba(161,161,170,0.15)' } },
      },
    },
  });
}

// â”€â”€ Utils â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function escapeHTML(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}
function formatDate(iso) {
  return new Date(iso).toLocaleDateString([], { month: 'short', day: 'numeric' });
}

function formatSize(bytes) {
  if (!bytes)          return '0 B';
  if (bytes < 1024)    return `${bytes} B`;
  if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1048576).toFixed(1)} MB`;
}



document.addEventListener('DOMContentLoaded', () => {
  initAuth();
  initChatUploadZone();
});



