// ── Base config ────────────────────────────────────────────────────────────────
const BASE = '/api';

// ── Token helpers ──────────────────────────────────────────────────────────────
const Auth = {
  token:   () => localStorage.getItem('rag_token'),
  user:    () => JSON.parse(localStorage.getItem('rag_user') || 'null'),
  save:    (token, user) => {
    localStorage.setItem('rag_token', token);
    localStorage.setItem('rag_user', JSON.stringify(user));
  },
  clear:   () => {
    localStorage.removeItem('rag_token');
    localStorage.removeItem('rag_user');
  },
  isAdmin: () => Auth.user()?.is_admin === true,
};

// ── Core fetch wrapper ─────────────────────────────────────────────────────────
async function api(method, path, body = null, isForm = false) {
  const headers = {};
  const token   = Auth.token();
  if (token) headers['Authorization'] = `Bearer ${token}`;

  let fetchBody = null;
  if (body) {
    if (isForm) {
      fetchBody = body;           // FormData — browser sets Content-Type automatically
    } else {
      headers['Content-Type'] = 'application/json';
      fetchBody = JSON.stringify(body);
    }
  }

  const res = await fetch(`${BASE}${path}`, { method, headers, body: fetchBody });

  if (res.status === 401) {
    Auth.clear();
    window.location.reload();
    return;
  }

  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
  return data;
}

// ── Auth API ───────────────────────────────────────────────────────────────────
const AuthAPI = {
  register: (payload) => api('POST', '/auth/register', payload),

  login: async (username, password) => {
    const form = new URLSearchParams();
    form.append('username', username);
    form.append('password', password);
    const res = await fetch(`${BASE}/auth/login`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body:    form,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || 'Login failed');
    return data;
  },

  me: () => api('GET', '/auth/me'),
};

// ── Chat API ───────────────────────────────────────────────────────────────────
const ChatAPI = {
  getKBs:          ()           => api('GET',    '/chat/knowledge-bases'),
  getKBPermission: (kbId)       => api('GET',    `/chat/kb-permission/${kbId}`),
  getSessions:     ()           => api('GET',    '/chat/sessions'),
  getMessages:     (sessionId)  => api('GET',    `/chat/sessions/${sessionId}/messages`),
  deleteSession:   (sessionId)  => api('DELETE', `/chat/sessions/${sessionId}`),
  sendMessage:     (payload)    => api('POST',   '/chat/message', payload),
  // Document management for manage-level users (reuses /documents/* routes)
  getDocuments:    (kbId)       => api('GET',    `/documents/${kbId}`),
  deleteDocument:  (docId)      => api('DELETE', `/documents/${docId}`),
  uploadDocument:  (kbId, file) => {
    const form = new FormData();
    form.append('file', file);
    return api('POST', `/documents/upload/${kbId}`, form, true);
  },
  uploadDocuments: (kbId, files) => {
    const form = new FormData();
    [...files].forEach(file => form.append('files', file));
    return api('POST', `/documents/upload-multiple/${kbId}`, form, true);
  },
};


// ── Admin API ──────────────────────────────────────────────────────────────────
const AdminAPI = {
  // System
  getStats:       ()               => api('GET', '/admin/stats'),
  getOllamaModels:()               => api('GET', '/admin/ollama/models'),

  // Personalities
  getPersonalities:  ()            => api('GET',    '/admin/personalities'),
  createPersonality: (p)           => api('POST',   '/admin/personalities', p),
  updatePersonality: (id, p)       => api('PUT',    `/admin/personalities/${id}`, p),
  deletePersonality: (id)          => api('DELETE', `/admin/personalities/${id}`),

  // Knowledge Bases
  getKBs:    ()            => api('GET',    '/admin/knowledge-bases'),
  createKB:  (payload)     => api('POST',   '/admin/knowledge-bases', payload),
  updateKB:  (id, payload) => api('PUT',    `/admin/knowledge-bases/${id}`, payload),
  deleteKB:  (id)          => api('DELETE', `/admin/knowledge-bases/${id}`),

  // Groups
  getGroups:          ()            => api('GET',    '/admin/groups'),
  createGroup:        (payload)     => api('POST',   '/admin/groups', payload),
  updateGroup:        (id, payload) => api('PUT',    `/admin/groups/${id}`, payload),
  deleteGroup:        (id)          => api('DELETE', `/admin/groups/${id}`),
  getGroupMembers:    (id)          => api('GET',    `/admin/groups/${id}/members`),
  setGroupMembers:    (id, userIds) => api('PUT',    `/admin/groups/${id}/members`, userIds),
  setGroupPermissions:(id, perms)   => api('PUT',    `/admin/groups/${id}/permissions`, perms),

  // Users
  getUsers:   ()               => api('GET',    '/admin/users'),
  updateUser: (id, payload)    => api('PUT',    `/admin/users/${id}`, payload),
  deleteUser: (id)             => api('DELETE', `/admin/users/${id}`),

  // Documents
  getDocuments:   (kbId)        => api('GET',    `/documents/${kbId}`),
  deleteDocument: (docId)       => api('DELETE', `/documents/${docId}`),
  uploadDocument: (kbId, file)  => {
    const form = new FormData();
    form.append('file', file);
    return api('POST', `/documents/upload/${kbId}`, form, true);
  },
  uploadDocuments: (kbId, files)  => {
    const form = new FormData();
    [...files].forEach(file => form.append('files', file));
    return api('POST', `/documents/upload-multiple/${kbId}`, form, true);
  },
};
