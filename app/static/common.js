function esc(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

async function apiFetch(url, opts = {}) {
  const res = await fetch(url, { ...opts, credentials: 'same-origin' });
  if (res.status === 401) {
    window.location.href = '/login';
    return new Promise(() => {}); // navigation is taking over; never resolve
  }
  return res;
}

async function requireAuth() {
  const res = await apiFetch('/api/v1/auth/me');
  return res.json();
}

function getStoredTheme() {
  return localStorage.getItem('cloner-theme');
}

function applyTheme(theme) {
  if (theme === 'dark' || theme === 'light') {
    document.documentElement.setAttribute('data-theme', theme);
  } else {
    document.documentElement.removeAttribute('data-theme');
  }
}

function setTheme(theme) {
  localStorage.setItem('cloner-theme', theme);
  applyTheme(theme);
}

function currentEffectiveTheme() {
  const stored = getStoredTheme();
  if (stored) return stored;
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

const THEME_ICONS = {
  dark: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>',
  light: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>',
};

function renderNav(activePage) {
  const nav = document.getElementById('nav');
  if (!nav) return;
  const links = [
    { id: 'generate', href: '/generate', label: 'Generate', icon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>' },
    { id: 'voices', href: '/voices', label: 'Voices', icon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="22"/></svg>' },
    { id: 'history', href: '/history', label: 'History', icon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>' },
  ];

  const effectiveTheme = currentEffectiveTheme();

  nav.innerHTML =
    '<div class="sidebar-brand">Shona Voice Cloner</div>' +
    '<div class="sidebar-links">' +
    links.map(l => `<a href="${l.href}" class="${l.id === activePage ? 'active' : ''}">${l.icon}${l.label}</a>`).join('') +
    '</div>' +
    `<button type="button" class="theme-toggle" id="themeToggle" aria-label="Toggle dark mode">${THEME_ICONS[effectiveTheme === 'dark' ? 'light' : 'dark']}<span id="themeToggleLabel">${effectiveTheme === 'dark' ? 'Light mode' : 'Dark mode'}</span></button>` +
    '<a href="#" id="logoutLink" class="sidebar-logout">Log out</a>';

  document.getElementById('themeToggle').addEventListener('click', () => {
    const next = currentEffectiveTheme() === 'dark' ? 'light' : 'dark';
    setTheme(next);
    renderNav(activePage);
  });

  document.getElementById('logoutLink').addEventListener('click', async (e) => {
    e.preventDefault();
    await fetch('/api/v1/auth/logout', { method: 'POST', credentials: 'same-origin' });
    window.location.href = '/login';
  });
}

// Apply the persisted/system theme immediately so pages don't flash the wrong theme.
applyTheme(getStoredTheme());
