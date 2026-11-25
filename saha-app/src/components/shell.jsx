// Shared dark window chrome + nav rail.
// Matched to Imagen AI desktop reference (sidebar tiles + topbar chrome).
// Window controls are delegated to Electron so the same shell works on macOS, Windows, and Linux.

import { useState } from 'react';
import SONNA from '../tokens.js';
import { ConfirmModal } from './confirm-modal.jsx';
import { UploaderPill, ManageUploadsPopover } from './uploader-popover.jsx';

// Clean vector icons matching Imagen AI desktop sidebar
function RailIcon({ kind, active, theme = 'dark' }) {
  const isLight = theme === 'light';
  const stroke = active
    ? (isLight ? '#1c1917' : '#FFFFFF')
    : (isLight ? '#71717a' : '#9E9689');
  const sw = 1.6;

  if (kind === 'home') {
    return (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke={stroke} strokeWidth={sw} strokeLinecap="round" strokeLinejoin="round">
        <path d="M3 10.5L12 3l9 7.5" />
        <path d="M5 9.5V20a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V9.5" />
        <path d="M10 21v-6a1 1 0 0 1 1-1h2a1 1 0 0 1 1 1v6" />
      </svg>
    );
  }
  if (kind === 'profiles' || kind === 'profile') {
    return (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke={stroke} strokeWidth={sw} strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="8" r="3" />
        <path d="M6 19c0-3.3 2.7-5 6-5s6 1.7 6 5" />
        <circle cx="4" cy="10" r="0.8" fill={stroke} stroke="none" />
        <circle cx="20" cy="10" r="0.8" fill={stroke} stroke="none" />
        <path d="M4.5 5.5l1 1M19.5 5.5l-1 1" strokeWidth="1.3" />
      </svg>
    );
  }
  if (kind === 'projects') {
    return (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke={stroke} strokeWidth={sw} strokeLinecap="round" strokeLinejoin="round">
        <rect x="3.5" y="4" width="17" height="15" rx="2.5" />
        <circle cx="8" cy="8.5" r="1.3" fill={stroke} stroke="none" />
        <path d="M4.5 16l4.5-4 4 3.5 2.5-2 4 3.5" />
      </svg>
    );
  }
  if (kind === 'logout') {
    return (
      <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke={SONNA.ochre} strokeWidth={sw} strokeLinecap="round" strokeLinejoin="round">
        <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
        <polyline points="16 17 21 12 16 7" />
        <line x1="21" y1="12" x2="9" y2="12" />
      </svg>
    );
  }
  return null;
}

// Side Menu Navigation Rail with exactly Home, AI Profiles, and Projects
export function NavRail({
  active = 'home',
  onNavigate,
  onLogout,
  theme = 'dark',
}) {
  const isLight = theme === 'light';
  const [logoutConfirmOpen, setLogoutConfirmOpen] = useState(false);

  const items = [
    { kind: 'home', label: 'Home' },
    { kind: 'profiles', label: 'AI Profiles' },
    { kind: 'projects', label: 'Projects' },
  ];

  return (
    <div style={{
      width: 78,
      flexShrink: 0,
      background: SONNA.bgPanel,
      borderRight: `1px solid ${SONNA.lineSoft}`,
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      padding: '12px 0 16px 0',
      userSelect: 'none',
    }}>
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: 8,
        width: '100%',
      }}>
        {items.map(({ kind, label }) => {
          const isActive = kind === active;
          const click = onNavigate ? () => onNavigate(kind) : undefined;
          return (
            <button
              key={kind}
              type="button"
              onClick={click}
              title={label}
              aria-label={label}
              style={{
                width: 64,
                height: 56,
                borderRadius: 8,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 3,
                background: isActive
                  ? (isLight ? 'rgba(0,0,0,0.06)' : 'rgba(255, 255, 255, 0.08)')
                  : 'transparent',
                border: isActive
                  ? (isLight ? '1px solid rgba(0,0,0,0.24)' : '1px solid rgba(255, 255, 255, 0.38)')
                  : '1px solid transparent',
                color: isActive
                  ? (isLight ? '#1c1917' : '#FFFFFF')
                  : (isLight ? '#71717a' : '#9E9689'),
                cursor: 'pointer',
                position: 'relative',
                padding: '6px 2px 4px 2px',
                transition: 'all 0.15s ease',
              }}
            >
              <RailIcon kind={kind} active={isActive} theme={theme} />
              <span
                style={{
                  fontSize: 10,
                  fontWeight: isActive ? 600 : 500,
                  lineHeight: 1.15,
                  textAlign: 'center',
                  whiteSpace: 'normal',
                  wordBreak: 'break-word',
                  maxWidth: 60,
                  color: 'inherit',
                  marginTop: 1,
                }}
              >
                {label}
              </span>
            </button>
          );
        })}
      </div>

      <div style={{ flex: 1 }} />

      {/* Logout button at bottom of sidebar */}
      <button
        type="button"
        onClick={() => setLogoutConfirmOpen(true)}
        title="Logout"
        aria-label="Logout"
        style={{
          width: 44,
          height: 38,
          borderRadius: 8,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          border: `1px solid ${isLight ? 'rgba(0,0,0,0.1)' : 'rgba(255,255,255,0.1)'}`,
          background: 'transparent',
          color: SONNA.ochre,
          cursor: 'pointer',
          padding: 0,
          transition: 'all 0.15s ease',
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.borderColor = SONNA.ochre;
          e.currentTarget.style.background = SONNA.ochreTint;
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.borderColor = isLight ? 'rgba(0,0,0,0.1)' : 'rgba(255,255,255,0.1)';
          e.currentTarget.style.background = 'transparent';
        }}
      >
        <RailIcon kind="logout" active={false} theme={theme} />
      </button>

      {/* Logout Confirm Modal matching theme */}
      <ConfirmModal
        isOpen={logoutConfirmOpen}
        title="Log out of Saha?"
        message="Are you sure you want to log out of your session?"
        confirmText="Log out"
        cancelText="Cancel"
        onConfirm={() => {
          setLogoutConfirmOpen(false);
          onLogout?.();
        }}
        onCancel={() => setLogoutConfirmOpen(false)}
        theme={theme}
      />
    </div>
  );
}

function ThemeIcon({ isLight }) {
  if (isLight) {
    return (
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="5" />
        <line x1="12" y1="1" x2="12" y2="3" />
        <line x1="12" y1="21" x2="12" y2="23" />
        <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
        <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
        <line x1="1" y1="12" x2="3" y2="12" />
        <line x1="21" y1="12" x2="23" y2="12" />
        <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
        <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
      </svg>
    );
  }
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
    </svg>
  );
}

// Window Chrome Header matching Imagen AI top bar
export function TitleBar({
  folder = '',
  theme = 'dark',
  onToggleTheme,
  onOpenNewProject,
  onNavigate,
  uploads = [],
  isUpdating = false,
  updatingProgress = 37,
  onCancelUpload,
  onPauseAll,
  onResumeAll,
  onDismissCompleted,
  isUploadsOpen: controlledIsOpen,
  onToggleUploadsOpen: controlledToggle,
}) {
  const isMac = typeof window !== 'undefined' && window.saha?.platform === 'darwin';
  const logoLeft = isMac ? 78 : 16;
  const isLight = theme === 'light';
  const [localUploadsOpen, setLocalUploadsOpen] = useState(false);
  const isUploadsOpen = controlledIsOpen !== undefined ? controlledIsOpen : localUploadsOpen;
  const toggleUploads = () => {
    if (controlledToggle) {
      controlledToggle(!isUploadsOpen);
    } else {
      setLocalUploadsOpen(!localUploadsOpen);
    }
  };

  return (
    <div style={{
      height: 48,
      flexShrink: 0,
      background: SONNA.bgDeep,
      borderBottom: `1px solid ${SONNA.lineSoft}`,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: `0 16px 0 ${logoLeft}px`,
      position: 'relative',
      WebkitAppRegion: 'drag',
      userSelect: 'none',
    }}>
      {/* Brand logo button: elegant serif lowercase typography in theme color */}
      <button
        type="button"
        onClick={() => onNavigate?.('home')}
        title="Go to Home"
        aria-label="Saha Home"
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          background: 'transparent',
          border: 'none',
          padding: 0,
          margin: 0,
          cursor: onNavigate ? 'pointer' : 'default',
          WebkitAppRegion: 'no-drag',
        }}
      >
        <span
          style={{
            fontFamily: 'Georgia, "Playfair Display", serif',
            fontStyle: 'italic',
            fontSize: 21,
            fontWeight: 700,
            letterSpacing: '-0.03em',
            color: SONNA.ochre,
            transition: 'opacity 0.15s ease',
          }}
          onMouseEnter={(e) => { e.currentTarget.style.opacity = '0.82'; }}
          onMouseLeave={(e) => { e.currentTarget.style.opacity = '1'; }}
        >
          saha
        </span>
      </button>

      {/* Center shoot / folder context */}
      <div style={{
        position: 'absolute',
        left: 0,
        right: 0,
        textAlign: 'center',
        fontSize: 12,
        fontWeight: 500,
        color: SONNA.fgMute,
        pointerEvents: 'none',
      }}>
        {folder && <span style={{ color: SONNA.fgDim }}>{folder}</span>}
      </div>

      {/* Right controls matching Imagen AI reference screenshots */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 12,
        WebkitAppRegion: 'no-drag',
        position: 'relative',
      }}>
        {/* Uploader Pill matching Imagen */}
        <UploaderPill
          activeUploads={uploads}
          isUpdating={isUpdating}
          isOpen={isUploadsOpen}
          onClick={toggleUploads}
          theme={theme}
        />

        {/* Theme toggle button */}
        {onToggleTheme && (
          <button
            type="button"
            onClick={onToggleTheme}
            title={isLight ? 'Switch to Dark Theme' : 'Switch to Light Theme'}
            aria-label="Toggle color theme"
            style={{
              width: 30,
              height: 30,
              borderRadius: 6,
              background: 'transparent',
              border: `1px solid ${isLight ? 'rgba(0,0,0,0.16)' : 'rgba(255,255,255,0.18)'}`,
              color: isLight ? '#71717a' : '#E8E2D6',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: 'pointer',
              padding: 0,
              fontSize: 13,
              transition: 'all 0.15s ease',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.borderColor = isLight ? 'rgba(0,0,0,0.4)' : 'rgba(255,255,255,0.45)';
              e.currentTarget.style.background = isLight ? 'rgba(0,0,0,0.06)' : 'rgba(255,255,255,0.08)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.borderColor = isLight ? 'rgba(0,0,0,0.16)' : 'rgba(255,255,255,0.18)';
              e.currentTarget.style.background = 'transparent';
            }}
          >
            <ThemeIcon isLight={isLight} />
          </button>
        )}

        {/* Floating Manage Uploads Popover */}
        <ManageUploadsPopover
          isOpen={isUploadsOpen}
          onClose={() => {
            if (controlledToggle) controlledToggle(false);
            else setLocalUploadsOpen(false);
          }}
          uploads={uploads}
          isUpdating={isUpdating}
          updatingProgress={updatingProgress}
          onCancelUpload={onCancelUpload}
          onPauseAll={onPauseAll}
          onResumeAll={onResumeAll}
          onDismissCompleted={onDismissCompleted}
          theme={theme}
        />
      </div>
    </div>
  );
}

// AppShell — outer frame; child renders below the rail.
export function AppShell({
  folder,
  activeNav = 'home',
  onNavigate,
  theme,
  onToggleTheme,
  onLogout,
  onOpenNewProject,
  uploads = [],
  isUpdating = false,
  updatingProgress = 37,
  onCancelUpload,
  onPauseAll,
  onResumeAll,
  onDismissCompleted,
  isUploadsOpen,
  onToggleUploadsOpen,
  children,
}) {
  return (
    <div style={{
      width: '100%',
      height: '100%',
      background: SONNA.bgDeep,
      color: SONNA.fg,
      fontFamily: SONNA.font,
      display: 'flex',
      flexDirection: 'column',
      overflow: 'hidden',
    }}>
      <TitleBar
        folder={folder}
        theme={theme}
        onToggleTheme={onToggleTheme}
        onOpenNewProject={onOpenNewProject}
        onNavigate={onNavigate}
        uploads={uploads}
        isUpdating={isUpdating}
        updatingProgress={updatingProgress}
        onCancelUpload={onCancelUpload}
        onPauseAll={onPauseAll}
        onResumeAll={onResumeAll}
        onDismissCompleted={onDismissCompleted}
        isUploadsOpen={isUploadsOpen}
        onToggleUploadsOpen={onToggleUploadsOpen}
      />
      <div style={{ flex: 1, display: 'flex', minHeight: 0 }}>
        <NavRail
          active={activeNav}
          onNavigate={onNavigate}
          onLogout={onLogout}
          theme={theme}
        />
        <div style={{ flex: 1, display: 'flex', minWidth: 0, overflow: 'hidden' }}>
          {children}
        </div>
      </div>
    </div>
  );
}

// Tiny utility components
export function Label({ children, style }) {
  return (
    <div style={{
      fontSize: 10,
      fontWeight: 600,
      color: SONNA.fgDim,
      textTransform: 'uppercase',
      letterSpacing: 0.6,
      ...style,
    }}>
      {children}
    </div>
  );
}

export function Hairline({ vertical, color = SONNA.line, style }) {
  return (
    <div style={{
      background: color,
      [vertical ? 'width' : 'height']: 1,
      [vertical ? 'height' : 'width']: '100%',
      flexShrink: 0,
      ...style,
    }} />
  );
}

export function Mono({ children, style }) {
  return <span style={{ fontFamily: SONNA.mono, ...style }}>{children}</span>;
}
