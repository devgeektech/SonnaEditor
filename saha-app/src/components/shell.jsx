// Shared dark window chrome + nav rail.
// Ported from "SAHA UI/shell.jsx" (IIFE -> ESM). Window controls are delegated
// to Electron so the same shell works on macOS, Windows, and Linux.

import SONNA from '../tokens.js';
import { SahaMark } from './logo.jsx';

// nav rail icons, 14×14, drawn as simple shapes
function RailIcon({ kind, active }) {
  const stroke = active ? SONNA.ochre : SONNA.fgMute;
  const sw = 1.4;
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
      {kind === 'process' && (
        <>
          <rect x="2" y="2.5" width="12" height="11" rx="1.5" stroke={stroke} strokeWidth={sw} />
          <path d="M2 6.5h12" stroke={stroke} strokeWidth={sw} />
          <circle cx="4.2" cy="4.5" r="0.5" fill={stroke} />
          <circle cx="6" cy="4.5" r="0.5" fill={stroke} />
        </>
      )}
      {kind === 'home' && (
        <>
          <path d="M2.5 7.2 8 2.7l5.5 4.5" stroke={stroke} strokeWidth={sw} strokeLinecap="round" strokeLinejoin="round" />
          <path d="M4 6.8v6h8v-6" stroke={stroke} strokeWidth={sw} strokeLinejoin="round" />
          <path d="M6.8 12.8V9.4h2.4v3.4" stroke={stroke} strokeWidth={sw} strokeLinecap="round" />
        </>
      )}
      {kind === 'profile' && (
        <>
          <circle cx="8" cy="6" r="2.6" stroke={stroke} strokeWidth={sw} />
          <path d="M3 13.5c0.6-2.6 2.6-4 5-4s4.4 1.4 5 4" stroke={stroke} strokeWidth={sw} strokeLinecap="round" />
        </>
      )}
      {kind === 'settings' && (
        <>
          <circle cx="8" cy="8" r="3.2" stroke={stroke} strokeWidth={sw} />
          <path d="M8 1.4v1.7M8 12.9v1.7M14.6 8h-1.7M3.1 8H1.4M12.7 3.3l-1.2 1.2M4.5 11.5l-1.2 1.2M12.7 12.7l-1.2-1.2M4.5 4.5 3.3 3.3" stroke={stroke} strokeWidth={sw} strokeLinecap="round" />
        </>
      )}
      {kind === 'projects' && (
        <>
          <path d="M2.5 4.5h4l1 1.5h6v6.5h-11v-8z" stroke={stroke} strokeWidth={sw} strokeLinejoin="round" />
          <path d="M4.5 8h7M4.5 10h4.5" stroke={stroke} strokeWidth={sw} strokeLinecap="round" />
        </>
      )}
      {kind === 'logout' && (
        <>
          <path d="M6.6 3.2h-3v9.6h3" stroke={stroke} strokeWidth={sw} strokeLinecap="round" strokeLinejoin="round" />
          <path d="M7.8 8h5.2M10.9 5.8 13.1 8l-2.2 2.2" stroke={stroke} strokeWidth={sw} strokeLinecap="round" strokeLinejoin="round" />
        </>
      )}
    </svg>
  );
}

function ThemeIcon({ active }) {
  const stroke = active ? SONNA.ochre : SONNA.fgMute;
  return (
    <svg width="15" height="15" viewBox="0 0 15 15" fill="none">
      <circle cx="7.5" cy="7.5" r="3.2" stroke={stroke} strokeWidth="1.3" />
      <path d="M7.5 1.2v1.7M7.5 12.1v1.7M13.8 7.5h-1.7M2.9 7.5H1.2M12 3l-1.2 1.2M4.2 10.8 3 12M12 12l-1.2-1.2M4.2 4.2 3 3" stroke={stroke} strokeWidth="1.3" strokeLinecap="round" />
    </svg>
  );
}

export function NavRail({
  active = 'process',
  accent = false,
  accentColor = SONNA.ochre,
  onNavigate,
  onLogout,
}) {
  const items = [
    { kind: 'home', enabled: true, label: 'Home' },
    { kind: 'profiles', enabled: true, label: 'AI Profile' },
    { kind: 'projects', enabled: true, label: 'Projects' },
  ];
  return (
    <div style={{
      width: 44, flexShrink: 0,
      background: SONNA.bgPanel,
      borderRight: `1px solid ${SONNA.line}`,
      display: 'flex', flexDirection: 'column', alignItems: 'center',
      padding: '14px 0',
    }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        {items.map(({ kind, enabled, label }) => {
          const isActive = kind === active;
          const click = enabled && onNavigate ? () => onNavigate(kind) : undefined;
          return (
            <div key={kind}
              onClick={click}
              title={label}
              aria-label={label}
              style={{
                width: 32, height: 32, borderRadius: 6,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                background: isActive ? SONNA.bgLifted : 'transparent',
                position: 'relative',
                cursor: enabled ? 'pointer' : 'default',
                opacity: enabled ? 1 : 0.4,
              }}>
              {accent && isActive && (
                <div style={{
                  position: 'absolute', left: -6, top: 8, bottom: 8, width: 2,
                  background: accentColor, borderRadius: 1,
                }} />
              )}
              <RailIcon kind={kind === 'profiles' ? 'profile' : kind} active={isActive} />
            </div>
          );
        })}
      </div>
      <div style={{ flex: 1 }} />
      <button
        type="button"
        onClick={() => {
          if (window.confirm('Do you want to logout?')) {
            onLogout?.();
          }
        }}
        title="Logout"
        aria-label="Logout"
        style={{
          width: 32, height: 32, borderRadius: 6,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          border: `1px solid ${SONNA.line}`,
          background: SONNA.bgPanel,
          color: SONNA.fg,
          opacity: 1,
          cursor: 'pointer',
          padding: 0,
        }}
      >
        <RailIcon kind="logout" active />
      </button>
    </div>
  );
}

function HeaderThemeButton({ theme = SONNA.theme, onToggleTheme }) {
  return (
    <div style={{
      position: 'absolute',
      right: 14,
      top: 6,
      WebkitAppRegion: 'no-drag',
    }}>
      <button
        type="button"
        onClick={onToggleTheme}
        title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}
        aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}
        style={{
          width: 32,
          height: 32,
          padding: 0,
          borderRadius: 6,
          border: `1px solid ${SONNA.line}`,
          background: SONNA.bgPanel,
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <ThemeIcon active={theme === 'light'} />
      </button>
    </div>
  );
}

export function TitleBar({ folder = '', theme, onToggleTheme }) {
  const isMac = typeof window !== 'undefined' && window.saha?.platform === 'darwin';
  const logoLeft = isMac ? 78 : 14;
  const contentLeftPadding = isMac ? 122 : 86;

  return (
    <div style={{
      // Keep enough drag area for native window controls across platforms.
      // The absolute-positioned title remains centered relative to the window.
      height: 44, flexShrink: 0,
      background: SONNA.bgDeep,
      borderBottom: `1px solid ${SONNA.line}`,
      display: 'flex', alignItems: 'center',
      padding: `0 14px 0 ${contentLeftPadding}px`,
      position: 'relative',
      WebkitAppRegion: 'drag',
    }}>
      <div style={{
        position: 'absolute',
        left: logoLeft,
        top: 7,
        WebkitAppRegion: 'no-drag',
      }}>
        <SahaMark size={30} mono />
      </div>
      <div style={{
        position: 'absolute', left: 0, right: 0, textAlign: 'center',
        fontSize: 13, fontWeight: 600, color: SONNA.fgMute, pointerEvents: 'none',
        letterSpacing: 0,
      }}>
        {folder && <span style={{ color: SONNA.fgDim, fontWeight: 400, letterSpacing: 0 }}>{folder}</span>}
      </div>
      <HeaderThemeButton theme={theme} onToggleTheme={onToggleTheme} />
    </div>
  );
}

// AppShell — outer frame; child renders below the rail.
// Width/height default to viewport so the Electron window fills its native frame
// instead of rendering a fixed-size 1280×780 island inside it.
export function AppShell({
  title,
  folder,
  activeNav = 'home',
  accent = false,
  accentColor,
  onNavigate,
  theme,
  onToggleTheme,
  onLogout,
  children,
}) {
  return (
    <div style={{
      width: '100%', height: '100%',
      background: SONNA.bgDeep,
      color: SONNA.fg,
      fontFamily: SONNA.font,
      display: 'flex', flexDirection: 'column',
      overflow: 'hidden',
    }}>
      <TitleBar title={title} folder={folder} theme={theme} onToggleTheme={onToggleTheme} />
      <div style={{ flex: 1, display: 'flex', minHeight: 0 }}>
        <NavRail
          active={activeNav}
          accent={accent}
          accentColor={accentColor}
          onNavigate={onNavigate}
          onLogout={onLogout}
        />
        <div style={{ flex: 1, display: 'flex', minWidth: 0 }}>
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
      fontSize: 10, fontWeight: 600, color: SONNA.fgDim,
      textTransform: 'uppercase', letterSpacing: 0.6,
      ...style,
    }}>{children}</div>
  );
}

export function Hairline({ vertical, color = SONNA.line, style }) {
  return <div style={{
    background: color,
    [vertical ? 'width' : 'height']: 1,
    [vertical ? 'height' : 'width']: '100%',
    flexShrink: 0,
    ...style,
  }} />;
}

export function Mono({ children, style }) {
  return <span style={{ fontFamily: SONNA.mono, ...style }}>{children}</span>;
}
