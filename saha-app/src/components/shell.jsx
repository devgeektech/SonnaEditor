// Shared dark window chrome + nav rail.
// Ported from "SAHA UI/shell.jsx" (IIFE → ESM). The fake TrafficLights from
// the design canvas have been removed — Electron's titleBarStyle: 'hiddenInset'
// overlays the real macOS traffic lights at top-left.

import SONNA from '../tokens.js';

const {
  bgDeep, bgPanel, bgLifted, bgHover, line, lineSoft,
  fg, fgMute, fgDim, fgFaint, ochre, font, mono,
} = SONNA;

// nav rail icons, 14×14, drawn as simple shapes
function RailIcon({ kind, active }) {
  const stroke = active ? fg : fgDim;
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
      {kind === 'profile' && (
        <>
          <circle cx="8" cy="6" r="2.6" stroke={stroke} strokeWidth={sw} />
          <path d="M3 13.5c0.6-2.6 2.6-4 5-4s4.4 1.4 5 4" stroke={stroke} strokeWidth={sw} strokeLinecap="round" />
        </>
      )}
      {kind === 'settings' && (
        <>
          <circle cx="8" cy="8" r="2" stroke={stroke} strokeWidth={sw} />
          <path d="M8 1.5v2M8 12.5v2M14.5 8h-2M3.5 8h-2M12.5 3.5l-1.4 1.4M4.9 11.1l-1.4 1.4M12.5 12.5l-1.4-1.4M4.9 4.9l-1.4-1.4" stroke={stroke} strokeWidth={sw} strokeLinecap="round" />
        </>
      )}
    </svg>
  );
}

export function NavRail({ active = 'process', accent = false, accentColor = ochre, onNavigate }) {
  const items = [
    { kind: 'process', enabled: true },
    { kind: 'profile', enabled: true },
  ];
  return (
    <div style={{
      width: 44, flexShrink: 0,
      background: bgPanel,
      borderRight: `1px solid ${line}`,
      display: 'flex', flexDirection: 'column', alignItems: 'center',
      padding: '14px 0',
    }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        {items.map(({ kind, enabled }) => {
          const isActive = kind === active;
          const click = enabled && onNavigate ? () => onNavigate(kind) : undefined;
          return (
            <div key={kind}
              onClick={click}
              style={{
                width: 32, height: 32, borderRadius: 6,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                background: isActive ? bgLifted : 'transparent',
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
              <RailIcon kind={kind} active={isActive} />
            </div>
          );
        })}
      </div>
      <div style={{ flex: 1 }} />
      <div style={{
        width: 32, height: 32, borderRadius: 6,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        opacity: 0.4,
      }}>
        <RailIcon kind="settings" active={false} />
      </div>
    </div>
  );
}

export function TitleBar({ title = 'saha', folder = '' }) {
  return (
    <div style={{
      // Bumped to 44px to accommodate the macOS native traffic lights overlay.
      // Padding-left clears the lights area; the absolute-positioned title
      // remains centered relative to the full window.
      height: 44, flexShrink: 0,
      background: bgDeep,
      borderBottom: `1px solid ${line}`,
      display: 'flex', alignItems: 'center',
      padding: '0 14px 0 86px',
      position: 'relative',
      WebkitAppRegion: 'drag',
    }}>
      <div style={{
        position: 'absolute', left: 0, right: 0, textAlign: 'center',
        fontSize: 13, fontWeight: 600, color: fgMute, pointerEvents: 'none',
        letterSpacing: -1,
      }}>
        {title}{folder && <span style={{ color: fgDim, fontWeight: 400, letterSpacing: 0 }}> &nbsp;—&nbsp; {folder}</span>}
      </div>
    </div>
  );
}

// MacShell — outer frame; child renders below the rail.
// Width/height default to viewport so the Electron window fills its native frame
// instead of rendering a fixed-size 1280×780 island inside it.
export function MacShell({ title, folder, activeNav = 'process', accent = false, accentColor, onNavigate, children }) {
  return (
    <div style={{
      width: '100%', height: '100%',
      background: bgDeep,
      color: fg,
      fontFamily: font,
      display: 'flex', flexDirection: 'column',
      overflow: 'hidden',
    }}>
      <TitleBar title={title} folder={folder} />
      <div style={{ flex: 1, display: 'flex', minHeight: 0 }}>
        <NavRail active={activeNav} accent={accent} accentColor={accentColor} onNavigate={onNavigate} />
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
      fontSize: 10, fontWeight: 600, color: fgDim,
      textTransform: 'uppercase', letterSpacing: 0.6,
      ...style,
    }}>{children}</div>
  );
}

export function Hairline({ vertical, color = line, style }) {
  return <div style={{
    background: color,
    [vertical ? 'width' : 'height']: 1,
    [vertical ? 'height' : 'width']: '100%',
    flexShrink: 0,
    ...style,
  }} />;
}

export function Mono({ children, style }) {
  return <span style={{ fontFamily: mono, ...style }}>{children}</span>;
}
