// Dismissable error banner. Lives at the top of the centre column when an
// API call fails. Per the UI/UX brief: no popup modals — banners only.

import SONNA from '../tokens.js';

export function ErrorBanner({ error, onDismiss }) {
  if (!error) return null;
  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: 12,
      padding: '10px 16px',
      background: 'rgba(156, 84, 84, 0.18)',
      borderBottom: `1px solid ${SONNA.red}`,
      color: SONNA.fg,
      fontFamily: SONNA.font,
      fontSize: 12.5,
    }}>
      <div style={{
        width: 16, height: 16, borderRadius: '50%',
        background: SONNA.red, color: '#1A1209',
        display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
        fontSize: 11, fontWeight: 700,
      }}>!</div>
      <span style={{ flex: 1, minWidth: 0 }}>
        {error.message || 'Something went wrong.'}
      </span>
      <button
        type="button"
        onClick={onDismiss}
        style={{
          background: 'transparent',
          border: `1px solid ${SONNA.line}`,
          borderRadius: 3,
          color: SONNA.fgMute,
          fontFamily: SONNA.font,
          fontSize: 11,
          padding: '4px 10px',
          cursor: 'pointer',
        }}
      >
        Dismiss
      </button>
    </div>
  );
}
