// Saha sign-in screen — two-column brand panel + form.
// Ported from "SAHA UI/login.jsx" (IIFE → ESM). Login is cosmetic in 7.2a:
// any of {Sign in, Continue with Apple, Continue with Google} fires onSubmit.
// Real auth is out of scope per the brief.

import SONNA from '../tokens.js';
import { SahaLockup } from './logo.jsx';

const F = SONNA.font;
const M = SONNA.mono;

const Tlabel = {
  fontSize: 10, fontWeight: 600, color: SONNA.fgDim,
  textTransform: 'uppercase', letterSpacing: 0.6,
};
const Tnum = { fontFamily: M, fontVariantNumeric: 'tabular-nums' };

function Field({ label, type = 'text', placeholder, value, hint, autoFocus }) {
  return (
    <label style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
      <span style={{ ...Tlabel }}>{label}</span>
      <span style={{
        display: 'flex', alignItems: 'center',
        height: 38, borderRadius: 3,
        background: SONNA.bgPanel,
        border: `1px solid ${SONNA.line}`,
        padding: '0 12px',
      }}>
        <input
          type={type}
          placeholder={placeholder}
          defaultValue={value}
          autoFocus={autoFocus}
          style={{
            flex: 1, background: 'transparent', border: 'none', outline: 'none',
            color: SONNA.fg, fontFamily: F, fontSize: 13, letterSpacing: 0.1,
          }}
        />
        {hint && <span style={{ ...Tnum, fontSize: 10, color: SONNA.fgFaint }}>{hint}</span>}
      </span>
    </label>
  );
}

function ProviderButton({ label, glyph, onClick }) {
  return (
    <button
      type="button"
      onClick={onClick}
      style={{
        height: 38, background: 'transparent',
        border: `1px solid ${SONNA.line}`, borderRadius: 3,
        color: SONNA.fgMute, fontFamily: F, fontSize: 13,
        display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 10,
        cursor: 'pointer',
      }}
    >
      {glyph}
      <span>{label}</span>
    </button>
  );
}

function GoogleGlyph() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
      <circle cx="7" cy="7" r="5.5" stroke={SONNA.fgMute} strokeWidth="1.2" />
      <path d="M7 4v3h3" stroke={SONNA.fgMute} strokeWidth="1.2" strokeLinecap="round" />
    </svg>
  );
}
function AppleGlyph() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
      <path d="M9.5 7.4c0-1.6 1.3-2.4 1.4-2.4-.7-1.1-1.9-1.2-2.3-1.3-1-.1-1.9.6-2.4.6s-1.3-.6-2.1-.6c-1.1 0-2.1.6-2.6 1.6-1.1 2-.3 4.9.8 6.5.5.8 1.2 1.7 2 1.6.8 0 1.1-.5 2.1-.5s1.3.5 2.1.5c.9 0 1.5-.8 2-1.6.6-.9.9-1.8.9-1.8s-1.7-.6-1.9-2.6Z M8.4 2.7c.4-.5.7-1.2.6-1.9-.6 0-1.3.4-1.7.9-.4.4-.7 1.1-.6 1.7.6.1 1.3-.3 1.7-.7Z"
        fill={SONNA.fgMute} />
    </svg>
  );
}

export function SahaLogin({ onSubmit }) {
  const handleSubmit = (e) => {
    if (e && typeof e.preventDefault === 'function') e.preventDefault();
    if (typeof onSubmit === 'function') onSubmit();
  };

  return (
    <form
      onSubmit={handleSubmit}
      style={{
        width: '100%', height: '100%',
        background: SONNA.bgDeep, color: SONNA.fg, fontFamily: F,
        display: 'flex', overflow: 'hidden', margin: 0,
      }}
    >
      {/* ── LEFT · brand panel ─────────────────────── */}
      <div style={{
        width: 560, flexShrink: 0,
        background: SONNA.bgPanel,
        borderRight: `1px solid ${SONNA.line}`,
        padding: '56px 56px 40px',
        display: 'flex', flexDirection: 'column', gap: 32,
      }}>
        <SahaLockup markSize={36} wordSize={26} gap={12} />

        <div style={{ flex: 1 }} />

        <div>
          <div style={{ ...Tlabel, color: SONNA.ochre, marginBottom: 14 }}>For working photographers</div>
          <div style={{
            fontSize: 38, fontWeight: 300, letterSpacing: -0.8, lineHeight: 1.15,
            color: SONNA.fg, maxWidth: 440,
          }}>
            Cull a wedding‑sized shoot
            <span style={{ color: SONNA.fgMute }}> before your coffee gets cold.</span>
          </div>
          <div style={{
            marginTop: 22, fontSize: 13, color: SONNA.fgMute, lineHeight: 1.65, maxWidth: 400,
          }}>
            Saha learns your taste from the photos you've already kept, then
            sorts the next shoot the way you would. Your edits, profiles and
            folders stay on your machine.
          </div>
        </div>

        {/* What's new card */}
        <div style={{
          marginTop: 4,
          border: `1px solid ${SONNA.line}`, borderRadius: 4,
          background: SONNA.bgDeep, padding: '14px 16px',
        }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 10 }}>
            <span style={{ ...Tlabel, color: SONNA.ochre }}>What's new</span>
            <span style={{ ...Tnum, fontSize: 10.5, color: SONNA.fgFaint }}>v0.4.2 · May 6</span>
          </div>
          <ul style={{
            margin: '10px 0 0', padding: 0, listStyle: 'none',
            display: 'flex', flexDirection: 'column', gap: 6,
            fontSize: 12.5, color: SONNA.fgMute, lineHeight: 1.5,
          }}>
            <li style={{ display: 'flex', gap: 10 }}>
              <span style={{ color: SONNA.ochre, ...Tnum }}>+</span>
              <span>Sony A1 II and Fujifilm X‑H2S support</span>
            </li>
            <li style={{ display: 'flex', gap: 10 }}>
              <span style={{ color: SONNA.ochre, ...Tnum }}>+</span>
              <span>GPU acceleration where available, CPU fallback everywhere</span>
            </li>
            <li style={{ display: 'flex', gap: 10 }}>
              <span style={{ color: SONNA.fgDim, ...Tnum }}>·</span>
              <span>Profile sync across devices is back, fixed in this build</span>
            </li>
          </ul>
        </div>

        <div style={{
          marginTop: 4, display: 'flex', alignItems: 'center', gap: 18,
          ...Tnum, fontSize: 10.5, color: SONNA.fgFaint,
        }}>
          <span>v0.4.2 · build 4218</span>
          <span style={{ width: 1, height: 9, background: SONNA.line }} />
          <span>RAF · NEF · ARW · CR3 · DNG</span>
          <span style={{ width: 1, height: 9, background: SONNA.line }} />
          <span>macOS · Windows · Linux</span>
        </div>
      </div>

      {/* ── RIGHT · sign-in form ───────────────────── */}
      <div style={{
        flex: 1, padding: '56px 88px',
        display: 'flex', flexDirection: 'column', justifyContent: 'center',
        position: 'relative',
      }}>
        <div style={{ maxWidth: 380, width: '100%', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: 24 }}>
          <div>
            <div style={{ fontSize: 22, fontWeight: 400, color: SONNA.fg, letterSpacing: -0.3 }}>
              Sign in
            </div>
            <div style={{ marginTop: 6, fontSize: 13, color: SONNA.fgMute }}>
              Welcome back. Pick up where you left off.
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            <Field label="Email" type="email" placeholder="you@studio.com" autoFocus />
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 7 }}>
                <span style={Tlabel}>Password</span>
                <a style={{ fontSize: 11, color: SONNA.ochre, textDecoration: 'none' }}>Forgot?</a>
              </div>
              <span style={{
                display: 'flex', alignItems: 'center',
                height: 38, borderRadius: 3,
                background: SONNA.bgPanel,
                border: `1px solid ${SONNA.line}`,
                padding: '0 12px',
              }}>
                <input type="password" defaultValue="••••••••••••" style={{
                  flex: 1, background: 'transparent', border: 'none', outline: 'none',
                  color: SONNA.fg, fontFamily: F, fontSize: 13, letterSpacing: 2,
                }} />
                <span style={{ ...Tnum, fontSize: 10, color: SONNA.fgFaint }}>show</span>
              </span>
            </div>
            <label style={{ display: 'flex', alignItems: 'center', gap: 9, fontSize: 12, color: SONNA.fgMute, marginTop: 2 }}>
              <span style={{
                width: 14, height: 14, borderRadius: 2,
                border: `1px solid ${SONNA.line}`, background: SONNA.ochre,
                display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
              }}>
                <svg width="9" height="9" viewBox="0 0 9 9" fill="none">
                  <path d="M1.5 4.5L3.5 6.5L7.5 2.5" stroke="#1A1209" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </span>
              Stay signed in on this device
            </label>
          </div>

          <button
            type="submit"
            style={{
              height: 40, background: SONNA.ochre, color: '#1A1209',
              border: 'none', borderRadius: 3,
              fontFamily: F, fontSize: 13, fontWeight: 600, letterSpacing: 0.2,
              display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 12,
              cursor: 'pointer',
            }}
          >
            <span>Sign in</span>
            <span style={{ ...Tnum, fontSize: 11, opacity: 0.65 }}>↵</span>
          </button>

          <div style={{ display: 'flex', alignItems: 'center', gap: 10, ...Tlabel }}>
            <div style={{ flex: 1, height: 1, background: SONNA.lineSoft }} />
            <span>or</span>
            <div style={{ flex: 1, height: 1, background: SONNA.lineSoft }} />
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            <ProviderButton label="Continue with Apple"  glyph={<AppleGlyph />} onClick={handleSubmit} />
            <ProviderButton label="Continue with Google" glyph={<GoogleGlyph />} onClick={handleSubmit} />
          </div>

          <div style={{
            marginTop: 4, fontSize: 12, color: SONNA.fgMute, textAlign: 'center',
          }}>
            No account?{' '}
            <a style={{ color: SONNA.fg, textDecoration: 'none', borderBottom: `1px solid ${SONNA.line}`, paddingBottom: 1 }}>
              Request access
            </a>
          </div>
        </div>

        {/* tiny footer */}
        <div style={{
          position: 'absolute', bottom: 24, right: 32,
          ...Tnum, fontSize: 10, color: SONNA.fgFaint,
          display: 'flex', gap: 14,
        }}>
          <a style={{ color: 'inherit', textDecoration: 'none' }}>Privacy</a>
          <a style={{ color: 'inherit', textDecoration: 'none' }}>Terms</a>
          <a style={{ color: 'inherit', textDecoration: 'none' }}>support@saha.app</a>
        </div>
      </div>
    </form>
  );
}
