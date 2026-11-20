// Saha sign-in screen — two-column brand panel + form.
// Ported from "SAHA UI/login.jsx" (IIFE → ESM). Login is cosmetic in 7.2a:
// any of {Sign in, Continue with Apple, Continue with Google} fires onSubmit.
// Real auth is out of scope per the brief.

import { useState } from 'react';
import SONNA from '../tokens.js';
import { SahaWordmark } from './logo.jsx';

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

function EyeIcon({ visible, color = SONNA.fgMute }) {
  if (visible) {
    return (
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24" />
        <line x1="1" y1="1" x2="23" y2="23" />
      </svg>
    );
  }
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
      <circle cx="12" cy="12" r="3" />
    </svg>
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

export function SahaLogin({ onSubmit, theme, onToggleTheme }) {
  const [showPassword, setShowPassword] = useState(false);
  const isLight = theme === 'light';

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
        <div>
          <SahaWordmark size={32} />
        </div>

        <div style={{ flex: 1 }} />

        <div>
          <div style={{
            fontSize: 38, fontWeight: 300, letterSpacing: 0, lineHeight: 1.15,
            color: SONNA.fg, maxWidth: 440,
          }}>
            Cull a wedding‑sized shoot
            <span style={{ color: SONNA.fgMute }}> before your coffee gets cold.</span>
          </div>
          <div style={{
            marginTop: 22, fontSize: 13, color: SONNA.ochre, lineHeight: 1.65, maxWidth: 400,
          }}>
            Saha learns your taste from the photos you've already kept, then
            sorts the next shoot the way you would. Your edits, profiles and
            folders stay on your machine.
          </div>
        </div>

        <div style={{ flex: 0.25 }} />
      </div>

      {/* ── RIGHT · sign-in form ───────────────────── */}
      <div style={{
        flex: 1, padding: '56px 88px',
        display: 'flex', flexDirection: 'column', justifyContent: 'center',
        position: 'relative',
      }}>
        {/* Right Corner Theme Button */}
        {onToggleTheme && (
          <button
            type="button"
            onClick={onToggleTheme}
            title={isLight ? 'Switch to Dark Theme' : 'Switch to Light Theme'}
            aria-label="Toggle theme"
            style={{
              position: 'absolute',
              top: 24,
              right: 28,
              width: 32,
              height: 32,
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
              zIndex: 10,
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
        <div style={{ maxWidth: 380, width: '100%', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: 24 }}>
          <div>
            <div style={{ fontSize: 22, fontWeight: 400, color: SONNA.fg, letterSpacing: 0 }}>
              Sign in
            </div>
            <div style={{ marginTop: 6, fontSize: 13, color: SONNA.fgMute }}>
              Welcome back. Pick up where you left off.
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            <Field
              label="Email"
              type="email"
              placeholder="you@studio.com"
              value="saha@studios.com"
              autoFocus
            />
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
                padding: '0 8px 0 12px',
              }}>
                <input
                  type={showPassword ? 'text' : 'password'}
                  defaultValue="sahastudios@123"
                  style={{
                    flex: 1, background: 'transparent', border: 'none', outline: 'none',
                    color: SONNA.fg, fontFamily: F, fontSize: 13,
                    letterSpacing: showPassword ? 0.2 : 1.5,
                  }}
                />
                <button
                  type="button"
                  onClick={() => setShowPassword((prev) => !prev)}
                  title={showPassword ? 'Hide password' : 'Show password'}
                  aria-label={showPassword ? 'Hide password' : 'Show password'}
                  style={{
                    background: 'transparent',
                    border: 'none',
                    outline: 'none',
                    padding: '4px 6px',
                    cursor: 'pointer',
                    display: 'inline-flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: SONNA.fgMute,
                  }}
                >
                  <EyeIcon visible={showPassword} />
                </button>
              </span>
            </div>
            <label style={{ display: 'flex', alignItems: 'center', gap: 9, fontSize: 12, color: SONNA.fgMute, marginTop: 2 }}>
              <span style={{
                width: 14, height: 14, borderRadius: 2,
                border: `1px solid ${SONNA.line}`, background: SONNA.ochre,
                display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
              }}>
                <svg width="9" height="9" viewBox="0 0 9 9" fill="none">
                  <path d="M1.5 4.5L3.5 6.5L7.5 2.5" stroke={SONNA.onAccent} strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </span>
              Stay signed in on this device
            </label>
          </div>

          <button
            type="submit"
            style={{
              height: 40, background: SONNA.cta, color: SONNA.onCta,
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
