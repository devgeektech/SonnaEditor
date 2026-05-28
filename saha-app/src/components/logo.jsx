// Saha brand mark + wordmark lockups.
// Ported from "SAHA UI/logo.jsx" (IIFE → ESM).

import SONNA from '../tokens.js';

const BONE = '#E8E2D6';

// ── Wordmark — the locked-in logo ───────────────────────────
export function SahaWordmark({ size = 26, color, weight = 600, tracking = -1.2 }) {
  return (
    <span style={{
      fontFamily: SONNA.font,
      fontSize: size,
      fontWeight: weight,
      letterSpacing: tracking,
      color: color ?? SONNA.fg,
      lineHeight: 1,
      whiteSpace: 'nowrap',
    }}>saha</span>
  );
}

// ── App icon ────────────────────────────────────────────────
// Default: flat panel-color squircle, bone wordmark, lots of side padding.
// gradient: dimensional warm ochre version (used in explorations only).
export function SahaMark({ size = 32, gradient = false, mono = false, bg, fg }) {
  const u = `mark-${size}-${gradient ? 'g' : 'f'}-${mono ? 'm' : 'c'}`;
  const textSize = gradient ? 62 : 50;
  const textColor = fg ?? (gradient ? '#FFF6E3' : (mono ? SONNA.fg : BONE));
  const flatBg = bg ?? (mono ? SONNA.bgDeep : SONNA.bgPanel);
  return (
    <div style={{
      width: size, height: size, display: 'inline-flex', flexShrink: 0,
      filter: size >= 40
        ? 'drop-shadow(0 4px 10px rgba(0,0,0,0.45)) drop-shadow(0 1px 2px rgba(0,0,0,0.35))'
        : 'none',
    }}>
      <svg width={size} height={size} viewBox="0 0 128 128" style={{ display: 'block' }}>
        <defs>
          {gradient && (
            <>
              <linearGradient id={`${u}-bg`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%"   stopColor="#D9A572" />
                <stop offset="55%"  stopColor="#B07746" />
                <stop offset="100%" stopColor="#7A4F2E" />
              </linearGradient>
              <linearGradient id={`${u}-glint`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%"   stopColor="rgba(255,255,255,0.22)" />
                <stop offset="100%" stopColor="rgba(255,255,255,0)" />
              </linearGradient>
            </>
          )}
          <clipPath id={`${u}-clip`}>
            <rect x="0" y="0" width="128" height="128" rx="29" ry="29" />
          </clipPath>
        </defs>
        <g clipPath={`url(#${u}-clip)`}>
          <rect width="128" height="128" fill={gradient ? `url(#${u}-bg)` : flatBg} />
          <text
            x="64" y={gradient ? 84 : 80} textAnchor="middle"
            fontFamily={SONNA.font}
            fontSize={textSize}
            fontWeight="600"
            letterSpacing="-2.4"
            fill={textColor}
          >saha</text>
          {gradient && <rect width="128" height="64" fill={`url(#${u}-glint)`} />}
        </g>
        <rect x="0.5" y="0.5" width="127" height="127" rx="29" ry="29"
          fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="1" />
      </svg>
    </div>
  );
}

// ── Lockup — mark + wordmark on one row ─────────────────────
export function SahaLockup({ markSize = 28, wordSize = 22, gap = 10, gradient = false, mono = false, color, bg }) {
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap }}>
      <SahaMark size={markSize} gradient={gradient} mono={mono} bg={bg} />
      <SahaWordmark size={wordSize} color={color} />
    </span>
  );
}

// ── Stacked lockup ──────────────────────────────────────────
export function SahaStack({ markSize = 56, wordSize = 26, gap = 14, gradient = false, color, mono = false }) {
  return (
    <div style={{ display: 'inline-flex', flexDirection: 'column', alignItems: 'center', gap }}>
      <SahaMark size={markSize} gradient={gradient} mono={mono} />
      <SahaWordmark size={wordSize} color={color} />
    </div>
  );
}
