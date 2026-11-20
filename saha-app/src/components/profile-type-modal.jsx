// Profile Type Choice Modal
// Reference: media_1794857193236_7fae887e.png (Modal matching Imagen AI Profiles choice)
// - Modal title: "Create a Personal AI Profile"
// - Left Card: "Lite Personal AI Profile" with lightning bolt illustration + motion lines + sparkles
// - Right Card: "Personal AI Profile" with "Recommended" lime ribbon badge + puzzle piece illustration + sparkles
// - Full Light and Dark theme support

import { useState } from 'react';
import SONNA from '../tokens.js';

const F = SONNA.font;

function LiteBoltIcon({ isLight }) {
  const motionStroke = isLight ? 'rgba(0, 0, 0, 0.4)' : 'rgba(255, 255, 255, 0.75)';

  return (
    <svg width="104" height="90" viewBox="0 0 104 90" fill="none" style={{ overflow: 'visible' }}>
      <defs>
        <linearGradient id="sahaBoltGrad" x1="0.2" y1="0" x2="0.85" y2="1">
          <stop offset="0%" stopColor="#FA586A" />
          <stop offset="42%" stopColor="#FF7A60" />
          <stop offset="100%" stopColor="#FFE0A0" />
        </linearGradient>
        <filter id="boltGlow" x="-15%" y="-15%" width="130%" height="130%">
          <feDropShadow dx="0" dy="2" stdDeviation="3" floodColor="rgba(250, 88, 106, 0.35)" />
        </filter>
      </defs>

      {/* Speed / motion lines on the left */}
      <line x1="8" y1="36" x2="38" y2="36" stroke={motionStroke} strokeWidth="2.8" strokeLinecap="round" />
      <line x1="16" y1="46" x2="44" y2="46" stroke={motionStroke} strokeWidth="2.8" strokeLinecap="round" />
      <line x1="24" y1="56" x2="48" y2="56" stroke={motionStroke} strokeWidth="2.8" strokeLinecap="round" />

      {/* Main Lightning Bolt */}
      <path
        d="M62 12 L40 42 H56 L44 76 L80 34 H60 L72 12 Z"
        fill="url(#sahaBoltGrad)"
        stroke="rgba(255, 255, 255, 0.4)"
        strokeWidth="1"
        strokeLinejoin="round"
        filter="url(#boltGlow)"
      />

      {/* Sparkles */}
      {/* Top right sparkle */}
      <path
        d="M86 14 Q86 20 92 20 Q86 20 86 26 Q86 20 80 20 Q86 20 86 14 Z"
        fill="#FFE4A0"
      />
      {/* Bottom left sparkle */}
      <path
        d="M32 64 Q32 68 36 68 Q32 68 32 72 Q32 68 28 68 Q32 68 32 64 Z"
        fill="#FF9A9A"
      />
    </svg>
  );
}

function PersonalPuzzleIcon() {
  return (
    <svg width="104" height="90" viewBox="0 0 104 90" fill="none" style={{ overflow: 'visible' }}>
      <defs>
        <linearGradient id="sahaPuzzleGrad" x1="0.1" y1="0.1" x2="0.9" y2="0.9">
          <stop offset="0%" stopColor="#FA586A" />
          <stop offset="45%" stopColor="#FF8B72" />
          <stop offset="100%" stopColor="#FFE3B4" />
        </linearGradient>
        <filter id="puzzleGlow" x="-15%" y="-15%" width="130%" height="130%">
          <feDropShadow dx="0" dy="2" stdDeviation="3" floodColor="rgba(250, 88, 106, 0.35)" />
        </filter>
      </defs>

      {/* Puzzle piece tilted ~18deg clockwise */}
      <g transform="translate(52, 45) rotate(18) translate(-27, -27)">
        <path
          d="M 8 8 
             H 21 C 20 1, 34 1, 33 8 
             H 46 
             V 21 C 53 20, 53 34, 46 33 
             V 46 
             H 33 C 34 53, 20 53, 21 46 
             H 8 
             V 33 C 1 34, 1 20, 8 21 
             Z"
          fill="url(#sahaPuzzleGrad)"
          stroke="rgba(255, 255, 255, 0.45)"
          strokeWidth="1.2"
          strokeLinejoin="round"
          filter="url(#puzzleGlow)"
        />
      </g>

      {/* Sparkles */}
      {/* Top left sparkle */}
      <path
        d="M22 24 Q22 28 26 28 Q22 28 22 32 Q22 28 18 28 Q22 28 22 24 Z"
        fill="#FFE4A0"
      />
      {/* Bottom right sparkle */}
      <path
        d="M82 62 Q82 65 85 65 Q82 65 82 68 Q82 65 79 65 Q82 65 82 62 Z"
        fill="#FFA0A0"
      />
      {/* Top right tiny glint */}
      <circle cx="80" cy="24" r="1.5" fill="#FFE4A0" />
    </svg>
  );
}

export function ChooseProfileTypeModal({
  isOpen,
  onClose,
  onSelectLite,
  onSelectPersonal,
  theme = 'dark',
}) {
  const [hoveredCard, setHoveredCard] = useState(null); // 'lite' | 'personal' | null

  if (!isOpen) return null;

  const isLight = theme === 'light';

  const modalBg = isLight ? '#FFFFFF' : '#22252A';
  const modalBorder = isLight ? 'rgba(0, 0, 0, 0.12)' : 'rgba(255, 255, 255, 0.08)';
  const headerBorder = isLight ? 'rgba(0, 0, 0, 0.08)' : 'rgba(255, 255, 255, 0.06)';
  const cardBg = isLight ? '#FAF7F2' : '#181A1F';
  const cardHoverBg = isLight ? '#EAE3D6' : '#2B3038';
  const cardBorder = isLight ? 'rgba(0, 0, 0, 0.12)' : 'rgba(255, 255, 255, 0.12)';
  const titleColor = isLight ? '#1c1917' : '#FFFFFF';
  const descColor = isLight ? '#78716c' : '#9E9E9E';

  const handleSelectLite = () => {
    onClose?.();
    onSelectLite?.();
  };

  const handleSelectPersonal = () => {
    onClose?.();
    onSelectPersonal?.();
  };

  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 1100,
        background: 'rgba(0, 0, 0, 0.72)',
        backdropFilter: 'blur(5px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 24,
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          width: '100%',
          maxWidth: 720,
          background: modalBg,
          border: `1px solid ${modalBorder}`,
          borderRadius: 12,
          boxShadow: isLight
            ? '0 24px 60px rgba(0, 0, 0, 0.18)'
            : '0 28px 70px rgba(0, 0, 0, 0.75)',
          overflow: 'hidden',
          display: 'flex',
          flexDirection: 'column',
          transition: 'all 0.2s ease',
        }}
      >
        {/* Header */}
        <div style={{
          padding: '20px 24px 18px',
          borderBottom: `1px solid ${headerBorder}`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}>
          <h2 style={{
            fontSize: 17,
            fontWeight: 600,
            color: titleColor,
            margin: 0,
            letterSpacing: -0.2,
          }}>
            Create a Personal AI Profile
          </h2>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            style={{
              width: 32,
              height: 32,
              background: 'transparent',
              border: 'none',
              color: isLight ? 'rgba(0,0,0,0.5)' : 'rgba(255, 255, 255, 0.65)',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              borderRadius: 6,
              transition: 'all 0.15s ease',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.color = titleColor;
              e.currentTarget.style.background = isLight ? 'rgba(0,0,0,0.06)' : 'rgba(255,255,255,0.08)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.color = isLight ? 'rgba(0,0,0,0.5)' : 'rgba(255, 255, 255, 0.65)';
              e.currentTarget.style.background = 'transparent';
            }}
          >
            <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <line x1="2" y1="2" x2="14" y2="14" />
              <line x1="14" y1="2" x2="2" y2="14" />
            </svg>
          </button>
        </div>

        {/* Modal Body: Two Side-by-Side Cards */}
        <div style={{
          padding: '24px 28px 32px',
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: 20,
        }}>
          {/* Card 1: Lite Personal AI Profile */}
          <div
            onClick={handleSelectLite}
            onMouseEnter={() => setHoveredCard('lite')}
            onMouseLeave={() => setHoveredCard(null)}
            style={{
              position: 'relative',
              background: hoveredCard === 'lite' ? cardHoverBg : cardBg,
              border: `1px solid ${hoveredCard === 'lite' ? (isLight ? 'rgba(0,0,0,0.35)' : 'rgba(255, 255, 255, 0.45)') : cardBorder}`,
              borderRadius: 8,
              padding: '38px 24px 34px',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              textAlign: 'center',
              cursor: 'pointer',
              transition: 'all 0.18s cubic-bezier(0.16, 1, 0.3, 1)',
              transform: hoveredCard === 'lite' ? 'translateY(-2px)' : 'none',
              boxShadow: hoveredCard === 'lite'
                ? (isLight ? '0 12px 28px rgba(0,0,0,0.08)' : '0 14px 32px rgba(0,0,0,0.5)')
                : 'none',
            }}
          >
            {/* Illustration */}
            <div style={{
              height: 100,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}>
              <LiteBoltIcon isLight={isLight} />
            </div>

            {/* Title */}
            <div style={{
              fontSize: 16,
              fontWeight: 700,
              color: titleColor,
              marginTop: 20,
              marginBottom: 10,
              letterSpacing: -0.2,
            }}>
              Lite Personal AI Profile
            </div>

            {/* Description */}
            <div style={{
              fontSize: 13,
              color: descColor,
              lineHeight: 1.5,
              maxWidth: 250,
            }}>
              Get started in just a few minutes. Upload your preset, answer the short style survey, and use your AI Profile right away!
            </div>
          </div>

          {/* Card 2: Personal AI Profile (Recommended) */}
          <div
            onClick={handleSelectPersonal}
            onMouseEnter={() => setHoveredCard('personal')}
            onMouseLeave={() => setHoveredCard(null)}
            style={{
              position: 'relative',
              background: hoveredCard === 'personal' ? cardHoverBg : cardBg,
              border: `1px solid ${hoveredCard === 'personal' ? (isLight ? 'rgba(0,0,0,0.35)' : 'rgba(255, 255, 255, 0.45)') : cardBorder}`,
              borderRadius: 8,
              padding: '38px 24px 34px',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              textAlign: 'center',
              cursor: 'pointer',
              transition: 'all 0.18s cubic-bezier(0.16, 1, 0.3, 1)',
              transform: hoveredCard === 'personal' ? 'translateY(-2px)' : 'none',
              boxShadow: hoveredCard === 'personal'
                ? (isLight ? '0 12px 28px rgba(0,0,0,0.08)' : '0 14px 32px rgba(0,0,0,0.5)')
                : 'none',
            }}
          >
            {/* "Recommended" Ribbon Badge at top-left */}
            <div style={{
              position: 'absolute',
              top: 14,
              left: 14,
              height: 22,
              padding: '0 12px 0 8px',
              background: '#D4F49C',
              color: '#2E4600',
              fontSize: 11,
              fontWeight: 700,
              display: 'flex',
              alignItems: 'center',
              letterSpacing: 0.15,
              clipPath: 'polygon(0% 0%, 100% 0%, 90% 50%, 100% 100%, 0% 100%)',
              borderTopLeftRadius: 4,
              borderBottomLeftRadius: 4,
            }}>
              Recommended
            </div>

            {/* Illustration */}
            <div style={{
              height: 100,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}>
              <PersonalPuzzleIcon />
            </div>

            {/* Title */}
            <div style={{
              fontSize: 16,
              fontWeight: 700,
              color: titleColor,
              marginTop: 20,
              marginBottom: 10,
              letterSpacing: -0.2,
            }}>
              Personal AI Profile
            </div>

            {/* Description */}
            <div style={{
              fontSize: 13,
              color: descColor,
              lineHeight: 1.5,
              maxWidth: 250,
            }}>
              Create an AI Profile with 2,000 photos from your Lightroom catalogs. Start today, even with less than 2,000 photos.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ChooseProfileTypeModal;
