// Profile Tuning & Override Settings Modal
// Reference: references/25_imagen_profile_update_settings_modal.png
// Allows fine-tuning profile baseline offsets (Exposure, White Balance, Contrast).

import { useState } from 'react';
import SONNA from '../tokens.js';

const F = SONNA.font;
const M = SONNA.mono;

export function ProfileTuningModal({
  isOpen,
  profile,
  onClose,
  onSaveOverrides,
}) {
  const [exposureOffset, setExposureOffset] = useState(0.0);
  const [tempBias, setTempBias] = useState(0);
  const [tintOffset, setTintOffset] = useState(0);
  const [contrastBoost, setContrastBoost] = useState(0);

  if (!isOpen || !profile) return null;

  const handleSave = () => {
    onSaveOverrides?.(profile.id, {
      exposureOffset,
      tempBias,
      tintOffset,
      contrastBoost,
    });
    onClose?.();
  };

  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 100,
        background: 'rgba(0, 0, 0, 0.65)',
        backdropFilter: 'blur(4px)',
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
          maxWidth: 580,
          background: SONNA.bgPanel,
          border: `1px solid ${SONNA.line}`,
          borderRadius: 8,
          boxShadow: '0 20px 50px rgba(0,0,0,0.5)',
          overflow: 'hidden',
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        {/* Header */}
        <div style={{
          padding: '18px 24px',
          borderBottom: `1px solid ${SONNA.lineSoft}`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}>
          <div>
            <div style={{ fontSize: 17, fontWeight: 600, color: SONNA.fg }}>
              Tune Profile Parameters
            </div>
            <div style={{ fontSize: 12, color: SONNA.fgDim, marginTop: 3 }}>
              {profile.display_name || profile.name} · Baseline adjustment offsets
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            style={{
              width: 30, height: 30,
              background: 'transparent',
              border: 'none',
              color: SONNA.fgMute,
              cursor: 'pointer',
              fontSize: 18,
              borderRadius: 4,
            }}
          >
            ×
          </button>
        </div>

        {/* Sliders Container */}
        <div style={{ padding: '24px 28px', display: 'flex', flexDirection: 'column', gap: 20 }}>
          {/* Exposure Offset */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
              <span style={{ fontSize: 13, fontWeight: 500, color: SONNA.fg }}>Exposure Bias</span>
              <span style={{ fontFamily: M, fontSize: 12, color: SONNA.ochre }}>
                {exposureOffset >= 0 ? `+${exposureOffset.toFixed(2)}` : exposureOffset.toFixed(2)} EV
              </span>
            </div>
            <input
              type="range"
              min="-0.50"
              max="0.50"
              step="0.05"
              value={exposureOffset}
              onChange={(e) => setExposureOffset(parseFloat(e.target.value))}
              style={{ width: '100%', accentColor: SONNA.ochre }}
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: SONNA.fgFaint, marginTop: 2 }}>
              <span>-0.50 EV (Darker)</span>
              <span>0.00 (Default)</span>
              <span>+0.50 EV (Brighter)</span>
            </div>
          </div>

          {/* Temperature Bias */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
              <span style={{ fontSize: 13, fontWeight: 500, color: SONNA.fg }}>Temperature Bias</span>
              <span style={{ fontFamily: M, fontSize: 12, color: SONNA.ochre }}>
                {tempBias >= 0 ? `+${tempBias}` : tempBias} K
              </span>
            </div>
            <input
              type="range"
              min="-300"
              max="300"
              step="25"
              value={tempBias}
              onChange={(e) => setTempBias(parseInt(e.target.value, 10))}
              style={{ width: '100%', accentColor: SONNA.ochre }}
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: SONNA.fgFaint, marginTop: 2 }}>
              <span>-300 K (Cooler / Blue)</span>
              <span>0 K</span>
              <span>+300 K (Warmer / Gold)</span>
            </div>
          </div>

          {/* Tint Offset */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
              <span style={{ fontSize: 13, fontWeight: 500, color: SONNA.fg }}>Tint Bias</span>
              <span style={{ fontFamily: M, fontSize: 12, color: SONNA.ochre }}>
                {tintOffset >= 0 ? `+${tintOffset}` : tintOffset}
              </span>
            </div>
            <input
              type="range"
              min="-5"
              max="5"
              step="1"
              value={tintOffset}
              onChange={(e) => setTintOffset(parseInt(e.target.value, 10))}
              style={{ width: '100%', accentColor: SONNA.ochre }}
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: SONNA.fgFaint, marginTop: 2 }}>
              <span>-5 (Green)</span>
              <span>0</span>
              <span>+5 (Magenta)</span>
            </div>
          </div>

          {/* Contrast Boost */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
              <span style={{ fontSize: 13, fontWeight: 500, color: SONNA.fg }}>Tone Contrast</span>
              <span style={{ fontFamily: M, fontSize: 12, color: SONNA.ochre }}>
                {contrastBoost >= 0 ? `+${contrastBoost}` : contrastBoost}
              </span>
            </div>
            <input
              type="range"
              min="-10"
              max="10"
              step="1"
              value={contrastBoost}
              onChange={(e) => setContrastBoost(parseInt(e.target.value, 10))}
              style={{ width: '100%', accentColor: SONNA.ochre }}
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: SONNA.fgFaint, marginTop: 2 }}>
              <span>-10 (Flatter)</span>
              <span>0 (Preset Standard)</span>
              <span>+10 (Punchier)</span>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div style={{
          padding: '16px 24px',
          borderTop: `1px solid ${SONNA.lineSoft}`,
          background: SONNA.bgDeep,
          display: 'flex',
          justifyContent: 'flex-end',
          gap: 12,
        }}>
          <button
            type="button"
            onClick={onClose}
            style={{
              height: 36,
              padding: '0 16px',
              background: 'transparent',
              border: `1px solid ${SONNA.line}`,
              borderRadius: 4,
              color: SONNA.fgMute,
              fontFamily: F,
              fontSize: 13,
              cursor: 'pointer',
            }}
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={handleSave}
            style={{
              height: 36,
              padding: '0 20px',
              background: SONNA.cta,
              color: SONNA.onCta,
              border: 'none',
              borderRadius: 4,
              fontFamily: F,
              fontSize: 13,
              fontWeight: 600,
              cursor: 'pointer',
            }}
          >
            Save Overrides
          </button>
        </div>
      </div>
    </div>
  );
}

export default ProfileTuningModal;

