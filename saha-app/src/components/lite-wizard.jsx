// Lite profile creation wizard
// Reference:
// - media_1794857986217_96aa8a28.png (Lite profile creation matching Imagen reference)
// - 11_imagen_lite_survey_exposure_bride_lake.jpg through 19_imagen_lite_survey_tint_bride_garden_build_cta.jpg
//   (Full 9-step benchmark visual style survey: 3 scenes for Exposure, 3 for Temperature, 3 for Tint)
// Full Light and Dark theme support via SONNA design tokens.

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import SONNA from '../tokens.js';
import { createLiteProfile } from '../api/client.js';
import { ConfirmModal } from './confirm-modal.jsx';

const F = SONNA.font;

// The 9-step Style Survey calibrated across 3 contrasting lighting conditions:
// Scene 1: Backlit high-key portrait by a lake (skin exposure vs sky highlight protection)
// Scene 2: Low-key / dim indoor shadow portrait (shadow lifting vs moody blacks)
// Scene 3: Direct sunlit garden portrait with green foliage (highlight clipping & chlorophyll green reflection)
const LITE_SURVEY_STEPS = [
  // 1-3: Exposure
  {
    stepIndex: 0,
    category: 'exposure',
    title: 'Exposure',
    prompt: 'Do you prefer your photos to look darker or brighter? Choose your preference below',
    leftLabel: 'Darker',
    rightLabel: 'Brighter',
    gradient: 'linear-gradient(to right, #1F2228, #E2E8F0)',
    scene: 'Backlit portrait by water',
    imagePrefix: 'q1',
  },
  {
    stepIndex: 1,
    category: 'exposure',
    title: 'Exposure',
    prompt: 'Do you prefer your photos to look darker or brighter? Choose your preference below',
    leftLabel: 'Darker',
    rightLabel: 'Brighter',
    gradient: 'linear-gradient(to right, #1F2228, #E2E8F0)',
    scene: 'Dim indoor / shadow portrait',
    imagePrefix: 'q2',
  },
  {
    stepIndex: 2,
    category: 'exposure',
    title: 'Exposure',
    prompt: 'Do you prefer your photos to look darker or brighter? Choose your preference below',
    leftLabel: 'Darker',
    rightLabel: 'Brighter',
    gradient: 'linear-gradient(to right, #1F2228, #E2E8F0)',
    scene: 'Full sun garden portrait',
    imagePrefix: 'q3',
  },
  // 4-6: Temperature
  {
    stepIndex: 3,
    category: 'temperature',
    title: 'Temperature',
    prompt: 'Do you prefer cooler or warmer tones in your photos? Choose your preference below',
    leftLabel: 'Cooler',
    rightLabel: 'Warmer',
    gradient: 'linear-gradient(to right, #38BDF8, #FBBF24)',
    scene: 'Backlit portrait by water',
    imagePrefix: 'q4',
  },
  {
    stepIndex: 4,
    category: 'temperature',
    title: 'Temperature',
    prompt: 'Do you prefer cooler or warmer tones in your photos? Choose your preference below',
    leftLabel: 'Cooler',
    rightLabel: 'Warmer',
    gradient: 'linear-gradient(to right, #38BDF8, #FBBF24)',
    scene: 'Dim indoor / shadow portrait',
    imagePrefix: 'q5',
  },
  {
    stepIndex: 5,
    category: 'temperature',
    title: 'Temperature',
    prompt: 'Do you prefer cooler or warmer tones in your photos? Choose your preference below',
    leftLabel: 'Cooler',
    rightLabel: 'Warmer',
    gradient: 'linear-gradient(to right, #38BDF8, #FBBF24)',
    scene: 'Full sun garden portrait',
    imagePrefix: 'q6',
  },
  // 7-9: Tint
  {
    stepIndex: 6,
    category: 'tint',
    title: 'Tint',
    prompt: 'Do you prefer more green or more magenta tones in your photos? Choose your preference below',
    leftLabel: 'Green',
    rightLabel: 'Magenta',
    gradient: 'linear-gradient(to right, #22C55E, #EC4899)',
    scene: 'Backlit portrait by water',
    imagePrefix: 'q7',
  },
  {
    stepIndex: 7,
    category: 'tint',
    title: 'Tint',
    prompt: 'Do you prefer more green or more magenta tones in your photos? Choose your preference below',
    leftLabel: 'Green',
    rightLabel: 'Magenta',
    gradient: 'linear-gradient(to right, #22C55E, #EC4899)',
    scene: 'Dim indoor / shadow portrait',
    imagePrefix: 'q8',
  },
  {
    stepIndex: 8,
    category: 'tint',
    title: 'Tint',
    prompt: 'Do you prefer more green or more magenta tones in your photos? Choose your preference below',
    leftLabel: 'Green',
    rightLabel: 'Magenta',
    gradient: 'linear-gradient(to right, #22C55E, #EC4899)',
    scene: 'Full sun garden portrait',
    imagePrefix: 'q9',
  },
];

const OPTION_VALUES = [-2, -1, 0, 1, 2];

function basenameOf(p) {
  if (!p) return '';
  const segs = p.replace(/\\/g, '/').split('/').filter(Boolean);
  return segs[segs.length - 1] || p;
}

// XMP Document Icon matching reference media_1794857986217_96aa8a28.png
function XmpDocIcon() {
  return (
    <svg width="48" height="56" viewBox="0 0 44 52" fill="none" style={{ overflow: 'visible' }}>
      <path
        d="M4 6C4 3.79086 5.79086 2 8 2H28L40 14V46C40 48.2091 38.2091 50 36 50H8C5.79086 50 4 48.2091 4 46V6Z"
        fill="#4A5568"
      />
      <path
        d="M28 2V14H40L28 2Z"
        fill="#2D3748"
      />
      <text
        x="22"
        y="33"
        fill="#FFFFFF"
        fontSize="9.5"
        fontWeight="700"
        fontFamily="sans-serif"
        textAnchor="middle"
        letterSpacing="0.8"
      >
        XMP
      </text>
      <circle cx="34" cy="42" r="8" fill="#3182CE" />
      <path
        d="M34 38V46M30 42H38"
        stroke="#FFFFFF"
        strokeWidth="1.6"
        strokeLinecap="round"
      />
    </svg>
  );
}

export function LiteProfileWizard({ onClose, onCreated, theme = 'dark', profiles = [] }) {
  const isLight = theme === 'light';

  // Steps: 'setup' (Name + Preset) | 'survey' (Style Survey 9 questions)
  const [currentStep, setCurrentStep] = useState('setup');
  const [surveyIndex, setSurveyIndex] = useState(0);

  const [profileName, setProfileName] = useState('');
  const [presetPath, setPresetPath] = useState(null);
  const [stepAnswers, setStepAnswers] = useState({});
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [confirmLeaveOpen, setConfirmLeaveOpen] = useState(false);

  const fileInputRef = useRef(null);
  const nameInputRef = useRef(null);
  const autoAdvanceTimerRef = useRef(null);

  // Auto-focus name input on mount
  useEffect(() => {
    if (currentStep === 'setup') {
      const t = setTimeout(() => nameInputRef.current?.focus(), 50);
      return () => clearTimeout(t);
    }
  }, [currentStep]);

  // Clean up auto-advance timer on unmount
  useEffect(() => {
    return () => {
      if (autoAdvanceTimerRef.current) {
        clearTimeout(autoAdvanceTimerRef.current);
      }
    };
  }, []);

  const hasAnyState = useMemo(() => (
    profileName.trim().length > 0
    || presetPath !== null
    || Object.keys(stepAnswers).length > 0
  ), [profileName, presetPath, stepAnswers]);

  const requestClose = useCallback(() => {
    if (submitting) return;
    if (hasAnyState) {
      setConfirmLeaveOpen(true);
      return;
    }
    onClose();
  }, [hasAnyState, onClose, submitting]);

  // Escape key handler
  useEffect(() => {
    const handler = (e) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        requestClose();
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [requestClose]);

  // File Picker
  const pickPreset = useCallback(async () => {
    if (window.saha?.pickFile) {
      const path = await window.saha.pickFile({
        title: 'Choose a Lightroom preset (.xmp)',
        filters: [{ name: 'Lightroom preset', extensions: ['xmp'] }],
      });
      if (path) setPresetPath(path);
    } else {
      fileInputRef.current?.click();
    }
  }, []);

  const handleNativeFileInput = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      setPresetPath(file.path || file.name);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer?.files?.[0];
    if (file) {
      if (file.name.toLowerCase().endsWith('.xmp')) {
        setPresetPath(file.path || file.name);
      } else {
        alert('Please select an XMP preset file (.xmp).');
      }
    }
  };

  const currentSurveyStep = LITE_SURVEY_STEPS[surveyIndex] || LITE_SURVEY_STEPS[0];

  // Selecting a card option: immediately highlight, then smoothly auto-advance if not the last question
  const selectOption = useCallback((optVal) => {
    setStepAnswers((prev) => ({ ...prev, [surveyIndex]: optVal }));

    if (autoAdvanceTimerRef.current) {
      clearTimeout(autoAdvanceTimerRef.current);
    }

    if (surveyIndex < 8) {
      autoAdvanceTimerRef.current = setTimeout(() => {
        setSurveyIndex((idx) => Math.min(idx + 1, 8));
      }, 250);
    }
  }, [surveyIndex]);

  const trimmedName = profileName.trim();
  const isDuplicateName = useMemo(() => {
    if (!trimmedName) return false;
    const lower = trimmedName.toLowerCase();
    return (profiles || []).some(
      (p) => (p.display_name || p.name || '').trim().toLowerCase() === lower
    );
  }, [trimmedName, profiles]);

  const canAdvance = useMemo(() => {
    if (currentStep === 'setup') {
      return trimmedName.length > 0 && !isDuplicateName && presetPath !== null;
    }
    if (currentStep === 'survey') {
      return stepAnswers[surveyIndex] !== undefined;
    }
    return false;
  }, [currentStep, trimmedName, isDuplicateName, presetPath, surveyIndex, stepAnswers]);

  const advance = useCallback(() => {
    if (!canAdvance || submitting) return;
    if (isDuplicateName) {
      setSubmitError(`A profile named "${trimmedName}" already exists. Every profile must have a unique name.`);
      return;
    }

    if (currentStep === 'setup') {
      setCurrentStep('survey');
      setSurveyIndex(0);
      return;
    }

    if (currentStep === 'survey') {
      if (surveyIndex < 8) {
        setSurveyIndex((i) => i + 1);
        return;
      }

      // Final step 9 answered → Aggregate and build profile with high precision
      const expAnswers = [stepAnswers[0], stepAnswers[1], stepAnswers[2]].filter((v) => v !== undefined);
      const tempAnswers = [stepAnswers[3], stepAnswers[4], stepAnswers[5]].filter((v) => v !== undefined);
      const tintAnswers = [stepAnswers[6], stepAnswers[7], stepAnswers[8]].filter((v) => v !== undefined);

      const avgExp = expAnswers.length ? Number((expAnswers.reduce((a, b) => a + b, 0) / expAnswers.length).toFixed(3)) : 0;
      const avgTemp = tempAnswers.length ? Number((tempAnswers.reduce((a, b) => a + b, 0) / tempAnswers.length).toFixed(3)) : 0;
      const avgTint = tintAnswers.length ? Number((tintAnswers.reduce((a, b) => a + b, 0) / tintAnswers.length).toFixed(3)) : 0;

      if (!presetPath || (!presetPath.includes('/') && !presetPath.includes('\\'))) {
        setSubmitError('Please select a valid .xmp preset file using the Browse button.');
        return;
      }

      setSubmitting(true);
      setSubmitError(null);
      createLiteProfile({
        profile_name: profileName.trim(),
        preset_path: presetPath,
        survey_answers: {
          exposure: avgExp,
          temperature: avgTemp,
          tint: avgTint,
          contrast: 0,
          saturation: 0,
          shadows: 0,
        },
      }).then((res) => {
        setSubmitting(false);
        onCreated(res);
      }).catch((e) => {
        setSubmitting(false);
        setSubmitError(e.message || 'Failed to create profile');
      });
    }
  }, [canAdvance, submitting, currentStep, surveyIndex, stepAnswers, profileName, presetPath, onCreated]);

  const goBack = useCallback(() => {
    if (submitting) return;
    if (autoAdvanceTimerRef.current) {
      clearTimeout(autoAdvanceTimerRef.current);
    }
    if (currentStep === 'survey') {
      if (surveyIndex > 0) {
        setSurveyIndex((i) => i - 1);
      } else {
        setCurrentStep('setup');
      }
    }
  }, [currentStep, surveyIndex, submitting]);

  // Theming colors
  const modalBg = isLight ? '#FFFFFF' : '#17191C';
  const modalBorder = isLight ? 'rgba(0, 0, 0, 0.12)' : 'rgba(255, 255, 255, 0.08)';
  const panelBg = isLight ? '#FBF9F5' : '#1F2228';
  const panelBorder = isLight ? 'rgba(0, 0, 0, 0.1)' : 'rgba(255, 255, 255, 0.08)';
  const titleColor = isLight ? '#1c1917' : '#FFFFFF';
  const subtextColor = isLight ? '#78716c' : '#8E95A2';
  const inputBg = isLight ? '#FFFFFF' : '#131417';
  const inputBorder = isLight ? 'rgba(0, 0, 0, 0.15)' : 'rgba(255, 255, 255, 0.14)';
  const accentColor = '#D4F49C'; // signature lime-yellow from Imagen style survey

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 1100,
        background: 'rgba(0, 0, 0, 0.78)',
        backdropFilter: 'blur(6px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 24,
      }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="lite-profile-creation-title"
        style={{
          width: '100%',
          maxWidth: 960,
          background: modalBg,
          border: `1px solid ${modalBorder}`,
          borderRadius: 10,
          boxShadow: isLight
            ? '0 24px 60px rgba(0, 0, 0, 0.18)'
            : '0 28px 70px rgba(0, 0, 0, 0.85)',
          overflow: 'hidden',
          display: 'flex',
          flexDirection: 'column',
          transition: 'all 0.2s ease',
        }}
      >
        {/* Hidden native file input fallback for browser dev mode */}
        <input
          ref={fileInputRef}
          type="file"
          accept=".xmp"
          style={{ display: 'none' }}
          onChange={handleNativeFileInput}
        />

        {/* Header Bar */}
        <div style={{
          padding: '16px 24px 14px',
          borderBottom: `1px solid ${isLight ? 'rgba(0,0,0,0.08)' : 'rgba(255, 255, 255, 0.08)'}`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}>
          <h2
            id="lite-profile-creation-title"
            style={{
              fontSize: 15,
              fontWeight: 600,
              color: titleColor,
              margin: 0,
              letterSpacing: -0.2,
            }}
          >
            Lite profile creation
          </h2>
          <button
            type="button"
            onClick={requestClose}
            disabled={submitting}
            aria-label="Close"
            style={{
              width: 28,
              height: 28,
              background: 'transparent',
              border: 'none',
              color: isLight ? 'rgba(0,0,0,0.5)' : 'rgba(255, 255, 255, 0.6)',
              cursor: submitting ? 'not-allowed' : 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              borderRadius: 6,
              transition: 'all 0.15s ease',
            }}
          >
            <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <line x1="2" y1="2" x2="14" y2="14" />
              <line x1="14" y1="2" x2="2" y2="14" />
            </svg>
          </button>
        </div>

        {/* Stepper Navigation: ① Set up > ② Style survey */}
        <div style={{
          padding: '10px 24px',
          borderBottom: `1px solid ${isLight ? 'rgba(0,0,0,0.06)' : 'rgba(255, 255, 255, 0.06)'}`,
          background: isLight ? '#FAF8F4' : '#141619',
          display: 'flex',
          alignItems: 'center',
          gap: 10,
        }}>
          {/* Step 1: Set up */}
          <div
            onClick={() => {
              if (currentStep === 'survey' && !submitting) {
                if (autoAdvanceTimerRef.current) clearTimeout(autoAdvanceTimerRef.current);
                setCurrentStep('setup');
              }
            }}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              cursor: currentStep === 'survey' ? 'pointer' : 'default',
            }}
          >
            <span style={{
              width: 18,
              height: 18,
              borderRadius: '50%',
              border: `1.5px solid ${currentStep === 'setup' ? accentColor : (isLight ? '#16A34A' : '#4ADE80')}`,
              color: currentStep === 'setup' ? accentColor : (isLight ? '#16A34A' : '#4ADE80'),
              display: 'inline-flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: 10.5,
              fontWeight: 700,
            }}>
              1
            </span>
            <span style={{
              fontSize: 13,
              fontWeight: currentStep === 'setup' ? 600 : 500,
              color: currentStep === 'setup' ? titleColor : subtextColor,
            }}>
              Set up
            </span>
          </div>

          <span style={{ color: isLight ? 'rgba(0,0,0,0.3)' : 'rgba(255,255,255,0.25)', fontSize: 13 }}>
            〉
          </span>

          {/* Step 2: Style survey */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{
              width: 18,
              height: 18,
              borderRadius: '50%',
              border: `1.5px solid ${currentStep === 'survey' ? accentColor : (isLight ? 'rgba(0,0,0,0.25)' : 'rgba(255,255,255,0.25)')}`,
              color: currentStep === 'survey' ? accentColor : (isLight ? 'rgba(0,0,0,0.4)' : 'rgba(255,255,255,0.35)'),
              display: 'inline-flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: 10.5,
              fontWeight: 700,
            }}>
              2
            </span>
            <span style={{
              fontSize: 13,
              fontWeight: currentStep === 'survey' ? 600 : 500,
              color: currentStep === 'survey' ? titleColor : subtextColor,
            }}>
              Style survey
            </span>
          </div>
        </div>

        {/* Modal Body */}
        <div style={{ padding: '24px 28px' }}>
          {currentStep === 'setup' ? (
            /* STEP 1: Two Side-by-Side Panels */
            <div className="saha-lite-setup-grid">
              {/* Left Panel: Name this profile & Capabilities */}
              <div style={{
                background: panelBg,
                border: `1px solid ${panelBorder}`,
                borderRadius: 8,
                padding: '24px',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'space-between',
              }}>
                <div>
                  <div style={{
                    fontSize: 13.5,
                    fontWeight: 600,
                    color: titleColor,
                    marginBottom: 10,
                  }}>
                    Name this profile
                  </div>
                  <input
                    ref={nameInputRef}
                    type="text"
                    value={profileName}
                    onChange={(e) => {
                      setProfileName(e.target.value);
                      if (submitError) setSubmitError(null);
                    }}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && canAdvance) {
                        e.preventDefault();
                        advance();
                      }
                    }}
                    placeholder="e.g. Wedding style 1"
                    style={{
                      width: '100%',
                      height: 38,
                      padding: '0 12px',
                      background: inputBg,
                      border: `1px solid ${isDuplicateName ? '#EF4444' : inputBorder}`,
                      borderRadius: 4,
                      color: titleColor,
                      fontFamily: F,
                      fontSize: 13,
                      outline: 'none',
                      transition: 'border-color 0.15s ease',
                      boxSizing: 'border-box',
                    }}
                    onFocus={(e) => {
                      e.currentTarget.style.borderColor = isDuplicateName ? '#EF4444' : SONNA.ochre;
                    }}
                    onBlur={(e) => {
                      e.currentTarget.style.borderColor = isDuplicateName ? '#EF4444' : inputBorder;
                    }}
                  />
                  {isDuplicateName && (
                    <div style={{
                      marginTop: 6,
                      fontSize: 11.5,
                      color: '#EF4444',
                      display: 'flex',
                      alignItems: 'center',
                      gap: 5,
                      fontWeight: 500,
                    }}>
                      <span>⚠️</span>
                      <span>A profile named &quot;{trimmedName}&quot; already exists. Every profile must have a unique name.</span>
                    </div>
                  )}
                </div>

                {/* Divider Line */}
                <div style={{
                  height: 1,
                  background: isLight ? 'rgba(0, 0, 0, 0.1)' : 'rgba(255, 255, 255, 0.1)',
                  margin: '24px 0 20px',
                }} />

                {/* Information bullet checklist matching Imagen reference */}
                <div>
                  <div style={{
                    fontSize: 12.5,
                    fontWeight: 600,
                    color: isLight ? '#292524' : '#E2E8F0',
                    marginBottom: 12,
                    lineHeight: 1.4,
                  }}>
                    You can use this AI Profile to edit photos that are both
                  </div>

                  <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 18 }}>
                    <div style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 8,
                      fontSize: 12.5,
                      fontWeight: 600,
                      color: titleColor,
                    }}>
                      <span style={{ fontSize: 14 }}>•</span>
                      <span>RAW files</span>
                    </div>
                    <div style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 8,
                      fontSize: 12.5,
                      fontWeight: 600,
                      color: titleColor,
                    }}>
                      <span style={{ fontSize: 14 }}>•</span>
                      <span>Color photos</span>
                    </div>
                  </div>

                  <div style={{
                    fontSize: 11.5,
                    color: subtextColor,
                    lineHeight: 1.5,
                  }}>
                    To edit JPEG, TIFF, or black and white photos, you'll need to create a Personal AI Profile
                  </div>
                </div>
              </div>

              {/* Right Panel: Upload Preset & Dropzone */}
              <div style={{
                background: panelBg,
                border: `1px solid ${panelBorder}`,
                borderRadius: 8,
                padding: '24px',
                display: 'flex',
                flexDirection: 'column',
              }}>
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  marginBottom: 14,
                }}>
                  <div style={{ fontSize: 13.5, fontWeight: 600, color: titleColor }}>
                    Upload preset
                  </div>
                </div>

                <div
                  onDragOver={(e) => {
                    e.preventDefault();
                    setIsDragging(true);
                  }}
                  onDragLeave={() => setIsDragging(false)}
                  onDrop={handleDrop}
                  style={{
                    flex: 1,
                    minHeight: 250,
                    borderRadius: 6,
                    border: `1.5px dashed ${isDragging ? (isLight ? '#16A34A' : accentColor) : (isLight ? 'rgba(0,0,0,0.2)' : 'rgba(255, 255, 255, 0.18)')}`,
                    background: isDragging
                      ? (isLight ? 'rgba(22, 163, 74, 0.05)' : 'rgba(212, 244, 156, 0.05)')
                      : (isLight ? 'rgba(0, 0, 0, 0.02)' : 'rgba(0, 0, 0, 0.2)'),
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    padding: '24px 20px',
                    textAlign: 'center',
                    transition: 'all 0.15s ease',
                  }}
                >
                  {presetPath ? (
                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                      <div style={{
                        width: 44,
                        height: 44,
                        borderRadius: '50%',
                        background: 'rgba(34, 197, 94, 0.15)',
                        color: isLight ? '#16A34A' : '#22C55E',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: 22,
                        marginBottom: 12,
                      }}>
                        ✓
                      </div>
                      <div style={{
                        fontSize: 14,
                        fontWeight: 600,
                        color: titleColor,
                        marginBottom: 4,
                      }}>
                        {basenameOf(presetPath)}
                      </div>
                      <div style={{
                        fontSize: 12,
                        color: subtextColor,
                        marginBottom: 16,
                      }}>
                        Preset loaded successfully
                      </div>
                      <button
                        type="button"
                        onClick={pickPreset}
                        style={{
                          height: 32,
                          padding: '0 16px',
                          background: isLight ? '#FFFFFF' : 'rgba(255, 255, 255, 0.08)',
                          border: `1px solid ${isLight ? 'rgba(0,0,0,0.2)' : 'rgba(255, 255, 255, 0.18)'}`,
                          borderRadius: 4,
                          color: titleColor,
                          fontFamily: F,
                          fontSize: 12,
                          fontWeight: 500,
                          cursor: 'pointer',
                        }}
                      >
                        Change preset (.xmp)
                      </button>
                    </div>
                  ) : (
                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                      <XmpDocIcon />

                      <button
                        type="button"
                        onClick={pickPreset}
                        style={{
                          height: 32,
                          padding: '0 16px',
                          background: isLight ? '#FFFFFF' : 'rgba(255, 255, 255, 0.08)',
                          border: `1px solid ${isLight ? 'rgba(0,0,0,0.18)' : 'rgba(255, 255, 255, 0.14)'}`,
                          borderRadius: 6,
                          color: titleColor,
                          fontFamily: F,
                          fontSize: 12.5,
                          fontWeight: 500,
                          cursor: 'pointer',
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: 6,
                          margin: '14px 0 10px',
                          transition: 'all 0.15s ease',
                          boxShadow: isLight ? '0 1px 3px rgba(0,0,0,0.05)' : 'none',
                        }}
                      >
                        <span>My presets</span>
                        <span style={{ fontSize: 10 }}>▼</span>
                      </button>

                      <div style={{
                        fontSize: 11.5,
                        color: subtextColor,
                        marginBottom: 6,
                      }}>
                        Or
                      </div>

                      <div style={{
                        fontSize: 12,
                        color: subtextColor,
                        maxWidth: 260,
                        lineHeight: 1.5,
                      }}>
                        Drag & drop or{' '}
                        <span
                          onClick={pickPreset}
                          style={{
                            color: isLight ? '#16A34A' : accentColor,
                            textDecoration: 'underline',
                            cursor: 'pointer',
                            fontWeight: 600,
                          }}
                        >
                          browse
                        </span>{' '}
                        to upload your preset (XMP file format only)
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          ) : (
            /* STEP 2: Full 9-Step Visual Style Survey */
            <div>
              {/* Question Title & Prompt matching Imagen references */}
              <div style={{ fontSize: 16, fontWeight: 700, color: titleColor, marginBottom: 4 }}>
                {currentSurveyStep.title}
              </div>
              <div style={{ fontSize: 12.5, color: subtextColor, marginBottom: 18 }}>
                {currentSurveyStep.prompt}
              </div>

              {/* Gradient Bar with Left & Right Preference Labels */}
              <div style={{ marginBottom: 18 }}>
                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  fontSize: 11.5,
                  fontWeight: 600,
                  color: isLight ? '#44403C' : '#D1D5DB',
                  marginBottom: 6,
                }}>
                  <span>{currentSurveyStep.leftLabel}</span>
                  <span>{currentSurveyStep.rightLabel}</span>
                </div>
                <div style={{
                  width: '100%',
                  height: 2.5,
                  borderRadius: 1.5,
                  background: currentSurveyStep.gradient,
                }} />
              </div>

              {/* 5-Option Benchmark Photos Side-by-Side */}
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))',
                gap: 12,
              }}>
                {OPTION_VALUES.map((optVal, optIdx) => {
                  const isSelected = stepAnswers[surveyIndex] === optVal;
                  const imgSrc = `survey/${currentSurveyStep.imagePrefix}_opt${optIdx}.webp`;

                  return (
                    <div
                      key={optVal}
                      onClick={() => selectOption(optVal)}
                      style={{
                        position: 'relative',
                        borderRadius: 6,
                        overflow: 'hidden',
                        cursor: 'pointer',
                        outline: isSelected ? `2px solid ${accentColor}` : '2px solid transparent',
                        boxShadow: isSelected
                          ? '0 0 16px rgba(212, 244, 156, 0.45)'
                          : (isLight ? '0 2px 6px rgba(0,0,0,0.06)' : '0 2px 6px rgba(0,0,0,0.4)'),
                        transition: 'transform 0.15s ease, box-shadow 0.15s ease, outline 0.15s ease',
                        background: '#1F2228',
                      }}
                      onMouseEnter={(e) => {
                        if (!isSelected) {
                          e.currentTarget.style.transform = 'translateY(-2px)';
                        }
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.transform = 'translateY(0)';
                      }}
                    >
                      <img
                        src={imgSrc}
                        alt={`Option ${optVal >= 0 ? `+${optVal}` : optVal}`}
                        draggable="false"
                        style={{
                          width: '100%',
                          aspectRatio: '120 / 148',
                          objectFit: 'cover',
                          display: 'block',
                        }}
                      />
                    </div>
                  );
                })}
              </div>

              {/* Bottom 9-Dash Progress Tracker */}
              <div style={{
                display: 'flex',
                gap: 6,
                justifyContent: 'center',
                alignItems: 'center',
                marginTop: 26,
                marginBottom: 8,
              }}>
                {LITE_SURVEY_STEPS.map((s, idx) => {
                  const isCurrent = idx === surveyIndex;
                  const isAnswered = stepAnswers[idx] !== undefined;

                  return (
                    <div
                      key={idx}
                      title={`Step ${idx + 1}: ${s.title}`}
                      onClick={() => {
                        if (isAnswered || idx <= surveyIndex) {
                          if (autoAdvanceTimerRef.current) clearTimeout(autoAdvanceTimerRef.current);
                          setSurveyIndex(idx);
                        }
                      }}
                      style={{
                        width: 26,
                        height: 3,
                        borderRadius: 2,
                        background: isCurrent
                          ? accentColor
                          : (isLight
                            ? (isAnswered ? '#78716c' : 'rgba(0,0,0,0.18)')
                            : (isAnswered ? '#6B7280' : 'rgba(255,255,255,0.18)')),
                        cursor: (isAnswered || idx <= surveyIndex) ? 'pointer' : 'default',
                        transition: 'all 0.2s ease',
                      }}
                    />
                  );
                })}
              </div>

              {submitError && (
                <div style={{
                  marginTop: 14,
                  padding: '10px 14px',
                  background: 'rgba(239, 68, 68, 0.12)',
                  border: `1px solid ${SONNA.red}`,
                  borderRadius: 4,
                  fontSize: 12.5,
                  color: SONNA.red,
                }}>
                  {submitError}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer Actions */}
        <div style={{
          padding: '14px 28px',
          borderTop: `1px solid ${isLight ? 'rgba(0,0,0,0.08)' : 'rgba(255, 255, 255, 0.08)'}`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: currentStep === 'survey' ? 'space-between' : 'flex-end',
          background: isLight ? '#FAF8F4' : '#141619',
        }}>
          {currentStep === 'survey' && (
            <button
              type="button"
              onClick={goBack}
              disabled={submitting}
              style={{
                height: 34,
                padding: '0 18px',
                background: 'transparent',
                border: `1px solid ${isLight ? 'rgba(0,0,0,0.2)' : 'rgba(255, 255, 255, 0.18)'}`,
                borderRadius: 4,
                color: titleColor,
                fontFamily: F,
                fontSize: 13,
                fontWeight: 500,
                cursor: submitting ? 'not-allowed' : 'pointer',
              }}
            >
              Back
            </button>
          )}

          {submitError && (
            <div
              title={submitError}
              style={{
                flex: 1,
                margin: '0 16px',
                padding: '6px 12px',
                background: 'rgba(239, 68, 68, 0.12)',
                border: `1px solid ${SONNA.red}`,
                borderRadius: 4,
                fontSize: 12,
                color: SONNA.red,
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
              }}
            >
              ⚠ {submitError}
            </div>
          )}

          {/* Setup Next Button OR Question 9 Build Profile Button OR optional Survey Next */}
          {(currentStep === 'setup' || surveyIndex === 8 || stepAnswers[surveyIndex] !== undefined) && (
            <button
              type="button"
              onClick={advance}
              disabled={!canAdvance || submitting}
              style={{
                height: 34,
                padding: '0 24px',
                background: (canAdvance && !submitting) ? SONNA.cta : (isLight ? '#E5E0D8' : '#2D323A'),
                border: 'none',
                borderRadius: 4,
                color: (canAdvance && !submitting) ? '#FFFFFF' : (isLight ? '#A8A29E' : '#6E7684'),
                fontFamily: F,
                fontSize: 13,
                fontWeight: 600,
                cursor: (canAdvance && !submitting) ? 'pointer' : 'not-allowed',
                transition: 'all 0.15s ease',
                boxShadow: (canAdvance && !submitting) ? '0 2px 8px rgba(224, 83, 83, 0.35)' : 'none',
              }}
            >
              {submitting
                ? 'Building…'
                : (currentStep === 'survey' && surveyIndex === 8
                  ? 'Build profile'
                  : 'Next')}
            </button>
          )}
        </div>
      </div>

      {/* Confirm Discard / Leave Modal */}
      <ConfirmModal
        isOpen={confirmLeaveOpen}
        title="Are you sure you want to leave?"
        message="Your progress won't be saved for later"
        confirmText="Yes, leave"
        cancelText="Cancel"
        onConfirm={() => {
          setConfirmLeaveOpen(false);
          onClose();
        }}
        onCancel={() => setConfirmLeaveOpen(false)}
        theme={theme}
      />
    </div>
  );
}

export default LiteProfileWizard;
