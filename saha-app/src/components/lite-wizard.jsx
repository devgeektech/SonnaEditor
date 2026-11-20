// Lite profile creation wizard
// Reference: media_1794857986217_96aa8a28.png (Lite profile creation matching Imagen reference)
// - Header: "Lite profile creation" with Stepper: "① Set up > ② Style survey"
// - Step 1: Two panels side-by-side
//   - Left panel: "Name this profile" input + supported formats checklist
//   - Right panel: "Upload preset" with XMP icon + drag-and-drop dropzone + "My presets ⌵" button + browse
// - Step 2: "Style survey" with visual slider questions (Exposure, Temperature, Tint)
// - Full Light and Dark theme support

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import SONNA from '../tokens.js';
import { createLiteProfile } from '../api/client.js';
import { ConfirmModal } from './confirm-modal.jsx';

const F = SONNA.font;
const M = SONNA.mono;

const Tnum = { fontFamily: M, fontVariantNumeric: 'tabular-nums' };

const SURVEY_QUESTIONS = [
  {
    key: 'exposure',
    title: 'Exposure',
    prompt: 'How bright are your edits relative to the preset baseline?',
    description: 'Controls overall image brightness. Affects Exposure2012.',
    options: [
      [-2, 'Much darker (moody, underexposed look)'],
      [-1, 'Slightly darker'],
      [0,  'Match the preset'],
      [+1, 'Slightly brighter'],
      [+2, 'Much brighter (bright, airy look)'],
    ],
  },
  {
    key: 'temperature',
    title: 'Temperature',
    prompt: "What's your typical white-balance bias?",
    description: 'Shifts the image cooler (blue) or warmer (yellow). Affects Temperature.',
    options: [
      [-2, 'Much cooler (blue-shifted, clean / editorial)'],
      [-1, 'Slightly cooler'],
      [0,  'Match the preset'],
      [+1, 'Slightly warmer'],
      [+2, 'Much warmer (golden, vintage)'],
    ],
  },
  {
    key: 'tint',
    title: 'Tint',
    prompt: "What's your typical green/magenta colour cast?",
    description: 'Global green/magenta cast across the whole image. Affects Tint.',
    options: [
      [-2, 'Strongly green-shifted (cooler greens, can flatten skin)'],
      [-1, 'Slightly green-shifted'],
      [0,  'Match the preset'],
      [+1, 'Slightly magenta-shifted'],
      [+2, 'Strongly magenta-shifted (warmer tones, lifted skin)'],
    ],
  },
];

const LEGACY_NEUTRAL_SURVEY_ANSWERS = {
  contrast: 0,
  saturation: 0,
  shadows: 0,
};

function basenameOf(p) {
  if (!p) return '';
  const segs = p.replace(/\\/g, '/').split('/').filter(Boolean);
  return segs[segs.length - 1] || p;
}

function previewSvg(questionKey, answer) {
  const exposure = questionKey === 'exposure' ? answer : 0;
  const temperature = questionKey === 'temperature' ? answer : 0;
  const tint = questionKey === 'tint' ? answer : 0;
  const brightness = 88 + exposure * 7;
  const sky = temperature > 0 ? '#efd7a7' : temperature < 0 ? '#b9d3ed' : '#d6ddd8';
  const mid = tint > 0 ? '#d9b7c5' : tint < 0 ? '#b7c9b1' : '#c7c0b5';
  const land = exposure < 0 ? '#413b31' : '#786d5c';
  const hazeOpacity = Math.max(0.08, 0.26 + exposure * 0.04);
  const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 84">
      <defs>
        <linearGradient id="sky" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stop-color="${sky}"/>
          <stop offset="1" stop-color="${mid}"/>
        </linearGradient>
        <linearGradient id="field" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0%" stop-color="${land}"/>
          <stop offset="1" stop-color="#9a8467"/>
        </linearGradient>
      </defs>
      <rect width="120" height="84" fill="url(#sky)"/>
      <circle cx="88" cy="22" r="12" fill="#fff0c4" opacity="${0.42 + exposure * 0.06}"/>
      <path d="M0 50 C22 39 39 43 56 35 C75 25 94 33 120 23 L120 84 L0 84 Z" fill="#5f6d64" opacity="0.72"/>
      <path d="M0 60 C22 53 42 57 61 49 C83 40 101 46 120 38 L120 84 L0 84 Z" fill="url(#field)"/>
      <path d="M10 72 C35 63 72 64 111 53" stroke="#f4efe6" stroke-width="5" opacity="${hazeOpacity}" fill="none"/>
      <rect width="120" height="84" fill="#fff" opacity="${Math.max(0, (brightness - 88) / 100)}"/>
      <rect width="120" height="84" fill="#000" opacity="${Math.max(0, (88 - brightness) / 110)}"/>
    </svg>
  `;
  return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
}

function SurveyPreview({ questionKey, answer }) {
  return (
    <img
      src={previewSvg(questionKey, answer)}
      alt=""
      draggable="false"
      style={{
        width: '100%',
        aspectRatio: '1.42 / 1',
        objectFit: 'cover',
        borderRadius: 4,
        display: 'block',
      }}
    />
  );
}

// XMP Document Icon matching reference media_1794857986217_96aa8a28.png
function XmpDocIcon() {
  return (
    <svg width="48" height="56" viewBox="0 0 44 52" fill="none" style={{ overflow: 'visible' }}>
      {/* File body with folded top-right corner */}
      <path
        d="M4 6C4 3.79086 5.79086 2 8 2H28L40 14V46C40 48.2091 38.2091 50 36 50H8C5.79086 50 4 48.2091 4 46V6Z"
        fill="#4A5568"
      />
      {/* Folded corner flap */}
      <path
        d="M28 2V14H40L28 2Z"
        fill="#2D3748"
      />
      {/* XMP text label */}
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
      {/* Blue upload badge at bottom-right */}
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

export function LiteProfileWizard({ onClose, onCreated, theme = 'dark' }) {
  const isLight = theme === 'light';

  // Steps: 'setup' (Name + Preset) | 'survey' (Style Survey)
  const [currentStep, setCurrentStep] = useState('setup');
  const [surveyIndex, setSurveyIndex] = useState(0);

  const [profileName, setProfileName] = useState('');
  const [presetPath, setPresetPath] = useState(null);
  const [surveyAnswers, setSurveyAnswers] = useState({});
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [confirmLeaveOpen, setConfirmLeaveOpen] = useState(false);

  const fileInputRef = useRef(null);
  const nameInputRef = useRef(null);

  // Auto-focus name input on mount
  useEffect(() => {
    if (currentStep === 'setup') {
      const t = setTimeout(() => nameInputRef.current?.focus(), 50);
      return () => clearTimeout(t);
    }
  }, [currentStep]);

  const hasAnyState = useMemo(() => (
    profileName.trim().length > 0
    || presetPath !== null
    || Object.keys(surveyAnswers).length > 0
  ), [profileName, presetPath, surveyAnswers]);

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

  const setAnswer = useCallback((key, value) => {
    setSurveyAnswers((s) => ({ ...s, [key]: value }));
  }, []);

  const canAdvance = useMemo(() => {
    if (currentStep === 'setup') {
      return profileName.trim().length > 0 && presetPath !== null;
    }
    if (currentStep === 'survey') {
      const q = SURVEY_QUESTIONS[surveyIndex];
      return q.key in surveyAnswers;
    }
    return false;
  }, [currentStep, profileName, presetPath, surveyIndex, surveyAnswers]);

  const advance = useCallback(() => {
    if (!canAdvance || submitting) return;

    if (currentStep === 'setup') {
      setCurrentStep('survey');
      setSurveyIndex(0);
      return;
    }

    if (currentStep === 'survey') {
      if (surveyIndex < SURVEY_QUESTIONS.length - 1) {
        setSurveyIndex((i) => i + 1);
        return;
      }

      // Survey complete → Submit
      setSubmitting(true);
      setSubmitError(null);
      createLiteProfile({
        profile_name: profileName.trim(),
        preset_path: presetPath,
        survey_answers: {
          ...LEGACY_NEUTRAL_SURVEY_ANSWERS,
          ...surveyAnswers,
        },
      }).then((res) => {
        setSubmitting(false);
        onCreated(res);
      }).catch((e) => {
        setSubmitting(false);
        setSubmitError(e.message || 'Failed to create profile');
      });
    }
  }, [canAdvance, submitting, currentStep, surveyIndex, profileName, presetPath, surveyAnswers, onCreated]);

  const goBack = useCallback(() => {
    if (submitting) return;
    if (currentStep === 'survey') {
      if (surveyIndex > 0) {
        setSurveyIndex((i) => i - 1);
      } else {
        setCurrentStep('setup');
      }
    }
  }, [currentStep, surveyIndex, submitting]);

  // Theming colors
  const modalBg = isLight ? '#FFFFFF' : '#1B1E22';
  const modalBorder = isLight ? 'rgba(0, 0, 0, 0.12)' : 'rgba(255, 255, 255, 0.08)';
  const panelBg = isLight ? '#FBF9F5' : '#202328';
  const panelBorder = isLight ? 'rgba(0, 0, 0, 0.1)' : 'rgba(255, 255, 255, 0.08)';
  const titleColor = isLight ? '#1c1917' : '#FFFFFF';
  const subtextColor = isLight ? '#78716c' : '#8E95A2';
  const inputBg = isLight ? '#FFFFFF' : '#16181B';
  const inputBorder = isLight ? 'rgba(0, 0, 0, 0.15)' : 'rgba(255, 255, 255, 0.14)';
  const accentColor = '#D4F49C'; // lime green from Imagen reference

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 1100,
        background: 'rgba(0, 0, 0, 0.75)',
        backdropFilter: 'blur(5px)',
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
          maxWidth: 880,
          background: modalBg,
          border: `1px solid ${modalBorder}`,
          borderRadius: 10,
          boxShadow: isLight
            ? '0 24px 60px rgba(0, 0, 0, 0.18)'
            : '0 28px 70px rgba(0, 0, 0, 0.75)',
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
          padding: '18px 24px 14px',
          borderBottom: `1px solid ${isLight ? 'rgba(0,0,0,0.08)' : 'rgba(255, 255, 255, 0.08)'}`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}>
          <h2
            id="lite-profile-creation-title"
            style={{
              fontSize: 16,
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
              width: 30,
              height: 30,
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
          padding: '12px 24px',
          borderBottom: `1px solid ${isLight ? 'rgba(0,0,0,0.06)' : 'rgba(255, 255, 255, 0.06)'}`,
          background: isLight ? '#FAF8F4' : '#191B1F',
          display: 'flex',
          alignItems: 'center',
          gap: 12,
        }}>
          {/* Step 1: Set up */}
          <div
            onClick={() => {
              if (currentStep === 'survey') setCurrentStep('setup');
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
            ›
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
            <div style={{
              display: 'grid',
              gridTemplateColumns: '1fr 1.35fr',
              gap: 20,
              alignItems: 'stretch',
            }}>
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
                    onChange={(e) => setProfileName(e.target.value)}
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
                      border: `1px solid ${inputBorder}`,
                      borderRadius: 4,
                      color: titleColor,
                      fontFamily: F,
                      fontSize: 13,
                      outline: 'none',
                      transition: 'border-color 0.15s ease',
                      boxSizing: 'border-box',
                    }}
                    onFocus={(e) => {
                      e.currentTarget.style.borderColor = SONNA.ochre;
                    }}
                    onBlur={(e) => {
                      e.currentTarget.style.borderColor = inputBorder;
                    }}
                  />
                </div>

                {/* Divider Line */}
                <div style={{
                  height: 1,
                  background: isLight ? 'rgba(0, 0, 0, 0.1)' : 'rgba(255, 255, 255, 0.1)',
                  margin: '24px 0 20px',
                }} />

                {/* Information bullet checklist */}
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
                {/* Header row */}
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  marginBottom: 14,
                }}>
                  <div style={{ fontSize: 13.5, fontWeight: 600, color: titleColor }}>
                    Upload preset
                  </div>
                </div>

                {/* Dropzone container */}
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
                    /* Preset selected state */
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
                    /* Empty Dropzone matching reference */
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
            /* STEP 2: Style Survey Questions */
            <div>
              <div style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                marginBottom: 10,
              }}>
                <div style={{
                  fontSize: 11,
                  fontWeight: 600,
                  textTransform: 'uppercase',
                  letterSpacing: 0.5,
                  color: SONNA.ochre,
                }}>
                  Question {surveyIndex + 1} of {SURVEY_QUESTIONS.length} · {SURVEY_QUESTIONS[surveyIndex].title}
                </div>
              </div>

              <div style={{ fontSize: 15, fontWeight: 600, color: titleColor, marginBottom: 6 }}>
                {SURVEY_QUESTIONS[surveyIndex].prompt}
              </div>
              <div style={{ fontSize: 12.5, color: subtextColor, marginBottom: 20 }}>
                {SURVEY_QUESTIONS[surveyIndex].description}
              </div>

              {/* 5-option Visual Card Grid */}
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(5, minmax(0, 1fr))',
                gap: 12,
              }}>
                {SURVEY_QUESTIONS[surveyIndex].options.map(([ans, label]) => {
                  const selected = surveyAnswers[SURVEY_QUESTIONS[surveyIndex].key] === ans;
                  return (
                    <div
                      key={ans}
                      onClick={() => setAnswer(SURVEY_QUESTIONS[surveyIndex].key, ans)}
                      style={{
                        padding: 6,
                        background: selected ? (isLight ? '#F5EFE6' : 'rgba(255, 255, 255, 0.08)') : panelBg,
                        border: `1.5px solid ${selected ? SONNA.ochre : panelBorder}`,
                        borderRadius: 6,
                        cursor: 'pointer',
                        display: 'flex',
                        flexDirection: 'column',
                        gap: 8,
                        transition: 'all 0.15s ease',
                      }}
                    >
                      <SurveyPreview questionKey={SURVEY_QUESTIONS[surveyIndex].key} answer={ans} />
                      <div style={{
                        height: 22,
                        borderRadius: 3,
                        background: selected ? SONNA.ochre : 'transparent',
                        color: selected ? '#FFFFFF' : subtextColor,
                        fontSize: 11.5,
                        fontWeight: 600,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        ...Tnum,
                      }}>
                        {ans > 0 ? `+${ans}` : ans}
                      </div>
                    </div>
                  );
                })}
              </div>

              {submitError && (
                <div style={{
                  marginTop: 18,
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
          padding: '16px 28px',
          borderTop: `1px solid ${isLight ? 'rgba(0,0,0,0.08)' : 'rgba(255, 255, 255, 0.08)'}`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: currentStep === 'survey' ? 'space-between' : 'flex-end',
          background: isLight ? '#FAF8F4' : '#191B1F',
        }}>
          {currentStep === 'survey' && (
            <button
              type="button"
              onClick={goBack}
              disabled={submitting}
              style={{
                height: 36,
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

          <button
            type="button"
            onClick={advance}
            disabled={!canAdvance || submitting}
            style={{
              height: 36,
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
            {submitting ? 'Creating…' : (currentStep === 'survey' && surveyIndex === SURVEY_QUESTIONS.length - 1 ? 'Create profile' : 'Next')}
          </button>
        </div>
      </div>

      {/* Confirm Discard / Leave Modal matching reference */}
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
