// Lite profile creation wizard — multi-step modal (4 visible steps; survey
// step paginates internally through the AI-edited Lite parameters).
//
// Step 1 — Profile name
// Step 2 — Preset upload (.xmp)
// Step 3 — style survey (one question per screen)
// Step 4 — Confirmation summary + submit
//
// Modal is non-dismissable via backdrop click to protect partially-entered
// progress. Escape and the explicit Cancel button trigger a native confirm
// when state has been entered.

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import SONNA from '../tokens.js';
import { createLiteProfile } from '../api/client.js';

const F = SONNA.font;
const M = SONNA.mono;

const Tlabel = {
  fontSize: 10, fontWeight: 600, color: SONNA.fgDim,
  textTransform: 'uppercase', letterSpacing: 0.6,
};
const Tnum = { fontFamily: M, fontVariantNumeric: 'tabular-nums' };

// The Lite initial profile path only adapts Exposure, Temperature, and Tint
// per photo; remaining legacy survey keys are submitted as neutral defaults
// for backend sidecar compatibility.
const SURVEY_QUESTIONS = [
  {
    key: 'exposure', title: 'Exposure',
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
    key: 'temperature', title: 'Temperature',
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
    key: 'tint', title: 'Tint',
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

const STEPS = ['name', 'preset', 'survey', 'confirm'];
const LEGACY_NEUTRAL_SURVEY_ANSWERS = {
  contrast: 0,
  saturation: 0,
  shadows: 0,
};

function basenameOf(p) {
  if (!p) return '';
  const segs = p.split('/').filter(Boolean);
  return segs[segs.length - 1] || p;
}

function labelForAnswer(questionKey, answer) {
  const q = SURVEY_QUESTIONS.find((x) => x.key === questionKey);
  if (!q) return String(answer);
  const opt = q.options.find(([v]) => v === answer);
  return opt ? opt[1] : String(answer);
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
          <stop offset="0" stop-color="${sky}"/>
          <stop offset="1" stop-color="${mid}"/>
        </linearGradient>
        <linearGradient id="field" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0" stop-color="${land}"/>
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
        borderRadius: 3,
        display: 'block',
      }}
    />
  );
}

// ── Step container chrome ────────────────────────────────
function StepHeader({ stepKind, surveyIndex }) {
  const stepIndex = STEPS.indexOf(stepKind);
  return (
    <div style={{
      padding: '14px 22px 12px',
      borderBottom: `1px solid ${SONNA.lineSoft}`,
    }}>
      <div style={{ ...Tlabel, marginBottom: 8 }}>
        New Lite profile · Step {stepIndex + 1} of {STEPS.length}
      </div>
      {/* Step dots */}
      <div style={{ display: 'flex', gap: 4 }}>
        {STEPS.map((k, i) => (
          <div key={k} style={{
            flex: 1, height: 3, borderRadius: 2,
            background: i <= stepIndex ? SONNA.ochre : SONNA.bgLifted,
          }} />
        ))}
      </div>
      {stepKind === 'survey' && (
        <div style={{
          marginTop: 8,
          ...Tnum, fontSize: 10.5, color: SONNA.fgFaint,
        }}>
          Question {surveyIndex + 1} of {SURVEY_QUESTIONS.length}
        </div>
      )}
    </div>
  );
}

function StepNameInput({ value, onChange, onNext }) {
  const inputRef = useRef(null);

  useEffect(() => {
    const id = window.setTimeout(() => inputRef.current?.focus(), 0);
    return () => window.clearTimeout(id);
  }, []);

  return (
    <div style={{ padding: '20px 22px', flex: 1, minHeight: 0, overflow: 'auto' }}>
      <div style={{ fontSize: 13, color: SONNA.fg, marginBottom: 8 }}>Profile name</div>
      <input
        ref={inputRef}
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && value.trim()) {
            e.preventDefault();
            onNext();
          }
        }}
        placeholder="e.g. Wedding Lite, Brand Campaign"
        style={{
          width: '100%', height: 38, padding: '0 12px',
          background: SONNA.bgPanel,
          border: `1px solid ${SONNA.line}`, borderRadius: 3,
          color: SONNA.fg, fontFamily: F, fontSize: 13,
          outline: 'none',
        }}
      />
      <div style={{ marginTop: 10, fontSize: 11.5, color: SONNA.fgFaint, lineHeight: 1.5 }}>
        Used as the profile's display name throughout Saha. You can rename it later.
      </div>
    </div>
  );
}

function StepPresetUpload({ presetPath, onPick }) {
  return (
    <div style={{ padding: '20px 22px', flex: 1, minHeight: 0, overflow: 'auto' }}>
      <div style={{ fontSize: 13, color: SONNA.fg, marginBottom: 12 }}>Lightroom preset</div>
      <button
        type="button"
        onClick={onPick}
        style={{
          height: 38, padding: '0 16px',
          background: SONNA.bgLifted,
          border: `1px solid ${SONNA.line}`, borderRadius: 3,
          color: SONNA.fg, fontFamily: F, fontSize: 13, fontWeight: 500,
          cursor: 'pointer',
          display: 'inline-flex', alignItems: 'center', gap: 10,
        }}
      >
        <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
          <path d="M2 7v3h8V7M6 2v6M3.5 4.5L6 2l2.5 2.5"
            stroke={SONNA.fgMute} strokeWidth="1.2"
            strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        <span>{presetPath ? 'Choose a different preset' : 'Choose preset (.xmp)'}</span>
      </button>
      {presetPath && (
        <div style={{
          marginTop: 12, padding: '10px 12px',
          background: SONNA.bgPanel,
          border: `1px solid ${SONNA.lineSoft}`, borderRadius: 3,
          ...Tnum, fontSize: 12, color: SONNA.fg,
        }}>
          {basenameOf(presetPath)}
        </div>
      )}
      <div style={{ marginTop: 12, fontSize: 11.5, color: SONNA.fgFaint, lineHeight: 1.5 }}>
        Pick a Lightroom develop preset exported as .xmp. Saha reads only its
        slider values; the original file is copied next to the new profile
        for provenance.
      </div>
    </div>
  );
}

function StepSurveyQuestion({ question, value, onChange }) {
  return (
    <div style={{ padding: '20px 22px', flex: 1, minHeight: 0, overflow: 'auto' }}>
      <div style={{ ...Tlabel, marginBottom: 4 }}>{question.title}</div>
      <div style={{ fontSize: 13, color: SONNA.fg, lineHeight: 1.5 }}>
        {question.prompt}
      </div>
      <div style={{
        marginTop: 6, fontSize: 11.5, color: SONNA.fgFaint, lineHeight: 1.5,
      }}>
        {question.description}
      </div>

      <div style={{
        marginTop: 16,
        display: 'grid',
        gridTemplateColumns: 'repeat(5, minmax(0, 1fr))',
        gap: 8,
      }}>
        {question.options.map(([ans, label]) => {
          const selected = value === ans;
          return (
            <label key={ans} style={{
              padding: 4,
              background: selected ? SONNA.bgLifted : SONNA.bgPanel,
              border: `1px solid ${selected ? SONNA.ochre : SONNA.lineSoft}`,
              borderRadius: 3,
              cursor: 'pointer',
              display: 'flex', flexDirection: 'column', gap: 6,
              transition: 'border-color 100ms, background 100ms',
            }}>
              <input
                type="radio"
                name={`survey-${question.key}`}
                checked={selected}
                onChange={() => onChange(ans)}
                aria-label={label}
                style={{ position: 'absolute', opacity: 0, pointerEvents: 'none' }}
              />
              <SurveyPreview questionKey={question.key} answer={ans} />
              <span style={{
                ...Tnum,
                height: 18,
                borderRadius: 2,
                background: selected ? SONNA.ochreTint : 'transparent',
                color: selected ? SONNA.fg : SONNA.fgMute,
                fontSize: 11,
                fontWeight: selected ? 600 : 500,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
              }}>
                {ans > 0 ? `+${ans}` : ans}
              </span>
            </label>
          );
        })}
      </div>
    </div>
  );
}

function StepConfirm({ profileName, presetPath, surveyAnswers, submitError }) {
  return (
    <div style={{ padding: '20px 22px', flex: 1, minHeight: 0, overflow: 'auto' }}>
      <div style={{ ...Tlabel, marginBottom: 12 }}>Review</div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
        <div>
          <div style={{ fontSize: 10.5, color: SONNA.fgDim, marginBottom: 2 }}>Name</div>
          <div style={{ fontSize: 13, color: SONNA.fg }}>{profileName}</div>
        </div>
        <div>
          <div style={{ fontSize: 10.5, color: SONNA.fgDim, marginBottom: 2 }}>Preset</div>
          <div style={{ ...Tnum, fontSize: 12, color: SONNA.fg }}>{basenameOf(presetPath)}</div>
        </div>
        <div>
          <div style={{ fontSize: 10.5, color: SONNA.fgDim, marginBottom: 6 }}>Style survey</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            {SURVEY_QUESTIONS.map((q) => (
              <div key={q.key} style={{
                display: 'grid', gridTemplateColumns: '90px 1fr', gap: 12,
                fontSize: 12, lineHeight: 1.45,
              }}>
                <span style={{ color: SONNA.fgMute }}>{q.title}</span>
                <span style={{ color: SONNA.fg }}>
                  {labelForAnswer(q.key, surveyAnswers[q.key])}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {submitError && (
        <div style={{
          marginTop: 16, padding: '10px 12px',
          background: 'rgba(156, 84, 84, 0.16)',
          border: `1px solid ${SONNA.red}`,
          borderRadius: 3,
          fontSize: 12, color: SONNA.fg, lineHeight: 1.5,
        }}>{submitError}</div>
      )}
    </div>
  );
}

// ── Top-level wizard ─────────────────────────────────────
export function LiteProfileWizard({ onClose, onCreated }) {
  const [stepKind, setStepKind] = useState('name');
  const [surveyIndex, setSurveyIndex] = useState(0);
  const [profileName, setProfileName] = useState('');
  const [presetPath, setPresetPath] = useState(null);
  const [surveyAnswers, setSurveyAnswers] = useState({});
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState(null);

  const hasAnyState = useMemo(() => (
    profileName.trim().length > 0
    || presetPath !== null
    || Object.keys(surveyAnswers).length > 0
  ), [profileName, presetPath, surveyAnswers]);

  const requestClose = useCallback(() => {
    if (submitting) return;
    if (hasAnyState) {
      const ok = window.confirm('Discard your progress and close?');
      if (!ok) return;
    }
    onClose();
  }, [hasAnyState, onClose, submitting]);

  // Escape triggers Cancel (with discard confirmation if state entered).
  // Backdrop click is intentionally NOT wired to close — protects mid-flow
  // progress from one accidental click.
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

  const pickPreset = useCallback(async () => {
    if (!window.saha?.pickFile) {
      setSubmitError('File picker unavailable (run via Electron)');
      return;
    }
    const path = await window.saha.pickFile({
      title: 'Choose a Lightroom preset',
      filters: [{ name: 'Lightroom preset', extensions: ['xmp'] }],
    });
    if (path) setPresetPath(path);
  }, []);

  const setAnswer = useCallback((key, value) => {
    setSurveyAnswers((s) => ({ ...s, [key]: value }));
  }, []);

  const canAdvance = (() => {
    if (stepKind === 'name') return profileName.trim().length > 0;
    if (stepKind === 'preset') return presetPath !== null;
    if (stepKind === 'survey') {
      const q = SURVEY_QUESTIONS[surveyIndex];
      return q.key in surveyAnswers;
    }
    if (stepKind === 'confirm') return true;
    return false;
  })();

  const advance = useCallback(() => {
    if (!canAdvance || submitting) return;
    if (stepKind === 'name') { setStepKind('preset'); return; }
    if (stepKind === 'preset') { setStepKind('survey'); setSurveyIndex(0); return; }
    if (stepKind === 'survey') {
      if (surveyIndex < SURVEY_QUESTIONS.length - 1) {
        setSurveyIndex((i) => i + 1);
      } else {
        setStepKind('confirm');
      }
      return;
    }
    // 'confirm' → submit
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
  }, [canAdvance, submitting, stepKind, surveyIndex, profileName, presetPath, surveyAnswers, onCreated]);

  const goBack = useCallback(() => {
    if (submitting) return;
    if (stepKind === 'preset') { setStepKind('name'); return; }
    if (stepKind === 'survey') {
      if (surveyIndex > 0) setSurveyIndex((i) => i - 1);
      else setStepKind('preset');
      return;
    }
    if (stepKind === 'confirm') {
      setStepKind('survey');
      setSurveyIndex(SURVEY_QUESTIONS.length - 1);
    }
  }, [stepKind, surveyIndex, submitting]);

  // Body content per step.
  let body;
  if (stepKind === 'name') {
    body = (
      <StepNameInput
        value={profileName}
        onChange={setProfileName}
        onNext={advance}
      />
    );
  } else if (stepKind === 'preset') {
    body = <StepPresetUpload presetPath={presetPath} onPick={pickPreset} />;
  } else if (stepKind === 'survey') {
    const q = SURVEY_QUESTIONS[surveyIndex];
    body = (
      <StepSurveyQuestion
        question={q}
        value={surveyAnswers[q.key]}
        onChange={(v) => setAnswer(q.key, v)}
      />
    );
  } else {
    body = (
      <StepConfirm
        profileName={profileName.trim()}
        presetPath={presetPath}
        surveyAnswers={surveyAnswers}
        submitError={submitError}
      />
    );
  }

  const showBack = stepKind !== 'name';
  const nextLabel = stepKind === 'confirm' ? 'Create profile' : 'Next';

  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 100,
      background: 'rgba(0, 0, 0, 0.55)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      padding: 24,
    }}>
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="lite-wizard-title"
        style={{
          width: 560, maxWidth: '100%', maxHeight: '85vh',
          background: SONNA.bgPanel,
          border: `1px solid ${SONNA.line}`,
          borderRadius: 4,
          boxShadow: '0 10px 32px rgba(0, 0, 0, 0.55)',
          display: 'flex', flexDirection: 'column',
        }}
      >
        {/* Title bar */}
        <div style={{
          padding: '14px 22px 0',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        }}>
          <div id="lite-wizard-title" style={{
            fontSize: 14, fontWeight: 600, color: SONNA.fg, letterSpacing: 0,
          }}>Create Lite profile</div>
          <button
            type="button"
            onClick={requestClose}
            disabled={submitting}
            aria-label="Cancel"
            style={{
              width: 22, height: 22, padding: 0,
              background: 'transparent', border: 'none',
              cursor: submitting ? 'not-allowed' : 'pointer',
              opacity: submitting ? 0.4 : 0.7,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}
          >
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
              <path d="M2.5 2.5l7 7M9.5 2.5l-7 7"
                stroke={SONNA.fgMute} strokeWidth="1.3" strokeLinecap="round" />
            </svg>
          </button>
        </div>

        <StepHeader stepKind={stepKind} surveyIndex={surveyIndex} />
        {body}

        {/* Footer: Back + Next/Create */}
        <div style={{
          padding: 14,
          borderTop: `1px solid ${SONNA.lineSoft}`,
          display: 'flex', justifyContent: 'space-between', gap: 10,
          flexShrink: 0,
        }}>
          {showBack ? (
            <button
              type="button"
              onClick={goBack}
              disabled={submitting}
              style={{
                height: 34, padding: '0 16px',
                background: 'transparent',
                border: `1px solid ${SONNA.line}`, borderRadius: 3,
                color: submitting ? SONNA.fgFaint : SONNA.fg,
                fontFamily: F, fontSize: 12.5, fontWeight: 500,
                cursor: submitting ? 'not-allowed' : 'pointer',
              }}
            >Back</button>
          ) : <span />}

          <button
            type="button"
            onClick={advance}
            disabled={!canAdvance || submitting}
            style={{
              height: 34, padding: '0 18px',
              background: (canAdvance && !submitting) ? SONNA.cta : SONNA.bgLifted,
              border: 'none', borderRadius: 3,
              color: (canAdvance && !submitting) ? SONNA.onCta : SONNA.fgFaint,
              fontFamily: F, fontSize: 13, fontWeight: 600, letterSpacing: 0.1,
              cursor: (canAdvance && !submitting) ? 'pointer' : 'not-allowed',
            }}
          >{submitting ? 'Creating…' : nextLabel}</button>
        </div>
      </div>
    </div>
  );
}
