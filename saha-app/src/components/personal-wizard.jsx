// Personal AI profile creation wizard. Trains a frontend-visible Mode A
// profile from a folder that contains RAW files and matching Lightroom XMPs.

import { useCallback, useEffect, useMemo, useState } from 'react';

import SONNA from '../tokens.js';
import { createPersonalProfile } from '../api/client.js';
import { useJob, isTerminal } from '../hooks/useJob.js';

const F = SONNA.font;
const M = SONNA.mono;

const Tlabel = {
  fontSize: 10, fontWeight: 600, color: SONNA.fgDim,
  textTransform: 'uppercase', letterSpacing: 0.6,
};
const Tnum = { fontFamily: M, fontVariantNumeric: 'tabular-nums' };

const STEPS = ['name', 'folder', 'settings', 'confirm'];

function basenameOf(p) {
  if (!p) return '';
  const segs = p.replace(/\\/g, '/').split('/').filter(Boolean);
  return segs[segs.length - 1] || p;
}

function StepHeader({ stepKind }) {
  const stepIndex = STEPS.indexOf(stepKind);
  return (
    <div style={{ padding: '14px 22px 12px', borderBottom: `1px solid ${SONNA.lineSoft}` }}>
      <div style={{ ...Tlabel, marginBottom: 8 }}>
        New Personal AI profile · Step {stepIndex + 1} of {STEPS.length}
      </div>
      <div style={{ display: 'flex', gap: 4 }}>
        {STEPS.map((k, i) => (
          <div key={k} style={{
            flex: 1, height: 3, borderRadius: 2,
            background: i <= stepIndex ? SONNA.ochre : SONNA.bgLifted,
          }} />
        ))}
      </div>
    </div>
  );
}

function StepName({ value, onChange, onNext }) {
  return (
    <div style={{ padding: '20px 22px', flex: 1, minHeight: 0, overflow: 'auto' }}>
      <div style={{ fontSize: 13, color: SONNA.fg, marginBottom: 8 }}>Profile name</div>
      <input
        type="text"
        autoFocus
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && value.trim()) {
            e.preventDefault();
            onNext();
          }
        }}
        placeholder="e.g. Sonna Weddings, Brand Shoots"
        style={{
          width: '100%', height: 38, padding: '0 12px',
          background: SONNA.bgPanel,
          border: `1px solid ${SONNA.line}`, borderRadius: 3,
          color: SONNA.fg, fontFamily: F, fontSize: 13,
          outline: 'none',
        }}
      />
      <div style={{ marginTop: 10, fontSize: 11.5, color: SONNA.fgFaint, lineHeight: 1.5 }}>
        This name is written into the checkpoint sidecar and shown in the profile list.
      </div>
    </div>
  );
}

function StepFolder({ folderPath, onPick }) {
  return (
    <div style={{ padding: '20px 22px', flex: 1, minHeight: 0, overflow: 'auto' }}>
      <div style={{ fontSize: 13, color: SONNA.fg, marginBottom: 12 }}>Training folder</div>
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
          <path d="M1.5 4h3l1-1h5v6.5h-9V4z"
            stroke={SONNA.fgMute} strokeWidth="1.2"
            strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        <span>{folderPath ? 'Choose a different folder' : 'Choose RAW + XMP folder'}</span>
      </button>
      {folderPath && (
        <div style={{
          marginTop: 12, padding: '10px 12px',
          background: SONNA.bgPanel,
          border: `1px solid ${SONNA.lineSoft}`, borderRadius: 3,
          ...Tnum, fontSize: 12, color: SONNA.fg,
          wordBreak: 'break-all',
        }}>
          {basenameOf(folderPath)}
        </div>
      )}
      <div style={{ marginTop: 12, fontSize: 11.5, color: SONNA.fgFaint, lineHeight: 1.5 }}>
        Pick the exported training folder. Saha reads RAW files and their matching Lightroom .xmp sidecars.
      </div>
    </div>
  );
}

function NumberField({ label, value, min, max, step = 1, onChange }) {
  return (
    <label style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      <span style={{ ...Tlabel, fontSize: 9.5 }}>{label}</span>
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{
          height: 34, padding: '0 10px',
          background: SONNA.bgPanel,
          border: `1px solid ${SONNA.line}`, borderRadius: 3,
          color: SONNA.fg, fontFamily: M, fontSize: 12.5,
          outline: 'none',
        }}
      />
    </label>
  );
}

function StepSettings({ maxEpochs, batchSize, workers, setMaxEpochs, setBatchSize, setWorkers }) {
  return (
    <div style={{ padding: '20px 22px', flex: 1, minHeight: 0, overflow: 'auto' }}>
      <div style={{ fontSize: 13, color: SONNA.fg, marginBottom: 14 }}>Training settings</div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12 }}>
        <NumberField label="Epochs" value={maxEpochs} min={1} max={200} onChange={setMaxEpochs} />
        <NumberField label="Batch" value={batchSize} min={1} max={64} onChange={setBatchSize} />
        <NumberField label="Workers" value={workers} min={0} max={16} onChange={setWorkers} />
      </div>
      <div style={{ marginTop: 14, fontSize: 11.5, color: SONNA.fgFaint, lineHeight: 1.5 }}>
        Defaults match the production recipe: 512px input, v2 slider set, fp32 precision, and published checkpoint output.
      </div>
    </div>
  );
}

function StepConfirm({ profileName, folderPath, maxEpochs, batchSize, workers, submitError }) {
  return (
    <div style={{ padding: '20px 22px', flex: 1, minHeight: 0, overflow: 'auto' }}>
      <div style={{ ...Tlabel, marginBottom: 12 }}>Review</div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
        <div>
          <div style={{ fontSize: 10.5, color: SONNA.fgDim, marginBottom: 2 }}>Name</div>
          <div style={{ fontSize: 13, color: SONNA.fg }}>{profileName}</div>
        </div>
        <div>
          <div style={{ fontSize: 10.5, color: SONNA.fgDim, marginBottom: 2 }}>Folder</div>
          <div style={{ ...Tnum, fontSize: 12, color: SONNA.fg, wordBreak: 'break-all' }}>
            {folderPath}
          </div>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12 }}>
          <div><div style={{ fontSize: 10.5, color: SONNA.fgDim }}>Epochs</div><div style={{ ...Tnum, color: SONNA.fg }}>{maxEpochs}</div></div>
          <div><div style={{ fontSize: 10.5, color: SONNA.fgDim }}>Batch</div><div style={{ ...Tnum, color: SONNA.fg }}>{batchSize}</div></div>
          <div><div style={{ fontSize: 10.5, color: SONNA.fgDim }}>Workers</div><div style={{ ...Tnum, color: SONNA.fg }}>{workers}</div></div>
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

function TrainingProgress({ job, onCancel, onDone }) {
  const snapshot = job.current?.snapshot;
  if (!snapshot) return null;
  const terminal = isTerminal(snapshot.state);
  const ok = snapshot.state === 'complete';
  const total = snapshot.epochs_total || 0;
  const done = snapshot.epochs_completed || 0;
  const pct = total ? Math.min(100, Math.round((done / total) * 100)) : 0;

  return (
    <div style={{ padding: '20px 22px', display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div style={{
        ...Tlabel,
        color: terminal ? (ok ? SONNA.green : SONNA.red) : SONNA.ochre,
      }}>
        {terminal ? (ok ? 'Training complete' : 'Training stopped') : 'Training profile'}
      </div>
      <div>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
          <span style={{ ...Tnum, fontSize: 38, fontWeight: 200, color: SONNA.fg, lineHeight: 1 }}>{pct}</span>
          <span style={{ fontSize: 16, color: SONNA.fgDim }}>%</span>
          <span style={{ flex: 1 }} />
          <span style={{ ...Tnum, fontSize: 12, color: SONNA.fgMute }}>
            epoch {done} of {total || '—'}
          </span>
        </div>
        <div style={{ marginTop: 10, height: 4, background: SONNA.bgLifted, borderRadius: 2, overflow: 'hidden' }}>
          <div style={{ width: `${pct}%`, height: '100%', background: SONNA.ochre }} />
        </div>
      </div>
      {snapshot.error && (
        <div style={{
          padding: '10px 12px',
          background: 'rgba(156, 84, 84, 0.18)',
          border: `1px solid ${SONNA.red}`,
          borderRadius: 3,
          fontSize: 12, color: SONNA.fg, lineHeight: 1.5,
        }}>{snapshot.error}</div>
      )}
      {snapshot.new_checkpoint_path && (
        <div style={{ fontSize: 11.5, color: SONNA.fgMute, lineHeight: 1.5 }}>
          Published checkpoint:
          <div style={{ marginTop: 5, ...Tnum, color: SONNA.fg, wordBreak: 'break-all' }}>
            {snapshot.new_checkpoint_path}
          </div>
        </div>
      )}
      {terminal ? (
        <button type="button" onClick={onDone} style={{
          height: 34, background: SONNA.ochre, color: '#1A1209',
          border: 'none', borderRadius: 3,
          fontFamily: F, fontSize: 13, fontWeight: 600,
          cursor: 'pointer',
        }}>Done</button>
      ) : (
        <button type="button" onClick={onCancel} disabled={snapshot.cancel_requested} style={{
          height: 32, background: 'transparent',
          border: `1px solid ${SONNA.line}`, borderRadius: 3,
          color: snapshot.cancel_requested ? SONNA.fgFaint : SONNA.fgMute,
          fontFamily: F, fontSize: 12,
          cursor: snapshot.cancel_requested ? 'not-allowed' : 'pointer',
        }}>
          {snapshot.cancel_requested ? 'Cancelling…' : 'Cancel'}
        </button>
      )}
    </div>
  );
}

export function PersonalProfileWizard({ onClose, onCreated }) {
  const [stepKind, setStepKind] = useState('name');
  const [profileName, setProfileName] = useState('');
  const [folderPath, setFolderPath] = useState(null);
  const [maxEpochs, setMaxEpochs] = useState(50);
  const [batchSize, setBatchSize] = useState(16);
  const [workers, setWorkers] = useState(4);
  const [submitError, setSubmitError] = useState(null);
  const job = useJob({ onError: (e) => setSubmitError(e?.message || String(e)) });

  const trainingStarted = !!job.current;
  const hasAnyState = useMemo(() => (
    profileName.trim().length > 0 || folderPath !== null || maxEpochs !== 50 || batchSize !== 16 || workers !== 4
  ), [profileName, folderPath, maxEpochs, batchSize, workers]);

  const requestClose = useCallback(() => {
    if (trainingStarted && !isTerminal(job.current.snapshot.state)) return;
    if (!trainingStarted && hasAnyState && !window.confirm('Discard your progress and close?')) return;
    onClose();
  }, [hasAnyState, job.current, onClose, trainingStarted]);

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

  const pickFolder = useCallback(async () => {
    if (!window.saha?.pickFolder) {
      setSubmitError('Folder picker unavailable (run via Electron)');
      return;
    }
    const path = await window.saha.pickFolder();
    if (path) setFolderPath(path);
  }, []);

  const canAdvance = (() => {
    if (stepKind === 'name') return profileName.trim().length > 0;
    if (stepKind === 'folder') return folderPath !== null;
    if (stepKind === 'settings') return maxEpochs >= 1 && batchSize >= 1 && workers >= 0;
    if (stepKind === 'confirm') return true;
    return false;
  })();

  const advance = useCallback(async () => {
    if (!canAdvance || trainingStarted) return;
    if (stepKind === 'name') { setStepKind('folder'); return; }
    if (stepKind === 'folder') { setStepKind('settings'); return; }
    if (stepKind === 'settings') { setStepKind('confirm'); return; }

    setSubmitError(null);
    try {
      await job.start({
        profile_name: profileName.trim(),
        input_dir: folderPath,
        max_epochs: maxEpochs,
        batch_size: batchSize,
        workers,
      }, { requestFn: createPersonalProfile });
    } catch (e) {
      setSubmitError(e.message || 'Failed to start training');
    }
  }, [batchSize, canAdvance, folderPath, job, maxEpochs, profileName, stepKind, trainingStarted, workers]);

  const goBack = useCallback(() => {
    if (trainingStarted) return;
    if (stepKind === 'folder') setStepKind('name');
    if (stepKind === 'settings') setStepKind('folder');
    if (stepKind === 'confirm') setStepKind('settings');
  }, [stepKind, trainingStarted]);

  let body;
  if (trainingStarted) {
    body = (
      <TrainingProgress
        job={job}
        onCancel={() => job.cancel()}
        onDone={() => {
          job.reset();
          onCreated?.();
        }}
      />
    );
  } else if (stepKind === 'name') {
    body = <StepName value={profileName} onChange={setProfileName} onNext={advance} />;
  } else if (stepKind === 'folder') {
    body = <StepFolder folderPath={folderPath} onPick={pickFolder} />;
  } else if (stepKind === 'settings') {
    body = (
      <StepSettings
        maxEpochs={maxEpochs}
        batchSize={batchSize}
        workers={workers}
        setMaxEpochs={setMaxEpochs}
        setBatchSize={setBatchSize}
        setWorkers={setWorkers}
      />
    );
  } else {
    body = (
      <StepConfirm
        profileName={profileName.trim()}
        folderPath={folderPath}
        maxEpochs={maxEpochs}
        batchSize={batchSize}
        workers={workers}
        submitError={submitError}
      />
    );
  }

  const showBack = !trainingStarted && stepKind !== 'name';
  const nextLabel = stepKind === 'confirm' ? 'Start training' : 'Next';

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
        aria-labelledby="personal-wizard-title"
        style={{
          width: 600, maxWidth: '100%', maxHeight: '85vh',
          background: SONNA.bgPanel,
          border: `1px solid ${SONNA.line}`,
          borderRadius: 4,
          boxShadow: '0 10px 32px rgba(0, 0, 0, 0.55)',
          display: 'flex', flexDirection: 'column',
        }}
      >
        <div style={{
          padding: '14px 22px 0',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        }}>
          <div id="personal-wizard-title" style={{
            fontSize: 14, fontWeight: 600, color: SONNA.fg, letterSpacing: -0.1,
          }}>Create Personal AI profile</div>
          <button
            type="button"
            onClick={requestClose}
            disabled={trainingStarted && !isTerminal(job.current?.snapshot?.state)}
            aria-label="Cancel"
            style={{
              width: 22, height: 22, padding: 0,
              background: 'transparent', border: 'none',
              cursor: trainingStarted ? 'not-allowed' : 'pointer',
              opacity: trainingStarted ? 0.35 : 0.7,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}
          >
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
              <path d="M2.5 2.5l7 7M9.5 2.5l-7 7"
                stroke={SONNA.fgMute} strokeWidth="1.3" strokeLinecap="round" />
            </svg>
          </button>
        </div>

        {!trainingStarted && <StepHeader stepKind={stepKind} />}
        {body}

        {!trainingStarted && (
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
                style={{
                  height: 34, padding: '0 16px',
                  background: 'transparent',
                  border: `1px solid ${SONNA.line}`, borderRadius: 3,
                  color: SONNA.fg,
                  fontFamily: F, fontSize: 12.5, fontWeight: 500,
                  cursor: 'pointer',
                }}
              >Back</button>
            ) : <span />}

            <button
              type="button"
              onClick={advance}
              disabled={!canAdvance}
              style={{
                height: 34, padding: '0 18px',
                background: canAdvance ? SONNA.ochre : SONNA.bgLifted,
                border: 'none', borderRadius: 3,
                color: canAdvance ? '#1A1209' : SONNA.fgFaint,
                fontFamily: F, fontSize: 13, fontWeight: 600, letterSpacing: 0.1,
                cursor: canAdvance ? 'pointer' : 'not-allowed',
              }}
            >{nextLabel}</button>
          </div>
        )}
      </div>
    </div>
  );
}
