// Profile view — three-section screen surfaced by the Profile rail icon.
//
// Section 1 (top bar):    Create new profile — two cards (Personal AI
//                         RAW+XMP training / Lite preset+survey entry point)
// Section 2 (left rail):  Your profiles — list of trained profiles, click
//                         to activate. Each row carries a type badge derived
//                         from profile_type ("Personal AI" / "Lite").
// Section 3 (centre+right): Fine-tuning. Captures dashboard + state-driven
//                         action panel stay live; a "Folder-based fine-tuning"
//                         placeholder sits below the captures summary as a
//                         visual preview of where the next Phase 5 backend
//                         work will land.
//
// All API mutations route through the same useJob hook the editor uses; we
// just feed it different request bodies and read the kind="finetune" snapshot.

import { useCallback, useEffect, useMemo, useState } from 'react';

import SONNA from '../tokens.js';
import { AppShell } from './shell.jsx';
import { ErrorBanner } from './error-banner.jsx';
import { LiteProfileWizard } from './lite-wizard.jsx';
import { PersonalProfileWizard } from './personal-wizard.jsx';
import { useJob, isTerminal } from '../hooks/useJob.js';
import { useCaptures } from '../hooks/useCaptures.js';
import { deleteProfile, startFineTune } from '../api/client.js';

const F = SONNA.font;
const M = SONNA.mono;

const Tlabel = {
  fontSize: 10, fontWeight: 600, color: SONNA.fgDim,
  textTransform: 'uppercase', letterSpacing: 0.6,
};
const Tnum = { fontFamily: M, fontVariantNumeric: 'tabular-nums' };

const FINETUNE_MIN_CAPTURES = 50;

const formatLoss = (v) => (v == null ? '—' : v.toFixed(4));

// Profile classification — driven by api/models.py:Profile.profile_type
// (landed in P1). `null` / missing covers Mode A trained ckpts (legacy v1.x
// production sidecars predate the field); "mode_b_initial" covers the
// preset-derived Lite profiles built via mode_b/checkpoint_builder.py.
// Future profile_type values flow through without changes here as long as the
// new badge wording is added.
function profileTypeLabel(p) {
  if (p.profile_type === 'mode_b_initial') return 'Lite';
  return 'Personal AI';
}
function isLiteProfile(p) {
  return p.profile_type === 'mode_b_initial';
}

// Order: active profile first, then by trained_at descending so the newest
// version of the same training lineage is just below the active one.
function sortProfiles(profiles) {
  return [...profiles].sort((a, b) => {
    if (a.is_active !== b.is_active) return a.is_active ? -1 : 1;
    const ta = a.trained_at || '1970-01-01T00:00:00Z';
    const tb = b.trained_at || '1970-01-01T00:00:00Z';
    return tb.localeCompare(ta);
  });
}

// ── Section 2 — Your profiles (left rail) ────────────────
function TypeBadge({ profile }) {
  const lite = isLiteProfile(profile);
  return (
    <span style={{
      fontSize: 9, fontWeight: 600,
      padding: '2px 5px', borderRadius: 2,
      textTransform: 'uppercase', letterSpacing: 0.5,
      background: lite ? SONNA.bgLifted : 'transparent',
      border: `1px solid ${SONNA.lineSoft}`,
      color: lite ? SONNA.fg : SONNA.fgDim,
      flexShrink: 0,
    }}>{profileTypeLabel(profile)}</span>
  );
}

function ProfileList({ profiles, onPick, onRevealDir, onDeleteProfile, activeProfileId }) {
  // Active-first, then trained_at descending. Memoised so re-sort only fires
  // when the upstream list changes identity.
  const sorted = useMemo(() => sortProfiles(profiles), [profiles]);

  return (
    <div style={{
      width: 384, flexShrink: 0,
      background: SONNA.bgPanel,
      borderRight: `1px solid ${SONNA.line}`,
      display: 'flex', flexDirection: 'column', minHeight: 0,
    }}>
      <div style={{ padding: '18px 20px 12px', borderBottom: `1px solid ${SONNA.lineSoft}` }}>
        <div style={Tlabel}>Your profiles</div>
        <div style={{ marginTop: 6, ...Tnum, fontSize: 11, color: SONNA.fgFaint }}>
          {sorted.length} {sorted.length === 1 ? 'profile' : 'profiles'}
        </div>
      </div>

      <div style={{ flex: 1, minHeight: 0, overflow: 'auto', padding: '8px 12px' }}>
        {sorted.map((p) => {
          const personalAI = !isLiteProfile(p);
          const canDelete = typeof onDeleteProfile === 'function' && p.id !== activeProfileId;
          return (
            <div key={p.id}
              onClick={() => onPick(p.id)}
              style={{
                padding: '12px 14px', marginBottom: 6,
                background: p.is_active ? SONNA.bgLifted : 'transparent',
                border: `1px solid ${p.is_active ? SONNA.line : 'transparent'}`,
                borderRadius: 4, cursor: 'pointer',
                display: 'flex', alignItems: 'flex-start', gap: 12,
              }}>
              <span style={{
                width: 8, height: 8, borderRadius: '50%',
                background: p.is_active ? SONNA.ochre : SONNA.fgFaint,
                flexShrink: 0, marginTop: 5,
              }} />
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{
                  display: 'flex', alignItems: 'center', gap: 8,
                  fontSize: 13, color: SONNA.fg, fontWeight: 500,
                }}>
                  <span style={{
                    minWidth: 0,
                    whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                  }}>
                    {p.display_name
                      ? p.display_name
                      : <>{p.name} <span style={{ color: SONNA.fgMute, fontWeight: 400 }}>{p.version}</span></>
                    }
                  </span>
                  <TypeBadge profile={p} />
                </div>
                {personalAI && (p.photo_count != null || p.val_loss != null) && (
                  <div style={{ ...Tnum, fontSize: 10.5, color: SONNA.fgFaint, marginTop: 3 }}>
                    {p.photo_count != null && `${p.photo_count.toLocaleString()} photos`}
                    {p.photo_count != null && p.val_loss != null && ' · '}
                    {p.val_loss != null && `val ${p.val_loss.toFixed(5)}`}
                  </div>
                )}
                {p.trained_at && (
                  <div style={{ ...Tnum, fontSize: 10.5, color: SONNA.fgFaint, marginTop: 2 }}>
                    {personalAI ? 'trained ' : 'created '}{p.trained_at.slice(0, 10)}
                  </div>
                )}
              </div>
              {canDelete && (
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    onDeleteProfile(p);
                  }}
                  title="Delete this generated profile"
                  style={{
                    flexShrink: 0,
                    width: 24, height: 24,
                    padding: 0,
                    border: 'none',
                    borderRadius: 3,
                    background: 'transparent',
                    color: SONNA.fgMute,
                    cursor: 'pointer',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                  }}
                >
                  <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
                    <path d="M2 2l6 6M8 2l-6 6" stroke={SONNA.fgMute} strokeWidth="1.3" strokeLinecap="round" />
                  </svg>
                </button>
              )}
            </div>
          );
        })}
        {sorted.length === 0 && (
          <div style={{ padding: 14, fontSize: 12, color: SONNA.fgFaint }}>
            No profiles yet. Use the "Create new profile" panel above.
          </div>
        )}
      </div>

      <div style={{ borderTop: `1px solid ${SONNA.line}`, padding: 14 }}>
        <button
          onClick={onRevealDir}
          disabled={!onRevealDir}
          style={{
            width: '100%', height: 28,
            background: 'transparent',
            border: `1px solid ${SONNA.line}`, borderRadius: 3,
            color: onRevealDir ? SONNA.fgMute : SONNA.fgFaint,
            fontFamily: F, fontSize: 11.5,
            cursor: onRevealDir ? 'pointer' : 'not-allowed',
            opacity: onRevealDir ? 1 : 0.6,
          }}
        >
          Profiles directory
        </button>
      </div>
    </div>
  );
}


// ── CENTRE column — captures summary ─────────────────────
function CapturesSummary({ data, loading }) {
  if (loading && !data) {
    return (
      <div style={{ flex: 1, padding: 28, color: SONNA.fgDim, fontSize: 13 }}>
        Loading captures…
      </div>
    );
  }
  const captures_count = data?.captures_count || 0;
  const since = data?.since;
  const most = data?.most_adjusted_fields || [];
  const correlations = data?.correlations || [];

  if (captures_count === 0) {
    return (
      <div style={{
        flex: 1, padding: 32,
        display: 'flex', flexDirection: 'column',
        background: SONNA.bgDeep, color: SONNA.fgDim,
      }}>
        <div style={Tlabel}>Captured edits</div>
        <div style={{
          marginTop: 24,
          fontSize: 14, color: SONNA.fgMute, lineHeight: 1.6, maxWidth: 420,
        }}>
          No corrections captured yet. Edit photos in Lightroom after Saha
          processes them, then save metadata; this view will populate.
        </div>
      </div>
    );
  }

  const maxAbs = Math.max(...most.map(([_, v]) => Math.abs(v)), 1e-6);

  return (
    <div style={{
      flex: 1, background: SONNA.bgDeep,
      display: 'flex', flexDirection: 'column', minWidth: 0, overflow: 'auto',
    }}>
      <div style={{ padding: '24px 28px 18px', borderBottom: `1px solid ${SONNA.lineSoft}` }}>
        <div style={Tlabel}>Captured edits</div>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 12, marginTop: 10 }}>
          <span style={{
            ...Tnum, fontSize: 42, fontWeight: 200,
            color: SONNA.fg, letterSpacing: -1, lineHeight: 1,
          }}>{captures_count}</span>
          {since && (
            <span style={{ ...Tnum, fontSize: 11.5, color: SONNA.fgFaint }}>
              since {since.slice(0, 10)}
            </span>
          )}
        </div>
      </div>

      {most.length > 0 && (
        <div style={{ padding: '20px 28px', borderBottom: `1px solid ${SONNA.lineSoft}` }}>
          <div style={{ ...Tlabel, marginBottom: 14 }}>Most adjusted fields</div>
          {most.slice(0, 8).map(([field, abs]) => {
            const pct = (Math.abs(abs) / maxAbs) * 100;
            return (
              <div key={field} style={{
                display: 'grid', gridTemplateColumns: '1fr 60px',
                alignItems: 'center', gap: 12,
                padding: '6px 0',
              }}>
                <div>
                  <div style={{ fontSize: 12, color: SONNA.fg }}>{field}</div>
                  <div style={{
                    marginTop: 4, height: 3, background: SONNA.bgLifted, borderRadius: 1.5,
                    position: 'relative', overflow: 'hidden',
                  }}>
                    <div style={{
                      position: 'absolute', left: 0, top: 0, bottom: 0,
                      width: `${pct}%`, background: SONNA.ochre,
                    }} />
                  </div>
                </div>
                <div style={{
                  ...Tnum, fontSize: 11.5, color: SONNA.fgMute, textAlign: 'right',
                }}>{abs.toFixed(2)}</div>
              </div>
            );
          })}
        </div>
      )}

      {correlations.length > 0 && (
        <div style={{ padding: '20px 28px' }}>
          <div style={{ ...Tlabel, marginBottom: 14 }}>Metadata correlations</div>
          {correlations.slice(0, 6).map((c, i) => (
            <div key={i} style={{
              padding: '6px 0', fontSize: 12, color: SONNA.fgMute, lineHeight: 1.5,
            }}>
              <span style={{ color: SONNA.fg }}>{c.field}</span>
              {' tracks with '}
              <span style={{ color: SONNA.fg }}>{c.metadata_col}</span>
              {' '}
              <span style={{ ...Tnum, color: SONNA.fgFaint }}>
                (r={c.spearman_r.toFixed(2)}, n={c.n})
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}


// ── RIGHT column — fine-tune action panel ────────────────
function RightActionPanel({
  capturesCount, activeProfile, onError,
  job, onActivateNew, capturesDir,
}) {
  const [confirming, setConfirming] = useState(false);
  const [weightRecent, setWeightRecent] = useState(2.0);

  const isRunning = !!(job.current && !isTerminal(job.current.snapshot.state));
  const isComplete = !!(job.current && isTerminal(job.current.snapshot.state));

  const startFinetune = useCallback(async () => {
    if (!activeProfile) return;
    if (!capturesDir) {
      onError({ source: 'finetune', message: 'Captures directory is not available yet.' });
      return;
    }
    onError(null);
    try {
      await job.start(
        {
          base_profile_id: activeProfile.id,
          captures_dir: capturesDir,
          weight_recent: weightRecent,
        },
        { requestFn: startFineTune },
      );
      setConfirming(false);
    } catch (e) {
      onError({ source: 'finetune', message: e.message });
    }
  }, [activeProfile, capturesDir, job, onError, weightRecent]);

  return (
    <div style={{
      width: 384, flexShrink: 0,
      background: SONNA.bgPanel,
      borderLeft: `1px solid ${SONNA.line}`,
      display: 'flex', flexDirection: 'column', minHeight: 0,
    }}>
      <div style={{ padding: '18px 20px', borderBottom: `1px solid ${SONNA.lineSoft}` }}>
        <div style={Tlabel}>Fine-tune</div>
      </div>

      {isRunning && job.current && (
        <FinetuneRunning snapshot={job.current.snapshot}
                         onCancel={() => job.cancel()} />
      )}

      {isComplete && job.current && (
        <FinetuneComplete snapshot={job.current.snapshot}
                          onActivate={onActivateNew}
                          onReset={() => job.reset()} />
      )}

      {!isRunning && !isComplete && capturesCount < FINETUNE_MIN_CAPTURES && (
        <div style={{ flex: 1, padding: '24px 20px',
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      textAlign: 'center' }}>
          <div style={{ maxWidth: 240 }}>
            <div style={{ fontSize: 13, color: SONNA.fgMute, lineHeight: 1.55 }}>
              Need at least {FINETUNE_MIN_CAPTURES} captured edits before a
              fine-tune is meaningful.
            </div>
            <div style={{ marginTop: 12, ...Tnum, fontSize: 11, color: SONNA.fgFaint }}>
              currently {capturesCount}
            </div>
          </div>
        </div>
      )}

      {!isRunning && !isComplete && capturesCount >= FINETUNE_MIN_CAPTURES && !confirming && (
        <div style={{ padding: '24px 20px' }}>
          <div style={{ fontSize: 13, color: SONNA.fgMute, lineHeight: 1.6, marginBottom: 18 }}>
            {capturesCount.toLocaleString()} captured edits ready.
            Fine-tuning produces a new profile version — the current one stays available.
          </div>
          <button onClick={() => setConfirming(true)} style={{
            width: '100%', height: 38,
            background: SONNA.ochre, color: '#1A1209',
            border: 'none', borderRadius: 3,
            fontFamily: F, fontSize: 13, fontWeight: 600, letterSpacing: 0.2,
            cursor: 'pointer',
          }}>
            Fine-tune profile
          </button>
        </div>
      )}

      {!isRunning && !isComplete && confirming && (
        <FinetuneConfirm
          activeProfile={activeProfile}
          capturesCount={capturesCount}
          weightRecent={weightRecent}
          setWeightRecent={setWeightRecent}
          onCancel={() => setConfirming(false)}
          onConfirm={startFinetune}
        />
      )}
    </div>
  );
}

function FinetuneConfirm({ activeProfile, capturesCount, weightRecent, setWeightRecent, onCancel, onConfirm }) {
  return (
    <div style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: 18 }}>
      <div style={{ fontSize: 13, color: SONNA.fg, lineHeight: 1.55 }}>
        Fine-tune <span style={{ color: SONNA.ochre }}>
          {activeProfile ? `${activeProfile.name} ${activeProfile.version}` : '—'}
        </span>{' '}
        with {capturesCount.toLocaleString()} captured edits?
      </div>

      <div>
        <div style={{ ...Tlabel, marginBottom: 8 }}>Recent edit weight</div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <input
            type="range"
            min="1" max="5" step="0.5"
            value={weightRecent}
            onChange={(e) => setWeightRecent(Number(e.target.value))}
            style={{ flex: 1, accentColor: SONNA.ochre }}
          />
          <span style={{ ...Tnum, fontSize: 13, color: SONNA.fg, minWidth: 24, textAlign: 'right' }}>
            {weightRecent.toFixed(1)}×
          </span>
        </div>
        <div style={{ marginTop: 6, fontSize: 11, color: SONNA.fgFaint, lineHeight: 1.5 }}>
          How much to favour recent edits over the original training data.
        </div>
      </div>

      <div style={{ display: 'flex', gap: 10 }}>
        <button onClick={onCancel} style={{
          flex: 1, height: 36,
          background: 'transparent',
          border: `1px solid ${SONNA.line}`, borderRadius: 3,
          color: SONNA.fgMute, fontFamily: F, fontSize: 12,
          cursor: 'pointer',
        }}>Cancel</button>
        <button onClick={onConfirm} style={{
          flex: 1, height: 36,
          background: SONNA.ochre, color: '#1A1209',
          border: 'none', borderRadius: 3,
          fontFamily: F, fontSize: 12, fontWeight: 600,
          cursor: 'pointer',
        }}>Confirm</button>
      </div>
    </div>
  );
}

function FinetuneRunning({ snapshot, onCancel }) {
  const total = snapshot.epochs_total || 0;
  const done = snapshot.epochs_completed || 0;
  const pct = total ? Math.round((done / total) * 100) : 0;
  return (
    <div style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
          <span style={{ ...Tnum, fontSize: 38, fontWeight: 200, color: SONNA.fg, lineHeight: 1 }}>{pct}</span>
          <span style={{ fontSize: 16, color: SONNA.fgDim }}>%</span>
          <span style={{ flex: 1 }} />
          <span style={{ ...Tnum, fontSize: 12, color: SONNA.fgMute }}>
            epoch {done} of {total}
          </span>
        </div>
        <div style={{
          marginTop: 10, height: 4,
          background: SONNA.bgLifted, borderRadius: 2, overflow: 'hidden',
        }}>
          <div style={{ width: `${pct}%`, height: '100%', background: SONNA.ochre }} />
        </div>
      </div>

      <div style={{ display: 'flex', gap: 18 }}>
        <div>
          <div style={{ ...Tlabel, fontSize: 9.5 }}>Train loss</div>
          <div style={{ marginTop: 2, ...Tnum, fontSize: 14, color: SONNA.fg }}>
            {formatLoss(snapshot.train_loss)}
          </div>
        </div>
        <div style={{ width: 1, background: SONNA.lineSoft }} />
        <div>
          <div style={{ ...Tlabel, fontSize: 9.5 }}>Val loss</div>
          <div style={{ marginTop: 2, ...Tnum, fontSize: 14, color: SONNA.fg }}>
            {formatLoss(snapshot.val_loss)}
          </div>
        </div>
      </div>

      <button onClick={onCancel} disabled={snapshot.cancel_requested} style={{
        height: 32,
        background: 'transparent',
        border: `1px solid ${SONNA.line}`, borderRadius: 3,
        color: snapshot.cancel_requested ? SONNA.fgFaint : SONNA.fgMute,
        fontFamily: F, fontSize: 12,
        cursor: snapshot.cancel_requested ? 'not-allowed' : 'pointer',
      }}>
        {snapshot.cancel_requested ? 'Cancelling…' : 'Cancel'}
      </button>

      <div style={{ ...Tnum, fontSize: 10.5, color: SONNA.fgFaint, lineHeight: 1.5 }}>
        Cancel takes effect at the next epoch boundary, which can be several
        minutes for large training sets.
      </div>
    </div>
  );
}

function FinetuneComplete({ snapshot, onActivate, onReset }) {
  const isOk = snapshot.state === 'complete';
  const isCancelled = snapshot.state === 'cancelled';
  const headline = isOk ? 'Fine-tune complete' : isCancelled ? 'Cancelled' : 'Failed';

  return (
    <div style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div style={{
        ...Tlabel,
        color: !isOk && !isCancelled ? SONNA.red : isCancelled ? SONNA.amber : SONNA.green,
      }}>{headline}</div>

      {isOk && snapshot.new_checkpoint_path && (
        <>
          <div style={{ fontSize: 12, color: SONNA.fgMute, lineHeight: 1.55 }}>
            New checkpoint:
            <div style={{
              marginTop: 6, ...Tnum, fontSize: 11, color: SONNA.fg,
              wordBreak: 'break-all',
            }}>{snapshot.new_checkpoint_path}</div>
          </div>
          {snapshot.val_loss != null && (
            <div style={{ display: 'flex', gap: 18 }}>
              <div>
                <div style={{ ...Tlabel, fontSize: 9.5 }}>Final val loss</div>
                <div style={{ marginTop: 2, ...Tnum, fontSize: 14, color: SONNA.fg }}>
                  {formatLoss(snapshot.val_loss)}
                </div>
              </div>
            </div>
          )}
          <button onClick={onActivate} style={{
            height: 36,
            background: SONNA.ochre, color: '#1A1209',
            border: 'none', borderRadius: 3,
            fontFamily: F, fontSize: 12, fontWeight: 600,
            cursor: 'pointer',
          }}>
            Activate new profile
          </button>
        </>
      )}

      {snapshot.error && (
        <div style={{
          padding: '10px 12px',
          background: 'rgba(156, 84, 84, 0.18)',
          border: `1px solid ${SONNA.red}`,
          borderRadius: 3,
          fontSize: 12, color: SONNA.fg, lineHeight: 1.5,
        }}>{snapshot.error}</div>
      )}

      <button onClick={onReset} style={{
        height: 32,
        background: 'transparent',
        border: `1px solid ${SONNA.line}`, borderRadius: 3,
        color: SONNA.fgMute, fontFamily: F, fontSize: 12,
        cursor: 'pointer',
      }}>
        Done
      </button>
    </div>
  );
}


// ── ProfileView — top-level component ────────────────────
// ── Section 1 — Create new profile (top bar) ─────────────
function CreateProfileBar({ onCreatePersonal, onCreateLite }) {
  return (
    <div style={{
      padding: '16px 24px',
      borderBottom: `1px solid ${SONNA.lineSoft}`,
      background: SONNA.bgDeep,
      flexShrink: 0,
    }}>
      <div style={{ ...Tlabel, marginBottom: 12 }}>Create new profile</div>
      <div style={{ display: 'flex', gap: 12 }}>
        <button
          type="button"
          onClick={onCreatePersonal}
          style={{
          flex: 1, padding: '14px 16px',
          background: SONNA.bgPanel,
          border: `1px solid ${SONNA.line}`,
          borderRadius: 4,
          cursor: 'pointer',
          textAlign: 'left',
          fontFamily: F,
          color: SONNA.fg,
          minWidth: 0,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ fontSize: 13, fontWeight: 600, color: SONNA.fg }}>
              Personal AI profile
            </span>
          </div>
          <div style={{ marginTop: 5, fontSize: 11.5, color: SONNA.fgMute, lineHeight: 1.5 }}>
            Train from RAW files and matching Lightroom XMP sidecars.
          </div>
        </button>

        <button
          type="button"
          onClick={onCreateLite}
          style={{
            flex: 1, padding: '14px 16px',
            background: SONNA.bgPanel,
            border: `1px solid ${SONNA.line}`,
            borderRadius: 4,
            cursor: 'pointer',
            textAlign: 'left',
            fontFamily: F,
            color: SONNA.fg,
            minWidth: 0,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ fontSize: 13, fontWeight: 600, color: SONNA.fg }}>
              Lite profile
            </span>
          </div>
          <div style={{ marginTop: 5, fontSize: 11.5, color: SONNA.fgMute, lineHeight: 1.5 }}>
            Preset + 6-question style calibration. Ready in minutes.
          </div>
        </button>
      </div>
    </div>
  );
}


// ── Section 3 placeholder — Folder-based fine-tuning ─────
// Visual preview of the future Phase 5 backend surface. The captures
// dashboard above this remains the live fine-tuning path; this section is
// purely an affordance preview.
function FolderFinetunePlaceholder() {
  return (
    <div style={{
      borderTop: `1px solid ${SONNA.lineSoft}`,
      padding: '18px 28px 22px',
      background: SONNA.bgDeep,
      flexShrink: 0,
      opacity: 0.6,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <div style={Tlabel}>Folder-based fine-tuning</div>
        <span style={{
          fontSize: 9, fontWeight: 600, color: SONNA.fgDim,
          textTransform: 'uppercase', letterSpacing: 0.5,
          padding: '2px 5px', border: `1px solid ${SONNA.lineSoft}`,
          borderRadius: 2,
        }}>Coming soon</span>
      </div>
      <div style={{
        marginTop: 8, fontSize: 12, color: SONNA.fgMute, lineHeight: 1.55,
        maxWidth: 520,
      }}>
        Fine-tuning will be available in a future update. You'll be able to
        mark folders / albums for the model to learn from your corrections.
      </div>

      {/* Skeleton rows — show the future shape without inventing folder names. */}
      <div style={{ marginTop: 14, display: 'flex', flexDirection: 'column', gap: 6 }}>
        {[0, 1].map((i) => (
          <div key={i} style={{
            padding: '9px 12px',
            background: SONNA.bgPanel,
            border: `1px solid ${SONNA.lineSoft}`,
            borderRadius: 3,
            display: 'flex', alignItems: 'center', gap: 12,
          }}>
            <div style={{
              flex: 1, height: 10,
              background: SONNA.bgLifted, borderRadius: 2, maxWidth: 220,
            }} />
            <div style={{
              width: 60, height: 9,
              background: SONNA.bgLifted, borderRadius: 2,
            }} />
            <button disabled style={{
              height: 22, padding: '0 10px',
              background: 'transparent',
              border: `1px solid ${SONNA.lineSoft}`, borderRadius: 3,
              color: SONNA.fgFaint, fontFamily: F, fontSize: 10.5,
              cursor: 'not-allowed',
            }}>Include</button>
          </div>
        ))}
        {/* Progress indicator skeleton */}
        <div style={{
          marginTop: 4,
          height: 3, background: SONNA.bgPanel,
          borderRadius: 2, overflow: 'hidden',
        }}>
          <div style={{ width: '0%', height: '100%', background: SONNA.fgFaint }} />
        </div>
      </div>
    </div>
  );
}


export function ProfileView({ profiles, activeProfile, onActivate, onProfilesChanged, onNavigate }) {
  const captures = useCaptures();
  const [error, setError] = useState(null);
  const [liteWizardOpen, setLiteWizardOpen] = useState(false);
  const [personalWizardOpen, setPersonalWizardOpen] = useState(false);
  const [appPaths, setAppPaths] = useState(null);

  useEffect(() => {
    let cancelled = false;
    async function loadPaths() {
      if (typeof window === 'undefined' || typeof window.saha?.getAppPaths !== 'function') {
        return;
      }
      try {
        const paths = await window.saha.getAppPaths();
        if (!cancelled) {
          setAppPaths(paths || null);
        }
      } catch (e) {
        if (!cancelled) {
          setAppPaths(null);
        }
      }
    }

    loadPaths();
    return () => {
      cancelled = true;
    };
  }, []);

  // useJob handles both kinds via a custom requestFn passed to start();
  // the hook's lifecycle (snapshot seeding, ws subscription, polling
  // fallback, terminal handling) is endpoint-agnostic.
  const job = useJob({ onError: setError });

  const handleActivateNew = useCallback(async () => {
    const path = job.current?.snapshot?.new_checkpoint_path;
    if (!path) return;
    // The new checkpoint lands in CHECKPOINTS_DIR. /api/profiles will
    // discover it on its next refetch. We need its profile id — derive
    // from the checkpoint filename (model-vX.Y.Z.ckpt → dp-event-vX.Y.Z).
    const m = /model-(v\d+\.\d+\.\d+)\.ckpt$/.exec(path);
    if (!m) {
      setError({ source: 'activate', message: `Could not derive profile id from ${path}` });
      return;
    }
    const newId = `dp-event-${m[1]}`;
    try {
      await onActivate(newId);
      job.reset();
      captures.refetch();
    } catch (e) {
      setError({ source: 'activate', message: e.message });
    }
  }, [job, onActivate, captures]);

  const handleCreateLite = useCallback(() => {
    setError(null);
    setLiteWizardOpen(true);
  }, []);

  const handleCreatePersonal = useCallback(() => {
    setError(null);
    setPersonalWizardOpen(true);
  }, []);

  const handleLiteWizardClose = useCallback(() => {
    setLiteWizardOpen(false);
  }, []);

  const handleLiteWizardCreated = useCallback(() => {
    setLiteWizardOpen(false);
    onProfilesChanged?.();
  }, [onProfilesChanged]);

  const handlePersonalWizardClose = useCallback(() => {
    setPersonalWizardOpen(false);
  }, []);

  const handlePersonalWizardCreated = useCallback(() => {
    setPersonalWizardOpen(false);
    onProfilesChanged?.();
  }, [onProfilesChanged]);

  const handleRevealDir = useCallback(async () => {
    if (!appPaths?.profilesDir) {
      setError({ source: 'paths', message: 'Profiles directory is not available yet.' });
      return;
    }
    try {
      const ok = await window.saha?.revealPath?.(appPaths.profilesDir);
      if (!ok) {
        setError({ source: 'paths', message: 'Could not open the profiles directory.' });
      }
    } catch (e) {
      setError({ source: 'paths', message: e.message });
    }
  }, [appPaths]);

  const handleDeleteProfile = useCallback(async (profile) => {
    if (!profile?.id) return;
    const label = profile.display_name || `${profile.name} ${profile.version}`;
    const ok = window.confirm(
      `Delete profile "${label}"? This removes its local checkpoint and sidecar files.`,
    );
    if (!ok) return;
    try {
      await deleteProfile(profile.id);
      onProfilesChanged?.();
      setError(null);
    } catch (e) {
      setError({ source: 'delete-profile', message: e.message });
    }
  }, [onProfilesChanged]);

  return (
    <AppShell title="saha — profile" activeNav="profile" onNavigate={onNavigate}>
      <div style={{
        flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0,
      }}>
        <CreateProfileBar
          onCreatePersonal={handleCreatePersonal}
          onCreateLite={handleCreateLite}
        />
        <div style={{ flex: 1, display: 'flex', minHeight: 0 }}>
          <ProfileList
            profiles={profiles}
            onPick={onActivate}
            onRevealDir={handleRevealDir}
            onDeleteProfile={handleDeleteProfile}
            activeProfileId={activeProfile?.id}
          />
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
            <ErrorBanner error={error} onDismiss={() => setError(null)} />
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
              <CapturesSummary data={captures.data} loading={captures.loading} />
              <FolderFinetunePlaceholder />
            </div>
          </div>
          <RightActionPanel
            capturesCount={captures.data?.captures_count || 0}
            activeProfile={activeProfile}
            onError={setError}
            job={job}
            onActivateNew={handleActivateNew}
            capturesDir={appPaths?.capturesDir}
          />
        </div>
      </div>
      {personalWizardOpen && (
        <PersonalProfileWizard
          onClose={handlePersonalWizardClose}
          onCreated={handlePersonalWizardCreated}
        />
      )}
      {liteWizardOpen && (
        <LiteProfileWizard
          onClose={handleLiteWizardClose}
          onCreated={handleLiteWizardCreated}
        />
      )}
    </AppShell>
  );
}
