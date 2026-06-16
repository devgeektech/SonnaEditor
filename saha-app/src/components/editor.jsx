// Editor — Saha "Process" view, fully wired to the FastAPI backend.
//
// Four-state UI keyed on (isQueueRunning, runResults, queue):
//   processing: isQueueRunning — queue dispatcher is active
//   complete:   !isQueueRunning && runResults.length > 0 — RightComplete
//               summary is shown. Process Selected button may still be
//               enabled (resume after cancel) iff queue.some(queued).
//   ready:      !isQueueRunning && runResults.length === 0 && queue.length > 0
//   empty:      no queue, no results
//
// canProcess decouples from visualState so the Process Selected button can be
// active both in 'ready' (fresh queue) and in 'complete-with-queued' (resume
// from cancel). See the dispatcher useEffect for the state-machine details.

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import SONNA from '../tokens.js';
import { AppShell } from './shell.jsx';
import { ErrorBanner } from './error-banner.jsx';
import { useRecentFolders } from '../hooks/useRecentFolders.js';
import { useJob, isTerminal } from '../hooks/useJob.js';
import { scanFolder } from '../api/client.js';

const F = SONNA.font;
const M = SONNA.mono;

const Tlabel = {
  fontSize: 10, fontWeight: 600, color: SONNA.fgDim,
  textTransform: 'uppercase', letterSpacing: 0.6,
};
const Tnum = { fontFamily: M, fontVariantNumeric: 'tabular-nums' };

const formatBytes = (n) => {
  if (n == null) return '';
  const mb = n / (1024 * 1024);
  return `${mb.toFixed(1)} MB`;
};

const formatRelative = (iso) => {
  if (!iso) return '';
  const then = new Date(iso).getTime();
  if (!isFinite(then)) return '';
  const diff = (Date.now() - then) / 1000;
  if (diff < 60) return 'just now';
  if (diff < 3600) return `${Math.round(diff / 60)} min ago`;
  if (diff < 86400) return `${Math.round(diff / 3600)} hr ago`;
  return `${Math.round(diff / 86400)} d ago`;
};

const formatDuration = (sec) => {
  if (!sec || sec < 1) return '—';
  if (sec < 60) return `${sec}s`;
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return s === 0 ? `${m}m` : `${m}m ${s}s`;
};

const computeDurationSec = (snap) => {
  if (!snap?.started_at || !snap?.ended_at) return 0;
  const start = new Date(snap.started_at).getTime();
  const end = new Date(snap.ended_at).getTime();
  if (!isFinite(start) || !isFinite(end) || end < start) return 0;
  return Math.round((end - start) / 1000);
};

const folderBasename = (p) => {
  const segs = (p || '').split(/[\\/]/).filter(Boolean);
  return segs[segs.length - 1] || p || '';
};

const dotColor = (status) =>
  status === 'flag' ? SONNA.amber : status === 'fail' ? SONNA.red : SONNA.green;


// ── LEFT column — multi-folder queue ─────────────────────
function ChevronIcon({ expanded }) {
  return (
    <svg width="10" height="10" viewBox="0 0 10 10" fill="none" style={{
      transition: 'transform 120ms',
      transform: expanded ? 'rotate(90deg)' : 'rotate(0deg)',
    }}>
      <path d="M3.5 2.5l3 2.5-3 2.5" stroke={SONNA.fgDim} strokeWidth="1.3"
        strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

// Per-folder queue status: clock / spinner / check / warning.
// The spinner uses the saha-spin keyframe declared in index.html.
function StatusIcon({ status }) {
  const size = 12;
  if (status === 'processing') {
    return (
      <div title="Processing" style={{
        width: size, height: size,
        borderRadius: '50%',
        border: `1.5px solid ${SONNA.ochre}`,
        borderTopColor: 'transparent',
        animation: 'saha-spin 0.8s linear infinite',
      }} />
    );
  }
  if (status === 'complete') {
    return (
      <svg title="Complete" width={size} height={size} viewBox="0 0 12 12" fill="none">
        <path d="M2.5 6.5L5 9l4.5-5.5" stroke={SONNA.green} strokeWidth="1.6"
          strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    );
  }
  if (status === 'failed') {
    return (
      <svg title="Failed" width={size} height={size} viewBox="0 0 12 12" fill="none">
        <circle cx="6" cy="6" r="5" stroke={SONNA.red} strokeWidth="1.2" />
        <path d="M6 3.2v3.4M6 8.4v0.4" stroke={SONNA.red} strokeWidth="1.5"
          strokeLinecap="round" />
      </svg>
    );
  }
  if (status === 'cancelled') {
    // Pause-mark inside a circle — signals "user interrupted", distinct from
    // the red error glyph above. Amber matches the cancelled-headline colour
    // in RightComplete.
    return (
      <svg title="Cancelled" width={size} height={size} viewBox="0 0 12 12" fill="none">
        <circle cx="6" cy="6" r="5" stroke={SONNA.amber} strokeWidth="1.2" />
        <path d="M4.5 4v4M7.5 4v4" stroke={SONNA.amber} strokeWidth="1.4"
          strokeLinecap="round" />
      </svg>
    );
  }
  // 'queued' (default) — clock face
  return (
    <svg title="Queued" width={size} height={size} viewBox="0 0 12 12" fill="none">
      <circle cx="6" cy="6" r="4.5" stroke={SONNA.fgFaint} strokeWidth="1.2" />
      <path d="M6 3.5V6l1.5 1.5" stroke={SONNA.fgFaint} strokeWidth="1.2"
        strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function FolderRow({
  folder,
  expanded,
  locked,
  selectable,
  onSelect,
  onToggle,
  onRemove,
}) {
  const folderName = folderBasename(folder.folderPath);

  return (
    <div style={{ padding: '4px 12px' }}>
      <div style={{
        display: 'flex', alignItems: 'center', gap: 8,
        padding: '8px 10px',
        background: SONNA.bgLifted,
        border: `1px solid ${SONNA.lineSoft}`,
        borderRadius: 3,
      }}>
        {selectable && (
          <input
            type="checkbox"
            checked={!!folder.selected}
            disabled={locked}
            onChange={(e) => onSelect(e.target.checked)}
            title="Include this folder in selected-folder processing"
            style={{
              width: 14,
              height: 14,
              accentColor: SONNA.ochre,
              cursor: locked ? 'not-allowed' : 'pointer',
              flexShrink: 0,
            }}
          />
        )}
        <button onClick={onToggle} style={{
          width: 18, height: 18, padding: 0,
          background: 'transparent', border: 'none',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          cursor: 'pointer', flexShrink: 0,
        }}>
          <ChevronIcon expanded={expanded} />
        </button>
        <div style={{
          width: 14, display: 'flex', alignItems: 'center', justifyContent: 'center',
          flexShrink: 0,
        }}>
          <StatusIcon status={folder.status} />
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{
            fontSize: 12, color: SONNA.fg, fontWeight: 500,
            whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
          }}>{folderName}</div>
          <div style={{ ...Tnum, fontSize: 10.5, color: SONNA.fgFaint, marginTop: 2 }}>
            {folder.fileCount} RAW
          </div>
        </div>
        <button
          onClick={onRemove}
          disabled={locked}
          title={locked ? 'Queue locked while processing' : 'Remove from queue'}
          style={{
            width: 22, height: 22, padding: 0,
            background: 'transparent', border: 'none',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            cursor: locked ? 'not-allowed' : 'pointer', flexShrink: 0,
            opacity: locked ? 0.3 : 0.7,
          }}
        >
          <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
            <path d="M2 2l6 6M8 2l-6 6" stroke={SONNA.fgMute} strokeWidth="1.3"
              strokeLinecap="round" />
          </svg>
        </button>
      </div>
      {expanded && (
        <div style={{ padding: '4px 0 6px 28px' }}>
          {folder.fileList.map((f, j) => (
            <div key={`${f.name}-${j}`} style={{
              display: 'flex', justifyContent: 'space-between', alignItems: 'center',
              padding: '3px 8px',
              ...Tnum, fontSize: 11, color: SONNA.fgMute,
            }}>
              <span style={{
                whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
              }}>{f.name}</span>
              <span style={{ color: SONNA.fgFaint, flexShrink: 0, marginLeft: 8 }}>
                {formatBytes(f.size_bytes)}
              </span>
            </div>
          ))}
          {folder.fileList.length === 0 && (
            <div style={{ padding: '6px 8px', fontSize: 11, color: SONNA.fgFaint }}>
              No files listed
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function LeftFolderQueue({
  queue,
  expandedSet,
  locked,
  onAddFolder,
  onRemove,
  onSelect,
  onToggleExpand,
}) {
  const totalRaws = queue.reduce((sum, f) => sum + (f.fileCount || 0), 0);

  return (
    <div style={{
      width: 384, flexShrink: 0,
      background: SONNA.bgPanel,
      borderRight: `1px solid ${SONNA.line}`,
      display: 'flex', flexDirection: 'column', minHeight: 0,
    }}>
      <div style={{ padding: '18px 20px 14px', borderBottom: `1px solid ${SONNA.lineSoft}` }}>
        <div style={{
          display: 'flex', alignItems: 'baseline', justifyContent: 'space-between',
        }}>
          <div style={Tlabel}>Queue</div>
          {locked && (
            <span style={{ fontSize: 10, color: SONNA.fgFaint, fontStyle: 'italic' }}>
              Locked while processing
            </span>
          )}
        </div>
        <button
          onClick={onAddFolder}
          disabled={locked}
          style={{
            marginTop: 12,
            width: '100%', height: 36,
            background: SONNA.bgLifted,
            border: `1px solid ${SONNA.line}`,
            borderRadius: 3,
            color: locked ? SONNA.fgFaint : SONNA.fg,
            fontFamily: F, fontSize: 12, fontWeight: 500,
            display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 10,
            cursor: locked ? 'not-allowed' : 'pointer',
            opacity: locked ? 0.5 : 1,
          }}
        >
          <svg width="11" height="11" viewBox="0 0 11 11" fill="none">
            <path d="M5.5 1.5v8M1.5 5.5h8"
              stroke={locked ? SONNA.fgFaint : SONNA.fgMute}
              strokeWidth="1.3" strokeLinecap="round" />
          </svg>
          <span>Add folder</span>
          <span style={{ ...Tnum, fontSize: 10, color: SONNA.fgFaint, marginLeft: 4 }}>⌘O</span>
        </button>
      </div>

      <div style={{ flex: 1, minHeight: 0, overflow: 'auto', padding: '8px 0' }}>
        {queue.length === 0 ? (
          <div style={{
            padding: '40px 24px', textAlign: 'center',
            fontSize: 12, color: SONNA.fgFaint, lineHeight: 1.5,
          }}>
            No folders queued.<br />Press Add folder to start.
          </div>
        ) : (
          queue.map((folder, i) => (
            <FolderRow
              key={`${folder.folderPath}-${i}`}
              folder={folder}
              expanded={expandedSet.has(i)}
              locked={locked}
              selectable={folder.status === 'queued'}
              onSelect={(selected) => onSelect(i, selected)}
              onToggle={() => onToggleExpand(i)}
              onRemove={() => onRemove(i)}
            />
          ))
        )}
      </div>

      {queue.length > 0 && (
        <div style={{
          borderTop: `1px solid ${SONNA.line}`,
          padding: '12px 20px',
          display: 'flex', alignItems: 'baseline', justifyContent: 'space-between',
        }}>
          <span style={Tlabel}>Total</span>
          <span style={{ ...Tnum, fontSize: 12, color: SONNA.fg }}>
            {totalRaws}
            <span style={{ color: SONNA.fgDim, fontSize: 11, marginLeft: 6 }}>
              RAW · {queue.length} {queue.length === 1 ? 'folder' : 'folders'}
            </span>
          </span>
        </div>
      )}
    </div>
  );
}


// ── CENTRE column ────────────────────────────────────────
function Section({ label, children, last }) {
  return (
    <div style={{
      padding: '20px 28px',
      borderBottom: last ? 'none' : `1px solid ${SONNA.lineSoft}`,
    }}>
      <div style={{ ...Tlabel, marginBottom: 12 }}>{label}</div>
      {children}
    </div>
  );
}

function ProfileSelect({ profiles, activeProfile, onPick, open, setOpen }) {
  const display = activeProfile;
  const isLite = display?.profile_type === 'mode_b_initial';
  const profileMeta = display
    ? (isLite
      ? 'Ready to Edit. Lite Profile.'
      : `Ready to Edit${display.photo_count != null ? `. ${display.photo_count.toLocaleString()} images trained.` : ''}`)
    : '';
  return (
    <div style={{ position: 'relative' }}>
      <div onClick={() => setOpen((v) => !v)} style={{
        height: 44,
        background: SONNA.bgPanel,
        border: `1px solid ${SONNA.line}`,
        borderRadius: 3,
        padding: '0 14px',
        display: 'flex', alignItems: 'center', gap: 12,
        cursor: 'pointer',
      }}>
        <span style={{
          width: 8, height: 8, borderRadius: '50%',
          background: display ? SONNA.ochre : SONNA.fgFaint, flexShrink: 0,
        }} />
        <div style={{ display: 'flex', flexDirection: 'column', flex: 1, minWidth: 0 }}>
          <span style={{ fontSize: 13, color: SONNA.fg, fontWeight: 500, letterSpacing: 0 }}>
            {display ? (display.display_name || `${display.name} ${display.version}`) : 'No profile available'}
          </span>
          {display && (
            <span style={{ ...Tnum, fontSize: 10.5, color: SONNA.fgFaint, marginTop: 2 }}>
              {profileMeta}
            </span>
          )}
        </div>
        <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
          <path d="M2.5 4l2.5 2.5L7.5 4" stroke={SONNA.fgDim} strokeWidth="1.3"
            strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </div>

      {open && profiles.length > 1 && (
        <div style={{
          position: 'absolute', top: 48, left: 0, right: 0, zIndex: 10,
          background: SONNA.bgPanel, border: `1px solid ${SONNA.line}`,
          borderRadius: 3, padding: 4,
          boxShadow: '0 4px 12px rgba(0,0,0,0.4)',
        }}>
          {profiles.map((p) => (
            <div key={p.id}
              onClick={() => { onPick(p.id); setOpen(false); }}
              style={{
                padding: '8px 10px', cursor: 'pointer',
                display: 'flex', alignItems: 'center', gap: 10, borderRadius: 2,
                background: p.is_active ? SONNA.bgLifted : 'transparent',
              }}>
              <span style={{
                width: 8, height: 8, borderRadius: '50%',
                background: p.is_active ? SONNA.ochre : SONNA.fgFaint,
              }} />
              <span style={{ fontSize: 12, color: SONNA.fg }}>{p.display_name || `${p.name} ${p.version}`}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function CentreAction({
  visualState,
  profiles, activeProfile,
  onPickProfile,
  onProcess,
  autoStraighten,
  onAutoStraightenChange,
  processRaws,
  selectedQueuedCount,
  canProcess,
  error, onDismissError,
}) {
  const [profileOpen, setProfileOpen] = useState(false);
  const isProcessing = visualState === 'processing';

  return (
    <div style={{
      flex: 1, background: SONNA.bgDeep,
      display: 'flex', flexDirection: 'column', minWidth: 0,
    }}>
      <ErrorBanner error={error} onDismiss={onDismissError} />

      <Section label="Profile">
        <ProfileSelect
          profiles={profiles} activeProfile={activeProfile}
          onPick={onPickProfile}
          open={profileOpen} setOpen={setProfileOpen}
        />
      </Section>

      <Section label="Options" last>
        <label style={{
          height: 38,
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          color: isProcessing ? SONNA.fgFaint : SONNA.fg,
          fontSize: 13,
          cursor: isProcessing ? 'not-allowed' : 'pointer',
          userSelect: 'none',
        }}>
          <input
            type="checkbox"
            checked={autoStraighten}
            disabled={isProcessing}
            onChange={(e) => onAutoStraightenChange(e.target.checked)}
            title="Apply Lightroom crop-angle straightening when confidence is high"
            style={{
              width: 15,
              height: 15,
              accentColor: SONNA.ochre,
              cursor: isProcessing ? 'not-allowed' : 'pointer',
            }}
          />
          <span>Auto straighten</span>
        </label>
      </Section>

      <div style={{ flex: 1 }} />

      <div style={{
        padding: 24,
        borderTop: `1px solid ${SONNA.line}`,
        background: SONNA.bgPanel,
      }}>
        {isProcessing ? (
          <button disabled style={{
            width: '100%', height: 42,
            background: SONNA.bgLifted, color: SONNA.fgMute,
            border: `1px solid ${SONNA.line}`, borderRadius: 3,
            fontFamily: F, fontSize: 13, fontWeight: 500,
            display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 10,
            cursor: 'not-allowed',
          }}>
            <span style={{ width: 7, height: 7, borderRadius: '50%', background: SONNA.ochre,
              boxShadow: `0 0 6px ${SONNA.ochre}` }} />
            <span>Processing in progress…</span>
          </button>
        ) : (
          <button
            onClick={onProcess}
            disabled={!canProcess}
            style={{
              width: '100%', height: 42,
              background: canProcess ? SONNA.cta : SONNA.bgLifted,
              color: canProcess ? SONNA.onCta : SONNA.fgMute,
              border: canProcess ? 'none' : `1px solid ${SONNA.line}`,
              borderRadius: 3,
              fontFamily: F, fontSize: 14, fontWeight: 600, letterSpacing: 0.2,
              display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 14,
              cursor: canProcess ? 'pointer' : 'not-allowed',
            }}>
            <span>Process Selected</span>
            <span style={{ ...Tnum, fontSize: 11, opacity: 0.7 }}>
              {processRaws > 0 ? `${processRaws.toLocaleString()} RAWs · ⌘R` : '⌘R'}
            </span>
          </button>
        )}
      </div>
    </div>
  );
}


// ── RIGHT column states ──────────────────────────────────
function RightEmpty({ lastRun }) {
  return (
    <div style={{
      width: 384, flexShrink: 0,
      background: SONNA.bgPanel,
      borderLeft: `1px solid ${SONNA.line}`,
      display: 'flex', flexDirection: 'column',
    }}>
      <div style={{ padding: '18px 20px', borderBottom: `1px solid ${SONNA.lineSoft}` }}>
        <div style={Tlabel}>Status</div>
      </div>
      <div style={{
        flex: 1,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        padding: 32,
      }}>
        <div style={{ textAlign: 'center', maxWidth: 240 }}>
          <div style={{
            width: 56, height: 56, margin: '0 auto 18px',
            borderRadius: '50%', border: `1px solid ${SONNA.line}`,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <div style={{ width: 6, height: 6, borderRadius: '50%', background: SONNA.fgFaint }} />
          </div>
          <div style={{ fontSize: 14, color: SONNA.fgMute, fontWeight: 400, lineHeight: 1.5 }}>
            Select a folder and<br />press Process
          </div>
          <div style={{ marginTop: 10, ...Tnum, fontSize: 10.5, color: SONNA.fgFaint, letterSpacing: 0.4 }}>
            ⌘R to start
          </div>
        </div>
      </div>
      {lastRun && (
        <div style={{ borderTop: `1px solid ${SONNA.lineSoft}`, padding: '14px 20px' }}>
          <div style={{ ...Tlabel, marginBottom: 8 }}>Last run</div>
          <div style={{
            display: 'flex', alignItems: 'baseline', gap: 8,
            ...Tnum, fontSize: 11.5, color: SONNA.fgMute,
          }}>
            <span style={{ color: SONNA.fg }}>{lastRun.raw_count} photos</span>
            <span style={{ color: SONNA.fgFaint }}>·</span>
            <span>{formatRelative(lastRun.last_processed_at)}</span>
          </div>
        </div>
      )}
    </div>
  );
}

// Briefly shown during the gap between dispatcher.job.reset() and the next
// folder's job.start() resolving. Without this the right column would crash
// trying to read job.current.snapshot during the transition.
function RightTransitioning({ queue }) {
  const nextFolder = queue.find((f) => f.status === 'queued');
  const folderName = folderBasename(nextFolder?.folderPath || '');
  return (
    <div style={{
      width: 384, flexShrink: 0,
      background: SONNA.bgPanel,
      borderLeft: `1px solid ${SONNA.line}`,
      display: 'flex', flexDirection: 'column',
    }}>
      <div style={{
        padding: '12px 20px',
        borderBottom: `1px solid ${SONNA.lineSoft}`,
      }}>
        <div style={Tlabel}>Processing</div>
      </div>
      <div style={{
        flex: 1,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        padding: 32,
      }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{
            width: 22, height: 22, margin: '0 auto 14px',
            borderRadius: '50%',
            border: `2px solid ${SONNA.ochre}`,
            borderTopColor: 'transparent',
            animation: 'saha-spin 0.8s linear infinite',
          }} />
          <div style={{ fontSize: 12, color: SONNA.fgDim, lineHeight: 1.5 }}>
            {folderName ? <>Starting<br /><span style={{ color: SONNA.fg }}>{folderName}</span>…</> : 'Starting next folder…'}
          </div>
        </div>
      </div>
    </div>
  );
}

function RightProcessing({ snapshot, liveLog, wsStatus, onCancel, queue }) {
  const total = snapshot.photos_total || 0;
  const processed = snapshot.photos_processed || 0;
  const prepared = snapshot.photos_prepared || 0;
  const done = Math.max(processed, prepared);
  const pct = total ? Math.round((done / total) * 100) : 0;

  // Queue position: 1-based index of the row currently running, derived from
  // queue statuses set by the dispatcher.
  const inFlightIndex = queue.findIndex((f) => f.status === 'processing');
  const position = inFlightIndex >= 0 ? inFlightIndex + 1 : 1;
  const queueTotal = queue.length || 1;
  const inFlightFolder = inFlightIndex >= 0 ? queue[inFlightIndex] : null;
  const folderName = folderBasename(inFlightFolder?.folderPath || snapshot.folder_path || '');

  return (
    <div style={{
      width: 384, flexShrink: 0,
      background: SONNA.bgPanel,
      borderLeft: `1px solid ${SONNA.line}`,
      display: 'flex', flexDirection: 'column', minHeight: 0,
    }}>
      {/* Header: state + queue position + currently-processing folder name */}
      <div style={{
        padding: '12px 20px 14px',
        borderBottom: `1px solid ${SONNA.lineSoft}`,
      }}>
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        }}>
          <div style={Tlabel}>{snapshot.cancel_requested ? 'Cancelling' : 'Processing'}</div>
          <div style={{
            ...Tnum, fontSize: 10.5, color: SONNA.ochre,
            display: 'flex', alignItems: 'center', gap: 6,
          }}>
            <span style={{
              width: 6, height: 6, borderRadius: '50%',
              background: SONNA.ochre, boxShadow: `0 0 6px ${SONNA.ochre}`,
            }} />
            {wsStatus === 'reconnecting' ? 'Reconnecting…'
              : wsStatus === 'polling' ? 'Polling'
              : 'Running'}
          </div>
        </div>
        <div style={{
          marginTop: 8, ...Tnum, fontSize: 10.5, color: SONNA.fgDim,
        }}>
          Folder {position} of {queueTotal}
        </div>
        <div style={{
          marginTop: 4, fontSize: 13, color: SONNA.fg, fontWeight: 500,
          whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
        }}>
          {folderName || '—'}
        </div>
      </div>

      {/* Per-folder progress (live updates land in P3; today the bar moves
          during the per-photo XMP-write phase only). */}
      <div style={{ padding: '12px 20px 10px' }}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
          <span style={{
            ...Tnum, fontSize: 26, fontWeight: 300,
            color: SONNA.fg, letterSpacing: 0, lineHeight: 1,
          }}>{pct}</span>
          <span style={{ fontSize: 13, color: SONNA.fgDim }}>%</span>
          <span style={{ flex: 1 }} />
          <span style={{ ...Tnum, fontSize: 11.5, color: SONNA.fgMute }}>
            {done} / {total}
          </span>
        </div>

        <div style={{
          marginTop: 8, height: 3,
          background: SONNA.bgLifted, borderRadius: 2, overflow: 'hidden',
        }}>
          <div style={{
            width: `${pct}%`, height: '100%', background: SONNA.ochre,
            transition: 'width 200ms ease-out',
          }} />
        </div>

        <div style={{
          marginTop: 10,
          display: 'flex', alignItems: 'center', gap: 8,
          ...Tnum, fontSize: 11,
        }}>
          <span style={{ color: SONNA.ochre }}>▸</span>
          <span style={{
            color: SONNA.fg, flex: 1, minWidth: 0,
            whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
          }}>{snapshot.current_photo || '—'}</span>
        </div>
      </div>

      {/* Compact rate · ETA · Cancel on one row */}
      <div style={{
        padding: '0 20px 12px',
        display: 'flex', alignItems: 'center', gap: 10,
      }}>
        <span style={{ ...Tnum, fontSize: 11, color: SONNA.fg }}>
          {(snapshot.photos_per_sec || 0).toFixed(1)}
          <span style={{ fontSize: 10, color: SONNA.fgDim, marginLeft: 2 }}>/s</span>
        </span>
        <span style={{ color: SONNA.fgFaint, fontSize: 11 }}>·</span>
        <span style={{ ...Tnum, fontSize: 11, color: SONNA.fg }}>
          {snapshot.eta_seconds || 0}s
          <span style={{ fontSize: 10, color: SONNA.fgDim, marginLeft: 3 }}>ETA</span>
        </span>
        <span style={{ flex: 1 }} />
        <button onClick={onCancel} disabled={snapshot.cancel_requested} style={{
          height: 24, padding: '0 12px',
          background: 'transparent',
          border: `1px solid ${SONNA.line}`, borderRadius: 3,
          color: snapshot.cancel_requested ? SONNA.fgFaint : SONNA.fgMute,
          fontFamily: F, fontSize: 11, fontWeight: 500,
          cursor: snapshot.cancel_requested ? 'not-allowed' : 'pointer',
          display: 'flex', alignItems: 'center', gap: 6,
        }}>
          <span>Cancel</span>
          <span style={{ ...Tnum, fontSize: 9.5, color: SONNA.fgFaint }}>⌘.</span>
        </button>
      </div>

      <div style={{
        borderTop: `1px solid ${SONNA.lineSoft}`,
        padding: '10px 20px 6px',
        display: 'flex', justifyContent: 'space-between',
      }}>
        <span style={Tlabel}>Live log</span>
        <span style={{ ...Tnum, fontSize: 10, color: SONNA.fgFaint }}>tail -f</span>
      </div>
      <div style={{ flex: 1, minHeight: 0, overflow: 'auto', padding: '4px 14px 12px' }}>
        {liveLog.slice(0, 30).map((l, i) => (
          <div key={i} style={{
            padding: '5px 6px',
            display: 'grid', gridTemplateColumns: '1fr 8px',
            alignItems: 'center', columnGap: 10,
            opacity: 1 - Math.min(i * 0.04, 0.6),
          }}>
            <div style={{ minWidth: 0 }}>
              <div style={{
                ...Tnum, fontSize: 11, color: SONNA.fg,
                whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
              }}>{l.name}</div>
              <div style={{
                ...Tnum, fontSize: 10, color: SONNA.fgFaint, marginTop: 1,
                whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
              }}>{l.edit_summary}</div>
            </div>
            <span style={{
              width: 8, height: 8, borderRadius: '50%',
              background: dotColor(l.status),
            }} />
          </div>
        ))}
        {liveLog.length === 0 && (
          <div style={{ padding: '12px 6px', fontSize: 11, color: SONNA.fgFaint }}>
            Waiting for first photo…
          </div>
        )}
      </div>
    </div>
  );
}

function RightComplete({ runResults, onProcessAnother }) {
  // End-of-run summary aggregated over every folder dispatched in this run.
  // 2b transitional state always produces exactly one entry; 2c's queue
  // dispatcher will populate N. Failed-folder list surfaces names + error
  // text so the user can decide whether to re-run individuals.
  const totalPhotos = runResults.reduce((s, r) => s + (r.photosProcessed || 0), 0);
  const totalPhotosFailed = runResults.reduce((s, r) => s + (r.photosFailed || 0), 0);
  const totalDuration = runResults.reduce((s, r) => s + (r.durationSec || 0), 0);
  const failedFolders = runResults.filter((r) => r.state !== 'complete');
  const cancelledOnly = failedFolders.length === runResults.length
    && runResults.every((r) => r.state === 'cancelled');
  const allFailed = runResults.length > 0 && failedFolders.length === runResults.length;

  const headline = cancelledOnly ? 'Cancelled'
    : allFailed ? 'Failed'
    : failedFolders.length > 0 ? 'Completed with errors'
    : 'Complete';
  const headlineColor = cancelledOnly ? SONNA.amber
    : allFailed ? SONNA.red
    : failedFolders.length > 0 ? SONNA.amber
    : SONNA.green;

  return (
    <div style={{
      width: 384, flexShrink: 0,
      background: SONNA.bgPanel,
      borderLeft: `1px solid ${SONNA.line}`,
      display: 'flex', flexDirection: 'column', minHeight: 0,
    }}>
      <div style={{
        padding: '12px 20px',
        borderBottom: `1px solid ${SONNA.lineSoft}`,
      }}>
        <div style={{ ...Tlabel, color: headlineColor }}>{headline}</div>
        <div style={{ ...Tnum, fontSize: 10.5, color: SONNA.fgDim, marginTop: 4 }}>
          {runResults.length} {runResults.length === 1 ? 'folder' : 'folders'}
        </div>
      </div>

      <div style={{ padding: '18px 20px 12px', flex: 1, minHeight: 0, overflow: 'auto' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div style={{ display: 'flex', gap: 16 }}>
            <div style={{ flex: 1 }}>
              <div style={Tlabel}>Photos</div>
              <div style={{ marginTop: 4, ...Tnum, fontSize: 22, fontWeight: 300, color: SONNA.fg }}>
                {totalPhotos.toLocaleString()}
              </div>
              {totalPhotosFailed > 0 && (
                <div style={{ marginTop: 2, ...Tnum, fontSize: 10.5, color: SONNA.red }}>
                  {totalPhotosFailed} failed
                </div>
              )}
            </div>
            <div style={{ flex: 1 }}>
              <div style={Tlabel}>Time</div>
              <div style={{ marginTop: 4, ...Tnum, fontSize: 22, fontWeight: 300, color: SONNA.fg }}>
                {formatDuration(totalDuration)}
              </div>
            </div>
          </div>

          {failedFolders.length > 0 && (
            <div>
              <div style={{ ...Tlabel, color: SONNA.red }}>
                {cancelledOnly ? 'Cancelled' : 'Failed'} · {failedFolders.length}
              </div>
              <div style={{ marginTop: 8, display: 'flex', flexDirection: 'column', gap: 6 }}>
                {failedFolders.map((r, i) => (
                  <div key={i} style={{
                    padding: '8px 10px',
                    background: 'rgba(156, 84, 84, 0.12)',
                    border: `1px solid ${SONNA.red}`,
                    borderRadius: 3,
                  }}>
                    <div style={{
                      ...Tnum, fontSize: 11.5, color: SONNA.fg,
                      whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                    }}>
                      {folderBasename(r.folderPath) || '(unknown folder)'}
                    </div>
                    {r.error && (
                      <div style={{
                        fontSize: 10.5, color: SONNA.fgDim,
                        marginTop: 3, lineHeight: 1.4,
                      }}>{r.error}</div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      <div style={{ padding: 14, borderTop: `1px solid ${SONNA.lineSoft}` }}>
        <button onClick={onProcessAnother} style={{
          width: '100%', height: 36,
          background: SONNA.bgLifted,
          border: `1px solid ${SONNA.line}`, borderRadius: 3,
          color: SONNA.fg, fontFamily: F, fontSize: 12, fontWeight: 500,
          cursor: 'pointer',
        }}>
          Process another folder
        </button>
      </div>
    </div>
  );
}


// Pre-process confirmation dialog. Fires when Process Selected is clicked
// against a queue where one or more 'queued' folders contain XMP sidecars
// that would be overwritten by the inference pipeline. Cancel is the
// keyboard-default action (Enter or Escape closes without proceeding) so
// muscle memory can't bypass the safety check.
function OverwriteConfirmDialog({ conflicts, totalCount, totalFolders, onCancel, onConfirm }) {
  return (
    <div
      onClick={onCancel}
      style={{
        position: 'fixed', inset: 0, zIndex: 100,
        background: 'rgba(0, 0, 0, 0.55)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        padding: 24,
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-labelledby="overwrite-dialog-title"
        style={{
          width: 480, maxWidth: '100%',
          background: SONNA.bgPanel,
          border: `1px solid ${SONNA.line}`,
          borderRadius: 4,
          boxShadow: '0 10px 32px rgba(0, 0, 0, 0.55)',
          display: 'flex', flexDirection: 'column',
        }}
      >
        <div style={{
          padding: '18px 22px 14px',
          borderBottom: `1px solid ${SONNA.lineSoft}`,
        }}>
          <div id="overwrite-dialog-title" style={{
            fontSize: 14, fontWeight: 600, color: SONNA.fg, letterSpacing: 0,
          }}>Existing XMP sidecars detected</div>
        </div>

        <div style={{
          padding: '16px 22px 18px',
          fontSize: 13, color: SONNA.fgMute, lineHeight: 1.55,
        }}>
          <p style={{ margin: 0 }}>
            The following folders contain XMP sidecars that will be replaced with new Saha predictions:
          </p>
          <ul style={{
            margin: '12px 0 0', paddingLeft: 22,
            ...Tnum, fontSize: 12.5, color: SONNA.fg,
          }}>
            {conflicts.map((c, i) => (
              <li key={`${c.folderName}-${i}`} style={{ marginBottom: 4 }}>
                <span>{c.folderName}</span>
                <span style={{ color: SONNA.fgDim, marginLeft: 8 }}>
                  — {c.count} existing {c.count === 1 ? 'XMP' : 'XMPs'}
                </span>
              </li>
            ))}
          </ul>
          <p style={{ margin: '14px 0 0', ...Tnum, fontSize: 12.5, color: SONNA.fg }}>
            Total: {totalCount} existing {totalCount === 1 ? 'XMP' : 'XMPs'} across {totalFolders} {totalFolders === 1 ? 'folder' : 'folders'} will be overwritten.
          </p>
          <p style={{ margin: '14px 0 0', color: SONNA.red, fontSize: 12.5 }}>
            This cannot be undone. Any manual edits in Lightroom for these photos will be lost.
          </p>
        </div>

        <div style={{
          padding: 14,
          borderTop: `1px solid ${SONNA.lineSoft}`,
          display: 'flex', justifyContent: 'flex-end', gap: 10,
        }}>
          <button
            autoFocus
            onClick={onCancel}
            style={{
              height: 34, padding: '0 18px',
              background: SONNA.bgLifted,
              border: `1px solid ${SONNA.line}`, borderRadius: 3,
              color: SONNA.fg, fontFamily: F, fontSize: 13, fontWeight: 500,
              cursor: 'pointer',
            }}
          >Cancel</button>
          <button
            onClick={onConfirm}
            style={{
              height: 34, padding: '0 18px',
              background: SONNA.red,
              border: 'none', borderRadius: 3,
              color: '#FFFFFF', fontFamily: F, fontSize: 13, fontWeight: 600,
              letterSpacing: 0.1,
              cursor: 'pointer',
            }}
          >Overwrite and continue</button>
        </div>
      </div>
    </div>
  );
}


// ── Editor — top-level component ─────────────────────────
export function Editor({
  profiles = [],
  activeProfile,
  onActivateProfile,
  onNavigate,
  theme,
  onToggleTheme,
  onLogout,
  onProjectsChange,
}) {
  // Multi-folder queue. Items shape:
  //   { folderPath, fileCount, fileList, status }
  // status ∈ "queued" | "processing" | "complete" | "failed" | "cancelled".
  // The dispatcher useEffect below drives sequential per-folder processing
  // through a backend that still serves one folder at a time.
  const [queue, setQueue] = useState([]);
  const [expandedSet, setExpandedSet] = useState(() => new Set());
  // Per-folder run results, appended once per job termination. Persists
  // across resume — a cancelled run's complete/cancelled entries stay in
  // place when the user clicks Process Selected again to dispatch the
  // remaining 'queued' folders.
  const [runResults, setRunResults] = useState([]);
  // Queue-level flags that drive the dispatcher state machine.
  const [isQueueRunning, setIsQueueRunning] = useState(false);
  const [cancelRequested, setCancelRequested] = useState(false);
  // Guards against double-recording a terminal job (state flips can fire the
  // effect twice in StrictMode dev double-invoke).
  const lastTerminalIdRef = useRef(null);
  // Guards Case C against re-dispatching during the gap between setQueue
  // marking a folder as 'processing' and useJob.start setting job.current.
  const dispatchInFlightRef = useRef(false);
  const dispatchScopeRef = useRef('selected');
  // null when no dialog; { conflicts: [{folderName, count}], totalCount,
  // totalFolders } when Process Selected has found existing XMPs that would
  // be overwritten and is waiting on the user.
  const [overwriteConfirm, setOverwriteConfirm] = useState(null);
  // Slider fields skipped from XMP write. Sourced entirely from the active
  // profile's `default_skip_fields` sidecar — the per-job UI overrides were
  // removed in the P0 cleanup (see UI rebuild plan). The model still predicts
  // these; the XMP writer omits them so Lightroom falls back to its defaults.
  const [skipFields, setSkipFields] = useState(() => new Set());
  useEffect(() => {
    setSkipFields(new Set(activeProfile?.default_skip_fields || []));
  }, [activeProfile?.id]);
  const [autoStraighten, setAutoStraighten] = useState(false);
  const [error, setError] = useState(null);

  // useRecentFolders is retained for RightEmpty's "Last run" tile only — the
  // left-panel Recent folders section was removed in 2a. Full hook removal is
  // queued for P7 when the last-run tile is reworked.
  const recentQ = useRecentFolders();
  const job = useJob({ onError: setError });

  // Derived values.
  const totalRaws = useMemo(
    () => queue.reduce((s, f) => s + (f.fileCount || 0), 0),
    [queue],
  );
  const selectedQueuedCount = useMemo(
    () => queue.filter((f) => f.status === 'queued' && f.selected).length,
    [queue],
  );
  const processRaws = useMemo(() => (
    queue.reduce((sum, f) => {
      if (f.status !== 'queued' || !f.selected) return sum;
      return sum + (f.fileCount || 0);
    }, 0)
  ), [queue]);
  const hasQueuedFolders = useMemo(
    () => queue.some((f) => f.status === 'queued'),
    [queue],
  );
  const hasProcessableFolders = selectedQueuedCount > 0;

  const visualState = useMemo(() => {
    if (isQueueRunning) return 'processing';
    if (runResults.length > 0) return 'complete';
    if (queue.length > 0) return 'ready';
    return 'empty';
  }, [isQueueRunning, runResults.length, queue.length]);

  // Decoupled from visualState — Process Selected is reachable both fresh
  // (ready) and as a resume affordance after cancel (complete-with-queued).
  // The disable check is the boundary between "nothing to do" and "resume".
  const canProcess = !isQueueRunning
    && hasProcessableFolders
    && !!activeProfile
    && processRaws > 0;

  // The in-flight folder during processing drives the app shell title bar.
  const inFlightFolderPath = useMemo(() => {
    if (!isQueueRunning) return '';
    return queue.find((f) => f.status === 'processing')?.folderPath || '';
  }, [isQueueRunning, queue]);

  // Full reset to ready/empty state. Used by "Process another folder" and
  // by Add-folder after a fully-completed run.
  const clearRunState = useCallback(() => {
    job.reset();
    setRunResults([]);
    setQueue((q) => q.map((f) => ({ ...f, status: 'queued', selected: false })));
    setIsQueueRunning(false);
    setCancelRequested(false);
    lastTerminalIdRef.current = null;
    dispatchInFlightRef.current = false;
    dispatchScopeRef.current = 'selected';
    recentQ.refetch();
  }, [job, recentQ]);

  const handleAddFolder = useCallback(async () => {
    if (!window.saha?.pickFolder) {
      setError({ source: 'pick', message: 'Folder picker unavailable (run via Electron)' });
      return;
    }
    const path = await window.saha.pickFolder();
    if (!path) return;
    setError(null);
    // Auto-clear only when the queue has no resumable work — i.e. the user
    // is past a fully-completed run. Mid-resume adds (some folders still
    // 'queued' after a cancel) preserve the existing complete/cancelled
    // history so the user doesn't lose their place.
    if (runResults.length > 0 && !hasQueuedFolders) {
      clearRunState();
    }
    try {
      const result = await scanFolder(path);
      if (!result.is_valid) {
        setError({ source: 'scan', message: result.error || 'Could not scan folder' });
        return;
      }
      setQueue((q) => [...q, {
        folderPath: result.folder_path,
        fileCount: result.raw_count,
        fileList: result.files || [],
        // Count of existing .xmp sidecars that match RAW basenames in this
        // folder — the inference pipeline would overwrite each one. Drives
        // the Process-Folders confirmation dialog. Cached at scan time;
        // staleness if the user edits in Lightroom between Add and Process
        // is accepted (worst case: one silent overwrite of a freshly-edited
        // photo). See API: FolderScanResponse.xmp_conflict_count.
        xmpConflictCount: result.xmp_conflict_count || 0,
        status: 'queued',
        selected: false,
        loadedAt: Date.now(),
      }]);
    } catch (e) {
      setError({ source: 'scan', message: e.message });
    }
  }, [runResults.length, hasQueuedFolders, clearRunState, queue.length]);

  useEffect(() => {
    if (typeof onProjectsChange !== 'function') return;
    onProjectsChange({
      queue: queue.map((f) => ({
        folderPath: f.folderPath,
        fileCount: f.fileCount,
        status: f.status,
        selected: !!f.selected,
        loadedAt: f.loadedAt || 0,
      })),
      runResults,
      currentJob: job.current?.snapshot || null,
    });
  }, [onProjectsChange, queue, runResults, job.current?.snapshot]);

  const handleRemove = useCallback((index) => {
    const folder = queue[index];
    const folderName = folderBasename(folder?.folderPath || '');
    const ok = window.confirm(
      `Remove this folder from the queue - ${folderName || 'selected folder'}?`,
    );
    if (!ok) return;
    setQueue((q) => q.filter((_, i) => i !== index));
    setExpandedSet((s) => {
      // Indices shift left after removal; rebuild so expansion state tracks
      // the same folder rows post-splice.
      const next = new Set();
      s.forEach((i) => {
        if (i < index) next.add(i);
        else if (i > index) next.add(i - 1);
      });
      return next;
    });
  }, [queue]);

  const handleToggleExpand = useCallback((index) => {
    setExpandedSet((s) => {
      const next = new Set(s);
      if (next.has(index)) next.delete(index);
      else next.add(index);
      return next;
    });
  }, []);

  const handleSelectFolder = useCallback((index, selected) => {
    setQueue((q) => q.map((f, i) => (
      i === index ? { ...f, selected } : f
    )));
  }, []);

  const beginDispatch = useCallback(() => {
    setError(null);
    setCancelRequested(false);
    dispatchScopeRef.current = 'selected';
    setIsQueueRunning(true);
  }, []);

  // Two-step gate: if any 'queued' folder has cached XMP conflicts, raise
  // the overwrite dialog and wait. Re-evaluated fresh on every click —
  // there's no "user already cancelled" memory; re-confirmation is cheap,
  // mis-remembering intent is expensive.
  const handleProcess = useCallback(() => {
    if (!canProcess) return;
    const shouldProcess = (f) => f.status === 'queued' && f.selected;
    const conflicts = queue
      .filter((f) => shouldProcess(f) && (f.xmpConflictCount || 0) > 0)
      .map((f) => ({
        folderName: folderBasename(f.folderPath),
        count: f.xmpConflictCount,
      }));
    if (conflicts.length > 0) {
      setOverwriteConfirm({
        conflicts,
        totalCount: conflicts.reduce((s, c) => s + c.count, 0),
        totalFolders: conflicts.length,
      });
      return;
    }
    beginDispatch();
  }, [canProcess, queue, beginDispatch]);

  const handleConfirmOverwrite = useCallback(() => {
    setOverwriteConfirm(null);
    beginDispatch();
  }, [beginDispatch]);

  const handleCancelOverwrite = useCallback(() => {
    setOverwriteConfirm(null);
  }, []);

  // Unified queue cancel: aborts the in-flight folder via job.cancel() and
  // raises the queue-level flag so the dispatcher stops advancing to the
  // next folder. Remaining 'queued' folders stay 'queued' so the user can
  // resume by clicking Process Selected again (accidental-cancel recovery).
  const handleCancel = useCallback(() => {
    if (!isQueueRunning) return;
    const snap = job.current?.snapshot;
    const inFlightIndex = queue.findIndex((f) => f.status === 'processing');
    if (inFlightIndex >= 0) {
      setQueue((q) => q.map((f, i) => (
        i === inFlightIndex ? { ...f, status: 'cancelled', selected: false } : f
      )));
      setRunResults((rs) => [...rs, {
        folderPath: snap?.folder_path || queue[inFlightIndex]?.folderPath || '',
        photosProcessed: snap?.photos_processed || 0,
        photosFailed: snap?.photos_failed || 0,
        durationSec: snap ? computeDurationSec(snap) : 0,
        state: 'cancelled',
        error: null,
      }]);
    }
    setCancelRequested(true);
    job.cancel();
    job.reset();
    setIsQueueRunning(false);
    setCancelRequested(false);
    dispatchInFlightRef.current = false;
    dispatchScopeRef.current = 'selected';
  }, [isQueueRunning, job, queue]);

  const handleProcessAnother = useCallback(() => {
    clearRunState();
  }, [clearRunState]);

  // Surface job-failed snapshot errors as a banner.
  useEffect(() => {
    const snap = job.current?.snapshot;
    if (snap?.state === 'failed' && snap.error) {
      setError({ source: 'job', message: snap.error });
    }
  }, [job.current?.snapshot?.state, job.current?.snapshot?.error]);

  // Queue dispatcher — single useEffect state machine.
  //
  // Case A: job in flight (non-terminal) → clear dispatch guard, wait.
  // Case B: job terminal → record once, then call job.reset() so Case C
  //         can advance. If cancelRequested, finalise the run instead
  //         (remaining 'queued' folders stay 'queued' per the
  //         accidental-cancel-recovery semantic).
  // Case C: no job in flight → dispatch the next 'queued' folder, or
  //         finalise the run if none remain.
  //
  // `job` is intentionally omitted from the dep array: job.start, job.reset,
  // and job.cancel are stable useCallback refs inside useJob, while job.current
  // is read via primitive deps (id, snapshot.state).
  useEffect(() => {
    if (!isQueueRunning) return;

    // Case A: job running.
    if (job.current && !isTerminal(job.current.snapshot.state)) {
      dispatchInFlightRef.current = false;
      return;
    }

    // Case B: job terminated.
    if (job.current && isTerminal(job.current.snapshot.state)) {
      if (lastTerminalIdRef.current === job.current.id) {
        // Already recorded — clear job.current so Case C can advance.
        job.reset();
        return;
      }
      lastTerminalIdRef.current = job.current.id;

      const snap = job.current.snapshot;
      const rowStatus = snap.state === 'complete' ? 'complete'
        : snap.state === 'cancelled' ? 'cancelled'
        : 'failed';
      const inFlightIndex = queue.findIndex((f) => f.status === 'processing');

      setQueue((q) => q.map((f, i) =>
        i === inFlightIndex ? { ...f, status: rowStatus } : f
      ));
      setRunResults((rs) => [...rs, {
        folderPath: snap.folder_path || queue[inFlightIndex]?.folderPath || '',
        photosProcessed: snap.photos_processed || 0,
        photosFailed: snap.photos_failed || 0,
        durationSec: computeDurationSec(snap),
        state: snap.state,
        error: snap.error || null,
      }]);

      if (cancelRequested) {
        // Finalise. Remaining 'queued' rows stay queued — Process Selected
        // becomes the resume affordance once visualState transitions to
        // 'complete' (because runResults is non-empty).
        setIsQueueRunning(false);
        setCancelRequested(false);
      } else if (dispatchScopeRef.current === 'single') {
        setIsQueueRunning(false);
      }
      // Always reset job.current — either to let Case C dispatch the next,
      // or to clear the terminal snapshot before the run ends.
      job.reset();
      return;
    }

    // Case C: no job in flight.
    if (dispatchInFlightRef.current) return;

    // Cancel raised in the between-folders gap (job.current was already null
    // when handleCancel fired). Finalise without dispatching.
    if (cancelRequested) {
      setIsQueueRunning(false);
      setCancelRequested(false);
      return;
    }

    const nextIndex = queue.findIndex((f) => (
      f.status === 'queued'
      && f.selected
    ));
    if (nextIndex < 0) {
      // Selected batch complete, or no queued work remains.
      setIsQueueRunning(false);
      return;
    }

    if (!activeProfile) {
      setError({ source: 'start', message: 'No active profile selected' });
      setIsQueueRunning(false);
      return;
    }

    const targetFolder = queue[nextIndex];
    dispatchInFlightRef.current = true;
    setQueue((q) => q.map((f, i) =>
      i === nextIndex ? { ...f, status: 'processing' } : f
    ));
    // Backend Pydantic defaults apply for omitted fields:
    //   write_xmp_in_place=true, confidence_threshold=0.65, preserve_wb=false.
    // flag_low_confidence is explicitly false — confidence flagging UI was
    // removed in P0 so there's no surface to react to flagged photos.
    job.start({
      folder_path: targetFolder.folderPath,
      profile_id: activeProfile.id,
      flag_low_confidence: false,
      skip_fields: Array.from(skipFields),
      auto_straighten: autoStraighten,
    }).catch((e) => {
      // job.start rejected before opening the WS (HTTP error). Mark the
      // folder failed and let the effect re-fire to advance.
      dispatchInFlightRef.current = false;
      setQueue((q) => q.map((f, i) =>
        i === nextIndex ? { ...f, status: 'failed' } : f
      ));
      setRunResults((rs) => [...rs, {
        folderPath: targetFolder.folderPath,
        photosProcessed: 0, photosFailed: 0, durationSec: 0,
        state: 'failed', error: e.message || 'Failed to start',
      }]);
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isQueueRunning, cancelRequested, job.current?.id, job.current?.snapshot?.state, queue, activeProfile, skipFields, autoStraighten, selectedQueuedCount]);


  // Keyboard shortcuts (⌘O / ⌘R / ⌘.) — global for the editor.
  useEffect(() => {
    const handler = (e) => {
      // While the overwrite-confirm dialog is open it owns the keyboard;
      // ⌘ shortcuts are suppressed so they can't re-trigger flows behind it.
      if (overwriteConfirm) return;
      if (!e.metaKey) return;
      if (e.key === 'o') {
        e.preventDefault();
        if (visualState !== 'processing') handleAddFolder();
        return;
      }
      if (e.key === 'r') {
        e.preventDefault();
        if (visualState === 'ready') handleProcess();
        return;
      }
      if (e.key === '.') {
        e.preventDefault();
        if (visualState === 'processing') handleCancel();
        return;
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [visualState, handleAddFolder, handleProcess, handleCancel, overwriteConfirm]);

  // Overwrite dialog: Enter and Escape both dismiss without proceeding.
  // Cancel is the keyboard-default action so muscle-memory can't bypass.
  useEffect(() => {
    if (!overwriteConfirm) return;
    const handler = (e) => {
      if (e.key === 'Enter' || e.key === 'Escape') {
        e.preventDefault();
        setOverwriteConfirm(null);
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [overwriteConfirm]);

  // Right-column choice.
  let rightColumn;
  if (visualState === 'processing') {
    // During the brief gap between job.reset() (Case B) and the next
    // job.start() setting job.current (Case C), render the transition
    // placeholder instead of RightProcessing (which would crash on
    // job.current.snapshot).
    if (job.current && !isTerminal(job.current.snapshot.state)) {
      rightColumn = (
        <RightProcessing
          snapshot={job.current.snapshot}
          liveLog={job.current.liveLog}
          wsStatus={job.current.wsStatus}
          onCancel={handleCancel}
          queue={queue}
        />
      );
    } else {
      rightColumn = <RightTransitioning queue={queue} />;
    }
  } else if (visualState === 'complete') {
    rightColumn = (
      <RightComplete
        runResults={runResults}
        onProcessAnother={handleProcessAnother}
      />
    );
  } else {
    rightColumn = <RightEmpty lastRun={recentQ.folders[0] || null} />;
  }

  return (
    <>
      <AppShell
        title="saha"
        folder={inFlightFolderPath}
        activeNav="home"
        onNavigate={onNavigate}
        theme={theme}
        onToggleTheme={onToggleTheme}
        onLogout={onLogout}
      >
        <LeftFolderQueue
          queue={queue}
          expandedSet={expandedSet}
          locked={isQueueRunning}
          onAddFolder={handleAddFolder}
          onRemove={handleRemove}
          onSelect={handleSelectFolder}
          onToggleExpand={handleToggleExpand}
        />
        <CentreAction
          visualState={visualState}
          profiles={profiles}
          activeProfile={activeProfile}
          onPickProfile={onActivateProfile}
          onProcess={handleProcess}
          autoStraighten={autoStraighten}
          onAutoStraightenChange={setAutoStraighten}
          processRaws={processRaws}
          selectedQueuedCount={selectedQueuedCount}
          canProcess={canProcess}
          error={error}
          onDismissError={() => setError(null)}
        />
        {rightColumn}
      </AppShell>
      {overwriteConfirm && (
        <OverwriteConfirmDialog
          conflicts={overwriteConfirm.conflicts}
          totalCount={overwriteConfirm.totalCount}
          totalFolders={overwriteConfirm.totalFolders}
          onCancel={handleCancelOverwrite}
          onConfirm={handleConfirmOverwrite}
        />
      )}
    </>
  );
}
