import SONNA from '../tokens.js';
import { AppShell } from './shell.jsx';

const F = SONNA.font;
const M = SONNA.mono;

const Tlabel = {
  fontSize: 10, fontWeight: 600, color: SONNA.fgDim,
  textTransform: 'uppercase', letterSpacing: 0.6,
};
const Tnum = { fontFamily: M, fontVariantNumeric: 'tabular-nums' };

const folderBasename = (p) => {
  const segs = (p || '').split(/[\\/]/).filter(Boolean);
  return segs[segs.length - 1] || p || '';
};

const formatLoadedAt = (ts) => {
  if (!ts) return '-';
  const d = new Date(ts);
  if (!isFinite(d.getTime())) return '-';
  return d.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
};

function StatusBadge({ status }) {
  const palette = {
    processing: [SONNA.ochre, SONNA.ochreTint],
    complete: [SONNA.green, 'transparent'],
    failed: [SONNA.red, 'transparent'],
    cancelled: [SONNA.amber, 'transparent'],
    cancelling: [SONNA.amber, SONNA.ochreTint],
    running: [SONNA.ochre, SONNA.ochreTint],
    queued: [SONNA.fgMute, 'transparent'],
  };
  const [color, bg] = palette[status] || palette.queued;
  return (
    <span style={{
      display: 'inline-flex',
      alignItems: 'center',
      height: 22,
      padding: '0 8px',
      border: `1px solid ${color}`,
      borderRadius: 999,
      background: bg,
      color,
      fontSize: 10.5,
      fontWeight: 600,
      textTransform: 'uppercase',
      letterSpacing: 0.5,
    }}>
      {status || 'queued'}
    </span>
  );
}

export function ProjectsView({ projects, onNavigate, theme, onToggleTheme, onLogout }) {
  const currentJob = projects?.currentJob || null;
  const queueRows = [...(projects?.queue || [])]
    .sort((a, b) => (a.loadedAt || 0) - (b.loadedAt || 0))
    .map((row) => ({
      id: `queue-${row.folderPath}`,
      folderPath: row.folderPath,
      photos: row.fileCount || 0,
      status: currentJob?.folder_path === row.folderPath
        ? (currentJob.cancel_requested ? 'cancelling' : currentJob.state || row.status || 'queued')
        : row.status || 'queued',
      loadedAt: row.loadedAt,
      selected: row.selected,
      processed: currentJob?.folder_path === row.folderPath ? currentJob.photos_processed : null,
      total: currentJob?.folder_path === row.folderPath ? currentJob.photos_total : null,
    }));
  const resultRows = [...(projects?.runResults || [])].map((row, i) => ({
    id: `run-${row.folderPath}-${i}`,
    folderPath: row.folderPath,
    photos: row.photosProcessed || 0,
    status: row.state || 'complete',
    loadedAt: 0,
    selected: false,
  }));
  const rows = queueRows.length > 0 ? queueRows : resultRows;

  return (
    <AppShell
      title="saha - projects"
      activeNav="projects"
      onNavigate={onNavigate}
      theme={theme}
      onToggleTheme={onToggleTheme}
      onLogout={onLogout}
    >
      <div style={{
        flex: 1,
        background: SONNA.bgDeep,
        display: 'flex',
        flexDirection: 'column',
        minWidth: 0,
      }}>
        <div style={{
          height: 72,
          padding: '18px 28px',
          borderBottom: `1px solid ${SONNA.lineSoft}`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}>
          <div>
            <div style={{ fontSize: 20, color: SONNA.fg, fontWeight: 600 }}>Projects</div>
            <div style={{ marginTop: 4, ...Tnum, fontSize: 11, color: SONNA.fgFaint }}>
              {rows.length} {rows.length === 1 ? 'folder' : 'folders'}
            </div>
          </div>
          <button
            type="button"
            onClick={() => onNavigate?.('home')}
            title="Go to Home to add folders or start processing"
            style={{
              height: 34,
              padding: '0 14px',
              background: SONNA.cta,
              color: SONNA.onCta,
              border: 'none',
              borderRadius: 4,
              fontFamily: F,
              fontSize: 12,
              fontWeight: 700,
              cursor: 'pointer',
            }}
          >
            New project
          </button>
        </div>

        <div style={{ padding: '18px 24px', overflow: 'auto' }}>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'minmax(220px, 1fr) 120px 90px 140px 90px',
            gap: 14,
            padding: '0 12px 10px',
            borderBottom: `1px solid ${SONNA.lineSoft}`,
            ...Tlabel,
          }}>
            <span>Project</span>
            <span>Status</span>
            <span>Photos</span>
            <span>Loaded</span>
            <span>Selected</span>
          </div>
          {rows.map((row) => (
            <div key={row.id} style={{
              display: 'grid',
              gridTemplateColumns: 'minmax(220px, 1fr) 120px 90px 140px 90px',
              gap: 14,
              alignItems: 'center',
              minHeight: 48,
              padding: '9px 12px',
              borderBottom: `1px solid ${SONNA.lineSoft}`,
              color: SONNA.fg,
            }}>
              <div style={{ minWidth: 0 }}>
                <div style={{
                  fontSize: 13,
                  fontWeight: 600,
                  whiteSpace: 'nowrap',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                }}>{folderBasename(row.folderPath)}</div>
                <div style={{
                  marginTop: 2,
                  ...Tnum,
                  fontSize: 10.5,
                  color: SONNA.fgFaint,
                  whiteSpace: 'nowrap',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                }}>{row.folderPath}</div>
              </div>
              <StatusBadge status={row.status} />
              <div style={{ ...Tnum, fontSize: 12, color: SONNA.fgMute }}>
                {row.total ? `${row.processed || 0}/${row.total}` : row.photos}
              </div>
              <div style={{ ...Tnum, fontSize: 11, color: SONNA.fgFaint }}>{formatLoadedAt(row.loadedAt)}</div>
              <div style={{ fontSize: 12, color: row.selected ? SONNA.ochre : SONNA.fgFaint }}>
                {row.selected ? 'Yes' : '-'}
              </div>
            </div>
          ))}
          {rows.length === 0 && (
            <div style={{
              padding: 32,
              color: SONNA.fgMute,
              fontSize: 13,
            }}>
              No projects yet.
            </div>
          )}
        </div>
      </div>
    </AppShell>
  );
}
