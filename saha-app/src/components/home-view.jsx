// Home View — Active Projects Dashboard
// Reference: media_1794844500224_98a831e1.jpg & Imagen AI references

import { useMemo } from 'react';
import SONNA from '../tokens.js';
import { AppShell } from './shell.jsx';

const F = SONNA.font;
const M = SONNA.mono;

const Tnum = { fontFamily: M, fontVariantNumeric: 'tabular-nums' };

export function HomeView({
  onNavigate,
  onOpenNewProject,
  onOpenCreateProfile,
  onOpenProject,
  projects = [],
  activeProfile = null,
  theme,
  onToggleTheme,
  onLogout,
  uploads = [],
  isUpdating = false,
  updatingProgress = 37,
  onCancelUpload,
  onPauseAll,
  onResumeAll,
  onDismissCompleted,
}) {
  const isLight = theme === 'light';
  const currentJob = projects?.currentJob || null;

  const allProjects = useMemo(() => {
    if (Array.isArray(projects)) {
      return projects.map((p) => ({
        id: p.id,
        name: p.name,
        path: p.path,
        count: p.photos || p.count || 21,
        status: p.status === 'processing' ? 'Processing' : (p.status || 'Complete'),
        date: p.date || 'Just now',
        raw: p,
      }));
    }
    if (Array.isArray(projects?.list)) {
      return projects.list.map((p) => ({
        id: p.id,
        name: p.name,
        path: p.path,
        count: p.photos || p.count || 21,
        status: p.status === 'processing' ? 'Processing' : (p.status || 'Complete'),
        date: p.date || 'Just now',
        raw: p,
      }));
    }
    const queue = projects?.queue || [];
    const recentRuns = projects?.runResults || [];
    return [
      ...queue.map((q) => ({
        id: q.id || q.folderPath,
        name: q.folderPath?.split(/[\\/]/).pop() || 'Untitled Project',
        path: q.folderPath,
        count: q.fileCount || 0,
        status: 'Processing',
        date: q.loadedAt ? new Date(q.loadedAt).toLocaleDateString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : 'Just now',
        raw: q,
      })),
      ...recentRuns.map((r, i) => ({
        id: `run-${i}`,
        name: r.folderPath?.split(/[\\/]/).pop() || 'Untitled Project',
        path: r.folderPath,
        count: r.photosProcessed || 0,
        status: 'Complete',
        date: 'Recent',
        raw: r,
      })),
    ];
  }, [projects]);

  return (
    <AppShell
      title="saha - home"
      activeNav="home"
      onNavigate={onNavigate}
      theme={theme}
      onToggleTheme={onToggleTheme}
      onLogout={onLogout}
      onOpenNewProject={onOpenNewProject}
      uploads={uploads}
      isUpdating={isUpdating}
      updatingProgress={updatingProgress}
      onCancelUpload={onCancelUpload}
      onPauseAll={onPauseAll}
      onResumeAll={onResumeAll}
      onDismissCompleted={onDismissCompleted}
    >
      <style>{`
        @keyframes pulseDot {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.35; transform: scale(0.85); }
        }
      `}</style>
      <div style={{
        flex: 1,
        background: SONNA.bgDeep,
        display: 'flex',
        flexDirection: 'column',
        minWidth: 0,
        overflowY: 'auto',
      }}>
        {/* Top Header Bar */}
        <div style={{
          height: 72,
          padding: '18px 32px',
          borderBottom: `1px solid ${SONNA.lineSoft}`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexShrink: 0,
        }}>
          <div>
            <div style={{ fontSize: 22, color: SONNA.fg, fontWeight: 700 }}>
              Welcome to Saha
            </div>
            <div style={{ marginTop: 3, fontSize: 12, color: SONNA.fgDim }}>
              Personalized AI photo editing for Sonna Studios
            </div>
          </div>
        </div>

        {/* Content Body: Only Active Projects */}
        <div style={{ padding: '28px 32px', display: 'flex', flexDirection: 'column', gap: 24, maxWidth: 1280 }}>
          {/* Active Job Processing Card (when running) */}
          {currentJob && currentJob.state === 'running' && (
            <div style={{
              background: SONNA.bgPanel,
              border: `1px solid ${SONNA.ochre}`,
              borderRadius: 8,
              padding: '20px 24px',
              boxShadow: '0 8px 24px rgba(240, 148, 79, 0.12)',
            }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <span style={{
                    display: 'inline-flex', alignItems: 'center', height: 22, padding: '0 8px',
                    borderRadius: 999, background: SONNA.ochreTint, color: SONNA.ochre,
                    fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 0.5,
                  }}>
                    ● Processing Project
                  </span>
                  <span style={{ fontSize: 14, fontWeight: 600, color: SONNA.fg }}>
                    {currentJob.folder_path?.split(/[\\/]/).pop()}
                  </span>
                </div>
                <span style={{ fontFamily: M, fontSize: 12, color: SONNA.fgDim }}>
                  {currentJob.photos_processed || 0} / {currentJob.photos_total || 0} photos
                </span>
              </div>

              {/* Progress Bar */}
              <div style={{
                height: 8, background: SONNA.bgLifted, borderRadius: 4, overflow: 'hidden', marginBottom: 10,
              }}>
                <div style={{
                  height: '100%',
                  width: `${Math.round(((currentJob.photos_processed || 0) / (currentJob.photos_total || 1)) * 100)}%`,
                  background: SONNA.ochre,
                  transition: 'width 200ms ease',
                }} />
              </div>

              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11.5, color: SONNA.fgFaint }}>
                <span>Writing XMP sidecars &amp; auto-straightening</span>
                <button
                  type="button"
                  onClick={() => onNavigate?.('projects')}
                  style={{
                    background: 'transparent', border: 'none', color: SONNA.ochre,
                    cursor: 'pointer', fontWeight: 600, fontSize: 11.5,
                  }}
                >
                  View live progress →
                </button>
              </div>
            </div>
          )}

          {/* Active Projects Section matching Imagen AI reference screenshot */}
          <div>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
              <div style={{ fontSize: 15, fontWeight: 700, color: SONNA.fg }}>
                Active projects ({allProjects.length})
              </div>
              <button
                type="button"
                onClick={() => onNavigate?.('projects')}
                style={{
                  background: 'transparent',
                  border: 'none',
                  color: isLight ? '#1c1917' : '#FFFFFF',
                  fontSize: 13,
                  fontWeight: 600,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 4,
                  opacity: 0.95,
                  transition: 'opacity 0.15s ease',
                }}
                onMouseEnter={(e) => { e.currentTarget.style.opacity = '1'; }}
                onMouseLeave={(e) => { e.currentTarget.style.opacity = '0.95'; }}
              >
                <span>View all</span>
                <span style={{ fontSize: 14 }}>›</span>
              </button>
            </div>

            {/* Grid of Cards: "+ Create a project" followed by project cards */}
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(210px, 1fr))',
              gap: 16,
            }}>
              {/* Card 1: Create a Project Card */}
              <button
                type="button"
                onClick={onOpenNewProject}
                style={{
                  height: 220,
                  borderRadius: 10,
                  background: isLight ? '#FFFFFF' : '#1F1D17',
                  border: `1.5px dashed ${isLight ? 'rgba(0,0,0,0.22)' : 'rgba(255,255,255,0.4)'}`,
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: 12,
                  cursor: 'pointer',
                  color: isLight ? '#1c1917' : '#FFFFFF',
                  transition: 'all 0.15s ease',
                  padding: 16,
                  boxShadow: isLight ? '0 2px 8px rgba(0,0,0,0.06)' : 'none',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = SONNA.ochre;
                  e.currentTarget.style.transform = 'translateY(-2px)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = isLight ? 'rgba(0,0,0,0.22)' : 'rgba(255,255,255,0.4)';
                  e.currentTarget.style.transform = 'translateY(0)';
                }}
              >
                <div style={{
                  width: 40,
                  height: 40,
                  borderRadius: '50%',
                  border: `1.5px solid ${isLight ? '#1c1917' : '#FFFFFF'}`,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: 22,
                  color: isLight ? '#1c1917' : '#FFFFFF',
                  fontWeight: 300,
                }}>
                  +
                </div>
                <span style={{ fontSize: 13.5, fontWeight: 600, color: isLight ? '#1c1917' : '#FFFFFF' }}>
                  Create a project
                </span>
              </button>

              {/* Existing Project Cards */}
              {allProjects.map((proj, idx) => {
                const isProc = proj.status?.toLowerCase() === 'processing';
                return (
                  <div
                    key={`proj-${proj.id || proj.path}-${idx}`}
                    onClick={() => {
                      if (onOpenProject) {
                        onOpenProject(proj.raw || proj);
                      } else {
                        onNavigate?.('projects');
                      }
                    }}
                    style={{
                      height: 220,
                      borderRadius: 10,
                      background: SONNA.bgPanel,
                      border: `1px solid ${SONNA.lineSoft}`,
                      display: 'flex',
                      flexDirection: 'column',
                      overflow: 'hidden',
                      cursor: 'pointer',
                      transition: 'border-color 0.15s ease, transform 0.15s ease',
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.borderColor = isLight ? '#D8CEC1' : 'rgba(255, 255, 255, 0.22)';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.borderColor = SONNA.lineSoft;
                    }}
                  >
                    {/* Top image preview area matching Imagen reference */}
                    <div style={{
                      flex: 1,
                      background: SONNA.bgLifted,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      position: 'relative',
                      borderBottom: `1px solid ${SONNA.lineSoft}`,
                    }}>
                      <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke={SONNA.fgFaint} strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round">
                        <rect x="3" y="3" width="18" height="18" rx="2" />
                        <circle cx="8.5" cy="8.5" r="1.5" />
                        <polyline points="21 15 16 10 5 21" />
                      </svg>
                    </div>

                    {/* Bottom info area matching Imagen reference */}
                    <div style={{ padding: '12px 14px', display: 'flex', flexDirection: 'column', gap: 5, background: SONNA.bgPanel }}>
                      <div style={{
                        fontSize: 13,
                        fontWeight: 600,
                        color: SONNA.fg,
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                      }}>
                        {proj.name}
                      </div>

                      {/* Status indicator row */}
                      <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 6,
                        fontSize: 11.5,
                        fontWeight: 500,
                        color: isProc ? '#60A5FA' : SONNA.fgDim,
                      }}>
                        <span style={{
                          width: 6,
                          height: 6,
                          borderRadius: '50%',
                          background: isProc ? '#3B82F6' : '#22C55E',
                          boxShadow: isProc ? '0 0 6px rgba(59, 130, 246, 0.6)' : 'none',
                          animation: isProc ? 'pulseDot 1.8s ease-in-out infinite' : 'none',
                          display: 'inline-block',
                        }} />
                        <span>{isProc ? 'Processing' : 'Complete'}</span>
                      </div>

                      <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        fontSize: 11,
                        color: SONNA.fgDim,
                        marginTop: 2,
                      }}>
                        <span>{proj.date}</span>
                        <span style={{ ...Tnum }}>{proj.count} photos</span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </AppShell>
  );
}

export default HomeView;
