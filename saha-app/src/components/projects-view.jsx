// Projects View — Projects Dashboard with Active / Archived Segmented Switcher & Zero State
// Reference:
// - media_1794847706957_276d9356.png (Top segmented pill "Active 0 | Archived 1", centered "Start with a project",
//   "+ New project" coral button)
// Full Light and Dark theme support via SONNA design tokens.

import { useMemo, useState, useEffect } from 'react';
import SONNA from '../tokens.js';
import { AppShell } from './shell.jsx';

const F = SONNA.font;
const M = SONNA.mono;

export function ProjectsView({
  projects,
  onNavigate,
  onOpenNewProject,
  onOpenProject,
  theme = 'dark',
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
  const [projectTab, setProjectTab] = useState('active'); // 'active' | 'archived'
  const [archivedIds, setArchivedIds] = useState(() => {
    try {
      const stored = localStorage.getItem('saha_archived_project_ids');
      return stored ? JSON.parse(stored) : ['archived-demo-1'];
    } catch {
      return ['archived-demo-1'];
    }
  });
  const [cardMenuOpenId, setCardMenuOpenId] = useState(null);

  // Close menus on outside click
  useEffect(() => {
    const handleClose = () => setCardMenuOpenId(null);
    window.addEventListener('click', handleClose);
    return () => window.removeEventListener('click', handleClose);
  }, []);

  const currentJob = projects?.currentJob || null;

  // Gather projects from array or queue and results
  const allProjects = useMemo(() => {
    if (Array.isArray(projects)) {
      return projects.map((p) => ({
        id: p.id || `proj-${p.path}`,
        name: p.name || p.path?.split(/[\\/]/).filter(Boolean).pop() || 'Untitled Project',
        path: p.path,
        photos: p.photos || p.count || 21,
        date: p.date || 'Just now',
        status: p.status || 'complete',
        raw: p,
      }));
    }
    if (Array.isArray(projects?.list)) {
      return projects.list.map((p) => ({
        id: p.id || `proj-${p.path}`,
        name: p.name || p.path?.split(/[\\/]/).filter(Boolean).pop() || 'Untitled Project',
        path: p.path,
        photos: p.photos || p.count || 21,
        date: p.date || 'Just now',
        status: p.status || 'complete',
        raw: p,
      }));
    }
    const list = [];
    if (projects?.queue) {
      projects.queue.forEach((row) => {
        list.push({
          id: `queue-${row.folderPath}`,
          name: row.folderPath?.split(/[\\/]/).filter(Boolean).pop() || 'Untitled Project',
          path: row.folderPath,
          photos: row.fileCount || 0,
          date: 'Just now',
          status: currentJob?.folder_path === row.folderPath ? (currentJob.state || 'processing') : 'processing',
          raw: row,
        });
      });
    }
    if (projects?.runResults) {
      projects.runResults.forEach((row, i) => {
        list.push({
          id: `run-${row.folderPath}-${i}`,
          name: row.folderPath?.split(/[\\/]/).filter(Boolean).pop() || `Project ${i + 1}`,
          path: row.folderPath,
          photos: row.photosProcessed || 21,
          date: 'Nov 14, 11:43 AM',
          status: 'complete',
          raw: row,
        });
      });
    }
    return list;
  }, [projects, currentJob]);

  // Separate into Active and Archived
  const activeProjects = useMemo(() => {
    return allProjects.filter((p) => !archivedIds.includes(p.id));
  }, [allProjects, archivedIds]);

  const archivedProjects = useMemo(() => {
    const archivedList = allProjects.filter((p) => archivedIds.includes(p.id));
    if (archivedIds.includes('archived-demo-1') && !archivedList.some((p) => p.id === 'archived-demo-1')) {
      archivedList.push({
        id: 'archived-demo-1',
        name: 'Wedding Ceremony Highlight',
        path: 'D:/Sonna/Shoots/2026/Wedding_Highlights',
        photos: 184,
        date: 'Oct 28, 2026',
        status: 'archived',
      });
    }
    return archivedList;
  }, [allProjects, archivedIds]);

  const handleArchive = (id) => {
    setCardMenuOpenId(null);
    const updated = [...archivedIds, id];
    setArchivedIds(updated);
    try {
      localStorage.setItem('saha_archived_project_ids', JSON.stringify(updated));
    } catch (e) {
      console.error(e);
    }
  };

  const handleUnarchive = (id) => {
    setCardMenuOpenId(null);
    const updated = archivedIds.filter((item) => item !== id);
    setArchivedIds(updated);
    try {
      localStorage.setItem('saha_archived_project_ids', JSON.stringify(updated));
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <AppShell
      title="saha - projects"
      activeNav="projects"
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
        color: SONNA.fg,
        transition: 'background 0.2s ease, color 0.2s ease',
      }}>
        {/* Top Header Bar */}
        <div style={{
          height: 64,
          padding: '0 32px',
          borderBottom: `1px solid ${SONNA.lineSoft}`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexShrink: 0,
        }}>
          {/* Left Title */}
          <div style={{
            fontSize: 16,
            fontWeight: 700,
            color: SONNA.fg,
            letterSpacing: -0.2,
          }}>
            Projects
          </div>

          {/* Center Segmented Pill: Active (count) | Archived (count) */}
          <div style={{
            display: 'inline-flex',
            alignItems: 'center',
            background: isLight ? '#EFE7DC' : '#1F1D17',
            borderRadius: 20,
            padding: 3,
            border: `1px solid ${SONNA.lineSoft}`,
            gap: 2,
          }}>
            {/* Active Pill */}
            <button
              type="button"
              onClick={() => setProjectTab('active')}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                padding: '5px 14px',
                borderRadius: 16,
                background: projectTab === 'active' ? (isLight ? '#FFFFFF' : '#2E2B23') : 'transparent',
                border: projectTab === 'active' ? `1px solid ${isLight ? '#D8CEC1' : 'rgba(255, 255, 255, 0.16)'}` : '1px solid transparent',
                color: projectTab === 'active' ? SONNA.fg : SONNA.fgDim,
                fontFamily: F,
                fontSize: 12.5,
                fontWeight: 600,
                cursor: 'pointer',
                transition: 'all 0.15s ease',
                boxShadow: projectTab === 'active' && isLight ? '0 1px 3px rgba(0,0,0,0.06)' : 'none',
              }}
            >
              <span>Active</span>
              <span style={{ fontSize: 11.5, opacity: 0.85, fontFamily: M }}>
                {activeProjects.length}
              </span>
            </button>

            {/* Archived Pill */}
            <button
              type="button"
              onClick={() => setProjectTab('archived')}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                padding: '5px 14px',
                borderRadius: 16,
                background: projectTab === 'archived' ? (isLight ? '#FFFFFF' : '#2E2B23') : 'transparent',
                border: projectTab === 'archived' ? `1px solid ${isLight ? '#D8CEC1' : 'rgba(255, 255, 255, 0.16)'}` : '1px solid transparent',
                color: projectTab === 'archived' ? SONNA.fg : SONNA.fgDim,
                fontFamily: F,
                fontSize: 12.5,
                fontWeight: 600,
                cursor: 'pointer',
                transition: 'all 0.15s ease',
                boxShadow: projectTab === 'archived' && isLight ? '0 1px 3px rgba(0,0,0,0.06)' : 'none',
              }}
            >
              <span>Archived</span>
              <span style={{ fontSize: 11.5, opacity: 0.85, fontFamily: M }}>
                {archivedProjects.length}
              </span>
            </button>
          </div>

          {/* Right spacer to keep center pill perfectly balanced */}
          <div style={{ width: 80 }} />
        </div>

        {/* Tab 1: Active Projects */}
        {projectTab === 'active' && (
          activeProjects.length === 0 ? (
            /* Empty State */
            <div style={{
              flex: 1,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              textAlign: 'center',
              padding: '60px 24px',
            }}>
              <h2 style={{
                fontSize: 22,
                fontWeight: 700,
                color: SONNA.fg,
                marginBottom: 8,
                letterSpacing: -0.3,
              }}>
                Start with a project
              </h2>
              <p style={{
                fontSize: 13.5,
                color: SONNA.fgMute,
                maxWidth: 440,
                margin: '0 0 24px 0',
                lineHeight: 1.5,
              }}>
                Every shoot lives inside a project. Upload photos to get started.
              </p>

              {/* Coral Primary Button: + New project */}
              <button
                type="button"
                onClick={onOpenNewProject}
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 6,
                  padding: '9px 24px',
                  background: '#E05353',
                  color: '#FFFFFF',
                  border: 'none',
                  borderRadius: 6,
                  fontFamily: F,
                  fontSize: 13,
                  fontWeight: 600,
                  cursor: 'pointer',
                  boxShadow: '0 4px 14px rgba(224, 83, 83, 0.35)',
                  transition: 'background 0.15s ease',
                }}
                onMouseEnter={(e) => { e.currentTarget.style.background = '#EC5A5A'; }}
                onMouseLeave={(e) => { e.currentTarget.style.background = '#E05353'; }}
              >
                <span>+</span>
                <span>New project</span>
              </button>
            </div>
          ) : (
            /* Active Projects Grid */
            <div style={{ padding: '28px 36px', maxWidth: 1280 }}>
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))',
                gap: 18,
              }}>
                {/* Create Project Card */}
                <button
                  type="button"
                  onClick={onOpenNewProject}
                  style={{
                    height: 220,
                    borderRadius: 10,
                    background: SONNA.bgPanel,
                    border: `1px dashed ${SONNA.line}`,
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: 12,
                    cursor: 'pointer',
                    color: SONNA.fg,
                    transition: 'all 0.15s ease',
                    padding: 16,
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor = '#E05353';
                    e.currentTarget.style.background = SONNA.bgHover;
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = SONNA.line;
                    e.currentTarget.style.background = SONNA.bgPanel;
                  }}
                >
                  <div style={{
                    width: 38,
                    height: 38,
                    borderRadius: '50%',
                    border: `1.5px solid ${SONNA.fgDim}`,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: 20,
                    fontWeight: 300,
                    color: SONNA.fgDim,
                  }}>
                    +
                  </div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: SONNA.fg }}>
                    Create a project
                  </div>
                </button>

                {/* Project Cards */}
                {activeProjects.map((p) => {
                  const isMenuOpen = cardMenuOpenId === p.id;
                  return (
                    <div
                      key={p.id}
                      style={{
                        height: 220,
                        borderRadius: 10,
                        background: SONNA.bgPanel,
                        border: `1px solid ${SONNA.lineSoft}`,
                        overflow: 'hidden',
                        display: 'flex',
                        flexDirection: 'column',
                        cursor: 'pointer',
                        transition: 'transform 0.15s ease, border-color 0.15s ease',
                        position: 'relative',
                      }}
                      onClick={() => {
                        if (onOpenProject) {
                          onOpenProject(p.raw || p);
                        } else if (p.path && typeof window !== 'undefined' && window.saha?.revealPath) {
                          window.saha.revealPath(p.path);
                        }
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.borderColor = isLight ? '#D8CEC1' : 'rgba(255, 255, 255, 0.22)';
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.borderColor = SONNA.lineSoft;
                      }}
                    >
                      {/* Image Thumbnail / Canvas */}
                      <div style={{
                        flex: 1,
                        background: SONNA.bgLifted,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        position: 'relative',
                        overflow: 'hidden',
                      }}>
                        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke={SONNA.fgFaint} strokeWidth="1.2">
                          <rect x="3" y="3" width="18" height="18" rx="2" />
                          <circle cx="8.5" cy="8.5" r="1.5" />
                          <polyline points="21 15 16 10 5 21" />
                        </svg>

                        {/* Top-Right Card Menu */}
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            setCardMenuOpenId(isMenuOpen ? null : p.id);
                          }}
                          style={{
                            position: 'absolute',
                            top: 8,
                            right: 8,
                            width: 26,
                            height: 26,
                            borderRadius: 4,
                            background: isLight ? 'rgba(255,255,255,0.85)' : 'rgba(0,0,0,0.5)',
                            border: 'none',
                            color: SONNA.fg,
                            fontSize: 16,
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            boxShadow: isLight ? '0 1px 3px rgba(0,0,0,0.1)' : 'none',
                          }}
                        >
                          ⋮
                        </button>

                        {isMenuOpen && (
                          <div
                            onClick={(e) => e.stopPropagation()}
                            style={{
                              position: 'absolute',
                              top: 38,
                              right: 8,
                              zIndex: 20,
                              background: SONNA.bgPanel,
                              border: `1px solid ${SONNA.line}`,
                              borderRadius: 6,
                              padding: 4,
                              boxShadow: isLight ? '0 8px 24px rgba(0,0,0,0.15)' : '0 8px 24px rgba(0,0,0,0.6)',
                              width: 140,
                            }}
                          >
                            <button
                              type="button"
                              onClick={() => handleArchive(p.id)}
                              style={{
                                width: '100%',
                                padding: '6px 10px',
                                background: 'transparent',
                                border: 'none',
                                color: SONNA.fg,
                                fontSize: 12,
                                textAlign: 'left',
                                cursor: 'pointer',
                                borderRadius: 4,
                              }}
                            >
                              Archive project
                            </button>
                          </div>
                        )}
                      </div>

                      {/* Card Details Footer matching Imagen reference */}
                      <div style={{ padding: '12px 14px', background: SONNA.bgPanel, display: 'flex', flexDirection: 'column', gap: 5 }}>
                        <div style={{
                          fontSize: 13,
                          fontWeight: 600,
                          color: SONNA.fg,
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                        }}>
                          {p.name}
                        </div>

                        {/* Status indicator row matching Imagen reference */}
                        <div style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: 6,
                          fontSize: 11.5,
                          fontWeight: 500,
                          color: p.status === 'processing' ? '#60A5FA' : SONNA.fgDim,
                        }}>
                          <span style={{
                            width: 6,
                            height: 6,
                            borderRadius: '50%',
                            background: p.status === 'processing' ? '#3B82F6' : '#22C55E',
                            boxShadow: p.status === 'processing' ? '0 0 6px rgba(59, 130, 246, 0.6)' : 'none',
                            animation: p.status === 'processing' ? 'pulseDot 1.8s ease-in-out infinite' : 'none',
                            display: 'inline-block',
                          }} />
                          <span>{p.status === 'processing' ? 'Processing' : 'Complete'}</span>
                        </div>

                        <div style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center',
                          marginTop: 2,
                          fontSize: 11,
                          color: SONNA.fgDim,
                        }}>
                          <span>{p.date}</span>
                          <span style={{ fontFamily: M }}>{p.photos} photos</span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )
        )}

        {/* Tab 2: Archived Projects */}
        {projectTab === 'archived' && (
          archivedProjects.length === 0 ? (
            <div style={{
              flex: 1,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              textAlign: 'center',
              padding: '60px 24px',
            }}>
              <h2 style={{ fontSize: 20, fontWeight: 700, color: SONNA.fg, marginBottom: 8 }}>
                No archived projects
              </h2>
              <p style={{ fontSize: 13, color: SONNA.fgMute, maxWidth: 400, margin: '0 0 20px 0' }}>
                Projects you archive from completed shoots will appear here.
              </p>
              <button
                type="button"
                onClick={() => setProjectTab('active')}
                style={{
                  padding: '7px 18px',
                  background: isLight ? '#FFFFFF' : 'transparent',
                  border: `1px solid ${isLight ? '#D8CEC1' : 'rgba(255, 255, 255, 0.35)'}`,
                  borderRadius: 6,
                  color: SONNA.fg,
                  fontSize: 12.5,
                  fontWeight: 600,
                  cursor: 'pointer',
                }}
              >
                View active projects
              </button>
            </div>
          ) : (
            <div style={{ padding: '28px 36px', maxWidth: 1280 }}>
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))',
                gap: 18,
              }}>
                {archivedProjects.map((p) => (
                  <div
                    key={p.id}
                    style={{
                      height: 220,
                      borderRadius: 10,
                      background: SONNA.bgPanel,
                      border: `1px solid ${SONNA.lineSoft}`,
                      overflow: 'hidden',
                      display: 'flex',
                      flexDirection: 'column',
                    }}
                  >
                    <div style={{
                      flex: 1,
                      background: SONNA.bgLifted,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      position: 'relative',
                    }}>
                      <span style={{ fontSize: 32, opacity: 0.4 }}>📦</span>
                    </div>
                    <div style={{ padding: '12px 14px', background: SONNA.bgPanel }}>
                      <div style={{ fontSize: 13, fontWeight: 600, color: SONNA.fg }}>{p.name}</div>
                      <div style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        marginTop: 6,
                        fontSize: 11,
                        color: SONNA.fgDim,
                      }}>
                        <span>{p.date}</span>
                        <button
                          type="button"
                          onClick={() => handleUnarchive(p.id)}
                          style={{
                            background: 'transparent',
                            border: 'none',
                            color: '#E05353',
                            fontSize: 11,
                            fontWeight: 600,
                            cursor: 'pointer',
                            padding: 0,
                          }}
                        >
                          Unarchive
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )
        )}
      </div>
    </AppShell>
  );
}

export default ProjectsView;
