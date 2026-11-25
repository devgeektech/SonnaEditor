// Projects View — Projects Dashboard with Active / Archived Segmented Switcher & Zero State
// Reference:
// - media_1794847706957_276d9356.png (Top segmented pill "Active 0 | Archived 1", centered "Start with a project",
//   "+ New project" coral button)
// Full Light and Dark theme support via SONNA design tokens.

import { useMemo, useState, useEffect } from 'react';
import SONNA from '../tokens.js';
import { AppShell } from './shell.jsx';
import { listCatalogs, scanFolder } from '../api/client.js';

const F = SONNA.font;
const M = SONNA.mono;
const Tnum = { fontFamily: SONNA.mono, fontVariantNumeric: 'tabular-nums' };

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
  isUploadsOpen,
  onToggleUploadsOpen,
  onUpdateProjectStatus,
  onOpenTrainProfile,
}) {
  const isLight = theme === 'light';
  const [projectTab, setProjectTab] = useState('active'); // 'active' | 'archived'
  const [viewMode, setViewMode] = useState('list'); // 'list' | 'grid' (Default to list per Imagen reference)
  const [searchQuery, setSearchQuery] = useState('');
  const [archivedIds, setArchivedIds] = useState(() => {
    try {
      const stored = localStorage.getItem('saha_archived_project_ids');
      return stored ? JSON.parse(stored) : ['archived-demo-1'];
    } catch {
      return ['archived-demo-1'];
    }
  });
  const [cardMenuOpenId, setCardMenuOpenId] = useState(null);

  // Review in Adobe (Lightroom Classic) modal state
  const [reviewDropdownOpenId, setReviewDropdownOpenId] = useState(null);
  const [reviewAdobeProject, setReviewAdobeProject] = useState(null);
  const [reviewAdobeError, setReviewAdobeError] = useState(null);
  const [downloadEditsProject, setDownloadEditsProject] = useState(null);
  const [downloadPath, setDownloadPath] = useState('');
  const [downloadSelectionTab, setDownloadSelectionTab] = useState('folders'); // 'folders' | 'collections'
  const [downloadSelectionOpen, setDownloadSelectionOpen] = useState(true);
  const [downloadCatalogFolders, setDownloadCatalogFolders] = useState([]);
  const [downloadCatalogCollections, setDownloadCatalogCollections] = useState([]);
  const [downloadSelectedIds, setDownloadSelectedIds] = useState(new Set());
  const [downloadScanning, setDownloadScanning] = useState(false);
  const [showDownloadOverwriteNote, setShowDownloadOverwriteNote] = useState(false);
  const [dontShowOverwriteNoteAgain, setDontShowOverwriteNoteAgain] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState(0.0);
  const [isDownloadComplete, setIsDownloadComplete] = useState(false);
  const [isBannerDismissed, setIsBannerDismissed] = useState(false);

  // Close menus on outside click
  useEffect(() => {
    const handleClose = () => {
      setCardMenuOpenId(null);
      setReviewDropdownOpenId(null);
    };
    window.addEventListener('click', handleClose);
    return () => window.removeEventListener('click', handleClose);
  }, []);

  const handleOpenReviewAdobe = (p) => {
    setReviewAdobeError(null);
    setReviewAdobeProject(p);
  };

  const handleLaunchLightroomClassic = async () => {
    if (!reviewAdobeProject) return;
    const targetPath = reviewAdobeProject.path || '';
    if (typeof window !== 'undefined' && typeof window.saha?.openInLightroom === 'function') {
      const res = await window.saha.openInLightroom(targetPath);
      if (res && res.ok === false && res.error) {
        setReviewAdobeError(res.error);
        return;
      }
    } else if (targetPath && typeof window !== 'undefined' && window.saha?.revealPath) {
      await window.saha.revealPath(targetPath);
    }
    setReviewAdobeProject(null);
  };

  const loadDownloadSelectionForPath = async (targetPath, proj) => {
    const cleanPath = (targetPath || '').trim();
    if (!cleanPath) {
      setDownloadCatalogFolders([]);
      setDownloadCatalogCollections([]);
      setDownloadSelectedIds(new Set());
      return;
    }
    if (cleanPath.toLowerCase().endsWith('.lrcat')) {
      setDownloadScanning(true);
      try {
        const scan = await scanFolder(cleanPath, 'catalog');
        const folders = (scan?.folders || []).map((f, idx) => ({
          id: f.id || `folder-${idx}`,
          name: f.name || (f.path ? f.path.split(/[\\/]/).filter(Boolean).pop() : `Folder ${idx + 1}`),
          path: f.path || cleanPath,
          totalPhotos: f.total_photos ?? f.count ?? f.raw_count ?? 0,
          filteredPhotos: f.filtered_photos ?? f.total_photos ?? f.count ?? 0,
          date: f.date || proj?.date || '—',
        }));
        const collections = (scan?.collections || []).map((c, idx) => ({
          id: c.id || `col-${idx}`,
          name: c.name || `Collection ${idx + 1}`,
          path: c.name || '',
          totalPhotos: c.total_photos ?? c.count ?? 0,
          filteredPhotos: c.filtered_photos ?? c.total_photos ?? c.count ?? 0,
          date: c.date || proj?.date || '—',
        }));
        setDownloadCatalogFolders(folders);
        setDownloadCatalogCollections(collections);
        const matching = folders.filter((f) =>
          (proj?.name && f.name.toLowerCase() === proj.name.toLowerCase()) ||
          (proj?.path && f.path && f.path.replace(/\\/g, '/').toLowerCase() === proj.path.replace(/\\/g, '/').toLowerCase())
        );
        const initialIds = (matching.length > 0 ? matching : folders).map((f) => f.id);
        setDownloadSelectedIds(new Set(initialIds));
      } catch (e) {
        console.warn('Catalog scan in download modal failed:', e);
        setDownloadCatalogFolders([{
          id: 'folder-0',
          name: proj?.name || 'Catalog photos',
          path: cleanPath,
          totalPhotos: proj?.photos || 0,
          filteredPhotos: proj?.photos || 0,
          date: proj?.date || '—',
        }]);
        setDownloadCatalogCollections([]);
        setDownloadSelectedIds(new Set(['folder-0']));
      } finally {
        setDownloadScanning(false);
      }
    } else {
      setDownloadCatalogFolders([{
        id: 'folder-0',
        name: proj?.name || cleanPath.split(/[\\/]/).filter(Boolean).pop() || 'Folder',
        path: cleanPath,
        totalPhotos: proj?.photos || 0,
        filteredPhotos: proj?.photos || 0,
        date: proj?.date || '—',
      }]);
      setDownloadCatalogCollections([]);
      setDownloadSelectedIds(new Set(['folder-0']));
    }
  };

  const handleStartDownloadEdits = async (p) => {
    let initialPath = p.path || `C:\\Users\\ASUS\\Downloads\\${p.name}`;
    setDownloadEditsProject(p);
    setDownloadPath(initialPath);
    setDownloadSelectionTab('folders');
    setDownloadSelectionOpen(true);
    setShowDownloadOverwriteNote(false);
    setIsDownloading(false);
    setIsDownloadComplete(false);
    setDownloadProgress(0.0);
    setIsBannerDismissed(false);

    // If project was created from a folder, auto-discover the user's Lightroom catalog
    // so Selection shows all catalog folders (matching Imagen) and Open LR catalog opens LrC.
    if (!initialPath.toLowerCase().endsWith('.lrcat')) {
      try {
        const catalogs = await listCatalogs();
        if (Array.isArray(catalogs) && catalogs.length > 0 && catalogs[0]?.path) {
          initialPath = catalogs[0].path;
          setDownloadPath(initialPath);
        }
      } catch (e) {
        console.warn('Could not auto-discover Lightroom catalog:', e);
      }
    }
    await loadDownloadSelectionForPath(initialPath, p);
  };

  const handleBrowseDownloadPath = async () => {
    if (typeof window === 'undefined') return;
    try {
      const isCatalog = (downloadPath || '').toLowerCase().endsWith('.lrcat');
      let chosen = null;
      if (isCatalog && window.saha?.pickFile) {
        chosen = await window.saha.pickFile({
          title: 'Choose a Lightroom catalog',
          filters: [{ name: 'Lightroom catalog', extensions: ['lrcat'] }],
        });
      } else if (window.saha?.pickFolder) {
        chosen = await window.saha.pickFolder();
      }
      if (chosen) {
        setDownloadPath(chosen);
        loadDownloadSelectionForPath(chosen, downloadEditsProject);
      }
    } catch (e) {
      console.error(e);
    }
  };

  const startDownloadEditsProgress = () => {
    setShowDownloadOverwriteNote(false);
    setIsDownloading(true);
    setDownloadProgress(0.0);

    let current = 0;
    const interval = setInterval(() => {
      current += Math.random() * 14 + 8;
      if (current >= 100) {
        clearInterval(interval);
        setDownloadProgress(100.0);
        setTimeout(() => {
          setIsDownloading(false);
          setIsDownloadComplete(true);
        }, 350);
      } else {
        setDownloadProgress(Math.min(99.4, current));
      }
    }, 120);
  };

  const handleConfirmDownloadEdits = () => {
    try {
      if (localStorage.getItem('saha_skip_download_overwrite_note') === 'true') {
        startDownloadEditsProgress();
        return;
      }
    } catch {
      // ignore localStorage read errors
    }
    setDontShowOverwriteNoteAgain(false);
    setShowDownloadOverwriteNote(true);
  };

  const handleContinueAfterOverwriteNote = () => {
    if (dontShowOverwriteNoteAgain) {
      try {
        localStorage.setItem('saha_skip_download_overwrite_note', 'true');
      } catch {
        // ignore localStorage write errors
      }
    }
    startDownloadEditsProgress();
  };

  const handleOpenLrCatalogFromComplete = async () => {
    const proj = downloadEditsProject;
    const targetPath = downloadPath || proj?.path || '';
    setDownloadEditsProject(null);
    setIsDownloading(false);
    setIsDownloadComplete(false);
    if (typeof window !== 'undefined' && typeof window.saha?.openInLightroom === 'function') {
      await window.saha.openInLightroom(targetPath);
    } else if (targetPath && typeof window !== 'undefined' && window.saha?.revealPath) {
      await window.saha.revealPath(targetPath);
    }
  };

  const currentJob = projects?.currentJob || null;

  // Gather projects from array or queue and results
  const allProjects = useMemo(() => {
    if (Array.isArray(projects)) {
      return projects.map((p) => ({
        id: p.id || `proj-${p.path}`,
        name: p.name || p.path?.split(/[\\/]/).filter(Boolean).pop() || 'Untitled Project',
        profile: p.profile || 'WARM SKIN TONES',
        path: p.path,
        photos: p.photos || p.count || 5,
        date: p.date || 'Feb 10, 2026',
        status: p.status || 'complete',
        reviewMode: p.reviewMode,
        progress: p.progress || 0,
        raw: p,
      }));
    }
    if (Array.isArray(projects?.list)) {
      return projects.list.map((p) => ({
        id: p.id || `proj-${p.path}`,
        name: p.name || p.path?.split(/[\\/]/).filter(Boolean).pop() || 'Untitled Project',
        profile: p.profile || 'WARM SKIN TONES',
        path: p.path,
        photos: p.photos || p.count || 5,
        date: p.date || 'Feb 10, 2026',
        status: p.status || 'complete',
        reviewMode: p.reviewMode,
        progress: p.progress || 0,
        raw: p,
      }));
    }
    const list = [];
    if (projects?.queue) {
      projects.queue.forEach((row) => {
        list.push({
          id: `queue-${row.folderPath}`,
          name: row.folderPath?.split(/[\\/]/).filter(Boolean).pop() || 'Untitled Project',
          profile: 'WARM SKIN TONES',
          path: row.folderPath,
          photos: row.fileCount || 5,
          date: 'Just now',
          status: currentJob?.folder_path === row.folderPath ? (currentJob.state || 'processing') : 'processing',
          progress: currentJob?.percent || 0,
          raw: row,
        });
      });
    }
    if (projects?.runResults) {
      projects.runResults.forEach((row, i) => {
        list.push({
          id: `run-${row.folderPath}-${i}`,
          name: row.folderPath?.split(/[\\/]/).filter(Boolean).pop() || `Project ${i + 1}`,
          profile: 'WARM SKIN TONES',
          path: row.folderPath,
          photos: row.photosProcessed || 21,
          date: 'Nov 17, 2:59 AM',
          status: 'complete',
          progress: 100,
          raw: row,
        });
      });
    }
    return list;
  }, [projects, currentJob]);

  // Separate into Active and Archived + Search query filter
  const activeProjects = useMemo(() => {
    return allProjects
      .filter((p) => !archivedIds.includes(p.id))
      .filter((p) => !searchQuery || p.name.toLowerCase().includes(searchQuery.toLowerCase()));
  }, [allProjects, archivedIds, searchQuery]);

  const archivedProjects = useMemo(() => {
    const archivedList = allProjects.filter((p) => archivedIds.includes(p.id));
    if (archivedIds.includes('archived-demo-1') && !archivedList.some((p) => p.id === 'archived-demo-1')) {
      archivedList.push({
        id: 'archived-demo-1',
        name: 'Wedding Ceremony Highlight',
        profile: 'WARM SKIN TONES',
        path: 'D:/Sonna/Shoots/2026/Wedding_Highlights',
        photos: 184,
        date: 'Oct 28, 2026',
        status: 'archived',
        progress: 100,
      });
    }
    return archivedList.filter((p) => !searchQuery || p.name.toLowerCase().includes(searchQuery.toLowerCase()));
  }, [allProjects, archivedIds, searchQuery]);

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
      isUploadsOpen={isUploadsOpen}
      onToggleUploadsOpen={onToggleUploadsOpen}
    >
      <style>{`
        @keyframes pulseDot {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.35; transform: scale(0.85); }
        }
        @keyframes editProgressAnim {
          0% { width: 25%; }
          50% { width: 75%; }
          100% { width: 95%; }
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
        {/* Top Header Bar matching Images 5, 6, 7 */}
        <div style={{
          minHeight: 64,
          padding: '10px 24px',
          borderBottom: `1px solid ${SONNA.lineSoft}`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          flexShrink: 0,
          gap: 12,
        }}>
          {/* Left: Search input */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            background: isLight ? '#FFFFFF' : '#14161B',
            border: `1px solid ${SONNA.lineSoft}`,
            borderRadius: 6,
            padding: '7px 12px',
            flex: '1 1 180px',
            maxWidth: 260,
            minWidth: 150,
            gap: 8,
          }}>
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search for a project"
              style={{
                background: 'transparent',
                border: 'none',
                color: SONNA.fg,
                fontFamily: F,
                fontSize: 12.5,
                outline: 'none',
                width: '100%',
              }}
            />
            <span style={{ color: SONNA.fgDim, fontSize: 13 }}>🔍</span>
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

          {/* Right: View mode switcher + "+ New project" Button */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            {/* View Mode Toggle: List / Grid */}
            <div style={{
              display: 'flex',
              alignItems: 'center',
              background: isLight ? '#EFE7DC' : '#1F1D17',
              borderRadius: 6,
              padding: 2,
              border: `1px solid ${SONNA.lineSoft}`,
              gap: 2,
            }}>
              <button
                type="button"
                title="List view"
                onClick={() => setViewMode('list')}
                style={{
                  width: 30,
                  height: 30,
                  borderRadius: 4,
                  background: viewMode === 'list' ? (isLight ? '#FFFFFF' : '#2E2B23') : 'transparent',
                  border: 'none',
                  color: viewMode === 'list' ? SONNA.fg : SONNA.fgDim,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  transition: 'all 0.15s ease',
                }}
              >
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="8" y1="6" x2="21" y2="6" />
                  <line x1="8" y1="12" x2="21" y2="12" />
                  <line x1="8" y1="18" x2="21" y2="18" />
                  <line x1="3" y1="6" x2="3.01" y2="6" />
                  <line x1="3" y1="12" x2="3.01" y2="12" />
                  <line x1="3" y1="18" x2="3.01" y2="18" />
                </svg>
              </button>
              <button
                type="button"
                title="Grid view"
                onClick={() => setViewMode('grid')}
                style={{
                  width: 30,
                  height: 30,
                  borderRadius: 4,
                  background: viewMode === 'grid' ? (isLight ? '#FFFFFF' : '#2E2B23') : 'transparent',
                  border: 'none',
                  color: viewMode === 'grid' ? SONNA.fg : SONNA.fgDim,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  transition: 'all 0.15s ease',
                }}
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <rect x="3" y="3" width="7" height="7" />
                  <rect x="14" y="3" width="7" height="7" />
                  <rect x="14" y="14" width="7" height="7" />
                  <rect x="3" y="14" width="7" height="7" />
                </svg>
              </button>
            </div>

            {/* Coral Primary Button: + New project */}
            <button
              type="button"
              onClick={onOpenNewProject}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: 6,
                height: 34,
                padding: '0 16px',
                background: '#E05353',
                color: '#FFFFFF',
                border: 'none',
                borderRadius: 6,
                fontFamily: F,
                fontSize: 12.5,
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
          ) : viewMode === 'list' ? (
            /* Table / List View matching Images 5, 6, 7 */
            <div style={{ padding: '20px 24px', maxWidth: 1400 }}>
              <div className="saha-table-scroll">
                <div style={{
                  border: `1px solid ${SONNA.lineSoft}`,
                  borderRadius: 8,
                  overflow: 'visible',
                  background: isLight ? '#FFFFFF' : '#191B20',
                  minWidth: 820,
                }}>
                {/* Table Header */}
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'minmax(220px, 2fr) minmax(140px, 1.2fr) minmax(180px, 1.5fr) 90px 140px 70px 140px 40px',
                  alignItems: 'center',
                  padding: '12px 18px',
                  background: isLight ? '#F9FAFB' : '#14161B',
                  borderBottom: `1px solid ${SONNA.lineSoft}`,
                  borderTopLeftRadius: 7,
                  borderTopRightRadius: 7,
                  fontSize: 12,
                  fontWeight: 600,
                  color: SONNA.fgDim,
                }}>
                  <div>Project name</div>
                  <div>Profile</div>
                  <div>Status</div>
                  <div style={{ textAlign: 'center' }}>Photos</div>
                  <div>Date</div>
                  <div style={{ textAlign: 'center' }}>Cloud</div>
                  <div>Next step</div>
                  <div />
                </div>

                {/* Table Rows */}
                {activeProjects.map((p) => {
                  const isMenuOpen = cardMenuOpenId === p.id;
                  const isUploading = p.status === 'uploading';
                  const isEditing = p.status === 'editing' || p.status === 'processing';
                  const isReady = p.status === 'complete' || p.status === 'ready' || p.status === 'fine_tune';

                  return (
                    <div
                      key={p.id}
                      style={{
                        display: 'grid',
                        gridTemplateColumns: 'minmax(220px, 2fr) minmax(140px, 1.2fr) minmax(180px, 1.5fr) 90px 140px 70px 140px 40px',
                        alignItems: 'center',
                        padding: '12px 18px',
                        borderBottom: `1px solid ${SONNA.lineSoft}`,
                        fontSize: 12.5,
                        color: SONNA.fg,
                        cursor: 'default',
                        transition: 'background 0.12s ease',
                        position: 'relative',
                      }}
                      onMouseEnter={(e) => { e.currentTarget.style.background = isLight ? '#F3F4F6' : '#20232A'; }}
                      onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; }}
                    >
                      {/* Project Name & Thumbnail */}
                      <div style={{ display: 'flex', alignItems: 'center', gap: 12, minWidth: 0 }}>
                        <div style={{
                          width: 38,
                          height: 38,
                          borderRadius: 6,
                          background: isLight ? '#E5E7EB' : '#2A2E36',
                          border: `1px solid ${SONNA.lineSoft}`,
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          flexShrink: 0,
                          overflow: 'hidden',
                        }}>
                          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke={SONNA.fgFaint} strokeWidth="1.6">
                            <rect x="3" y="3" width="18" height="18" rx="2" />
                            <circle cx="8.5" cy="8.5" r="1.5" />
                            <polyline points="21 15 16 10 5 21" />
                          </svg>
                        </div>
                        <span style={{
                          fontWeight: 600,
                          color: SONNA.fg,
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                        }}>
                          {p.name}
                        </span>
                      </div>

                      {/* Profile Name */}
                      <div style={{
                        fontSize: 12,
                        fontWeight: 700,
                        letterSpacing: 0.3,
                        color: isLight ? '#374151' : '#CBD5E1',
                        textTransform: 'uppercase',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                      }}>
                        {p.profile || 'WARM SKIN TONES'}
                      </div>

                      {/* Status Column with Live States */}
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        {isUploading ? (
                          <>
                            <span style={{
                              width: 7,
                              height: 7,
                              borderRadius: '50%',
                              background: '#3B82F6',
                              boxShadow: '0 0 8px rgba(59, 130, 246, 0.7)',
                              animation: 'pulseDot 1.6s ease-in-out infinite',
                              display: 'inline-block',
                            }} />
                            <span style={{ color: '#60A5FA', fontSize: 12, fontWeight: 500 }}>
                              Uploading photos · {p.progress || 0}%
                            </span>
                          </>
                        ) : isEditing ? (
                          <div style={{ display: 'flex', alignItems: 'center', gap: 8, width: '100%', maxWidth: 160 }}>
                            <span style={{
                              width: 7,
                              height: 7,
                              borderRadius: '50%',
                              background: '#3B82F6',
                              boxShadow: '0 0 8px rgba(59, 130, 246, 0.7)',
                              animation: 'pulseDot 1.6s ease-in-out infinite',
                              display: 'inline-block',
                              flexShrink: 0,
                            }} />
                            <span style={{ color: '#60A5FA', fontSize: 12, fontWeight: 500, flexShrink: 0 }}>
                              Editing
                            </span>
                            <div style={{
                              flex: 1,
                              height: 4,
                              background: 'rgba(255, 255, 255, 0.1)',
                              borderRadius: 2,
                              overflow: 'hidden',
                            }}>
                              <div style={{
                                height: '100%',
                                background: '#3B82F6',
                                borderRadius: 2,
                                animation: 'editProgressAnim 3s ease-in-out infinite',
                              }} />
                            </div>
                          </div>
                        ) : (
                          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                            <span style={{
                              width: 7,
                              height: 7,
                              borderRadius: '50%',
                              background: '#22C55E',
                              boxShadow: '0 0 8px rgba(34, 197, 94, 0.7)',
                              display: 'inline-block',
                            }} />
                            <span style={{ color: '#4ADE80', fontSize: 12, fontWeight: 500 }}>
                              Ready for review
                            </span>
                          </div>
                        )}
                      </div>

                      {/* Photos Count */}
                      <div style={{ textAlign: 'center', ...Tnum, color: SONNA.fgDim }}>
                        {p.photos}
                      </div>

                      {/* Date */}
                      <div style={{ color: SONNA.fgDim, fontSize: 12 }}>
                        {p.date}
                      </div>

                      {/* Cloud Indicator */}
                      <div style={{ textAlign: 'center', color: SONNA.fgFaint }}>
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M17.5 19H9a7 7 0 1 1 6.71-9h1.79a4.5 4.5 0 1 1 0 9Z" />
                        </svg>
                      </div>

                      {/* Next Step Column with Review edits button & Adobe dropdown */}
                      <div>
                        {isReady ? (
                          p.reviewMode === 'adobe' ? (
                            <button
                              type="button"
                              onClick={(e) => {
                                e.stopPropagation();
                                handleStartDownloadEdits(p.raw || p);
                              }}
                              style={{
                                display: 'inline-flex',
                                alignItems: 'center',
                                padding: '5px 14px',
                                background: '#E05353',
                                color: '#FFFFFF',
                                border: 'none',
                                borderRadius: 4,
                                fontFamily: F,
                                fontSize: 11.5,
                                fontWeight: 700,
                                cursor: 'pointer',
                                boxShadow: '0 2px 8px rgba(224, 83, 83, 0.35)',
                                transition: 'background 0.15s ease',
                              }}
                              onMouseEnter={(e) => { e.currentTarget.style.background = '#EC5A5A'; }}
                              onMouseLeave={(e) => { e.currentTarget.style.background = '#E05353'; }}
                            >
                              Review edits in Adobe
                            </button>
                          ) : (
                            <div style={{ position: 'relative', display: 'inline-block' }}>
                              <button
                                type="button"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setCardMenuOpenId(null);
                                  setReviewDropdownOpenId(reviewDropdownOpenId === p.id ? null : p.id);
                                }}
                                style={{
                                  display: 'inline-flex',
                                  alignItems: 'center',
                                  gap: 6,
                                  padding: '5px 12px',
                                  background: '#E05353',
                                  color: '#FFFFFF',
                                  border: 'none',
                                  borderRadius: 4,
                                  fontFamily: F,
                                  fontSize: 11.5,
                                  fontWeight: 700,
                                  cursor: 'pointer',
                                  boxShadow: '0 2px 8px rgba(224, 83, 83, 0.35)',
                                  transition: 'background 0.15s ease',
                                }}
                                onMouseEnter={(e) => { e.currentTarget.style.background = '#EC5A5A'; }}
                                onMouseLeave={(e) => { e.currentTarget.style.background = '#E05353'; }}
                              >
                                <span>Review edits</span>
                                <span style={{ fontSize: 9 }}>▼</span>
                              </button>

                              {reviewDropdownOpenId === p.id && (
                                <div
                                  onClick={(e) => e.stopPropagation()}
                                  style={{
                                    position: 'absolute',
                                    top: 'calc(100% + 4px)',
                                    right: 0,
                                    zIndex: 100,
                                    minWidth: 165,
                                    background: isLight ? '#FFFFFF' : '#1F2228',
                                    border: `1px solid ${isLight ? 'rgba(0,0,0,0.14)' : 'rgba(255,255,255,0.14)'}`,
                                    borderRadius: 6,
                                    boxShadow: isLight ? '0 10px 25px rgba(0,0,0,0.15)' : '0 12px 36px rgba(0,0,0,0.7)',
                                    overflow: 'hidden',
                                    padding: '4px 0',
                                    animation: 'saha-popover-in 0.12s ease-out',
                                  }}
                                >
                                  {/* Review in Adobe -> Download edits first, then "Where do you want to review?" (LrC) */}
                                  <button
                                    type="button"
                                    onClick={() => {
                                      setReviewDropdownOpenId(null);
                                      handleStartDownloadEdits(p.raw || p);
                                    }}
                                    style={{
                                      width: '100%',
                                      padding: '8px 14px',
                                      display: 'flex',
                                      alignItems: 'center',
                                      gap: 9,
                                      background: 'transparent',
                                      border: 'none',
                                      color: isLight ? '#1F2937' : '#F3F4F6',
                                      fontFamily: F,
                                      fontSize: 12,
                                      fontWeight: 500,
                                      textAlign: 'left',
                                      cursor: 'pointer',
                                      transition: 'background 0.1s ease',
                                    }}
                                    onMouseEnter={(e) => { e.currentTarget.style.background = isLight ? '#F3F4F6' : '#2A2E36'; }}
                                    onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; }}
                                  >
                                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                                      <polyline points="7 10 12 15 17 10" />
                                      <line x1="12" y1="15" x2="12" y2="3" />
                                    </svg>
                                    <span>Review in Adobe</span>
                                  </button>

                                  {/* Review in Saha with Eye Icon matching Image 1 */}
                                  <button
                                    type="button"
                                    onClick={() => {
                                      setReviewDropdownOpenId(null);
                                      onOpenProject?.(p.raw || p);
                                    }}
                                    style={{
                                      width: '100%',
                                      padding: '8px 14px',
                                      display: 'flex',
                                      alignItems: 'center',
                                      gap: 9,
                                      background: 'transparent',
                                      border: 'none',
                                      color: isLight ? '#1F2937' : '#F3F4F6',
                                      fontFamily: F,
                                      fontSize: 12,
                                      fontWeight: 500,
                                      textAlign: 'left',
                                      cursor: 'pointer',
                                      transition: 'background 0.1s ease',
                                    }}
                                    onMouseEnter={(e) => { e.currentTarget.style.background = isLight ? '#F3F4F6' : '#2A2E36'; }}
                                    onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; }}
                                  >
                                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                      <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
                                      <circle cx="12" cy="12" r="3" />
                                    </svg>
                                    <span>Review in Saha</span>
                                  </button>
                                </div>
                              )}
                            </div>
                          )
                        ) : (
                          <span style={{ fontSize: 11.5, color: SONNA.fgFaint }}>
                            In progress
                          </span>
                        )}
                      </div>

                      {/* ⋮ Action Menu */}
                      <div>
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            setReviewDropdownOpenId(null);
                            setCardMenuOpenId(isMenuOpen ? null : p.id);
                          }}
                          style={{
                            background: 'transparent',
                            border: 'none',
                            color: SONNA.fgDim,
                            fontSize: 16,
                            cursor: 'pointer',
                            padding: 4,
                            borderRadius: 4,
                          }}
                        >
                          ⋮
                        </button>

                        {isMenuOpen && (
                          <div
                            onClick={(e) => e.stopPropagation()}
                            style={{
                              position: 'absolute',
                              top: 40,
                              right: 18,
                              zIndex: 50,
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
                    </div>
                  );
                })}
                </div>
              </div>
            </div>
          ) : (
            /* Active Projects Grid */
            <div style={{ padding: '24px 28px', maxWidth: 1280 }}>
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
                  const isUploading = p.status === 'uploading';
                  const isEditing = p.status === 'editing' || p.status === 'processing';
                  const isReady = p.status === 'complete' || p.status === 'ready' || p.status === 'fine_tune';

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
                            setReviewDropdownOpenId(null);
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
                          color: (isUploading || isEditing) ? '#60A5FA' : '#4ADE80',
                        }}>
                          <span style={{
                            width: 6,
                            height: 6,
                            borderRadius: '50%',
                            background: (isUploading || isEditing) ? '#3B82F6' : '#22C55E',
                            boxShadow: (isUploading || isEditing) ? '0 0 6px rgba(59, 130, 246, 0.6)' : 'none',
                            animation: (isUploading || isEditing) ? 'pulseDot 1.8s ease-in-out infinite' : 'none',
                            display: 'inline-block',
                          }} />
                          <span>
                            {isUploading ? `Uploading · ${p.progress || 0}%` : (isEditing ? 'Editing' : 'Ready for review')}
                          </span>
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
            <div style={{ padding: '24px 28px', maxWidth: 1280 }}>
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

      {/* ── Review in Adobe Modal ("Where do you want to review the photos?" -> LrC only) ── */}
      {reviewAdobeProject && (
        <div
          onClick={() => setReviewAdobeProject(null)}
          style={{
            position: 'fixed',
            inset: 0,
            zIndex: 2000,
            background: 'rgba(0, 0, 0, 0.72)',
            backdropFilter: 'blur(3px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: 20,
          }}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            style={{
              width: '100%',
              maxWidth: 'min(480px, 94vw)',
              background: isLight ? '#FFFFFF' : '#23262B',
              border: `1px solid ${isLight ? 'rgba(0,0,0,0.12)' : 'rgba(255,255,255,0.12)'}`,
              borderRadius: 12,
              boxShadow: isLight ? '0 20px 48px rgba(0,0,0,0.2)' : '0 24px 64px rgba(0,0,0,0.85)',
              overflow: 'hidden',
              fontFamily: F,
              color: isLight ? '#1F2937' : '#F3F4F6',
              animation: 'saha-modal-in 0.15s ease-out',
            }}
          >
            {/* Header */}
            <div style={{
              padding: '18px 22px 14px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
            }}>
              <div style={{ fontSize: 16, fontWeight: 700, letterSpacing: -0.1 }}>
                Where do you want to review the photos?
              </div>
              <button
                type="button"
                onClick={() => setReviewAdobeProject(null)}
                style={{
                  background: 'transparent',
                  border: 'none',
                  color: isLight ? '#6B7280' : '#9CA3AF',
                  fontSize: 16,
                  cursor: 'pointer',
                  padding: 4,
                  lineHeight: 1,
                }}
              >
                ✕
              </button>
            </div>

            {/* Body: Single Lightroom Classic (LrC) Option */}
            <div style={{ padding: '4px 22px 22px', display: 'flex', flexDirection: 'column', gap: 12 }}>
              <button
                type="button"
                onClick={handleLaunchLightroomClassic}
                style={{
                  width: '100%',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 14,
                  padding: '14px 16px',
                  borderRadius: 8,
                  background: isLight ? '#F9FAFB' : '#2C3037',
                  border: `1px solid ${isLight ? 'rgba(0,0,0,0.12)' : 'rgba(255,255,255,0.14)'}`,
                  cursor: 'pointer',
                  textAlign: 'left',
                  transition: 'background 0.15s ease, border-color 0.15s ease',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = isLight ? '#F3F4F6' : '#353A42';
                  e.currentTarget.style.borderColor = isLight ? 'rgba(0,0,0,0.22)' : 'rgba(255,255,255,0.28)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = isLight ? '#F9FAFB' : '#2C3037';
                  e.currentTarget.style.borderColor = isLight ? 'rgba(0,0,0,0.12)' : 'rgba(255,255,255,0.14)';
                }}
              >
                {/* LrC Badge */}
                <div style={{
                  width: 34,
                  height: 34,
                  borderRadius: 6,
                  background: '#001E36',
                  border: '1.5px solid #31A8FF',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: '#31A8FF',
                  fontSize: 13,
                  fontWeight: 800,
                  letterSpacing: -0.3,
                  flexShrink: 0,
                }}>
                  LrC
                </div>
                <div style={{ minWidth: 0, flex: 1 }}>
                  <div style={{ fontSize: 14.5, fontWeight: 700, color: isLight ? '#111827' : '#FFFFFF' }}>
                    Lightroom Classic
                  </div>
                  <div style={{
                    fontSize: 12,
                    color: isLight ? '#6B7280' : '#9CA3AF',
                    marginTop: 2,
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                  }}>
                    {reviewAdobeProject.path && reviewAdobeProject.path.toLowerCase().endsWith('.lrcat')
                      ? `Last opened: ${reviewAdobeProject.path.split(/[\\/]/).filter(Boolean).pop()}`
                      : `Open ${reviewAdobeProject.name || 'project'} in Lightroom Classic`}
                  </div>
                </div>
              </button>

              {reviewAdobeError && (
                <div style={{
                  padding: '8px 12px',
                  borderRadius: 6,
                  background: 'rgba(224, 83, 83, 0.14)',
                  border: '1px solid rgba(224, 83, 83, 0.45)',
                  color: '#FCA5A5',
                  fontSize: 12,
                }}>
                  {reviewAdobeError}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ── Modal 1: Download edit metadata Modal (matching reference, without Filters) ─────────────── */}
      {downloadEditsProject && !isDownloading && !isDownloadComplete && (() => {
        const activeItems = downloadSelectionTab === 'folders' ? downloadCatalogFolders : downloadCatalogCollections;
        const allChecked = activeItems.length > 0 && activeItems.every((item) => downloadSelectedIds.has(item.id));
        const someChecked = activeItems.some((item) => downloadSelectedIds.has(item.id));
        const selectedPhotoCount = activeItems
          .filter((item) => downloadSelectedIds.has(item.id))
          .reduce((sum, item) => sum + (Number(item.totalPhotos ?? item.count) || 0), 0);

        return (
        <div style={{
          position: 'fixed',
          inset: 0,
          zIndex: 2000,
          background: 'rgba(0, 0, 0, 0.75)',
          backdropFilter: 'blur(3px)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: 20,
        }}>
          <div style={{
            width: '100%',
            maxWidth: 'min(700px, 94vw)',
            maxHeight: '90vh',
            background: isLight ? '#FFFFFF' : '#1E2025',
            border: `1px solid ${isLight ? 'rgba(0,0,0,0.12)' : 'rgba(255,255,255,0.12)'}`,
            borderRadius: 10,
            boxShadow: isLight ? '0 20px 48px rgba(0,0,0,0.2)' : '0 24px 64px rgba(0,0,0,0.85)',
            overflow: 'hidden',
            display: 'flex',
            flexDirection: 'column',
            fontFamily: F,
            color: isLight ? '#1F2937' : '#F3F4F6',
            animation: 'saha-modal-in 0.15s ease-out',
          }}>
            {/* Header */}
            <div style={{
              padding: '18px 24px 14px',
              display: 'flex',
              alignItems: 'flex-start',
              justifyContent: 'space-between',
              borderBottom: `1px solid ${isLight ? 'rgba(0,0,0,0.08)' : 'rgba(255,255,255,0.08)'}`,
            }}>
              <div>
                <div style={{ fontSize: 16, fontWeight: 700, letterSpacing: -0.1 }}>
                  Download edit metadata
                </div>
                <div style={{ fontSize: 12, color: isLight ? '#6B7280' : '#9CA3AF', marginTop: 3 }}>
                  {(downloadPath || '').toLowerCase().endsWith('.lrcat')
                    ? 'Download the edit metadata to the catalog where your photos are'
                    : 'Download the edit metadata to the folder where your photos are'}
                </div>
              </div>
              <button
                type="button"
                onClick={() => setDownloadEditsProject(null)}
                style={{
                  background: 'transparent',
                  border: 'none',
                  color: isLight ? '#6B7280' : '#9CA3AF',
                  fontSize: 16,
                  cursor: 'pointer',
                  padding: 4,
                  lineHeight: 1,
                }}
              >
                ✕
              </button>
            </div>

            {/* Body */}
            <div style={{ padding: '18px 24px', display: 'flex', flexDirection: 'column', gap: 16, overflowY: 'auto' }}>
              {/* Project Name */}
              <div>
                <div style={{ fontSize: 12.5, fontWeight: 600, color: isLight ? '#111827' : '#FFFFFF', marginBottom: 4 }}>
                  Project name
                </div>
                <div style={{ fontSize: 13, color: isLight ? '#374151' : '#D1D5DB' }}>
                  {downloadEditsProject.name}
                </div>
              </div>

              {/* Lightroom Catalog / Folder Location */}
              <div>
                <div style={{ fontSize: 12.5, fontWeight: 600, color: isLight ? '#111827' : '#FFFFFF', marginBottom: 8 }}>
                  {(downloadPath || '').toLowerCase().endsWith('.lrcat') ? 'Lightroom catalog' : 'Photos folder'}
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <div style={{
                    flex: 1,
                    display: 'flex',
                    alignItems: 'center',
                    background: isLight ? '#F9FAFB' : '#14161B',
                    border: `1px solid ${isLight ? 'rgba(0,0,0,0.14)' : 'rgba(255,255,255,0.14)'}`,
                    borderRadius: 6,
                    padding: '7px 12px',
                  }}>
                    <input
                      type="text"
                      value={downloadPath}
                      onChange={(e) => setDownloadPath(e.target.value)}
                      style={{
                        flex: 1,
                        background: 'transparent',
                        border: 'none',
                        outline: 'none',
                        color: isLight ? '#111827' : '#E5E7EB',
                        fontFamily: M,
                        fontSize: 12,
                      }}
                    />
                  </div>
                  <button
                    type="button"
                    onClick={handleBrowseDownloadPath}
                    style={{
                      background: 'transparent',
                      border: 'none',
                      color: isLight ? '#111827' : '#FFFFFF',
                      fontFamily: F,
                      fontSize: 12.5,
                      fontWeight: 600,
                      cursor: 'pointer',
                      padding: '6px 10px',
                      borderRadius: 4,
                    }}
                    onMouseEnter={(e) => { e.currentTarget.style.background = isLight ? 'rgba(0,0,0,0.06)' : 'rgba(255,255,255,0.08)'; }}
                    onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; }}
                  >
                    Browse
                  </button>
                </div>
              </div>

              {/* Selection Section (No Filters section per user request) */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                <button
                  type="button"
                  onClick={() => setDownloadSelectionOpen((v) => !v)}
                  style={{
                    width: '100%',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 8,
                    padding: '9px 14px',
                    borderRadius: 6,
                    background: isLight ? '#F3F4F6' : '#262930',
                    border: `1px solid ${isLight ? 'rgba(0,0,0,0.08)' : 'rgba(255,255,255,0.08)'}`,
                    color: isLight ? '#111827' : '#E5E7EB',
                    fontFamily: F,
                    fontSize: 12.5,
                    fontWeight: 600,
                    cursor: 'pointer',
                    textAlign: 'left',
                  }}
                >
                  <span style={{ fontSize: 11, opacity: 0.8 }}>{downloadSelectionOpen ? '⌄' : '›'}</span>
                  <span>Selection</span>
                </button>

                {downloadSelectionOpen && (
                  <>
                    {/* Show catalog content by Folders | Collections */}
                    {(downloadPath || '').toLowerCase().endsWith('.lrcat') && (
                      <div style={{ display: 'flex', alignItems: 'center', gap: 24, fontSize: 12.5 }}>
                        <span style={{ fontWeight: 600, color: isLight ? '#111827' : '#E5E7EB' }}>
                          Show catalog content by
                        </span>
                        <label style={{ display: 'flex', alignItems: 'center', gap: 7, cursor: 'pointer' }}>
                          <input
                            type="radio"
                            name="download_selection_tab"
                            checked={downloadSelectionTab === 'folders'}
                            onChange={() => setDownloadSelectionTab('folders')}
                            style={{ accentColor: '#E05353', cursor: 'pointer' }}
                          />
                          <span>Folders</span>
                        </label>
                        <label style={{ display: 'flex', alignItems: 'center', gap: 7, cursor: 'pointer' }}>
                          <input
                            type="radio"
                            name="download_selection_tab"
                            checked={downloadSelectionTab === 'collections'}
                            onChange={() => setDownloadSelectionTab('collections')}
                            style={{ accentColor: '#E05353', cursor: 'pointer' }}
                          />
                          <span>Collections</span>
                        </label>
                      </div>
                    )}

                    {/* Folders / Collections Table */}
                    <div className="saha-table-scroll">
                      <div style={{
                        borderRadius: 6,
                        border: `1px solid ${isLight ? 'rgba(0,0,0,0.08)' : 'rgba(255,255,255,0.08)'}`,
                        overflow: 'hidden',
                        minWidth: 520,
                      }}>
                        <div style={{
                          display: 'grid',
                          gridTemplateColumns: '38px 1fr 90px 100px 105px',
                          alignItems: 'center',
                          padding: '9px 12px',
                          background: isLight ? '#F9FAFB' : '#181A1F',
                          borderBottom: `1px solid ${isLight ? 'rgba(0,0,0,0.08)' : 'rgba(255,255,255,0.08)'}`,
                          fontSize: 11.5,
                          fontWeight: 600,
                          color: isLight ? '#6B7280' : '#9CA3AF',
                        }}>
                          <input
                            type="checkbox"
                            checked={allChecked}
                            ref={(el) => { if (el) el.indeterminate = !allChecked && someChecked; }}
                            onChange={(e) => {
                              if (e.target.checked) {
                                setDownloadSelectedIds(new Set(activeItems.map((i) => i.id)));
                              } else {
                                setDownloadSelectedIds(new Set());
                              }
                            }}
                            style={{ accentColor: '#E05353', cursor: 'pointer' }}
                          />
                          <span>{downloadSelectionTab === 'folders' ? 'Folder name & path' : 'Collection name'}</span>
                          <span>Total Photos</span>
                          <span>Filtered Photos</span>
                          <span>Project date</span>
                        </div>

                        {downloadScanning ? (
                          <div style={{ padding: '16px 14px', fontSize: 12, color: isLight ? '#6B7280' : '#9CA3AF' }}>
                            Scanning catalog...
                          </div>
                        ) : activeItems.length === 0 ? (
                          <div style={{ padding: '16px 14px', fontSize: 12, color: isLight ? '#6B7280' : '#9CA3AF' }}>
                            No {downloadSelectionTab} found.
                          </div>
                        ) : (
                          <div style={{ maxHeight: 190, overflowY: 'auto' }}>
                            {activeItems.map((item) => {
                              const checked = downloadSelectedIds.has(item.id);
                              return (
                                <div
                                  key={item.id}
                                  onClick={() => {
                                    setDownloadSelectedIds((prev) => {
                                      const next = new Set(prev);
                                      if (next.has(item.id)) next.delete(item.id);
                                      else next.add(item.id);
                                      return next;
                                    });
                                  }}
                                  style={{
                                    display: 'grid',
                                    gridTemplateColumns: '38px 1fr 90px 100px 105px',
                                    alignItems: 'center',
                                    padding: '9px 12px',
                                    borderBottom: `1px solid ${isLight ? 'rgba(0,0,0,0.06)' : 'rgba(255,255,255,0.06)'}`,
                                    background: checked ? (isLight ? 'rgba(224, 83, 83, 0.06)' : 'rgba(255,255,255,0.04)') : 'transparent',
                                    fontSize: 12,
                                    cursor: 'pointer',
                                  }}
                                >
                                  <input
                                    type="checkbox"
                                    checked={checked}
                                    onChange={() => {}}
                                    style={{ accentColor: '#E05353', cursor: 'pointer' }}
                                  />
                                  <span style={{
                                    whiteSpace: 'nowrap',
                                    overflow: 'hidden',
                                    textOverflow: 'ellipsis',
                                    paddingRight: 10,
                                    color: isLight ? '#111827' : '#E5E7EB',
                                  }}>
                                    {item.path || item.name}
                                  </span>
                                  <span style={Tnum}>{item.totalPhotos ?? item.count ?? 0}</span>
                                  <span style={Tnum}>{item.filteredPhotos ?? item.totalPhotos ?? item.count ?? 0}</span>
                                  <span style={{ color: isLight ? '#6B7280' : '#9CA3AF' }}>{item.date}</span>
                                </div>
                              );
                            })}
                          </div>
                        )}
                      </div>
                    </div>
                  </>
                )}
              </div>
            </div>

            {/* Footer */}
            <div style={{
              padding: '14px 24px',
              borderTop: `1px solid ${isLight ? 'rgba(0,0,0,0.08)' : 'rgba(255,255,255,0.08)'}`,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: 12,
            }}>
              <div style={{ fontSize: 12, fontWeight: 600, color: isLight ? '#111827' : '#E5E7EB' }}>
                {selectedPhotoCount || downloadEditsProject.photos || 0} photos selected
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <button
                  type="button"
                  onClick={() => setDownloadEditsProject(null)}
                  style={{
                    background: 'transparent',
                    border: 'none',
                    color: isLight ? '#4B5563' : '#D1D5DB',
                    fontFamily: F,
                    fontSize: 12.5,
                    fontWeight: 600,
                    cursor: 'pointer',
                    padding: '8px 14px',
                  }}
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={handleConfirmDownloadEdits}
                  style={{
                    padding: '9px 22px',
                    background: '#E05353',
                    color: '#FFFFFF',
                    border: 'none',
                    borderRadius: 6,
                    fontFamily: F,
                    fontSize: 12.5,
                    fontWeight: 700,
                    cursor: 'pointer',
                    boxShadow: '0 4px 14px rgba(224, 83, 83, 0.35)',
                    transition: 'background 0.15s ease',
                  }}
                  onMouseEnter={(e) => { e.currentTarget.style.background = '#EC5A5A'; }}
                  onMouseLeave={(e) => { e.currentTarget.style.background = '#E05353'; }}
                >
                  Download edits
                </button>
              </div>
            </div>
          </div>
        </div>
        );
      })()}

      {/* ── Note Alert Overlay ("Downloading the edits will overwrite...") ── */}
      {downloadEditsProject && showDownloadOverwriteNote && (
        <div
          onClick={() => setShowDownloadOverwriteNote(false)}
          style={{
            position: 'fixed',
            inset: 0,
            zIndex: 2100,
            background: 'rgba(0, 0, 0, 0.55)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: 20,
          }}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            style={{
              width: '100%',
              maxWidth: 'min(420px, 92vw)',
              background: isLight ? '#FFFFFF' : '#23262B',
              border: `1px solid ${isLight ? 'rgba(0,0,0,0.12)' : 'rgba(255,255,255,0.14)'}`,
              borderRadius: 10,
              boxShadow: isLight ? '0 20px 48px rgba(0,0,0,0.2)' : '0 24px 64px rgba(0,0,0,0.85)',
              padding: '22px 24px 20px',
              fontFamily: F,
              color: isLight ? '#1F2937' : '#F3F4F6',
              animation: 'saha-modal-in 0.14s ease-out',
            }}
          >
            <div style={{ fontSize: 16, fontWeight: 700, color: isLight ? '#111827' : '#FFFFFF', marginBottom: 12 }}>
              Note
            </div>
            <div style={{ fontSize: 13, lineHeight: 1.5, color: isLight ? '#4B5563' : '#D1D5DB', marginBottom: 22 }}>
              Downloading the edits will overwrite any previous edit of the photos submitted for editing.
            </div>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12 }}>
              <label style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                fontSize: 12.5,
                color: isLight ? '#374151' : '#E5E7EB',
                cursor: 'pointer',
              }}>
                <input
                  type="checkbox"
                  checked={dontShowOverwriteNoteAgain}
                  onChange={(e) => setDontShowOverwriteNoteAgain(e.target.checked)}
                  style={{ accentColor: '#E05353', cursor: 'pointer', width: 15, height: 15 }}
                />
                <span>Don&apos;t show this again</span>
              </label>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <button
                  type="button"
                  onClick={() => setShowDownloadOverwriteNote(false)}
                  style={{
                    background: 'transparent',
                    border: 'none',
                    color: isLight ? '#4B5563' : '#E5E7EB',
                    fontFamily: F,
                    fontSize: 12.5,
                    fontWeight: 600,
                    cursor: 'pointer',
                    padding: '7px 12px',
                  }}
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={handleContinueAfterOverwriteNote}
                  style={{
                    padding: '8px 20px',
                    background: '#E05353',
                    color: '#FFFFFF',
                    border: 'none',
                    borderRadius: 6,
                    fontFamily: F,
                    fontSize: 12.5,
                    fontWeight: 700,
                    cursor: 'pointer',
                    boxShadow: '0 4px 14px rgba(224, 83, 83, 0.35)',
                  }}
                  onMouseEnter={(e) => { e.currentTarget.style.background = '#EC5A5A'; }}
                  onMouseLeave={(e) => { e.currentTarget.style.background = '#E05353'; }}
                >
                  Continue
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Modal 2: Downloading edit metadata Modal matching Image 3 & Image 4 ── */}
      {downloadEditsProject && (isDownloading || isDownloadComplete) && (
        <div style={{
          position: 'fixed',
          inset: 0,
          zIndex: 2000,
          background: 'rgba(0, 0, 0, 0.75)',
          backdropFilter: 'blur(3px)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: 20,
        }}>
          <div style={{
            width: '100%',
            maxWidth: 'min(560px, 94vw)',
            maxHeight: '90vh',
            background: isLight ? '#FFFFFF' : '#1C1F24',
            border: `1px solid ${isLight ? 'rgba(0,0,0,0.12)' : 'rgba(255,255,255,0.12)'}`,
            borderRadius: 10,
            boxShadow: isLight ? '0 20px 48px rgba(0,0,0,0.2)' : '0 24px 64px rgba(0,0,0,0.85)',
            overflow: 'hidden',
            display: 'flex',
            flexDirection: 'column',
            fontFamily: F,
            color: isLight ? '#1F2937' : '#F3F4F6',
            animation: 'saha-modal-in 0.15s ease-out',
          }}>
            {/* Header */}
            <div style={{
              padding: '16px 22px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              borderBottom: `1px solid ${isLight ? 'rgba(0,0,0,0.08)' : 'rgba(255,255,255,0.08)'}`,
            }}>
              <div style={{ fontSize: 15, fontWeight: 700, letterSpacing: -0.1 }}>
                Downloading edit metadata
              </div>
              {isDownloadComplete && (
                <button
                  type="button"
                  onClick={() => setDownloadEditsProject(null)}
                  style={{
                    background: 'transparent',
                    border: 'none',
                    color: isLight ? '#6B7280' : '#9CA3AF',
                    fontSize: 16,
                    cursor: 'pointer',
                    padding: 4,
                    lineHeight: 1,
                  }}
                >
                  ✕
                </button>
              )}
            </div>

            {/* Body */}
            <div style={{ padding: '20px 24px', display: 'flex', flexDirection: 'column', gap: 14 }}>
              {isDownloading ? (
                /* In Progress State matching Image 3 */
                <>
                  <div style={{
                    background: isLight ? '#F9FAFB' : '#23272E',
                    border: `1px solid ${isLight ? 'rgba(0,0,0,0.08)' : 'rgba(255,255,255,0.08)'}`,
                    borderRadius: 8,
                    padding: '16px 20px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    gap: 16,
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                      <span style={{
                        display: 'inline-block',
                        width: 14,
                        height: 14,
                        borderRadius: '50%',
                        border: '2px solid rgba(16, 185, 129, 0.3)',
                        borderTopColor: '#10B981',
                        animation: 'saha-spin 0.8s linear infinite',
                      }} />
                      <div>
                        <div style={{ fontSize: 13, fontWeight: 600, color: isLight ? '#111827' : '#FFFFFF' }}>
                          {downloadEditsProject.name}
                        </div>
                        <div style={{ fontSize: 11, color: isLight ? '#6B7280' : '#9CA3AF', marginTop: 2 }}>
                          Reading metadata. This might take a while.
                        </div>
                      </div>
                    </div>

                    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                      <div style={{
                        width: 140,
                        height: 4,
                        background: isLight ? 'rgba(0,0,0,0.1)' : 'rgba(255,255,255,0.1)',
                        borderRadius: 2,
                        overflow: 'hidden',
                      }}>
                        <div style={{
                          width: `${downloadProgress}%`,
                          height: '100%',
                          background: '#10B981',
                          borderRadius: 2,
                          transition: 'width 0.15s ease',
                        }} />
                      </div>
                      <span style={{ ...Tnum, fontSize: 11.5, color: isLight ? '#4B5563' : '#9CA3AF', minWidth: 42, textAlign: 'right' }}>
                        {downloadProgress.toFixed(1)}%
                      </span>
                    </div>
                  </div>

                  <div style={{ fontSize: 11.5, color: isLight ? '#6B7280' : '#9CA3AF', marginTop: 2 }}>
                    Downloading can take time. Keep the app open for now.
                  </div>
                </>
              ) : (
                /* Completed State matching Image 4 */
                <>
                  {!isBannerDismissed && (
                    <div style={{
                      background: isLight ? 'rgba(209, 250, 229, 0.7)' : 'rgba(6, 78, 59, 0.45)',
                      border: `1px solid ${isLight ? 'rgba(52, 211, 153, 0.7)' : 'rgba(16, 185, 129, 0.5)'}`,
                      borderRadius: 6,
                      padding: '10px 14px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                    }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12.5, color: isLight ? '#065F46' : '#34D399', fontWeight: 600 }}>
                        <span style={{ fontSize: 14 }}>✓</span>
                        <span>Download Complete! {downloadEditsProject.photos || 21} photos were downloaded.</span>
                      </div>
                      <button
                        type="button"
                        onClick={() => setIsBannerDismissed(true)}
                        style={{
                          background: 'transparent',
                          border: 'none',
                          color: isLight ? '#065F46' : '#34D399',
                          cursor: 'pointer',
                          padding: 2,
                          fontSize: 12,
                        }}
                      >
                        ✕
                      </button>
                    </div>
                  )}

                  <div style={{
                    background: isLight ? '#F9FAFB' : '#23272E',
                    border: `1px solid ${isLight ? 'rgba(0,0,0,0.08)' : 'rgba(255,255,255,0.08)'}`,
                    borderRadius: 8,
                    padding: '14px 18px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 12,
                  }}>
                    <span style={{
                      width: 20,
                      height: 20,
                      borderRadius: '50%',
                      background: '#10B981',
                      color: '#FFFFFF',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: 12,
                      fontWeight: 800,
                      flexShrink: 0,
                    }}>
                      ✓
                    </span>
                    <div>
                      <div style={{ fontSize: 13, fontWeight: 600, color: isLight ? '#111827' : '#FFFFFF' }}>
                        {downloadEditsProject.name}
                      </div>
                      <div style={{ fontSize: 11, color: isLight ? '#6B7280' : '#9CA3AF', marginTop: 2 }}>
                        {downloadEditsProject.photos || 21} photos
                      </div>
                    </div>
                  </div>
                </>
              )}
            </div>

            {/* Footer (Complete state buttons: Close + Open LR catalog) */}
            {isDownloadComplete && (
              <div style={{
                padding: '14px 24px',
                borderTop: `1px solid ${isLight ? 'rgba(0,0,0,0.08)' : 'rgba(255,255,255,0.08)'}`,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'flex-end',
                gap: 12,
              }}>
                <button
                  type="button"
                  onClick={() => {
                    setDownloadEditsProject(null);
                    setIsDownloading(false);
                    setIsDownloadComplete(false);
                  }}
                  style={{
                    background: 'transparent',
                    border: 'none',
                    color: isLight ? '#4B5563' : '#E5E7EB',
                    fontFamily: F,
                    fontSize: 12.5,
                    fontWeight: 600,
                    cursor: 'pointer',
                    padding: '7px 14px',
                  }}
                >
                  Close
                </button>

                <button
                  type="button"
                  onClick={handleOpenLrCatalogFromComplete}
                  style={{
                    padding: '8px 18px',
                    background: '#E05353',
                    color: '#FFFFFF',
                    border: 'none',
                    borderRadius: 6,
                    fontFamily: F,
                    fontSize: 12.5,
                    fontWeight: 700,
                    cursor: 'pointer',
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: 7,
                    boxShadow: '0 4px 14px rgba(224, 83, 83, 0.35)',
                    transition: 'background 0.15s ease',
                  }}
                  onMouseEnter={(e) => { e.currentTarget.style.background = '#EC5A5A'; }}
                  onMouseLeave={(e) => { e.currentTarget.style.background = '#E05353'; }}
                >
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
                    <polyline points="15 3 21 3 21 9" />
                    <line x1="10" y1="14" x2="21" y2="3" />
                  </svg>
                  <span>Open LR catalog</span>
                </button>
              </div>
            )}
          </div>
        </div>
      )}
    </AppShell>
  );
}

export default ProjectsView;
