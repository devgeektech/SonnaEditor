// App — Modernized Imagen AI Workflow Architecture
// Screens: login → home ⇄ projects ⇄ profiles ⇄ project-detail
// Modals: ChooseProfileTypeModal, NewProjectModal, LiteProfileWizard, PersonalProfileWizard

import { Component, useCallback, useLayoutEffect, useState, useEffect } from 'react';
import { SahaLogin } from './components/login.jsx';
import { HomeView } from './components/home-view.jsx';
import { ProjectsView } from './components/projects-view.jsx';
import { ProfileView } from './components/profile-view.jsx';
import { ProjectDetailView } from './components/project-detail-view.jsx';
import { ChooseProfileTypeModal } from './components/profile-type-modal.jsx';
import { NewProjectModal } from './components/new-project-modal.jsx';
import { LiteProfileWizard } from './components/lite-wizard.jsx';
import { PersonalProfileWizard } from './components/personal-wizard.jsx';
import { useProfiles } from './hooks/useProfiles.js';
import { applyTheme } from './tokens.js';
import { scanFolder } from './api/client.js';

class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }
  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }
  componentDidCatch(error, errorInfo) {
    console.error('[saha] Component error caught by ErrorBoundary:', error, errorInfo);
  }
  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: 40, color: '#f87171', background: '#1c1917', height: '100%', fontFamily: 'sans-serif' }}>
          <h2 style={{ fontSize: 18, fontWeight: 700 }}>Something went wrong in this view</h2>
          <pre style={{ whiteSpace: 'pre-wrap', marginTop: 12, fontSize: 12.5, color: '#fca5a5' }}>
            {this.state.error?.message || String(this.state.error)}
          </pre>
          <button
            type="button"
            onClick={() => this.setState({ hasError: false, error: null })}
            style={{ marginTop: 20, padding: '8px 16px', background: '#e05353', color: '#fff', border: 'none', borderRadius: 6, cursor: 'pointer' }}
          >
            Retry View
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

export function App() {
  const [screen, setScreen] = useState('login');
  const [hasEnteredApp, setHasEnteredApp] = useState(false);
  const [theme, setTheme] = useState('dark');
  const [projectSnapshot, setProjectSnapshot] = useState({ queue: [], runResults: [] });
  const [activeProject, setActiveProject] = useState(null);
  const [projectsList, setProjectsList] = useState(() => {
    try {
      const stored = localStorage.getItem('saha_user_projects');
      if (stored) {
        const parsed = JSON.parse(stored);
        if (Array.isArray(parsed) && parsed.length > 0) return parsed;
      }
    } catch (e) {}
    return [
      {
        id: 'proj-sample-raw',
        name: 'Sample_Raw',
        path: 'D:/Sonna/Shoots/2026/Sample_Raw',
        photos: 21,
        date: 'Nov 17, 2:58 AM',
        status: 'complete',
        createdAt: Date.now() - 3600000,
      },
    ];
  });
  const [uploads, setUploads] = useState([]);
  const [isUpdating, setIsUpdating] = useState(false);
  const [updatingProgress, setUpdatingProgress] = useState(37);

  // Sync projectsList to localStorage
  useEffect(() => {
    try {
      localStorage.setItem('saha_user_projects', JSON.stringify(projectsList));
    } catch (e) {}
  }, [projectsList]);

  // Modal States
  const [isChooseProfileOpen, setIsChooseProfileOpen] = useState(false);
  const [isLiteWizardOpen, setIsLiteWizardOpen] = useState(false);
  const [isPersonalWizardOpen, setIsPersonalWizardOpen] = useState(false);
  const [isNewProjectOpen, setIsNewProjectOpen] = useState(false);
  const [newlyCreatedProfileId, setNewlyCreatedProfileId] = useState(null);

  const profilesQ = useProfiles();

  const handleProfileCreated = useCallback(async (res) => {
    const prevIds = new Set((profilesQ.profiles || []).map((p) => p.id));
    const createdId = res?.profile_id || res?.id;
    if (createdId) {
      setNewlyCreatedProfileId(createdId);
    }
    const fresh = await profilesQ.refetch();
    if (!createdId && fresh && Array.isArray(fresh)) {
      const added = fresh.find((p) => !prevIds.has(p.id));
      if (added) {
        setNewlyCreatedProfileId(added.id);
      }
    }
  }, [profilesQ]);

  useLayoutEffect(() => {
    applyTheme(theme);
    if (typeof window !== 'undefined') {
      try {
        if (theme === 'dark') {
          window.localStorage?.removeItem('saha-theme');
        }
      } catch (e) {}
    }
  }, [theme]);

  const handleNavigate = useCallback((kind) => {
    if (kind === 'home') setScreen('home');
    else if (kind === 'profiles') setScreen('profiles');
    else if (kind === 'projects') setScreen('projects');
    else if (kind === 'logout') setScreen('login');
  }, []);

  const handleToggleTheme = useCallback(() => {
    setTheme((current) => current === 'dark' ? 'light' : 'dark');
  }, []);

  const handleLogout = useCallback(() => {
    setScreen('login');
  }, []);

  const handleLogin = useCallback(() => {
    setHasEnteredApp(true);
    setScreen('home');
  }, []);

  // Pick folder helper for NewProjectModal
  const handlePickFolder = useCallback(async () => {
    if (typeof window === 'undefined' || !window.saha?.pickFolder) return null;
    try {
      const path = await window.saha.pickFolder();
      if (!path) return null;
      const scan = await scanFolder(path, 'folder');
      return { path, raw_count: scan.raw_count || 0 };
    } catch (e) {
      console.error('Failed to pick folder', e);
      return null;
    }
  }, []);

  // Pick catalog helper for NewProjectModal
  const handlePickCatalog = useCallback(async () => {
    if (typeof window === 'undefined' || !window.saha?.pickFile) return null;
    try {
      const path = await window.saha.pickFile({
        title: 'Choose a Lightroom catalog',
        filters: [{ name: 'Lightroom catalog', extensions: ['lrcat'] }],
      });
      if (!path) return null;
      const scan = await scanFolder(path, 'catalog');
      return { path, raw_count: scan.raw_count || 0 };
    } catch (e) {
      console.error('Failed to pick catalog', e);
      return null;
    }
  }, []);

  const handleCancelUpload = useCallback((id) => {
    setUploads((prev) => prev.filter((item) => item.id !== id));
    setProjectSnapshot((prev) => ({
      ...prev,
      queue: (prev.queue || []).filter((q) => q.id !== id),
    }));
  }, []);

  const handlePauseAll = useCallback(() => {
    setUploads((prev) =>
      prev.map((u) => (u.status === 'uploading' ? { ...u, status: 'paused' } : u))
    );
  }, []);

  const handleResumeAll = useCallback(() => {
    setUploads((prev) =>
      prev.map((u) => (u.status === 'paused' ? { ...u, status: 'uploading' } : u))
    );
  }, []);

  const handleDismissCompleted = useCallback(() => {
    setUploads((prev) => prev.filter((u) => u.status !== 'complete'));
  }, []);

  // Open Project workspace
  const handleOpenProject = useCallback((proj) => {
    setActiveProject(proj);
    setScreen('project-detail');
  }, []);

  // Create Project handler matching Imagen workflow
  const handleCreateProject = useCallback(async (projectData) => {
    // If a specific profile was selected, activate it
    if (projectData.profileId && projectData.profileId !== profilesQ.activeProfile?.id) {
      try {
        await profilesQ.activate(projectData.profileId);
      } catch (e) {
        console.warn('Failed to switch profile', e);
      }
    }

    const uploadId = `upload-${Date.now()}`;
    const projId = `proj-${Date.now()}`;
    const total = projectData.rawCount || 21;
    const projName = projectData.name || (projectData.sourcePath ? projectData.sourcePath.split(/[\\/]/).filter(Boolean).pop() : 'Sample_Raw');
    const nowStr = new Date().toLocaleDateString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });

    const newProject = {
      id: projId,
      uploadId,
      name: projName,
      path: projectData.sourcePath || `D:/Sonna/Shoots/2026/${projName}`,
      photos: total,
      date: nowStr,
      status: 'processing',
      createdAt: Date.now(),
    };

    // Prepend to projectsList
    setProjectsList((prev) => [newProject, ...prev]);

    // Open project workspace directly to show photos editing in progress
    setActiveProject(newProject);
    setScreen('project-detail');

    // Create new upload item for Uploader top pill
    const newUpload = {
      id: uploadId,
      name: projName,
      folderPath: projectData.sourcePath,
      totalPhotos: total,
      processedPhotos: 0,
      percent: 0,
      status: 'uploading',
      timeLeft: 'Calculating...',
      createdAt: Date.now(),
    };

    setUploads((prev) => [newUpload, ...prev]);

    // Simulate smooth upload & editing progression
    let currentProcessed = 0;
    const interval = setInterval(() => {
      currentProcessed += 1;
      const pct = Math.min(100, Math.round((currentProcessed / total) * 100));
      const remainingSec = Math.max(1, Math.round((total - currentProcessed) * 0.8));

      setUploads((prev) =>
        prev.map((item) => {
          if (item.id !== uploadId) return item;
          if (item.status === 'paused' || item.status === 'cancelled') return item;
          const isDone = currentProcessed >= total;
          return {
            ...item,
            processedPhotos: currentProcessed,
            percent: pct,
            timeLeft: isDone ? 'Done' : `${remainingSec}s`,
            status: isDone ? 'complete' : 'uploading',
          };
        })
      );

      if (currentProcessed >= total) {
        clearInterval(interval);
        // Mark project complete in projectsList and activeProject
        setProjectsList((prev) =>
          prev.map((p) => (p.id === projId ? { ...p, status: 'complete' } : p))
        );
        setActiveProject((prev) => (prev && prev.id === projId ? { ...prev, status: 'complete' } : prev));
      }
    }, 400);
  }, [profilesQ]);

  if (screen === 'login' && !hasEnteredApp) {
    return (
      <SahaLogin
        onSubmit={handleLogin}
        theme={theme}
        onToggleTheme={handleToggleTheme}
      />
    );
  }

  return (
    <ErrorBoundary>
      {/* Login Screen */}
      <div style={{ display: screen === 'login' ? 'flex' : 'none', width: '100%', height: '100%' }}>
        <SahaLogin
          onSubmit={handleLogin}
          theme={theme}
          onToggleTheme={handleToggleTheme}
        />
      </div>

      {/* Screen 1: Home View */}
      <div style={{ display: screen === 'home' ? 'flex' : 'none', width: '100%', height: '100%' }}>
        <HomeView
          onNavigate={handleNavigate}
          onOpenNewProject={() => setIsNewProjectOpen(true)}
          onOpenCreateProfile={() => setIsChooseProfileOpen(true)}
          onOpenProject={handleOpenProject}
          projects={projectsList}
          activeProfile={profilesQ.activeProfile}
          theme={theme}
          onToggleTheme={handleToggleTheme}
          onLogout={handleLogout}
          uploads={uploads}
          isUpdating={isUpdating}
          updatingProgress={updatingProgress}
          onCancelUpload={handleCancelUpload}
          onPauseAll={handlePauseAll}
          onResumeAll={handleResumeAll}
          onDismissCompleted={handleDismissCompleted}
        />
      </div>

      {/* Screen 2: Projects Dashboard */}
      <div style={{ display: screen === 'projects' ? 'flex' : 'none', width: '100%', height: '100%' }}>
        <ProjectsView
          projects={projectsList}
          onNavigate={handleNavigate}
          onOpenNewProject={() => setIsNewProjectOpen(true)}
          onOpenProject={handleOpenProject}
          theme={theme}
          onToggleTheme={handleToggleTheme}
          onLogout={handleLogout}
          uploads={uploads}
          isUpdating={isUpdating}
          updatingProgress={updatingProgress}
          onCancelUpload={handleCancelUpload}
          onPauseAll={handlePauseAll}
          onResumeAll={handleResumeAll}
          onDismissCompleted={handleDismissCompleted}
        />
      </div>

      {/* Screen: Project Detail / Photo Inspection Workspace */}
      <div style={{ display: screen === 'project-detail' ? 'flex' : 'none', width: '100%', height: '100%' }}>
        <ProjectDetailView
          project={activeProject}
          onBack={() => setScreen('projects')}
          theme={theme}
          onToggleTheme={handleToggleTheme}
          uploads={uploads}
          isUpdating={isUpdating}
          updatingProgress={updatingProgress}
          onCancelUpload={handleCancelUpload}
          onPauseAll={handlePauseAll}
          onResumeAll={handleResumeAll}
          onDismissCompleted={handleDismissCompleted}
        />
      </div>

      {/* Screen 3: AI Profiles */}
      <div style={{ display: screen === 'profiles' ? 'flex' : 'none', width: '100%', height: '100%' }}>
        <ProfileView
          profiles={profilesQ.profiles}
          activeProfile={profilesQ.activeProfile}
          onActivate={profilesQ.activate}
          onProfilesChanged={profilesQ.refetch}
          onNavigate={handleNavigate}
          onOpenNewProject={() => setIsNewProjectOpen(true)}
          theme={theme}
          onToggleTheme={handleToggleTheme}
          onLogout={handleLogout}
          uploads={uploads}
          isUpdating={isUpdating}
          updatingProgress={updatingProgress}
          onCancelUpload={handleCancelUpload}
          onPauseAll={handlePauseAll}
          onResumeAll={handleResumeAll}
          onDismissCompleted={handleDismissCompleted}
        />
      </div>

      {/* Modal 1: Choose Profile Type Modal */}
      <ChooseProfileTypeModal
        isOpen={isChooseProfileOpen}
        onClose={() => setIsChooseProfileOpen(false)}
        onSelectLite={() => {
          setIsChooseProfileOpen(false);
          setIsLiteWizardOpen(true);
        }}
        onSelectPersonal={() => {
          setIsChooseProfileOpen(false);
          setIsPersonalWizardOpen(true);
        }}
        theme={theme}
      />

      {/* Modal 2: New Project Modal */}
      <NewProjectModal
        isOpen={isNewProjectOpen}
        onClose={() => setIsNewProjectOpen(false)}
        profiles={profilesQ.profiles}
        activeProfile={profilesQ.activeProfile}
        onPickFolder={handlePickFolder}
        onPickCatalog={handlePickCatalog}
        onCreateProject={handleCreateProject}
        onOpenCreateProfile={() => setIsChooseProfileOpen(true)}
        newlyCreatedProfileId={newlyCreatedProfileId}
      />

      {/* Modal 3: Lite Profile Creation Wizard */}
      {isLiteWizardOpen && (
        <LiteProfileWizard
          onClose={() => setIsLiteWizardOpen(false)}
          onCreated={async (res) => {
            setIsLiteWizardOpen(false);
            await handleProfileCreated(res);
          }}
          theme={theme}
        />
      )}

      {/* Modal 4: Personal AI Profile Creation Wizard */}
      {isPersonalWizardOpen && (
        <PersonalProfileWizard
          onClose={() => setIsPersonalWizardOpen(false)}
          onCreated={async (res) => {
            setIsPersonalWizardOpen(false);
            await handleProfileCreated(res);
          }}
        />
      )}
    </ErrorBoundary>
  );
}

export default App;
