// App — three screens: login → editor ⇄ profile.
// useProfiles is hoisted here so editor and profile share one source of truth
// for the active profile (activate from profile view → editor sees the change
// without needing its own refetch).

import { useCallback, useLayoutEffect, useState } from 'react';
import { SahaLogin } from './components/login.jsx';
import { Editor } from './components/editor.jsx';
import { ProfileView } from './components/profile-view.jsx';
import { ProjectsView } from './components/projects-view.jsx';
import { useProfiles } from './hooks/useProfiles.js';
import { applyTheme } from './tokens.js';

export function App() {
  const [screen, setScreen] = useState('login');
  const [hasEnteredApp, setHasEnteredApp] = useState(false);
  const [theme, setTheme] = useState('light');
  const [projectSnapshot, setProjectSnapshot] = useState({ queue: [], runResults: [] });
  const profilesQ = useProfiles();

  useLayoutEffect(() => {
    applyTheme(theme);
    if (typeof window !== 'undefined') {
      window.localStorage?.setItem('saha-theme', theme);
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
    <>
      <div style={{ display: screen === 'login' ? 'flex' : 'none', width: '100%', height: '100%' }}>
        <SahaLogin
          onSubmit={handleLogin}
          theme={theme}
          onToggleTheme={handleToggleTheme}
        />
      </div>
      <div style={{ display: screen === 'home' ? 'flex' : 'none', width: '100%', height: '100%' }}>
        <Editor
          profiles={profilesQ.profiles}
          activeProfile={profilesQ.activeProfile}
          onActivateProfile={profilesQ.activate}
          onNavigate={handleNavigate}
          theme={theme}
          onToggleTheme={handleToggleTheme}
          onLogout={handleLogout}
          onProjectsChange={setProjectSnapshot}
        />
      </div>
      <div style={{ display: screen === 'profiles' ? 'flex' : 'none', width: '100%', height: '100%' }}>
      <ProfileView
        profiles={profilesQ.profiles}
        activeProfile={profilesQ.activeProfile}
        onActivate={profilesQ.activate}
        onProfilesChanged={profilesQ.refetch}
        onNavigate={handleNavigate}
        theme={theme}
        onToggleTheme={handleToggleTheme}
        onLogout={handleLogout}
      />
      </div>
      <div style={{ display: screen === 'projects' ? 'flex' : 'none', width: '100%', height: '100%' }}>
        <ProjectsView
          projects={projectSnapshot}
          onNavigate={handleNavigate}
          theme={theme}
          onToggleTheme={handleToggleTheme}
          onLogout={handleLogout}
        />
      </div>
    </>
  );
}
