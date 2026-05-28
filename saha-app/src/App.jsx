// App — three screens: login → editor ⇄ profile.
// useProfiles is hoisted here so editor and profile share one source of truth
// for the active profile (activate from profile view → editor sees the change
// without needing its own refetch).

import { useCallback, useState } from 'react';
import { SahaLogin } from './components/login.jsx';
import { Editor } from './components/editor.jsx';
import { ProfileView } from './components/profile-view.jsx';
import { useProfiles } from './hooks/useProfiles.js';

export function App() {
  const [screen, setScreen] = useState('login');
  const profilesQ = useProfiles();

  const handleNavigate = useCallback((kind) => {
    if (kind === 'process') setScreen('editor');
    else if (kind === 'profile') setScreen('profile');
  }, []);

  if (screen === 'login') {
    return <SahaLogin onSubmit={() => setScreen('editor')} />;
  }
  if (screen === 'profile') {
    return (
      <ProfileView
        profiles={profilesQ.profiles}
        activeProfile={profilesQ.activeProfile}
        onActivate={profilesQ.activate}
        onProfilesChanged={profilesQ.refetch}
        onNavigate={handleNavigate}
      />
    );
  }
  return (
    <Editor
      profiles={profilesQ.profiles}
      activeProfile={profilesQ.activeProfile}
      onActivateProfile={profilesQ.activate}
      onNavigate={handleNavigate}
    />
  );
}
