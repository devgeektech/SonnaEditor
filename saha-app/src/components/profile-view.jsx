// Profile View — Modern AI Profiles Table & Management
// Reference:
// - media_1794847665263_8f3e1c11.png (AI Profiles list with 'Your profiles' / 'Profile library' tabs,
//   'Create your own profile' button, and card rows with Ready to use / Draft states)
// Full Light and Dark theme support via SONNA design tokens.

import { useMemo, useState, useEffect } from 'react';
import SONNA from '../tokens.js';
import { AppShell } from './shell.jsx';
import { ErrorBanner } from './error-banner.jsx';
import { ChooseProfileTypeModal } from './profile-type-modal.jsx';
import { LiteProfileWizard } from './lite-wizard.jsx';
import { PersonalProfileWizard } from './personal-wizard.jsx';
import { ConfirmModal } from './confirm-modal.jsx';
import { deleteProfile } from '../api/client.js';

const F = SONNA.font;
const M = SONNA.mono;

// Helper to format dates matching Imagen AI reference (e.g. "Nov 14, 2026")
function formatDate(d) {
  if (!d) return 'Nov 14, 2026';
  try {
    const date = new Date(d);
    if (isNaN(date.getTime())) return 'Nov 14, 2026';
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
  } catch {
    return 'Nov 14, 2026';
  }
}

// Empty state folder illustration matching Imagen AI reference (media_1794855536397_e28a20ab.png)
function EmptyProfilesFolder({ isLight }) {
  const backFill = isLight ? '#DDD2C3' : '#2D3540';
  const frontFill = isLight ? '#C9BCAB' : '#3E4856';
  const emblemStroke = isLight ? 'rgba(0, 0, 0, 0.18)' : 'rgba(255, 255, 255, 0.18)';
  const highlight = isLight ? 'rgba(255, 255, 255, 0.45)' : 'rgba(255, 255, 255, 0.08)';

  return (
    <div style={{
      width: 76,
      height: 64,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      marginBottom: 16,
    }}>
      <svg width="76" height="64" viewBox="0 0 88 76" fill="none">
        {/* Back tab & body */}
        <path
          d="M8 18C8 12.4772 12.4772 8 18 8H34.3431C37.0173 8 39.5822 9.06209 41.4728 10.9526L45.5272 15.0074C47.4178 16.8979 49.9827 17.96 52.6569 17.96H70C75.5228 17.96 80 22.4372 80 27.96V58C80 63.5228 75.5228 68 70 68H18C12.4772 68 8 63.5228 8 58V18Z"
          fill={backFill}
        />
        {/* Front pocket */}
        <path
          d="M8 30C8 24.4772 12.4772 20 18 20H70C75.5228 20 80 24.4772 80 30V58C80 63.5228 75.5228 68 70 68H18C12.4772 68 8 63.5228 8 58V30Z"
          fill={frontFill}
        />
        {/* Top edge highlight of front pocket */}
        <path
          d="M18 21H70"
          stroke={highlight}
          strokeWidth="1.5"
          strokeLinecap="round"
        />
        {/* Center circular empty emblem */}
        <circle
          cx="44"
          cy="44"
          r="12"
          stroke={emblemStroke}
          strokeWidth="2.5"
          fill="none"
        />
        <line
          x1="35.5"
          y1="35.5"
          x2="52.5"
          y2="52.5"
          stroke={emblemStroke}
          strokeWidth="2.5"
          strokeLinecap="round"
        />
      </svg>
    </div>
  );
}


export function ProfileView({
  profiles = [],
  activeProfile = null,
  onActivate,
  onProfilesChanged,
  onNavigate,
  onOpenNewProject,
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
  const [error, setError] = useState(null);
  const [sortOrder, setSortOrder] = useState('desc'); // 'desc' | 'asc'
  const [chooseTypeOpen, setChooseTypeOpen] = useState(false);
  const [liteWizardOpen, setLiteWizardOpen] = useState(false);
  const [personalWizardOpen, setPersonalWizardOpen] = useState(false);
  const [activeMenuId, setActiveMenuId] = useState(null);
  const [refreshKey, setRefreshKey] = useState(0);
  const [deleteConfirmProfile, setDeleteConfirmProfile] = useState(null);
  const [renameModalProfile, setRenameModalProfile] = useState(null);

  // Close menus on outside click
  useEffect(() => {
    const handleWindowClick = () => setActiveMenuId(null);
    window.addEventListener('click', handleWindowClick);
    return () => window.removeEventListener('click', handleWindowClick);
  }, []);

  // Merge server profiles with local custom names and drafts
  const allProfiles = useMemo(() => {
    let customNames = {};
    let localDrafts = [];
    let deletedIds = [];
    try {
      customNames = JSON.parse(localStorage.getItem('saha_profile_custom_names') || '{}');
      deletedIds = JSON.parse(localStorage.getItem('saha_deleted_profile_ids') || '[]');
      const stored = localStorage.getItem('saha_draft_profiles');
      if (stored) {
        localDrafts = JSON.parse(stored);
      } else if (!deletedIds.includes('draft-wedw') && !deletedIds.includes('wedw')) {
        // Initialize default draft matching reference image media_1794847665263_8f3e1c11.png
        localDrafts = [
          {
            id: 'draft-wedw',
            name: 'wedw',
            display_name: 'wedw',
            photo_count: 189,
            trained_at: '2026-11-14T10:00:00Z',
            status: 'draft',
            isDraft: true,
          },
        ];
        localStorage.setItem('saha_draft_profiles', JSON.stringify(localDrafts));
      }
    } catch {
      // ignore JSON errors
    }

    // Process backend profiles (server is the single source of truth for files on disk)
    const processedServer = profiles.map((p) => ({
      ...p,
      display_name: customNames[p.id] || p.display_name || p.name,
      status: 'ready',
      isDraft: false,
      photo_count: p.photo_count != null ? p.photo_count : 0,
      trained_at: p.trained_at || '2026-11-14T10:00:00Z',
    }));

    // If server has no profiles yet and 'vi-profile' hasn't been deleted by user, provide placeholder
    const defaultPlaceholder = (!deletedIds.includes('vi-profile') && !deletedIds.includes('vi')) ? [
      {
        id: 'vi-profile',
        name: 'vi',
        display_name: 'vi',
        photo_count: 0,
        trained_at: '2026-11-14T10:00:00Z',
        status: 'ready',
        isDraft: false,
        is_active: true,
      },
    ] : [];

    const displayServer = processedServer.length > 0 ? processedServer : defaultPlaceholder;
    const filteredDrafts = localDrafts.filter((p) => !deletedIds.includes(p.id) && !deletedIds.includes(p.name));
    const combined = [...displayServer, ...filteredDrafts];

    return combined.sort((a, b) => {
      const da = new Date(a.trained_at || 0).getTime();
      const db = new Date(b.trained_at || 0).getTime();
      return sortOrder === 'desc' ? db - da : da - db;
    });
  }, [profiles, sortOrder, refreshKey]);

  // Actions
  const handleUseToEdit = async (profile) => {
    try {
      if (profile.id && onActivate && !profile.isDraft) {
        await onActivate(profile.id);
      }
      if (onOpenNewProject) {
        onOpenNewProject();
      } else if (onNavigate) {
        onNavigate('home');
      }
    } catch (e) {
      setError({ source: 'activate', message: e.message });
    }
  };

  const handleUploadPhotos = () => {
    setPersonalWizardOpen(true);
  };

  const handleRename = (profile) => {
    setActiveMenuId(null);
    setRenameModalProfile(profile);
  };

  const executeRename = (newName) => {
    const profile = renameModalProfile;
    setRenameModalProfile(null);
    if (!profile) return;
    const current = profile.display_name || profile.name;
    const trimmed = (newName || '').trim();
    if (trimmed && trimmed.toLowerCase() !== current.toLowerCase()) {
      const isDup = allProfiles.some(
        (p) => p.id !== profile.id && (p.display_name || p.name || '').trim().toLowerCase() === trimmed.toLowerCase()
      );
      if (isDup) {
        setError({ source: 'rename', message: `A profile named "${trimmed}" already exists. Every profile must have a unique name.` });
        return;
      }
      try {
        if (profile.isDraft) {
          const drafts = JSON.parse(localStorage.getItem('saha_draft_profiles') || '[]');
          const updated = drafts.map((d) => d.id === profile.id ? { ...d, name: trimmed, display_name: trimmed } : d);
          localStorage.setItem('saha_draft_profiles', JSON.stringify(updated));
        } else {
          const customNames = JSON.parse(localStorage.getItem('saha_profile_custom_names') || '{}');
          customNames[profile.id] = trimmed;
          localStorage.setItem('saha_profile_custom_names', JSON.stringify(customNames));
        }
        setRefreshKey((k) => k + 1);
        onProfilesChanged?.();
      } catch (e) {
        console.error('Failed to rename profile', e);
      }
    }
  };

  const handleDuplicate = (profile) => {
    setActiveMenuId(null);
    const current = profile.display_name || profile.name;
    const existingNames = new Set(allProfiles.map((p) => (p.display_name || p.name || '').trim().toLowerCase()));
    let counter = 1;
    let dupName = `${current} (Copy)`;
    while (existingNames.has(dupName.toLowerCase())) {
      counter += 1;
      dupName = `${current} (Copy ${counter})`;
    }
    try {
      const drafts = JSON.parse(localStorage.getItem('saha_draft_profiles') || '[]');
      const newProfile = {
        id: `profile-${Date.now()}`,
        name: dupName,
        display_name: dupName,
        photo_count: profile.photo_count || 0,
        trained_at: new Date().toISOString(),
        status: profile.status || 'ready',
        isDraft: profile.isDraft || false,
      };
      drafts.push(newProfile);
      localStorage.setItem('saha_draft_profiles', JSON.stringify(drafts));
      setRefreshKey((k) => k + 1);
    } catch (e) {
      console.error('Failed to duplicate profile', e);
    }
  };

  const handleDelete = (profile) => {
    setActiveMenuId(null);
    setDeleteConfirmProfile(profile);
  };

  const executeDelete = async () => {
    const profile = deleteConfirmProfile;
    setDeleteConfirmProfile(null);
    if (!profile) return;

    try {
      // 1. Remove from local drafts if present
      try {
        const drafts = JSON.parse(localStorage.getItem('saha_draft_profiles') || '[]');
        const updated = drafts.filter((d) => d.id !== profile.id && d.name !== profile.name);
        localStorage.setItem('saha_draft_profiles', JSON.stringify(updated));
      } catch (err) {
        console.warn('Failed to update drafts:', err);
      }

      // 2. If this is a draft or placeholder, mark as deleted in localStorage so it doesn't reappear
      if (profile.isDraft || profile.id === 'vi-profile') {
        try {
          const deletedIds = JSON.parse(localStorage.getItem('saha_deleted_profile_ids') || '[]');
          if (profile.id && !deletedIds.includes(profile.id)) deletedIds.push(profile.id);
          localStorage.setItem('saha_deleted_profile_ids', JSON.stringify(deletedIds));
        } catch (err) {
          console.warn('Failed to update deleted IDs:', err);
        }
      }

      // 3. Clean up custom renamed label
      try {
        const customNames = JSON.parse(localStorage.getItem('saha_profile_custom_names') || '{}');
        if (customNames[profile.id]) {
          delete customNames[profile.id];
          localStorage.setItem('saha_profile_custom_names', JSON.stringify(customNames));
        }
      } catch (err) {
        console.warn('Failed to remove custom name:', err);
      }

      // 4. If this is a server profile (and not placeholder), delete on server
      if (!profile.isDraft && profile.id !== 'vi-profile') {
        try {
          await deleteProfile(profile.id);
        } catch (serverErr) {
          console.warn('Backend deleteProfile note:', serverErr);
        }
      }

      onProfilesChanged?.();
      setRefreshKey((k) => k + 1);
      setError(null);
    } catch (e) {
      console.error('Failed to delete profile', e);
      setError({ source: 'delete-profile', message: e.message });
    }
  };

  return (
    <AppShell
      title="saha - profiles"
      activeNav="profiles"
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
      <div style={{
        flex: 1,
        background: SONNA.bgDeep,
        display: 'flex',
        flexDirection: 'column',
        minWidth: 0,
        overflowY: 'auto',
        padding: '28px 36px',
        color: SONNA.fg,
        transition: 'background 0.2s ease, color 0.2s ease',
      }}>
        {/* Header Bar */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: 16,
        }}>
          <h1 style={{
            fontSize: 22,
            fontWeight: 700,
            color: SONNA.fg,
            margin: 0,
            letterSpacing: -0.3,
          }}>
            AI Profiles
          </h1>

          {/* Create your own profile outline button only shown when profiles exist */}
          {allProfiles.length > 0 && (
            <button
              type="button"
              onClick={() => setChooseTypeOpen(true)}
              style={{
                height: 36,
                padding: '0 18px',
                background: isLight ? '#FFFFFF' : 'transparent',
                border: `1px solid ${isLight ? 'rgba(0,0,0,0.2)' : 'rgba(255, 255, 255, 0.45)'}`,
                borderRadius: 6,
                color: SONNA.fg,
                fontFamily: F,
                fontSize: 13,
                fontWeight: 600,
                cursor: 'pointer',
                display: 'inline-flex',
                alignItems: 'center',
                gap: 6,
                justifyContent: 'center',
                transition: 'all 0.15s ease',
                boxShadow: isLight ? '0 1px 3px rgba(0,0,0,0.06)' : 'none',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = isLight ? 'rgba(0,0,0,0.45)' : 'rgba(255, 255, 255, 0.85)';
                e.currentTarget.style.background = isLight ? '#F5EFE6' : 'rgba(255, 255, 255, 0.05)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = isLight ? 'rgba(0,0,0,0.2)' : 'rgba(255, 255, 255, 0.45)';
                e.currentTarget.style.background = isLight ? '#FFFFFF' : 'transparent';
              }}
            >
              <span>+</span>
              <span>Create your own profile</span>
            </button>
          )}
        </div>

        {/* Sub-navigation Tab: Your profiles */}
        <div style={{
          display: 'flex',
          gap: 28,
          borderBottom: `1px solid ${SONNA.lineSoft}`,
          marginBottom: 18,
        }}>
          <button
            type="button"
            style={{
              padding: '8px 0 12px 0',
              background: 'transparent',
              border: 'none',
              borderBottom: `2px solid ${SONNA.fg}`,
              color: SONNA.fg,
              fontSize: 13,
              fontWeight: 600,
              cursor: 'default',
              transition: 'all 0.15s ease',
            }}
          >
            Your profiles
          </button>
        </div>

        <ErrorBanner error={error} onDismiss={() => setError(null)} />

        {/* Your profiles Table or Zero State matching Imagen AI reference */}
        {allProfiles.length === 0 ? (
          <div style={{
            flex: 1,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            textAlign: 'center',
            padding: '60px 24px',
          }}>
            <EmptyProfilesFolder isLight={isLight} />
            <h2 style={{
              fontSize: 22,
              fontWeight: 700,
              color: SONNA.fg,
              marginBottom: 8,
              letterSpacing: -0.3,
            }}>
              No personal AI profiles yet
            </h2>
            <p style={{
              fontSize: 13.5,
              color: SONNA.fgMute,
              maxWidth: 440,
              lineHeight: 1.5,
              margin: '0 0 24px 0',
            }}>
              Create your first Personal AI Profile and train it with your editing style by uploading previous edited photos.
            </p>
            <button
              type="button"
              onClick={() => setChooseTypeOpen(true)}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: 6,
                height: 36,
                padding: '0 20px',
                background: SONNA.ochre,
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
              onMouseEnter={(e) => {
                e.currentTarget.style.background = SONNA.ochreSoft;
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = SONNA.ochre;
              }}
            >
              <span>+</span>
              <span>Create your own profile</span>
            </button>
          </div>
        ) : (
          <div className="saha-table-scroll">
          <div style={{ display: 'flex', flexDirection: 'column', minWidth: 740 }}>
            {/* Table Header Row */}
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'minmax(180px, 1.8fr) 180px 150px 170px 240px',
              padding: '8px 24px 12px 24px',
              fontSize: 12,
              fontWeight: 500,
              color: SONNA.fgDim,
            }}>
              <span>Profile name</span>
              <span>Number of photos</span>
              <span
                onClick={() => setSortOrder((prev) => (prev === 'desc' ? 'asc' : 'desc'))}
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 4,
                  cursor: 'pointer',
                  userSelect: 'none',
                }}
              >
                Date {sortOrder === 'desc' ? '↓' : '↑'}
              </span>
              <span>Status</span>
              <span />
            </div>

            {/* Profile Card Rows */}
            {allProfiles.map((p) => {
              const isDraft = p.status === 'draft' || p.isDraft;
              const menuOpen = activeMenuId === p.id;

              return (
                <div
                  key={p.id}
                  style={{
                    display: 'grid',
                    gridTemplateColumns: 'minmax(180px, 1.8fr) 180px 150px 170px 240px',
                    padding: '16px 24px',
                    marginBottom: 8,
                    background: SONNA.bgPanel,
                    borderRadius: 8,
                    border: `1px solid ${SONNA.lineSoft}`,
                    alignItems: 'center',
                    position: 'relative',
                    transition: 'background 0.15s ease, border-color 0.15s ease',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background = SONNA.bgHover;
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = SONNA.bgPanel;
                  }}
                >
                  {/* Column 1: Profile Name */}
                  <div style={{
                    fontSize: 13.5,
                    fontWeight: 600,
                    color: SONNA.fg,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                    paddingRight: 12,
                  }}>
                    {p.display_name || p.name}
                  </div>

                  {/* Column 2: Number of Photos */}
                  <div style={{
                    fontSize: 13,
                    color: SONNA.fgMute,
                    fontFamily: M,
                  }}>
                    {p.photo_count != null ? p.photo_count.toLocaleString() : 0}
                  </div>

                  {/* Column 3: Date */}
                  <div style={{
                    fontSize: 13,
                    color: SONNA.fgMute,
                  }}>
                    {formatDate(p.trained_at)}
                  </div>

                  {/* Column 4: Status */}
                  <div>
                    {isDraft ? (
                      <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 6,
                        color: SONNA.fgDim,
                        fontSize: 13,
                        fontWeight: 500,
                      }}>
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M12 20h9" />
                          <path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z" />
                        </svg>
                        Draft
                      </div>
                    ) : (
                      <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 6,
                        color: isLight ? '#16A34A' : '#22C55E',
                        fontSize: 13,
                        fontWeight: 500,
                      }}>
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                          <circle cx="12" cy="12" r="10" />
                          <path d="m9 12 2 2 4-4" />
                        </svg>
                        Ready to use
                      </div>
                    )}
                  </div>

                  {/* Column 5: Right Action Buttons */}
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'flex-end',
                    gap: 10,
                  }}>
                    {isDraft ? (
                      <button
                        type="button"
                        onClick={() => handleUploadPhotos(p)}
                        style={{
                          height: 32,
                          padding: '0 16px',
                          background: isLight ? '#FFFFFF' : 'transparent',
                          border: `1px solid ${isLight ? 'rgba(0,0,0,0.2)' : 'rgba(255, 255, 255, 0.35)'}`,
                          borderRadius: 6,
                          color: SONNA.fg,
                          fontFamily: F,
                          fontSize: 12.5,
                          fontWeight: 600,
                          cursor: 'pointer',
                          transition: 'all 0.15s ease',
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.borderColor = isLight ? 'rgba(0,0,0,0.4)' : 'rgba(255, 255, 255, 0.7)';
                          e.currentTarget.style.background = isLight ? '#EFE7DC' : 'rgba(255, 255, 255, 0.06)';
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.borderColor = isLight ? 'rgba(0,0,0,0.2)' : 'rgba(255, 255, 255, 0.35)';
                          e.currentTarget.style.background = isLight ? '#FFFFFF' : 'transparent';
                        }}
                      >
                        Upload photos
                      </button>
                    ) : (
                      <>
                        <button
                          type="button"
                          onClick={() => handleUseToEdit(p)}
                          style={{
                            height: 32,
                            padding: '0 18px',
                            background: '#E05353',
                            border: 'none',
                            borderRadius: 6,
                            color: '#FFFFFF',
                            fontFamily: F,
                            fontSize: 12.5,
                            fontWeight: 600,
                            cursor: 'pointer',
                            boxShadow: '0 2px 8px rgba(224, 83, 83, 0.35)',
                            transition: 'background 0.15s ease',
                          }}
                          onMouseEnter={(e) => {
                            e.currentTarget.style.background = '#EC5A5A';
                          }}
                          onMouseLeave={(e) => {
                            e.currentTarget.style.background = '#E05353';
                          }}
                        >
                          Use to edit
                        </button>
                      </>
                    )}

                    {/* Three-dots menu button */}
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        setActiveMenuId(menuOpen ? null : p.id);
                      }}
                      title="More options"
                      style={{
                        width: 30,
                        height: 32,
                        borderRadius: 6,
                        background: 'transparent',
                        border: 'none',
                        color: SONNA.fgMute,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        cursor: 'pointer',
                        fontSize: 18,
                        transition: 'color 0.15s ease',
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.color = SONNA.fg;
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.color = SONNA.fgMute;
                      }}
                    >
                      ⋮
                    </button>

                    {/* Dropdown Menu */}
                    {menuOpen && (
                      <div
                        onClick={(e) => e.stopPropagation()}
                        style={{
                          position: 'absolute',
                          right: 20,
                          top: 50,
                          zIndex: 60,
                          width: 170,
                          background: SONNA.bgPanel,
                          border: `1px solid ${SONNA.line}`,
                          borderRadius: 8,
                          boxShadow: isLight ? '0 10px 28px rgba(0,0,0,0.12)' : '0 12px 32px rgba(0, 0, 0, 0.65)',
                          padding: 5,
                          textAlign: 'left',
                        }}
                      >
                        <button
                          type="button"
                          onClick={() => handleRename(p)}
                          style={{
                            width: '100%',
                            padding: '8px 10px',
                            background: 'transparent',
                            border: 'none',
                            color: SONNA.fg,
                            fontSize: 12,
                            textAlign: 'left',
                            cursor: 'pointer',
                            borderRadius: 4,
                          }}
                        >
                          Rename Profile
                        </button>
                        <button
                          type="button"
                          onClick={() => handleDuplicate(p)}
                          style={{
                            width: '100%',
                            padding: '8px 10px',
                            background: 'transparent',
                            border: 'none',
                            color: SONNA.fg,
                            fontSize: 12,
                            textAlign: 'left',
                            cursor: 'pointer',
                            borderRadius: 4,
                          }}
                        >
                          Duplicate Profile
                        </button>
                        <div style={{ height: 1, background: SONNA.lineSoft, margin: '4px 0' }} />
                        <button
                          type="button"
                          onClick={() => handleDelete(p)}
                          style={{
                            width: '100%',
                            padding: '8px 10px',
                            background: 'transparent',
                            border: 'none',
                            color: SONNA.red,
                            fontSize: 12,
                            textAlign: 'left',
                            cursor: 'pointer',
                            borderRadius: 4,
                          }}
                        >
                          Delete Profile
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
          </div>
        )
        }
      </div>

      {/* Choice Modal (Lite vs Personal AI Profile) */}
      <ChooseProfileTypeModal
        isOpen={chooseTypeOpen}
        onClose={() => setChooseTypeOpen(false)}
        onSelectLite={() => setLiteWizardOpen(true)}
        onSelectPersonal={() => setPersonalWizardOpen(true)}
        theme={theme}
      />

      {/* Lite Wizard */}
      {liteWizardOpen && (
        <LiteProfileWizard
          onClose={() => setLiteWizardOpen(false)}
          onCreated={async (res) => {
            setLiteWizardOpen(false);
            if (onProfilesChanged) {
              await onProfilesChanged(res);
            }
            setRefreshKey((k) => k + 1);
          }}
          theme={theme}
          profiles={profiles}
        />
      )}

      {/* Personal AI Wizard */}
      {personalWizardOpen && (
        <PersonalProfileWizard
          onClose={() => setPersonalWizardOpen(false)}
          onCreated={async (res) => {
            setPersonalWizardOpen(false);
            if (onProfilesChanged) {
              await onProfilesChanged(res);
            }
            setRefreshKey((k) => k + 1);
          }}
          profiles={profiles}
        />
      )}
      {/* Delete Profile Confirmation Modal */}
      <ConfirmModal
        isOpen={!!deleteConfirmProfile}
        title="Delete profile?"
        message={`Do you want to delete profile "${deleteConfirmProfile?.display_name || deleteConfirmProfile?.name}"? This cannot be undone.`}
        confirmText="Delete"
        cancelText="Cancel"
        onConfirm={executeDelete}
        onCancel={() => setDeleteConfirmProfile(null)}
        theme={theme}
      />

      {/* Rename Profile Prompt Modal */}
      <ConfirmModal
        isOpen={!!renameModalProfile}
        title="Rename profile"
        message="Enter a new name for this profile:"
        confirmText="Save"
        cancelText="Cancel"
        isPrompt={true}
        promptInitialValue={renameModalProfile?.display_name || renameModalProfile?.name || ''}
        promptPlaceholder="Profile name"
        onConfirm={executeRename}
        onCancel={() => setRenameModalProfile(null)}
        theme={theme}
      />
    </AppShell>
  );
}

export default ProfileView;
