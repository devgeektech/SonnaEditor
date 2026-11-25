// Uploader & Manage Uploads popover matching Imagen AI reference
// Reference:
// - media_1794864773172_e1452fc9.jpg (Idle top-bar pill "Uploader" + empty state "You will see all project uploads here")
// - media_1794864748523_90932ae2.jpg (Active top-bar "Uploading 1/1" + "Manage uploads" panel with DNG banner, summary, progress bar, Cancel upload)
// - media_1794864635431_f92b20ea.jpg (Updating top-bar pill + "We're updating your packages" progress card)

import { useState, useEffect, useRef } from 'react';
import SONNA from '../tokens.js';

const F = SONNA.font;
const M = SONNA.mono;
const Tnum = { fontFamily: SONNA.mono, fontVariantNumeric: 'tabular-nums' };

export function UploaderPill({
  activeUploads = [],
  isUpdating = false,
  isOpen = false,
  onClick,
  theme = 'dark',
}) {
  const isLight = theme === 'light';
  const activeCount = activeUploads.filter((u) => u.status === 'uploading').length;
  const totalCount = activeUploads.length;
  const hasActive = activeCount > 0 || isUpdating;

  let label = 'Uploader';
  if (isUpdating) {
    label = 'Updating';
  } else if (activeCount > 0) {
    label = `Uploading ${activeCount}/${totalCount}`;
  }

  return (
    <button
      type="button"
      onClick={onClick}
      title="Manage uploads and background tasks"
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 7,
        height: 30,
        padding: '0 12px',
        borderRadius: 16,
        background: isOpen
          ? (isLight ? 'rgba(0, 0, 0, 0.08)' : 'rgba(255, 255, 255, 0.12)')
          : (hasActive
            ? (isLight ? 'rgba(37, 99, 235, 0.1)' : 'rgba(37, 99, 235, 0.18)')
            : 'transparent'),
        border: `1px solid ${hasActive ? '#3B82F6' : (isLight ? 'rgba(0, 0, 0, 0.18)' : 'rgba(255, 255, 255, 0.2)')}`,
        color: hasActive ? '#60A5FA' : (isLight ? '#374151' : '#E5E7EB'),
        fontFamily: F,
        fontSize: 12.5,
        fontWeight: 600,
        cursor: 'pointer',
        transition: 'all 0.15s ease',
        boxShadow: hasActive ? '0 0 10px rgba(59, 130, 246, 0.25)' : 'none',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.background = isLight ? 'rgba(0,0,0,0.06)' : 'rgba(255,255,255,0.08)';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background = isOpen
          ? (isLight ? 'rgba(0, 0, 0, 0.08)' : 'rgba(255, 255, 255, 0.12)')
          : (hasActive
            ? (isLight ? 'rgba(37, 99, 235, 0.1)' : 'rgba(37, 99, 235, 0.18)')
            : 'transparent');
      }}
    >
      {/* Upload Arrow Icon matching Imagen */}
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
        <line x1="12" y1="19" x2="12" y2="5" />
        <polyline points="5 12 12 5 19 12" />
      </svg>
      <span>{label}</span>

      {/* Rotating spinner or active dot */}
      {hasActive && (
        <span
          style={{
            display: 'inline-block',
            width: 10,
            height: 10,
            borderRadius: '50%',
            border: '2px solid rgba(96, 165, 250, 0.3)',
            borderTopColor: '#60A5FA',
            animation: 'saha-spin 0.8s linear infinite',
            marginLeft: 2,
          }}
        />
      )}
    </button>
  );
}

export function ManageUploadsPopover({
  isOpen = false,
  onClose,
  uploads = [],
  isUpdating = false,
  updatingProgress = 37,
  onCancelUpload,
  onPauseAll,
  onResumeAll,
  onDismissCompleted,
  theme = 'dark',
}) {
  const isLight = theme === 'light';
  const popoverRef = useRef(null);
  const [showTipBanner, setShowTipBanner] = useState(() => {
    try {
      return localStorage.getItem('saha_hide_dng_tip') !== 'true';
    } catch {
      return true;
    }
  });
  const [isExpanded, setIsExpanded] = useState(false);

  // Close on outside click
  useEffect(() => {
    if (!isOpen) return;
    const handleClickOutside = (e) => {
      if (popoverRef.current && !popoverRef.current.contains(e.target)) {
        onClose?.();
      }
    };
    window.addEventListener('mousedown', handleClickOutside);
    return () => window.removeEventListener('mousedown', handleClickOutside);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const handleDismissTip = () => {
    setShowTipBanner(false);
    try {
      localStorage.setItem('saha_hide_dng_tip', 'true');
    } catch {}
  };

  const activeCount = uploads.filter((u) => u.status === 'uploading').length;
  const isPaused = uploads.length > 0 && uploads.every((u) => u.status === 'paused');
  const totalPhotos = uploads.reduce((acc, u) => acc + (u.totalPhotos || 0), 0);
  const hasItems = uploads.length > 0 || isUpdating;

  return (
    <div
      ref={popoverRef}
      style={{
        position: 'fixed',
        top: 50,
        right: 16,
        width: 480,
        maxHeight: isExpanded ? 'calc(100vh - 80px)' : 540,
        background: isLight ? '#FFFFFF' : '#1C1F24',
        border: `1px solid ${isLight ? 'rgba(0, 0, 0, 0.14)' : 'rgba(255, 255, 255, 0.14)'}`,
        borderRadius: 10,
        boxShadow: isLight
          ? '0 16px 40px rgba(0, 0, 0, 0.16)'
          : '0 24px 64px rgba(0, 0, 0, 0.85)',
        zIndex: 2500,
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
        color: isLight ? '#1F2937' : '#F3F4F6',
        fontFamily: F,
        animation: 'saha-popover-in 0.15s ease-out',
      }}
    >
      {/* ── Top Header ────────────────────────────────────────────── */}
      <div style={{
        padding: '13px 18px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        borderBottom: `1px solid ${isLight ? 'rgba(0,0,0,0.08)' : 'rgba(255,255,255,0.08)'}`,
      }}>
        <div style={{ fontSize: 14, fontWeight: 700, letterSpacing: -0.1 }}>
          Manage uploads
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          {/* Expand / Minimize toggle button */}
          <button
            type="button"
            onClick={() => setIsExpanded(!isExpanded)}
            title={isExpanded ? 'Collapse' : 'Expand'}
            style={{
              background: 'transparent',
              border: 'none',
              color: isLight ? '#6B7280' : '#9CA3AF',
              cursor: 'pointer',
              padding: 4,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              borderRadius: 4,
            }}
          >
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="15 3 21 3 21 9" />
              <polyline points="9 21 3 21 3 15" />
              <line x1="21" y1="3" x2="14" y2="10" />
              <line x1="3" y1="21" x2="10" y2="14" />
            </svg>
          </button>

          {/* Close button */}
          <button
            type="button"
            onClick={onClose}
            title="Close"
            style={{
              background: 'transparent',
              border: 'none',
              color: isLight ? '#6B7280' : '#9CA3AF',
              cursor: 'pointer',
              padding: 4,
              fontSize: 16,
              lineHeight: 1,
              borderRadius: 4,
            }}
          >
            ✕
          </button>
        </div>
      </div>

      <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column' }}>
        {/* ── Blue Tip Banner matching Image 2 ────────────────────── */}
        {showTipBanner && (
          <div style={{
            margin: '12px 16px 0',
            padding: '9px 14px',
            borderRadius: 6,
            background: isLight ? 'rgba(219, 234, 254, 0.7)' : 'rgba(30, 58, 138, 0.35)',
            border: `1px solid ${isLight ? 'rgba(147, 197, 253, 0.8)' : 'rgba(59, 130, 246, 0.3)'}`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: 12,
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 11.5, color: isLight ? '#1E40AF' : '#BFDBFE' }}>
              <span style={{ fontSize: 13 }}>💡</span>
              <span>Adobe DNG Converter compresses your photos during upload</span>
            </div>
            <button
              type="button"
              onClick={handleDismissTip}
              style={{
                background: 'transparent',
                border: 'none',
                color: isLight ? '#2563EB' : '#93C5FD',
                fontFamily: F,
                fontSize: 11,
                fontWeight: 600,
                cursor: 'pointer',
                whiteSpace: 'nowrap',
                padding: '2px 4px',
              }}
            >
              Don't show again
            </button>
          </div>
        )}

        {/* ── Package Updating Section matching Image 1 ──────────── */}
        {isUpdating && (
          <div style={{ padding: '14px 18px 4px' }}>
            <div style={{ fontSize: 12.5, color: isLight ? '#4B5563' : '#9CA3AF', marginBottom: 10 }}>
              We're updating your packages.
            </div>
            <div style={{
              background: isLight ? '#F3F4F6' : '#23272E',
              border: `1px solid ${isLight ? 'rgba(0,0,0,0.08)' : 'rgba(255,255,255,0.08)'}`,
              borderRadius: 6,
              padding: '12px 14px',
              display: 'flex',
              alignItems: 'center',
              gap: 12,
            }}>
              <span style={{
                display: 'inline-block',
                width: 12,
                height: 12,
                borderRadius: '50%',
                border: '2px solid rgba(16, 185, 129, 0.3)',
                borderTopColor: '#10B981',
                animation: 'saha-spin 0.8s linear infinite',
              }} />
              <span style={{ fontSize: 12.5, fontWeight: 600, color: isLight ? '#111827' : '#FFFFFF' }}>
                Updating
              </span>
              <div style={{ flex: 1, margin: '0 8px', height: 4, background: 'rgba(255,255,255,0.1)', borderRadius: 2, overflow: 'hidden' }}>
                <div style={{ width: `${updatingProgress}%`, height: '100%', background: '#10B981', borderRadius: 2 }} />
              </div>
              <span style={{ ...Tnum, fontSize: 11, color: isLight ? '#6B7280' : '#9CA3AF' }}>
                {updatingProgress}% | 22s
              </span>
            </div>
          </div>
        )}

        {/* ── Active Projects List or Empty State ────────────────── */}
        {uploads.length > 0 ? (
          <>
            {/* Summary Row */}
            <div style={{
              padding: '12px 18px 8px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              fontSize: 11.5,
              color: isLight ? '#6B7280' : '#9CA3AF',
            }}>
              <div>
                {uploads.length} project{uploads.length > 1 ? 's' : ''} | {totalPhotos} photos | Time left: {activeCount > 0 ? 'Calculating...' : 'Done'}
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                {activeCount > 0 && (
                  <button
                    type="button"
                    onClick={isPaused ? onResumeAll : onPauseAll}
                    style={{
                      background: 'transparent',
                      border: 'none',
                      color: isLight ? '#2563EB' : '#93C5FD',
                      fontFamily: F,
                      fontSize: 11.5,
                      fontWeight: 600,
                      cursor: 'pointer',
                      padding: 0,
                    }}
                  >
                    {isPaused ? 'Resume all' : 'Pause all'}
                  </button>
                )}
                {uploads.some((u) => u.status === 'complete') && (
                  <>
                    <span>|</span>
                    <button
                      type="button"
                      onClick={onDismissCompleted}
                      style={{
                        background: 'transparent',
                        border: 'none',
                        color: isLight ? '#2563EB' : '#93C5FD',
                        fontFamily: F,
                        fontSize: 11.5,
                        fontWeight: 600,
                        cursor: 'pointer',
                        padding: 0,
                      }}
                    >
                      Dismiss completed
                    </button>
                  </>
                )}
              </div>
            </div>

            {/* Project Upload Cards */}
            <div style={{ padding: '0 16px 16px', display: 'flex', flexDirection: 'column', gap: 10 }}>
              {uploads.map((item) => {
                const isItemComplete = item.status === 'complete';
                const isItemPaused = item.status === 'paused';
                const progressPct = Math.min(100, Math.round(item.percent || (isItemComplete ? 100 : 0)));

                return (
                  <div
                    key={item.id}
                    style={{
                      background: isLight ? '#F9FAFB' : '#23272E',
                      border: `1px solid ${isLight ? 'rgba(0,0,0,0.08)' : 'rgba(255,255,255,0.08)'}`,
                      borderRadius: 8,
                      padding: '12px 16px',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: 8,
                    }}
                  >
                    {/* Project Header */}
                    <div style={{ fontSize: 13, fontWeight: 600, color: isLight ? '#111827' : '#FFFFFF' }}>
                      {item.name || 'Untitled Project'}
                    </div>

                    {/* Status & Action matching Image 5 & Image 6 */}
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8, flex: 1, minWidth: 0 }}>
                        {isItemComplete ? (
                          <span style={{ color: '#10B981', fontSize: 13, fontWeight: 700 }}>✓</span>
                        ) : isItemPaused ? (
                          <span style={{ color: '#F59E0B', fontSize: 12 }}>⏸</span>
                        ) : (
                          <span style={{
                            display: 'inline-block',
                            width: 10,
                            height: 10,
                            borderRadius: '50%',
                            border: '1.5px solid rgba(16, 185, 129, 0.3)',
                            borderTopColor: '#10B981',
                            animation: 'saha-spin 0.8s linear infinite',
                          }} />
                        )}

                        <span style={{ fontSize: 12, color: isLight ? '#4B5563' : '#D1D5DB' }}>
                          {isItemComplete
                            ? `Upload for editing Upload complete. You uploaded ${item.totalPhotos} photos.`
                            : isItemPaused
                            ? `Paused at ${item.processedPhotos || 0}/${item.totalPhotos} photos`
                            : `Upload for editing ${item.totalPhotos || 5} photos`}
                        </span>

                        {/* Progress Bar (Visible while uploading) */}
                        {!isItemComplete && (
                          <>
                            <div style={{
                              flex: 1,
                              maxWidth: 160,
                              height: 5,
                              borderRadius: 3,
                              background: isLight ? 'rgba(0,0,0,0.1)' : 'rgba(255,255,255,0.12)',
                              overflow: 'hidden',
                              margin: '0 6px',
                            }}>
                              <div style={{
                                width: `${progressPct}%`,
                                height: '100%',
                                background: '#10B981',
                                borderRadius: 3,
                                transition: 'width 0.2s ease',
                              }} />
                            </div>

                            <span style={{ ...Tnum, fontSize: 11, color: isLight ? '#4B5563' : '#9CA3AF' }}>
                              {progressPct}%
                            </span>
                          </>
                        )}
                      </div>

                      {/* Cancel / Dismiss Button */}
                      {isItemComplete ? (
                        <button
                          type="button"
                          onClick={() => onCancelUpload?.(item.id)}
                          style={{
                            background: 'transparent',
                            border: 'none',
                            color: isLight ? '#6B7280' : '#9CA3AF',
                            fontFamily: F,
                            fontSize: 11.5,
                            fontWeight: 600,
                            cursor: 'pointer',
                            padding: '2px 4px',
                            transition: 'color 0.15s ease',
                          }}
                          onMouseEnter={(e) => { e.currentTarget.style.color = '#FFFFFF'; }}
                          onMouseLeave={(e) => { e.currentTarget.style.color = isLight ? '#6B7280' : '#9CA3AF'; }}
                        >
                          Dismiss
                        </button>
                      ) : (
                        <button
                          type="button"
                          onClick={() => onCancelUpload?.(item.id)}
                          style={{
                            background: 'transparent',
                            border: 'none',
                            color: isLight ? '#6B7280' : '#9CA3AF',
                            fontFamily: F,
                            fontSize: 11.5,
                            cursor: 'pointer',
                            padding: '2px 4px',
                            transition: 'color 0.15s ease',
                          }}
                          onMouseEnter={(e) => { e.currentTarget.style.color = '#EF4444'; }}
                          onMouseLeave={(e) => { e.currentTarget.style.color = isLight ? '#6B7280' : '#9CA3AF'; }}
                        >
                          Cancel upload
                        </button>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </>
        ) : !isUpdating ? (
          /* ── Empty State matching Image 3 ───────────────────────── */
          <div style={{
            padding: '52px 24px 60px',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 16,
            textAlign: 'center',
          }}>
            {/* Stylized Folder / Cloud upload icon matching Image 3 */}
            <div style={{
              width: 64,
              height: 64,
              borderRadius: 14,
              background: isLight ? '#F3F4F6' : '#2A2F38',
              border: `1px solid ${isLight ? 'rgba(0,0,0,0.08)' : 'rgba(255,255,255,0.1)'}`,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              position: 'relative',
              boxShadow: '0 8px 24px rgba(0, 0, 0, 0.25)',
            }}>
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke={isLight ? '#6B7280' : '#8B93A2'} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
                <line x1="12" y1="11" x2="12" y2="17" />
                <polyline points="9 14 12 11 15 14" />
              </svg>
            </div>

            <div style={{ fontSize: 13, color: isLight ? '#6B7280' : '#9CA3AF', maxWidth: 260, lineHeight: 1.4 }}>
              You will see all project uploads here
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}
