// Project Detail & Photo Inspection View matching Imagen AI reference
// Reference: media_1794864635431_f92b20ea.jpg (Shoot grid with "Editing...", bottom "• Editing in progress" pill,
// left filter sidebar, right tool rail, breadcrumb "Projects / Project Name")

import { useState, useMemo } from 'react';
import SONNA from '../tokens.js';
import { TitleBar } from './shell.jsx';

const F = SONNA.font;
const M = SONNA.mono;
const Tnum = { fontFamily: SONNA.mono, fontVariantNumeric: 'tabular-nums' };

// Event and wedding photo scenes for visual realism matching reference
const PHOTO_SCENES = [
  { id: 1, title: 'Banquet Entrance', aspect: '4/3', gradient: 'linear-gradient(135deg, #2D2721 0%, #4A3E31 50%, #3B3025 100%)' },
  { id: 2, title: 'Catering Display', aspect: '4/3', gradient: 'linear-gradient(135deg, #1F2730 0%, #354354 50%, #222B36 100%)' },
  { id: 3, title: 'Toast & Cheers', aspect: '4/3', gradient: 'linear-gradient(135deg, #32252B 0%, #523A44 50%, #36262E 100%)' },
  { id: 4, title: 'Couple Reception', aspect: '4/3', gradient: 'linear-gradient(135deg, #2E2A20 0%, #4D4636 50%, #332E23 100%)' },
  { id: 5, title: 'Guest Arrival', aspect: '4/3', gradient: 'linear-gradient(135deg, #1C2B27 0%, #2E4640 50%, #1F302B 100%)' },
  { id: 6, title: 'Hall Interior', aspect: '4/3', gradient: 'linear-gradient(135deg, #2A2430 0%, #463B51 50%, #2E2736 100%)' },
  { id: 7, title: 'First Dance', aspect: '4/3', gradient: 'linear-gradient(135deg, #332621 0%, #563E36 50%, #3A2A24 100%)' },
  { id: 8, title: 'Family Table', aspect: '4/3', gradient: 'linear-gradient(135deg, #22292E 0%, #38444D 50%, #262E34 100%)' },
  { id: 9, title: 'Speeches', aspect: '4/3', gradient: 'linear-gradient(135deg, #2D2423 0%, #4A3A38 50%, #342928 100%)' },
  { id: 10, title: 'Decor & Flowers', aspect: '4/3', gradient: 'linear-gradient(135deg, #252A22 0%, #3D4538 50%, #2B3027 100%)' },
  { id: 11, title: 'Celebration', aspect: '4/3', gradient: 'linear-gradient(135deg, #31272D 0%, #503E49 50%, #372A32 100%)' },
  { id: 12, title: 'Portraits', aspect: '4/3', gradient: 'linear-gradient(135deg, #2B2822 0%, #484338 50%, #322E27 100%)' },
];

export function ProjectDetailView({
  project,
  onBack,
  theme = 'dark',
  onToggleTheme,
  uploads = [],
  isUpdating = false,
  updatingProgress = 37,
  onCancelUpload,
  onPauseAll,
  onResumeAll,
  onDismissCompleted,
  isUploadsOpen,
  onToggleUploadsOpen,
}) {
  const isLight = theme === 'light';
  const photoCount = project?.photos || 21;
  const isProcessing = project?.status === 'processing' || project?.status === 'editing' || project?.status === 'uploading' || isUpdating;

  const [zoomLevel, setZoomLevel] = useState(1);
  const [selectedPhotoId, setSelectedPhotoId] = useState(null);
  const [activeRightTool, setActiveRightTool] = useState(null);
  const [seeOriginal, setSeeOriginal] = useState(false);

  // Generate list of photos matching project count
  const photos = useMemo(() => {
    return Array.from({ length: photoCount }, (_, i) => {
      const scene = PHOTO_SCENES[i % PHOTO_SCENES.length];
      return {
        id: i + 1,
        filename: `DSC_${String(1001 + i).padStart(4, '0')}.RAW`,
        scene: scene.title,
        gradient: scene.gradient,
        isEditing: isProcessing,
      };
    });
  }, [photoCount, isProcessing]);

  return (
    <div style={{
      width: '100%',
      height: '100%',
      background: isLight ? '#F9FAFB' : '#141619',
      color: isLight ? '#111827' : '#F3F4F6',
      fontFamily: F,
      display: 'flex',
      flexDirection: 'column',
      overflow: 'hidden',
    }}>
      {/* ── Top Bar with TitleBar & Breadcrumb Controls ────────── */}
      <TitleBar
        theme={theme}
        onToggleTheme={onToggleTheme}
        onNavigate={onBack}
        uploads={uploads}
        isUpdating={isUpdating}
        updatingProgress={updatingProgress}
        onCancelUpload={onCancelUpload}
        onPauseAll={onPauseAll}
        onResumeAll={onResumeAll}
        onDismissCompleted={onDismissCompleted}
        isUploadsOpen={isUploadsOpen}
        onToggleUploadsOpen={onToggleUploadsOpen}
      />

      {/* ── Subheader Bar matching Imagen reference ─────────────── */}
      <div style={{
        height: 44,
        flexShrink: 0,
        background: isLight ? '#FFFFFF' : '#1A1D21',
        borderBottom: `1px solid ${isLight ? 'rgba(0,0,0,0.08)' : 'rgba(255,255,255,0.08)'}`,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 16px',
      }}>
        {/* Breadcrumb: Projects / Project Name */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13 }}>
          <button
            type="button"
            onClick={onBack}
            style={{
              background: 'transparent',
              border: 'none',
              color: isLight ? '#6B7280' : '#9CA3AF',
              fontFamily: F,
              fontSize: 13,
              fontWeight: 500,
              cursor: 'pointer',
              padding: 0,
            }}
            onMouseEnter={(e) => { e.currentTarget.style.color = isLight ? '#111827' : '#FFFFFF'; }}
            onMouseLeave={(e) => { e.currentTarget.style.color = isLight ? '#6B7280' : '#9CA3AF'; }}
          >
            Projects
          </button>
          <span style={{ color: isLight ? '#D1D5DB' : '#4B5563' }}>/</span>
          <span style={{ fontWeight: 700, color: isLight ? '#111827' : '#FFFFFF' }}>
            {project?.name || 'Untitled Project'}
          </span>
        </div>

        {/* Action Controls matching Imagen */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <button
            type="button"
            style={{
              background: 'transparent',
              border: 'none',
              color: isLight ? '#4B5563' : '#D1D5DB',
              fontFamily: F,
              fontSize: 12.5,
              fontWeight: 500,
              cursor: 'pointer',
              padding: '6px 10px',
              borderRadius: 6,
            }}
          >
            Actions ▾
          </button>

          <button
            type="button"
            style={{
              background: 'transparent',
              border: 'none',
              color: isLight ? '#4B5563' : '#D1D5DB',
              fontFamily: F,
              fontSize: 12.5,
              fontWeight: 500,
              cursor: 'pointer',
              padding: '6px 10px',
              borderRadius: 6,
            }}
          >
            Edit project again ▾
          </button>

          <button
            type="button"
            style={{
              padding: '5px 14px',
              background: isLight ? '#E5E7EB' : 'rgba(255, 255, 255, 0.1)',
              border: `1px solid ${isLight ? 'rgba(0,0,0,0.12)' : 'rgba(255,255,255,0.14)'}`,
              borderRadius: 6,
              color: isLight ? '#111827' : '#FFFFFF',
              fontFamily: F,
              fontSize: 12,
              fontWeight: 600,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: 4,
            }}
          >
            <span>Export</span>
            <span style={{ fontSize: 10 }}>▾</span>
          </button>
        </div>
      </div>

      {/* ── Main Workspace: 3-column Layout ────────────────────── */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        {/* Left Sidebar: Filters & Sorters matching Image 1 */}
        <div style={{
          width: 210,
          flexShrink: 0,
          background: isLight ? '#F9FAFB' : '#171A1E',
          borderRight: `1px solid ${isLight ? 'rgba(0,0,0,0.08)' : 'rgba(255,255,255,0.08)'}`,
          display: 'flex',
          flexDirection: 'column',
          overflowY: 'auto',
          padding: '14px 12px',
          gap: 16,
          fontSize: 12,
        }}>
          {/* Sort By Dropdown */}
          <div>
            <div style={{ fontSize: 11, fontWeight: 700, color: isLight ? '#6B7280' : '#9CA3AF', marginBottom: 6 }}>
              Sort by
            </div>
            <button
              type="button"
              style={{
                width: '100%',
                padding: '6px 10px',
                background: isLight ? '#FFFFFF' : '#21252C',
                border: `1px solid ${isLight ? 'rgba(0,0,0,0.12)' : 'rgba(255,255,255,0.12)'}`,
                borderRadius: 6,
                color: isLight ? '#111827' : '#FFFFFF',
                fontFamily: F,
                fontSize: 12,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                cursor: 'pointer',
              }}
            >
              <span>Capture date</span>
              <span style={{ fontSize: 10 }}>▾</span>
            </button>
          </div>

          {/* Filters Header */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', paddingTop: 4 }}>
            <span style={{ fontWeight: 700, fontSize: 12, color: isLight ? '#111827' : '#FFFFFF' }}>Filters</span>
            <span style={{ fontSize: 11, color: isLight ? '#9CA3AF' : '#6B7280', cursor: 'pointer' }}>Clear all</span>
          </div>

          {/* Accordion 1: Collections */}
          <div style={{ borderTop: `1px solid ${isLight ? 'rgba(0,0,0,0.06)' : 'rgba(255,255,255,0.06)'}`, paddingTop: 10 }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontWeight: 600, color: isLight ? '#374151' : '#D1D5DB' }}>
              <span>⌄ Collections</span>
              <span style={{ color: '#E05353', cursor: 'pointer', fontSize: 11 }}>+ Add</span>
            </div>
            <div style={{ fontSize: 11, color: isLight ? '#9CA3AF' : '#6B7280', marginTop: 6, paddingLeft: 10 }}>
              No collections yet
            </div>
          </div>

          {/* Accordion 2: Camera type */}
          <div style={{ borderTop: `1px solid ${isLight ? 'rgba(0,0,0,0.06)' : 'rgba(255,255,255,0.06)'}`, paddingTop: 10 }}>
            <div style={{ fontWeight: 600, color: isLight ? '#374151' : '#D1D5DB', marginBottom: 6 }}>
              ⌄ Camera type
            </div>
            <button
              type="button"
              style={{
                width: '100%',
                padding: '5px 8px',
                background: isLight ? '#FFFFFF' : '#21252C',
                border: `1px solid ${isLight ? 'rgba(0,0,0,0.1)' : 'rgba(255,255,255,0.1)'}`,
                borderRadius: 5,
                color: isLight ? '#111827' : '#FFFFFF',
                fontFamily: F,
                fontSize: 11.5,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
              }}
            >
              <span>All cameras</span>
              <span style={{ fontSize: 10 }}>▾</span>
            </button>
          </div>

          {/* Accordion 3: Stars */}
          <div style={{ borderTop: `1px solid ${isLight ? 'rgba(0,0,0,0.06)' : 'rgba(255,255,255,0.06)'}`, paddingTop: 10 }}>
            <div style={{ fontWeight: 600, color: isLight ? '#374151' : '#D1D5DB', marginBottom: 6 }}>
              ⌄ Stars
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4, paddingLeft: 6 }}>
              {[
                { stars: '★★★★★', count: 0 },
                { stars: '★★★★', count: 0 },
                { stars: '★★★', count: 0 },
                { stars: '★★', count: 0 },
                { stars: '★', count: 0 },
                { stars: 'Unrated', count: photoCount },
              ].map((row) => (
                <div key={row.stars} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: isLight ? '#6B7280' : '#9CA3AF' }}>
                  <span>{row.stars}</span>
                  <span style={{ ...Tnum }}>{row.count}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Accordion 4: Colors */}
          <div style={{ borderTop: `1px solid ${isLight ? 'rgba(0,0,0,0.06)' : 'rgba(255,255,255,0.06)'}`, paddingTop: 10 }}>
            <div style={{ fontWeight: 600, color: isLight ? '#374151' : '#D1D5DB', marginBottom: 6 }}>
              ⌄ Colors
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4, paddingLeft: 6 }}>
              {[
                { label: 'Blue', color: '#3B82F6', count: 0 },
                { label: 'Green', color: '#10B981', count: 0 },
                { label: 'Yellow', color: '#F59E0B', count: 0 },
                { label: 'Red', color: '#EF4444', count: 0 },
                { label: 'Purple', color: '#8B5CF6', count: 0 },
                { label: 'No label', color: '#6B7280', count: photoCount },
              ].map((row) => (
                <div key={row.label} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: 11, color: isLight ? '#6B7280' : '#9CA3AF' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <span style={{ width: 8, height: 8, borderRadius: '50%', background: row.color }} />
                    <span>{row.label}</span>
                  </div>
                  <span style={{ ...Tnum }}>{row.count}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Accordion 5: Flags */}
          <div style={{ borderTop: `1px solid ${isLight ? 'rgba(0,0,0,0.06)' : 'rgba(255,255,255,0.06)'}`, paddingTop: 10 }}>
            <div style={{ fontWeight: 600, color: isLight ? '#374151' : '#D1D5DB', marginBottom: 6 }}>
              ⌄ Flags
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4, paddingLeft: 6, fontSize: 11, color: isLight ? '#6B7280' : '#9CA3AF' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>⚐ Flagged</span>
                <span style={{ ...Tnum }}>0</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>Unflagged</span>
                <span style={{ ...Tnum }}>{photoCount}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Center: Photos Grid with "Editing..." overlay badge matching Image 1 */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          <div style={{
            flex: 1,
            overflowY: 'auto',
            padding: 16,
            background: isLight ? '#F3F4F6' : '#141619',
          }}>
            <div style={{
              display: 'grid',
              gridTemplateColumns: `repeat(auto-fill, minmax(${Math.round(200 * zoomLevel)}px, 1fr))`,
              gap: 12,
            }}>
              {photos.map((item) => {
                const isSelected = selectedPhotoId === item.id;
                return (
                  <div
                    key={item.id}
                    onClick={() => setSelectedPhotoId(item.id)}
                    style={{
                      aspectRatio: '4/3',
                      borderRadius: 8,
                      background: item.gradient,
                      border: isSelected ? '2px solid #3B82F6' : `1px solid ${isLight ? 'rgba(0,0,0,0.1)' : 'rgba(255,255,255,0.08)'}`,
                      position: 'relative',
                      overflow: 'hidden',
                      cursor: 'pointer',
                      boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
                      transition: 'transform 0.15s ease, border-color 0.15s ease',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                    }}
                  >
                    {/* Subtle photo frame illustration */}
                    <div style={{ opacity: 0.25, color: '#FFFFFF', textAlign: 'center' }}>
                      <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.2">
                        <rect x="3" y="3" width="18" height="18" rx="2" />
                        <circle cx="8.5" cy="8.5" r="1.5" />
                        <polyline points="21 15 16 10 5 21" />
                      </svg>
                      <div style={{ fontSize: 10, marginTop: 4 }}>{item.scene}</div>
                    </div>

                    {/* Signature Overlay Badge: "Editing..." matching Imagen Image 1 */}
                    {item.isEditing ? (
                      <div style={{
                        position: 'absolute',
                        top: 8,
                        right: 8,
                        background: 'rgba(0, 0, 0, 0.72)',
                        backdropFilter: 'blur(4px)',
                        color: '#FFFFFF',
                        padding: '3px 9px',
                        borderRadius: 4,
                        fontSize: 11,
                        fontWeight: 600,
                        display: 'flex',
                        alignItems: 'center',
                        gap: 5,
                        boxShadow: '0 2px 6px rgba(0, 0, 0, 0.4)',
                        border: '1px solid rgba(255, 255, 255, 0.15)',
                      }}>
                        <span style={{
                          width: 6,
                          height: 6,
                          borderRadius: '50%',
                          background: '#60A5FA',
                          animation: 'saha-spin 1s linear infinite',
                        }} />
                        <span>Editing...</span>
                      </div>
                    ) : (
                      <div style={{
                        position: 'absolute',
                        top: 8,
                        right: 8,
                        background: 'rgba(16, 185, 129, 0.85)',
                        color: '#FFFFFF',
                        padding: '2px 7px',
                        borderRadius: 4,
                        fontSize: 10,
                        fontWeight: 700,
                      }}>
                        ✓ Edited
                      </div>
                    )}

                    {/* Bottom Filename tag on hover */}
                    <div style={{
                      position: 'absolute',
                      bottom: 6,
                      left: 8,
                      fontSize: 10,
                      fontFamily: M,
                      color: 'rgba(255,255,255,0.7)',
                      background: 'rgba(0,0,0,0.5)',
                      padding: '2px 5px',
                      borderRadius: 3,
                    }}>
                      {item.filename}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* ── Bottom Footer Toolbar matching Image 1 ──────────────── */}
          <div style={{
            height: 48,
            flexShrink: 0,
            background: isLight ? '#FFFFFF' : '#1A1D21',
            borderTop: `1px solid ${isLight ? 'rgba(0,0,0,0.08)' : 'rgba(255,255,255,0.08)'}`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '0 16px',
            fontSize: 12,
          }}>
            {/* Left Status Pill: "• Editing in progress" matching Imagen */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              {isProcessing ? (
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 7,
                  padding: '4px 12px',
                  borderRadius: 14,
                  background: 'rgba(224, 83, 83, 0.12)',
                  border: '1px solid rgba(224, 83, 83, 0.45)',
                  color: '#F87171',
                  fontSize: 11.5,
                  fontWeight: 600,
                }}>
                  <span style={{
                    width: 7,
                    height: 7,
                    borderRadius: '50%',
                    background: '#EF4444',
                    animation: 'saha-spin 1s linear infinite',
                  }} />
                  <span>Editing in progress</span>
                </div>
              ) : (
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  padding: '4px 12px',
                  borderRadius: 14,
                  background: 'rgba(16, 185, 129, 0.12)',
                  border: '1px solid rgba(16, 185, 129, 0.45)',
                  color: '#34D399',
                  fontSize: 11.5,
                  fontWeight: 600,
                }}>
                  <span>✓</span>
                  <span>{photoCount} photos edited</span>
                </div>
              )}

              {/* View layout icons */}
              <button
                type="button"
                title="Filter view"
                style={{
                  background: 'transparent',
                  border: 'none',
                  color: isLight ? '#6B7280' : '#9CA3AF',
                  cursor: 'pointer',
                  padding: 4,
                }}
              >
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3" />
                </svg>
              </button>

              <button
                type="button"
                title="Grid view"
                style={{
                  background: 'transparent',
                  border: 'none',
                  color: isLight ? '#111827' : '#FFFFFF',
                  cursor: 'pointer',
                  padding: 4,
                }}
              >
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <rect x="3" y="3" width="7" height="7" />
                  <rect x="14" y="3" width="7" height="7" />
                  <rect x="14" y="14" width="7" height="7" />
                  <rect x="3" y="14" width="7" height="7" />
                </svg>
              </button>
            </div>

            {/* Center zoom slider */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <input
                type="range"
                min="0.7"
                max="1.6"
                step="0.1"
                value={zoomLevel}
                onChange={(e) => setZoomLevel(parseFloat(e.target.value))}
                style={{
                  width: 90,
                  accentColor: '#E05353',
                  cursor: 'pointer',
                }}
              />
            </div>

            {/* Right photo count and "See original" toggle */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 14, color: isLight ? '#6B7280' : '#9CA3AF' }}>
              <span style={{ ...Tnum }}>{photoCount} photos • {photoCount} total</span>
              <span>↑↓ Capture date ↑</span>
              <button
                type="button"
                onClick={() => setSeeOriginal(!seeOriginal)}
                style={{
                  padding: '3px 10px',
                  borderRadius: 4,
                  background: seeOriginal ? '#2563EB' : 'transparent',
                  border: `1px solid ${seeOriginal ? '#2563EB' : (isLight ? 'rgba(0,0,0,0.14)' : 'rgba(255,255,255,0.18)')}`,
                  color: seeOriginal ? '#FFFFFF' : (isLight ? '#374151' : '#D1D5DB'),
                  fontSize: 11.5,
                  fontWeight: 600,
                  cursor: 'pointer',
                }}
              >
                See original
              </button>
            </div>
          </div>
        </div>

        {/* Right Tools Sidebar matching Imagen Image 1 */}
        <div style={{
          width: 58,
          flexShrink: 0,
          background: isLight ? '#F9FAFB' : '#171A1E',
          borderLeft: `1px solid ${isLight ? 'rgba(0,0,0,0.08)' : 'rgba(255,255,255,0.08)'}`,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          padding: '12px 0',
          gap: 14,
        }}>
          {[
            { id: 'edit', label: 'Edit', icon: '✎' },
            { id: 'crop', label: 'Crop', icon: '◫' },
            { id: 'masks', label: 'Masks', icon: '◎' },
            { id: 'cleanup', label: 'Cleanup', icon: '✦' },
            { id: 'retouch', label: 'Retouch', icon: '🪄' },
            { id: 'presets', label: 'Presets', icon: '🗂' },
          ].map((tool) => (
            <button
              key={tool.id}
              type="button"
              onClick={() => setActiveRightTool(tool.id)}
              title={tool.label}
              style={{
                width: 44,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: 2,
                background: activeRightTool === tool.id ? (isLight ? '#E5E7EB' : '#2A2E36') : 'transparent',
                border: 'none',
                borderRadius: 6,
                padding: '6px 0',
                color: activeRightTool === tool.id ? '#FFFFFF' : (isLight ? '#6B7280' : '#9CA3AF'),
                cursor: 'pointer',
                transition: 'all 0.15s ease',
              }}
            >
              <span style={{ fontSize: 14 }}>{tool.icon}</span>
              <span style={{ fontSize: 9.5, fontWeight: 500 }}>{tool.label}</span>
            </button>
          ))}

          <div style={{ flex: 1 }} />

          {[
            { id: 'versions', label: 'Versions', icon: '🗎' },
            { id: 'history', label: 'History', icon: '↺' },
            { id: 'info', label: 'Info', icon: 'ⓘ' },
          ].map((tool) => (
            <button
              key={tool.id}
              type="button"
              onClick={() => setActiveRightTool(tool.id)}
              title={tool.label}
              style={{
                width: 44,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: 2,
                background: activeRightTool === tool.id ? (isLight ? '#E5E7EB' : '#2A2E36') : 'transparent',
                border: 'none',
                borderRadius: 6,
                padding: '6px 0',
                color: activeRightTool === tool.id ? '#FFFFFF' : (isLight ? '#6B7280' : '#9CA3AF'),
                cursor: 'pointer',
              }}
            >
              <span style={{ fontSize: 14 }}>{tool.icon}</span>
              <span style={{ fontSize: 9.5, fontWeight: 500 }}>{tool.label}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
