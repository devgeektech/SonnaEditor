// New Project Ingestion & 4-Step Setup Wizard
// References:
// - media_1794859550508_b0c44c08.png (Step 2: Project setup)
// - media_1794859805052_6208f135.png (Step 3: Your workflow - AI Profile selection only)
// - media_1794860095467_820710b1.jpg (Step 4: AI tools - Crop, Straighten, Denoise)
// - media_1794847047939_843f4cc2.jpg & media_1794847055275_aa059f34.jpg (Step 1: Choose a source)

import { useEffect, useRef, useState } from 'react';
import SONNA from '../tokens.js';

const F = SONNA.font;
const M = SONNA.mono;

const DENOISE_STRENGTHS = [
  { id: 'subtle', label: 'Subtle', val: 15 },
  { id: 'moderate', label: 'Moderate', val: 25 },
  { id: 'strong', label: 'Strong', val: 40 },
  { id: 'stronger', label: 'Stronger', val: 60 },
];

const CROP_ASPECT_RATIOS = [
  { id: 'original', label: 'Original' },
  { id: '1:1', label: '1:1' },
  { id: '4:5', label: '4:5' },
  { id: '16:9', label: '16:9' },
  { id: 'free', label: 'Free' },
];

const PHOTOGRAPHY_TYPES = [
  { id: 'wedding', label: 'Wedding' },
  { id: 'school', label: 'School' },
  { id: 'family', label: 'Family / newborn' },
  { id: 'sports', label: 'Sports' },
  { id: 'events', label: 'Events' },
  { id: 'real_estate', label: 'Real estate' },
  { id: 'portraits', label: 'Portraits' },
  { id: 'boudoir', label: 'Boudoir' },
  { id: 'landscape', label: 'Landscape / nature' },
  { id: 'other', label: 'Other' },
];

function renderGenreIcon(id, size = 22) {
  const stroke = 'currentColor';
  const strokeWidth = 1.6;

  switch (id) {
    case 'wedding':
      return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={stroke} strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round">
          <circle cx="9" cy="14" r="5" />
          <circle cx="15" cy="10" r="5" />
          <path d="M12 5l1.5-2 1.5 2" />
        </svg>
      );
    case 'school':
      return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={stroke} strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round">
          <path d="M22 10v6M2 10l10-5 10 5-10 5z" />
          <path d="M6 12v5c0 2 3 3 6 3s6-1 6-3v-5" />
        </svg>
      );
    case 'family':
      return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={stroke} strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round">
          <path d="M19 12h2l-3-6H8a5 5 0 0 0-5 5v5h12" />
          <circle cx="7" cy="18" r="2" />
          <circle cx="17" cy="18" r="2" />
        </svg>
      );
    case 'sports':
      return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={stroke} strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="9" />
          <path d="M12 3a9 9 0 0 1 9 9" />
          <path d="M3 12a9 9 0 0 1 9 9" />
          <path d="M7 7l10 10" />
        </svg>
      );
    case 'events':
      return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={stroke} strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round">
          <rect x="3" y="6" width="18" height="12" rx="2" />
          <path d="M7 6v12M17 6v12M12 11v2" />
        </svg>
      );
    case 'real_estate':
      return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={stroke} strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round">
          <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
          <polyline points="9 22 9 12 15 12 15 22" />
        </svg>
      );
    case 'portraits':
      return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={stroke} strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="9" r="4" />
          <path d="M6 20v-1a6 6 0 0 1 12 0v1" />
        </svg>
      );
    case 'boudoir':
      return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={stroke} strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round">
          <path d="M8 4c0 2 2 3 4 3s4-1 4-3" />
          <path d="M7 10c0 4 3 6 5 6s5-2 5-6" />
          <path d="M6 21c2-5 3-7 6-7s4 2 6 7" />
        </svg>
      );
    case 'landscape':
      return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={stroke} strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round">
          <path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z" />
          <path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12" />
        </svg>
      );
    case 'other':
    default:
      return (
        <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={stroke} strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round">
          <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
          <circle cx="12" cy="13" r="4" />
        </svg>
      );
  }
}

export function NewProjectModal({
  isOpen,
  onClose,
  profiles = [],
  activeProfile = null,
  onPickFolder,
  onPickCatalog,
  onCreateProject,
  onOpenCreateProfile,
  newlyCreatedProfileId = null,
}) {
  // Step state: 'source' (1) | 'setup' (2) | 'workflow' (3) | 'tools' (4)
  const [step, setStep] = useState('source');
  const [sourceType, setSourceType] = useState('folder'); // 'folder' | 'catalog'
  const [sourcePath, setSourcePath] = useState('');
  const [projectName, setProjectName] = useState('');
  const [selectedProfileId, setSelectedProfileId] = useState(activeProfile?.id || (profiles[0]?.id || ''));
  const [genre, setGenre] = useState('wedding');
  const [detectedPhotoCount, setDetectedPhotoCount] = useState(0);
  const [isDragging, setIsDragging] = useState(false);

  // Step 4: AI Tools (Strictly Crop, Straighten, Denoise per user prompt)
  const [cropEnabled, setCropEnabled] = useState(true);
  const [cropAspectRatio, setCropAspectRatio] = useState('original');
  const [autoStraighten, setAutoStraighten] = useState(true);
  const [denoiseEnabled, setDenoiseEnabled] = useState(true);
  const [denoiseStrength, setDenoiseStrength] = useState('moderate');
  const [denoiseIsoThreshold, setDenoiseIsoThreshold] = useState(1200);

  // Step 4: AI Tools category and preview
  const [toolCategory, setToolCategory] = useState('recommended'); // 'recommended' | 'composition'
  const [previewTool, setPreviewTool] = useState('straighten'); // 'straighten' | 'crop'
  const [splitPos, setSplitPos] = useState(50); // 0-100

  // Step 3 Profile Dropdown state
  const [isProfileDropdownOpen, setIsProfileDropdownOpen] = useState(false);
  const dropdownRef = useRef(null);

  useEffect(() => {
    const handleOutsideClick = (e) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
        setIsProfileDropdownOpen(false);
      }
    };
    if (isProfileDropdownOpen) {
      document.addEventListener('mousedown', handleOutsideClick);
      return () => document.removeEventListener('mousedown', handleOutsideClick);
    }
  }, [isProfileDropdownOpen]);

  // Track modal open transitions so we only reset when opening fresh,
  // NOT when child modals (like profile creation) open and close over it.
  const wasOpenRef = useRef(false);

  useEffect(() => {
    if (isOpen && !wasOpenRef.current) {
      setStep('source');
      setSourceType('folder');
      setSourcePath('');
      setProjectName('');
      setDetectedPhotoCount(0);
      setIsDragging(false);
      setPreviewTool('crop');
      setSplitPos(50);
      if (activeProfile?.id) {
        setSelectedProfileId(activeProfile.id);
      } else if (profiles[0]?.id) {
        setSelectedProfileId(profiles[0].id);
      } else {
        setSelectedProfileId('');
      }
    }
    wasOpenRef.current = isOpen;
  }, [isOpen]);

  // When a new profile is created, auto-select it immediately
  useEffect(() => {
    if (newlyCreatedProfileId) {
      setSelectedProfileId(newlyCreatedProfileId);
    }
  }, [newlyCreatedProfileId]);

  // Keep selectedProfileId in sync when profiles change or first load
  useEffect(() => {
    if (!selectedProfileId && profiles.length > 0) {
      setSelectedProfileId(newlyCreatedProfileId || activeProfile?.id || profiles[0].id);
    }
  }, [profiles, selectedProfileId, activeProfile, newlyCreatedProfileId]);

  if (!isOpen) return null;

  const handleSelectFolder = async () => {
    if (onPickFolder) {
      const res = await onPickFolder();
      if (res && res.path) {
        setSourceType('folder');
        setSourcePath(res.path);
        const name = res.path.split(/[\\/]/).filter(Boolean).pop() || 'Untitled Project';
        setProjectName(name);
        setDetectedPhotoCount(res.raw_count || 0);
        setStep('setup');
      }
    }
  };

  const handleSelectCatalog = async () => {
    if (onPickCatalog) {
      const res = await onPickCatalog();
      if (res && res.path) {
        setSourceType('catalog');
        setSourcePath(res.path);
        const name = res.path.split(/[\\/]/).filter(Boolean).pop()?.replace(/\.lrcat$/i, '') || 'Catalog Project';
        setProjectName(name);
        setDetectedPhotoCount(res.raw_count || 0);
        setStep('setup');
      }
    }
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    setIsDragging(false);
    const files = e.dataTransfer?.files;
    if (files && files.length > 0) {
      const droppedPath = files[0].path;
      if (droppedPath) {
        const isLrcat = droppedPath.toLowerCase().endsWith('.lrcat');
        setSourceType(isLrcat ? 'catalog' : 'folder');
        setSourcePath(droppedPath);
        const name = droppedPath.split(/[\\/]/).filter(Boolean).pop()?.replace(/\.lrcat$/i, '') || 'Project';
        setProjectName(name);
        setDetectedPhotoCount(files.length);
        setStep('setup');
      }
    }
  };

  const handleStart = () => {
    const finalName = projectName?.trim() || (sourcePath ? sourcePath.split(/[\\/]/).filter(Boolean).pop() : 'Sample_Raw');
    onCreateProject?.({
      sourceType,
      sourcePath,
      folderPath: sourcePath,
      name: finalName,
      projectName: finalName,
      rawCount: detectedPhotoCount || 21,
      detectedPhotoCount: detectedPhotoCount || 21,
      profileId: selectedProfileId,
      genre,
      cropEnabled,
      cropAspectRatio,
      autoStraighten,
      denoiseEnabled,
      denoiseStrength,
      denoiseIsoThreshold,
    });
    onClose?.();
  };

  const currentGenreObj = PHOTOGRAPHY_TYPES.find((g) => g.id === genre) || PHOTOGRAPHY_TYPES[0];
  const selectedProfile = profiles.find((p) => p.id === selectedProfileId);
  const compActiveCount = (cropEnabled ? 1 : 0) + (autoStraighten ? 1 : 0);

  // Stepper definition
  const STEPS = [
    { id: 'source', label: 'Photos', num: 1 },
    { id: 'setup', label: 'Project setup', num: 2 },
    { id: 'workflow', label: 'Your workflow', num: 3 },
    { id: 'tools', label: 'AI tools', num: 4 },
  ];

  const stepOrder = ['source', 'setup', 'workflow', 'tools'];
  const currentStepIdx = stepOrder.indexOf(step);

  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 1000,
        background: 'rgba(0, 0, 0, 0.78)',
        backdropFilter: 'blur(6px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 24,
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          width: '100%',
          maxWidth: 940,
          background: '#16181C',
          border: '1px solid rgba(255, 255, 255, 0.12)',
          borderRadius: 14,
          boxShadow: '0 28px 72px rgba(0, 0, 0, 0.75)',
          overflow: 'hidden',
          display: 'flex',
          flexDirection: 'column',
          maxHeight: '92vh',
          minHeight: 600,
        }}
      >
        {/* Universal Header with Stepper (Exact match to Imagen AI screenshots) */}
        <div style={{
          padding: '16px 24px',
          borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          background: '#131417',
        }}>
          {/* Left Title */}
          <div style={{ fontSize: 16, fontWeight: 700, color: '#FFFFFF', letterSpacing: -0.2 }}>
            {step === 'source' && 'Choose a source'}
            {step === 'setup' && 'Project setup'}
            {step === 'workflow' && 'Your workflow'}
            {step === 'tools' && 'AI tools'}
          </div>

          {/* Stepper (Visible across steps 2-4) */}
          {step !== 'source' && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              {STEPS.map((s, idx) => {
                const isPassed = currentStepIdx > idx;
                const isCurrent = currentStepIdx === idx;

                return (
                  <div key={s.id} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <div style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 6,
                    }}>
                      <div style={{
                        width: 22,
                        height: 22,
                        borderRadius: '50%',
                        background: isPassed || isCurrent ? '#2563EB' : 'rgba(255, 255, 255, 0.08)',
                        color: isPassed || isCurrent ? '#FFFFFF' : '#888D96',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: 11,
                        fontWeight: 700,
                      }}>
                        {isPassed ? '✓' : s.num}
                      </div>
                      <span style={{
                        fontSize: 12.5,
                        fontWeight: isCurrent ? 600 : 500,
                        color: isCurrent || isPassed ? '#FFFFFF' : '#888D96',
                      }}>
                        {s.label}
                      </span>
                    </div>

                    {idx < STEPS.length - 1 && (
                      <div style={{
                        width: 28,
                        height: 1,
                        background: currentStepIdx > idx ? '#2563EB' : 'rgba(255, 255, 255, 0.16)',
                        margin: '0 4px',
                      }} />
                    )}
                  </div>
                );
              })}
            </div>
          )}

          {/* Close button */}
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            style={{
              width: 32,
              height: 32,
              background: 'transparent',
              border: 'none',
              color: '#9CA3AF',
              cursor: 'pointer',
              fontSize: 20,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              borderRadius: 6,
              transition: 'color 0.15s ease',
            }}
          >
            ×
          </button>
        </div>

        {/* Modal Body */}
        <div style={{ padding: '24px 28px', overflowY: 'auto', flex: 1, display: 'flex', flexDirection: 'column' }}>
          {/* STEP 1: Choose a source */}
          {step === 'source' && (
            <div style={{ display: 'flex', flexDirection: 'column', flex: 1, justifyContent: 'center' }}>
              {/* Segmented bar */}
              <div style={{
                background: '#1F2228',
                border: '1px solid rgba(255, 255, 255, 0.08)',
                borderRadius: 10,
                padding: '16px 20px',
                marginBottom: 20,
              }}>
                <div style={{ fontSize: 13, fontWeight: 600, color: '#E5E7EB', marginBottom: 12 }}>
                  Where are your photos?
                </div>
                <div style={{
                  display: 'inline-flex',
                  background: '#14161B',
                  border: '1px solid rgba(255, 255, 255, 0.07)',
                  borderRadius: 8,
                  padding: 3,
                  gap: 4,
                }}>
                  {/* Tab 1: Lightroom Classic catalog */}
                  <button
                    type="button"
                    onClick={() => setSourceType('catalog')}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 8,
                      padding: '7px 16px',
                      borderRadius: 6,
                      background: sourceType === 'catalog' ? '#2A2E36' : 'transparent',
                      border: sourceType === 'catalog' ? '1px solid rgba(255, 255, 255, 0.18)' : '1px solid transparent',
                      color: sourceType === 'catalog' ? '#FFFFFF' : '#8C929C',
                      fontSize: 12.5,
                      fontWeight: 600,
                      cursor: 'pointer',
                      transition: 'all 0.15s ease',
                    }}
                  >
                    <span style={{
                      fontSize: 10,
                      fontWeight: 800,
                      background: '#0284C7',
                      color: '#FFFFFF',
                      padding: '2px 5px',
                      borderRadius: 3,
                      letterSpacing: -0.2,
                    }}>
                      LrC
                    </span>
                    Lightroom Classic catalog
                  </button>

                  {/* Tab 2: Folder */}
                  <button
                    type="button"
                    onClick={() => setSourceType('folder')}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 8,
                      padding: '7px 16px',
                      borderRadius: 6,
                      background: sourceType === 'folder' ? '#2A2E36' : 'transparent',
                      border: sourceType === 'folder' ? '1px solid rgba(255, 255, 255, 0.18)' : '1px solid transparent',
                      color: sourceType === 'folder' ? '#FFFFFF' : '#8C929C',
                      fontSize: 12.5,
                      fontWeight: 600,
                      cursor: 'pointer',
                      transition: 'all 0.15s ease',
                    }}
                  >
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M4 20h16a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.93a2 2 0 0 1-1.66-.9l-.82-1.2A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13c0 1.1.9 2 2 2Z" />
                    </svg>
                    Folder
                  </button>
                </div>
              </div>

              {/* Dashed Drop Zone Box */}
              <div
                onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={handleDrop}
                style={{
                  border: `1.5px dashed ${isDragging ? '#E05353' : 'rgba(255, 255, 255, 0.18)'}`,
                  background: isDragging ? 'rgba(224, 83, 83, 0.05)' : 'rgba(255, 255, 255, 0.015)',
                  borderRadius: 10,
                  padding: '50px 24px',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  textAlign: 'center',
                  gap: 14,
                  transition: 'border-color 0.15s ease, background 0.15s ease',
                }}
              >
                {sourceType === 'catalog' ? (
                  <>
                    <div style={{ fontSize: 18, fontWeight: 700, color: '#FFFFFF' }}>
                      Choose a Lightroom Classic catalog
                    </div>
                    <div style={{ fontSize: 12, color: '#9CA3AF', display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer' }}>
                      <span>How do I find my catalog?</span>
                      <span style={{ fontSize: 13 }}>ⓘ</span>
                    </div>

                    <div style={{
                      margin: '10px 0',
                      width: 66,
                      height: 66,
                      borderRadius: 10,
                      background: '#1E293B',
                      border: '1px solid rgba(255, 255, 255, 0.16)',
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      justifyContent: 'center',
                      position: 'relative',
                      boxShadow: '0 8px 24px rgba(0, 0, 0, 0.4)',
                    }}>
                      <span style={{ fontSize: 17, fontWeight: 800, color: '#38BDF8', letterSpacing: -0.5 }}>LrC</span>
                      <span style={{ fontSize: 9.5, fontWeight: 700, color: '#94A3B8', letterSpacing: 0.5 }}>CAT</span>
                      <div style={{
                        position: 'absolute',
                        bottom: -7,
                        width: 22,
                        height: 22,
                        borderRadius: '50%',
                        background: '#3B82F6',
                        color: '#FFFFFF',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: 13,
                        fontWeight: 700,
                      }}>
                        ↑
                      </div>
                    </div>

                    <button
                      type="button"
                      onClick={handleSelectCatalog}
                      style={{
                        padding: '10px 24px',
                        background: '#E05353',
                        color: '#FFFFFF',
                        border: 'none',
                        borderRadius: 6,
                        fontFamily: F,
                        fontSize: 13,
                        fontWeight: 600,
                        cursor: 'pointer',
                        marginTop: 4,
                        boxShadow: '0 4px 14px rgba(224, 83, 83, 0.35)',
                      }}
                    >
                      My Lightroom catalogs ▾
                    </button>

                    <span style={{ fontSize: 11, color: '#6B7280', fontWeight: 600, letterSpacing: 0.5 }}>OR</span>

                    <button
                      type="button"
                      onClick={handleSelectCatalog}
                      style={{
                        padding: '8px 22px',
                        background: 'transparent',
                        border: '1px solid rgba(255, 255, 255, 0.22)',
                        borderRadius: 20,
                        color: '#E5E7EB',
                        fontFamily: F,
                        fontSize: 12.5,
                        fontWeight: 600,
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        gap: 6,
                      }}
                    >
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <circle cx="11" cy="11" r="8" />
                        <line x1="21" y1="21" x2="16.65" y2="16.65" />
                      </svg>
                      Browse catalog
                    </button>

                    <span style={{ fontSize: 11, color: '#6B7280', fontWeight: 600, letterSpacing: 0.5 }}>OR</span>
                    <span style={{ fontSize: 12, color: '#9CA3AF' }}>Drag &amp; drop</span>
                  </>
                ) : (
                  <>
                    <div style={{ fontSize: 18, fontWeight: 700, color: '#FFFFFF' }}>
                      Choose a folder
                    </div>
                    <div style={{ fontSize: 12, color: '#9CA3AF', display: 'flex', alignItems: 'center', gap: 4 }}>
                      <span>We support JPEG, TIFF, and RAW formats</span>
                      <span style={{ fontSize: 13, cursor: 'pointer' }}>ⓘ</span>
                    </div>

                    <div style={{
                      margin: '10px 0',
                      width: 66,
                      height: 66,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      position: 'relative',
                    }}>
                      <svg width="58" height="58" viewBox="0 0 24 24" fill="#334155" stroke="#475569" strokeWidth="1.2">
                        <path d="M2 6a2 2 0 0 1 2-2h4l2 2h10a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V6z" />
                      </svg>
                      <div style={{
                        position: 'absolute',
                        bottom: 0,
                        width: 22,
                        height: 22,
                        borderRadius: '50%',
                        background: '#3B82F6',
                        color: '#FFFFFF',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: 13,
                        fontWeight: 700,
                      }}>
                        ↑
                      </div>
                    </div>

                    <button
                      type="button"
                      onClick={handleSelectFolder}
                      style={{
                        padding: '10px 32px',
                        background: '#E05353',
                        color: '#FFFFFF',
                        border: 'none',
                        borderRadius: 6,
                        fontFamily: F,
                        fontSize: 13,
                        fontWeight: 600,
                        cursor: 'pointer',
                        marginTop: 4,
                        display: 'flex',
                        alignItems: 'center',
                        gap: 8,
                        boxShadow: '0 4px 14px rgba(224, 83, 83, 0.35)',
                      }}
                    >
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                        <circle cx="11" cy="11" r="8" />
                        <line x1="21" y1="21" x2="16.65" y2="16.65" />
                      </svg>
                      Browse
                    </button>

                    <span style={{ fontSize: 11, color: '#6B7280', fontWeight: 600, letterSpacing: 0.5 }}>OR</span>
                    <span style={{ fontSize: 12, color: '#9CA3AF' }}>Drag &amp; drop</span>
                  </>
                )}
              </div>
            </div>
          )}

          {/* STEP 2: Project setup (Exact match to media_1794859550508_b0c44c08.png) */}
          {step === 'setup' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
              {/* Top Card: Folder path & Project Name */}
              <div style={{
                background: '#1F2228',
                border: '1px solid rgba(255, 255, 255, 0.08)',
                borderRadius: 10,
                padding: '20px 24px',
                display: 'flex',
                flexDirection: 'column',
                gap: 14,
              }}>
                <div>
                  <div style={{ fontSize: 12.5, fontWeight: 600, color: '#FFFFFF', marginBottom: 4 }}>
                    Folder path
                  </div>
                  <div style={{ fontSize: 12.5, color: '#9CA3AF', fontFamily: M, wordBreak: 'break-all' }}>
                    {sourcePath || 'C:\\Photos\\Shoot'}
                  </div>
                  <div style={{ fontSize: 12, color: '#6B7280', marginTop: 4 }}>
                    Total photos: {detectedPhotoCount}
                  </div>
                </div>

                <div style={{ marginTop: 6 }}>
                  <div style={{ fontSize: 12.5, fontWeight: 600, color: '#FFFFFF', marginBottom: 8 }}>
                    Name your project
                  </div>
                  <input
                    type="text"
                    value={projectName}
                    onChange={(e) => setProjectName(e.target.value)}
                    placeholder="e.g. Wedding_Raw_Sample"
                    style={{
                      width: '100%',
                      maxWidth: 480,
                      height: 38,
                      padding: '0 14px',
                      background: '#131417',
                      border: '1px solid rgba(255, 255, 255, 0.16)',
                      borderRadius: 6,
                      color: '#FFFFFF',
                      fontFamily: F,
                      fontSize: 13,
                      outline: 'none',
                    }}
                  />
                </div>
              </div>

              {/* Bottom Card: Choose a photography type (10 genres in 2x5 grid) */}
              <div style={{
                background: '#1F2228',
                border: '1px solid rgba(255, 255, 255, 0.08)',
                borderRadius: 10,
                padding: '20px 24px',
              }}>
                <div style={{ fontSize: 13, fontWeight: 600, color: '#FFFFFF', marginBottom: 16 }}>
                  Choose a photography type
                </div>

                <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(5, 1fr)',
                  gap: 12,
                }}>
                  {PHOTOGRAPHY_TYPES.map((t) => {
                    const isSelected = genre === t.id;
                    return (
                      <button
                        key={t.id}
                        type="button"
                        onClick={() => setGenre(t.id)}
                        style={{
                          height: 84,
                          padding: '12px 8px',
                          borderRadius: 8,
                          background: isSelected ? 'rgba(224, 83, 83, 0.12)' : '#282C34',
                          border: isSelected ? '1.5px solid #E05353' : '1px solid rgba(255, 255, 255, 0.07)',
                          color: isSelected ? '#FFFFFF' : '#9CA3AF',
                          cursor: 'pointer',
                          display: 'flex',
                          flexDirection: 'column',
                          alignItems: 'center',
                          justifyContent: 'center',
                          gap: 8,
                          transition: 'all 0.15s ease',
                        }}
                      >
                        <div style={{ color: isSelected ? '#E05353' : '#CBD5E1' }}>
                          {renderGenreIcon(t.id, 24)}
                        </div>
                        <span style={{
                          fontSize: 11.5,
                          fontWeight: isSelected ? 600 : 500,
                          textAlign: 'center',
                          lineHeight: 1.2,
                        }}>
                          {t.label}
                        </span>
                      </button>
                    );
                  })}
                </div>
              </div>
            </div>
          )}

          {/* STEP 3: Your workflow (Exact match to media_1794861157298_5b5557e9.png - AI Profile ONLY with Dropdown) */}
          {step === 'workflow' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 18, minHeight: 340 }}>
              {/* Large AI Profile card matching Imagen reference */}
              <div style={{
                background: '#1F2228',
                border: '1px solid rgba(255, 255, 255, 0.08)',
                borderRadius: 10,
                padding: '24px 28px',
                display: 'flex',
                flexDirection: 'column',
                position: 'relative',
              }}>
                {/* Header row */}
                <div style={{ display: 'flex', alignItems: 'flex-start', gap: 18 }}>
                  <div style={{
                    width: 44,
                    height: 44,
                    borderRadius: 10,
                    background: '#282C34',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: '#9CA3AF',
                    flexShrink: 0,
                  }}>
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                      <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                      <circle cx="8.5" cy="8.5" r="1.5" />
                      <polyline points="21 15 16 10 5 21" />
                    </svg>
                  </div>
                  <div>
                    <div style={{ fontSize: 16, fontWeight: 700, color: '#FFFFFF', marginBottom: 4 }}>
                      AI Profile
                    </div>
                    <div style={{ fontSize: 13, color: '#9CA3AF', lineHeight: 1.4 }}>
                      Edits all {detectedPhotoCount || 21} photos for review. You only pay for what you export.
                    </div>
                  </div>
                </div>

                {/* Bottom row matching media_1794861157298_5b5557e9.png */}
                <div style={{
                  marginTop: 36,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  paddingTop: 16,
                  borderTop: '1px solid rgba(255, 255, 255, 0.06)',
                }}>
                  {/* Left: Selected Profile or 'No profile selected' */}
                  <div>
                    {selectedProfile ? (
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <div style={{
                          width: 8,
                          height: 8,
                          borderRadius: '50%',
                          background: '#22C55E',
                          boxShadow: '0 0 8px rgba(34, 197, 94, 0.6)',
                        }} />
                        <span style={{ fontSize: 13.5, fontWeight: 700, color: '#FFFFFF', letterSpacing: '0.01em' }}>
                          {selectedProfile.display_name || selectedProfile.name}
                        </span>
                        <span style={{
                          fontSize: 10,
                          fontWeight: 700,
                          padding: '2px 7px',
                          borderRadius: 4,
                          background: selectedProfile.profile_type === 'mode_b_initial' ? 'rgba(234, 179, 8, 0.15)' : 'rgba(59, 130, 246, 0.15)',
                          color: selectedProfile.profile_type === 'mode_b_initial' ? '#FACC15' : '#60A5FA',
                        }}>
                          {selectedProfile.profile_type === 'mode_b_initial' ? 'Lite' : 'Personal'}
                        </span>
                      </div>
                    ) : (
                      <span style={{ fontSize: 13, color: '#8E95A2' }}>
                        No profile selected
                      </span>
                    )}
                  </div>

                  {/* Right: Pill Dropdown button if profiles exist, or Create button if no profile present */}
                  {profiles && profiles.length > 0 ? (
                    <div ref={dropdownRef} style={{ position: 'relative' }}>
                      <button
                        type="button"
                        onClick={() => setIsProfileDropdownOpen((prev) => !prev)}
                        style={{
                          padding: '7px 20px',
                          borderRadius: 20,
                          background: isProfileDropdownOpen ? '#2A2E36' : '#16181C',
                          border: '1px solid rgba(255, 255, 255, 0.3)',
                          color: '#FFFFFF',
                          fontFamily: F,
                          fontSize: 12.5,
                          fontWeight: 600,
                          cursor: 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          gap: 8,
                          transition: 'all 0.15s ease',
                        }}
                      >
                        <span>Choose profile</span>
                        <span style={{
                          fontSize: 10,
                          display: 'inline-block',
                          transition: 'transform 0.15s ease',
                          transform: isProfileDropdownOpen ? 'rotate(180deg)' : 'none',
                        }}>
                          ▼
                        </span>
                      </button>

                      {/* Dropdown Menu */}
                      {isProfileDropdownOpen && (
                        <div
                          style={{
                            position: 'absolute',
                            top: 'calc(100% + 8px)',
                            right: 0,
                            minWidth: 280,
                            background: '#1F2228',
                            border: '1px solid rgba(255, 255, 255, 0.16)',
                            borderRadius: 8,
                            boxShadow: '0 16px 48px rgba(0, 0, 0, 0.75)',
                            zIndex: 100,
                            overflow: 'hidden',
                            padding: '6px 0',
                          }}
                        >
                          <div style={{
                            padding: '8px 16px 6px',
                            fontSize: 11,
                            fontWeight: 600,
                            color: '#888D96',
                            letterSpacing: '0.02em',
                          }}>
                            Your AI Profiles
                          </div>

                          <div style={{ maxHeight: 250, overflowY: 'auto' }}>
                            {profiles.map((p) => {
                              const isSelected = selectedProfileId === p.id;
                              const isLite = p.profile_type === 'mode_b_initial';
                              return (
                                <div
                                  key={p.id}
                                  onClick={() => {
                                    setSelectedProfileId(p.id);
                                    setIsProfileDropdownOpen(false);
                                  }}
                                  style={{
                                    padding: '9px 16px',
                                    cursor: 'pointer',
                                    background: isSelected ? 'rgba(224, 83, 83, 0.15)' : 'transparent',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'space-between',
                                    transition: 'background 0.12s ease',
                                  }}
                                  onMouseEnter={(e) => {
                                    if (!isSelected) e.currentTarget.style.background = '#2A2E36';
                                  }}
                                  onMouseLeave={(e) => {
                                    if (!isSelected) e.currentTarget.style.background = 'transparent';
                                  }}
                                >
                                  <div style={{ minWidth: 0, display: 'flex', alignItems: 'center', gap: 8 }}>
                                    <span style={{
                                      fontSize: 12.5,
                                      fontWeight: 700,
                                      color: isSelected ? '#FFFFFF' : '#E2E8F0',
                                      textTransform: 'uppercase',
                                      letterSpacing: '0.02em',
                                      whiteSpace: 'nowrap',
                                      overflow: 'hidden',
                                      textOverflow: 'ellipsis',
                                    }}>
                                      {p.display_name || p.name}
                                    </span>
                                    <span style={{
                                      fontSize: 9.5,
                                      fontWeight: 700,
                                      padding: '1px 5px',
                                      borderRadius: 3,
                                      background: isLite ? 'rgba(234, 179, 8, 0.15)' : 'rgba(59, 130, 246, 0.15)',
                                      color: isLite ? '#FACC15' : '#60A5FA',
                                    }}>
                                      {isLite ? 'Lite' : 'Personal'}
                                    </span>
                                  </div>
                                  {isSelected && (
                                    <span style={{ fontSize: 13, color: '#E05353', fontWeight: 800 }}>✓</span>
                                  )}
                                </div>
                              );
                            })}
                          </div>

                          {onOpenCreateProfile && (
                            <>
                              <div style={{ height: 1, background: 'rgba(255, 255, 255, 0.08)', margin: '4px 0' }} />
                              <div
                                onClick={() => {
                                  setIsProfileDropdownOpen(false);
                                  onOpenCreateProfile();
                                }}
                                style={{
                                  padding: '9px 16px',
                                  cursor: 'pointer',
                                  color: '#E05353',
                                  fontSize: 12,
                                  fontWeight: 600,
                                  display: 'flex',
                                  alignItems: 'center',
                                  gap: 6,
                                  transition: 'background 0.12s ease',
                                }}
                                onMouseEnter={(e) => {
                                  e.currentTarget.style.background = '#2A2E36';
                                }}
                                onMouseLeave={(e) => {
                                  e.currentTarget.style.background = 'transparent';
                                }}
                              >
                                <span>+</span> Create new profile
                              </div>
                            </>
                          )}
                        </div>
                      )}
                    </div>
                  ) : (
                    /* If no profile present: display button directly per user request */
                    <button
                      type="button"
                      onClick={onOpenCreateProfile}
                      style={{
                        padding: '8px 20px',
                        background: '#E05353',
                        color: '#FFFFFF',
                        border: 'none',
                        borderRadius: 6,
                        fontFamily: F,
                        fontSize: 12.5,
                        fontWeight: 600,
                        cursor: 'pointer',
                        boxShadow: '0 4px 14px rgba(224, 83, 83, 0.35)',
                        display: 'flex',
                        alignItems: 'center',
                        gap: 6,
                      }}
                    >
                      <span>+</span>
                      <span>Create AI profile</span>
                    </button>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* STEP 4: AI tools (Exact match to media_1794861659425_f693902c.png) */}
          {step === 'tools' && (
            <div style={{
              display: 'grid',
              gridTemplateColumns: '195px 1.25fr 1fr',
              gap: 16,
              flex: 1,
              minHeight: 420,
            }}>
              {/* Column 1: AI tools Sidebar Navigation */}
              <div style={{
                display: 'flex',
                flexDirection: 'column',
                gap: 4,
                borderRight: '1px solid rgba(255, 255, 255, 0.08)',
                paddingRight: 10,
              }}>
                <div style={{
                  fontSize: 12.5,
                  fontWeight: 700,
                  color: '#FFFFFF',
                  padding: '4px 10px 10px',
                  letterSpacing: -0.1,
                }}>
                  AI tools
                </div>

                {/* Tab 1: Recommended */}
                <button
                  type="button"
                  onClick={() => setToolCategory('recommended')}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 10,
                    width: '100%',
                    padding: '9px 12px',
                    borderRadius: 6,
                    background: toolCategory === 'recommended' ? '#2A2E36' : 'transparent',
                    border: 'none',
                    color: toolCategory === 'recommended' ? '#FFFFFF' : '#888D96',
                    fontFamily: F,
                    fontSize: 13,
                    fontWeight: toolCategory === 'recommended' ? 600 : 500,
                    cursor: 'pointer',
                    transition: 'all 0.15s ease',
                    textAlign: 'left',
                  }}
                >
                  <span style={{ fontSize: 13 }}>✨</span>
                  <span>Recommended</span>
                </button>

                {/* Tab 2: Composition */}
                <button
                  type="button"
                  onClick={() => setToolCategory('composition')}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 8,
                    width: '100%',
                    padding: '9px 12px',
                    borderRadius: 6,
                    background: toolCategory === 'composition' ? '#2A2E36' : 'transparent',
                    border: 'none',
                    color: toolCategory === 'composition' ? '#FFFFFF' : '#888D96',
                    fontFamily: F,
                    fontSize: 13,
                    fontWeight: toolCategory === 'composition' ? 600 : 500,
                    cursor: 'pointer',
                    transition: 'all 0.15s ease',
                    textAlign: 'left',
                  }}
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
                    <rect x="3" y="3" width="18" height="18" rx="2" />
                    <path d="M3 9h18M9 21V9" />
                  </svg>
                  <span style={{ whiteSpace: 'nowrap' }}>Composition</span>
                  {compActiveCount > 0 && (
                    <span style={{
                      fontSize: 10,
                      fontWeight: 700,
                      padding: '2px 7px',
                      borderRadius: 10,
                      background: '#2563EB',
                      color: '#FFFFFF',
                      marginLeft: 'auto',
                      whiteSpace: 'nowrap',
                      flexShrink: 0,
                      lineHeight: '13px',
                      display: 'inline-flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                    }}>
                      {compActiveCount} active
                    </span>
                  )}
                </button>
              </div>

              {/* Column 2: Center Tools List matching screenshot & video hover */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12, overflowY: 'auto', paddingRight: 6 }}>
                {toolCategory === 'recommended' ? (
                  <>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 13, fontWeight: 700, color: '#FFFFFF' }}>
                      <span>✨</span> Recommended for {currentGenreObj.label}
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                      {/* Crop Card with instant hover preview */}
                      <div
                        onMouseEnter={() => setPreviewTool('crop')}
                        onClick={() => setPreviewTool('crop')}
                        style={{
                          background: previewTool === 'crop' ? '#2A2E36' : '#202329',
                          border: previewTool === 'crop' ? '1px solid rgba(255, 255, 255, 0.24)' : '1px solid rgba(255, 255, 255, 0.08)',
                          borderRadius: 8,
                          padding: '11px 16px',
                          cursor: 'pointer',
                          transition: 'all 0.15s ease',
                        }}
                      >
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                          <div style={{ fontSize: 13.5, fontWeight: 600, color: '#FFFFFF' }}>
                            Crop
                          </div>
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              setCropEnabled(!cropEnabled);
                            }}
                            style={{
                              width: 24,
                              height: 24,
                              borderRadius: '50%',
                              background: cropEnabled ? '#2563EB' : 'transparent',
                              border: cropEnabled ? 'none' : '1.5px solid rgba(255, 255, 255, 0.35)',
                              color: '#FFFFFF',
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                              fontSize: 13,
                              fontWeight: 700,
                              cursor: 'pointer',
                              flexShrink: 0,
                            }}
                          >
                            {cropEnabled ? '✓' : '+'}
                          </button>
                        </div>
                      </div>

                      {/* Straighten Card with instant hover preview */}
                      <div
                        onMouseEnter={() => setPreviewTool('straighten')}
                        onClick={() => setPreviewTool('straighten')}
                        style={{
                          background: previewTool === 'straighten' ? '#2A2E36' : '#202329',
                          border: previewTool === 'straighten' ? '1px solid rgba(255, 255, 255, 0.24)' : '1px solid rgba(255, 255, 255, 0.08)',
                          borderRadius: 8,
                          padding: '11px 16px',
                          cursor: 'pointer',
                          transition: 'all 0.15s ease',
                        }}
                      >
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                            <span style={{ fontSize: 13.5, fontWeight: 600, color: '#FFFFFF' }}>
                              Straighten
                            </span>
                            <span style={{
                              fontSize: 10.5,
                              fontWeight: 700,
                              padding: '2px 7px',
                              borderRadius: 4,
                              background: '#2563EB',
                              color: '#FFFFFF',
                            }}>
                              Top used
                            </span>
                          </div>
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              setAutoStraighten(!autoStraighten);
                            }}
                            style={{
                              width: 24,
                              height: 24,
                              borderRadius: '50%',
                              background: autoStraighten ? '#2563EB' : 'transparent',
                              border: autoStraighten ? 'none' : '1.5px solid rgba(255, 255, 255, 0.35)',
                              color: '#FFFFFF',
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                              fontSize: 13,
                              fontWeight: 700,
                              cursor: 'pointer',
                              flexShrink: 0,
                            }}
                          >
                            {autoStraighten ? '✓' : '+'}
                          </button>
                        </div>
                      </div>
                    </div>
                  </>
                ) : (
                  <>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 13, fontWeight: 700, color: '#FFFFFF' }}>
                      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <rect x="3" y="3" width="18" height="18" rx="2" />
                        <path d="M3 9h18M9 21V9" />
                      </svg>
                      <span>Composition</span>
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                      {/* Crop Card with instant hover preview */}
                      <div
                        onMouseEnter={() => setPreviewTool('crop')}
                        onClick={() => setPreviewTool('crop')}
                        style={{
                          background: previewTool === 'crop' ? '#2A2E36' : '#202329',
                          border: previewTool === 'crop' ? '1px solid rgba(255, 255, 255, 0.24)' : '1px solid rgba(255, 255, 255, 0.08)',
                          borderRadius: 8,
                          padding: '11px 16px',
                          cursor: 'pointer',
                          transition: 'all 0.15s ease',
                        }}
                      >
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                          <div style={{ fontSize: 13.5, fontWeight: 600, color: '#FFFFFF' }}>
                            Crop
                          </div>
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              setCropEnabled(!cropEnabled);
                            }}
                            style={{
                              width: 24,
                              height: 24,
                              borderRadius: '50%',
                              background: cropEnabled ? '#2563EB' : 'transparent',
                              border: cropEnabled ? 'none' : '1.5px solid rgba(255, 255, 255, 0.35)',
                              color: '#FFFFFF',
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                              fontSize: 13,
                              fontWeight: 700,
                              cursor: 'pointer',
                              flexShrink: 0,
                            }}
                          >
                            {cropEnabled ? '✓' : '+'}
                          </button>
                        </div>
                      </div>

                      {/* Straighten Card with instant hover preview */}
                      <div
                        onMouseEnter={() => setPreviewTool('straighten')}
                        onClick={() => setPreviewTool('straighten')}
                        style={{
                          background: previewTool === 'straighten' ? '#2A2E36' : '#202329',
                          border: previewTool === 'straighten' ? '1px solid rgba(255, 255, 255, 0.24)' : '1px solid rgba(255, 255, 255, 0.08)',
                          borderRadius: 8,
                          padding: '11px 16px',
                          cursor: 'pointer',
                          transition: 'all 0.15s ease',
                        }}
                      >
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                            <span style={{ fontSize: 13.5, fontWeight: 600, color: '#FFFFFF' }}>
                              Straighten
                            </span>
                            <span style={{
                              fontSize: 10.5,
                              fontWeight: 700,
                              padding: '2px 7px',
                              borderRadius: 4,
                              background: '#2563EB',
                              color: '#FFFFFF',
                            }}>
                              Top used
                            </span>
                          </div>
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              setAutoStraighten(!autoStraighten);
                            }}
                            style={{
                              width: 24,
                              height: 24,
                              borderRadius: '50%',
                              background: autoStraighten ? '#2563EB' : 'transparent',
                              border: autoStraighten ? 'none' : '1.5px solid rgba(255, 255, 255, 0.35)',
                              color: '#FFFFFF',
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                              fontSize: 13,
                              fontWeight: 700,
                              cursor: 'pointer',
                              flexShrink: 0,
                            }}
                          >
                            {autoStraighten ? '✓' : '+'}
                          </button>
                        </div>
                      </div>
                    </div>
                  </>
                )}
              </div>

              {/* Column 3: Preview Pane (Exact match to video recording) */}
              <div style={{
                background: '#1F2228',
                border: '1px solid rgba(255, 255, 255, 0.08)',
                borderRadius: 10,
                padding: 16,
                display: 'flex',
                flexDirection: 'column',
              }}>
                <div style={{ fontSize: 13, fontWeight: 700, color: '#FFFFFF', marginBottom: 12 }}>
                  Preview
                </div>

                {/* Preview Image Visual Container with light neutral background */}
                <div style={{
                  width: '100%',
                  height: 220,
                  borderRadius: 6,
                  overflow: 'hidden',
                  position: 'relative',
                  background: '#CBD5E1', // exact light neutral background from video
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}>
                  {previewTool === 'straighten' ? (
                    /* Straighten Beach sunset photo (tilted with horizon line) */
                    <div style={{
                      position: 'relative',
                      width: '80%',
                      height: '74%',
                      transform: 'rotate(-4deg)',
                      borderRadius: 2,
                      overflow: 'hidden',
                      boxShadow: '0 8px 24px rgba(0, 0, 0, 0.28)',
                      background: 'linear-gradient(180deg, #93C5FD 0%, #BAE6FD 35%, #FDE047 62%, #94A3B8 100%)',
                      transition: 'all 0.2s ease',
                    }}>
                      {/* Sun glow on horizon */}
                      <div style={{
                        position: 'absolute',
                        top: '42%',
                        left: '28%',
                        width: 22,
                        height: 22,
                        borderRadius: '50%',
                        background: '#FFFBEB',
                        boxShadow: '0 0 18px #FEF08A',
                      }} />

                      {/* Beach couple walking holding hands */}
                      <div style={{
                        position: 'absolute',
                        bottom: '22%',
                        left: '50%',
                        display: 'flex',
                        alignItems: 'flex-end',
                        gap: 3,
                      }}>
                        <div style={{ width: 8, height: 26, background: '#1E293B', borderRadius: '3px 3px 0 0' }} />
                        <div style={{ width: 10, height: 24, background: '#FFFFFF', borderRadius: '3px 3px 0 0' }} />
                      </div>

                      {/* Shoreline / wet sand */}
                      <div style={{
                        position: 'absolute',
                        bottom: 0,
                        left: 0,
                        right: 0,
                        height: '35%',
                        background: '#64748B',
                        opacity: 0.85,
                      }} />

                      {/* White 3x3 rule of thirds grid lines */}
                      <div style={{ position: 'absolute', left: '33.3%', top: 0, bottom: 0, width: 1, background: 'rgba(255,255,255,0.45)' }} />
                      <div style={{ position: 'absolute', left: '66.6%', top: 0, bottom: 0, width: 1, background: 'rgba(255,255,255,0.45)' }} />
                      <div style={{ position: 'absolute', top: '33.3%', left: 0, right: 0, height: 1, background: 'rgba(255,255,255,0.45)' }} />
                      <div style={{ position: 'absolute', top: '66.6%', left: 0, right: 0, height: 1, background: 'rgba(255,255,255,0.45)' }} />

                      {/* White Corner crop brackets */}
                      <div style={{ position: 'absolute', top: 4, left: 4, width: 12, height: 12, borderTop: '2px solid #FFFFFF', borderLeft: '2px solid #FFFFFF' }} />
                      <div style={{ position: 'absolute', top: 4, right: 4, width: 12, height: 12, borderTop: '2px solid #FFFFFF', borderRight: '2px solid #FFFFFF' }} />
                      <div style={{ position: 'absolute', bottom: 4, left: 4, width: 12, height: 12, borderBottom: '2px solid #FFFFFF', borderLeft: '2px solid #FFFFFF' }} />
                      <div style={{ position: 'absolute', bottom: 4, right: 4, width: 12, height: 12, borderBottom: '2px solid #FFFFFF', borderRight: '2px solid #FFFFFF' }} />
                    </div>
                  ) : (
                    /* Crop Golden field sunset photo (matches frame 01 from user video) */
                    <div style={{
                      position: 'relative',
                      width: '82%',
                      height: '76%',
                      borderRadius: 2,
                      overflow: 'hidden',
                      boxShadow: '0 8px 24px rgba(0, 0, 0, 0.28)',
                      background: 'linear-gradient(180deg, #E2E8F0 0%, #FEF08A 38%, #B45309 68%, #451A03 100%)',
                      transition: 'all 0.2s ease',
                    }}>
                      {/* Sun on field horizon */}
                      <div style={{
                        position: 'absolute',
                        top: '42%',
                        left: '38%',
                        width: 26,
                        height: 26,
                        borderRadius: '50%',
                        background: '#FEF9C3',
                        boxShadow: '0 0 20px #FDE047',
                      }} />

                      {/* Golden grass field */}
                      <div style={{
                        position: 'absolute',
                        bottom: 0,
                        left: 0,
                        right: 0,
                        height: '42%',
                        background: 'linear-gradient(180deg, #78350F 0%, #291003 100%)',
                      }} />

                      {/* Embracing couple silhouette */}
                      <div style={{
                        position: 'absolute',
                        bottom: '22%',
                        left: '54%',
                        display: 'flex',
                        alignItems: 'flex-end',
                      }}>
                        <div style={{ width: 9, height: 26, background: '#18181B', borderRadius: '3px 3px 0 0' }} />
                        <div style={{ width: 12, height: 25, background: '#FFFFFF', borderRadius: '4px 4px 0 0', marginLeft: -3 }} />
                      </div>

                      {/* White 3x3 rule of thirds grid lines */}
                      <div style={{ position: 'absolute', left: '33.3%', top: 0, bottom: 0, width: 1, background: 'rgba(255,255,255,0.45)' }} />
                      <div style={{ position: 'absolute', left: '66.6%', top: 0, bottom: 0, width: 1, background: 'rgba(255,255,255,0.45)' }} />
                      <div style={{ position: 'absolute', top: '33.3%', left: 0, right: 0, height: 1, background: 'rgba(255,255,255,0.45)' }} />
                      <div style={{ position: 'absolute', top: '66.6%', left: 0, right: 0, height: 1, background: 'rgba(255,255,255,0.45)' }} />

                      {/* White Corner crop brackets */}
                      <div style={{ position: 'absolute', top: 4, left: 4, width: 14, height: 14, borderTop: '2.5px solid #FFFFFF', borderLeft: '2.5px solid #FFFFFF' }} />
                      <div style={{ position: 'absolute', top: 4, right: 4, width: 14, height: 14, borderTop: '2.5px solid #FFFFFF', borderRight: '2.5px solid #FFFFFF' }} />
                      <div style={{ position: 'absolute', bottom: 4, left: 4, width: 14, height: 14, borderBottom: '2.5px solid #FFFFFF', borderLeft: '2.5px solid #FFFFFF' }} />
                      <div style={{ position: 'absolute', bottom: 4, right: 4, width: 14, height: 14, borderBottom: '2.5px solid #FFFFFF', borderRight: '2.5px solid #FFFFFF' }} />

                      {/* Edge center indicators */}
                      <div style={{ position: 'absolute', top: 4, left: '50%', transform: 'translateX(-50%)', width: 16, height: 2.5, background: '#FFFFFF' }} />
                      <div style={{ position: 'absolute', bottom: 4, left: '50%', transform: 'translateX(-50%)', width: 16, height: 2.5, background: '#FFFFFF' }} />
                      <div style={{ position: 'absolute', left: 4, top: '50%', transform: 'translateY(-50%)', width: 2.5, height: 16, background: '#FFFFFF' }} />
                      <div style={{ position: 'absolute', right: 4, top: '50%', transform: 'translateY(-50%)', width: 2.5, height: 16, background: '#FFFFFF' }} />

                      {/* Diagonal resize arrow icon at bottom-left corner matching video */}
                      <div style={{
                        position: 'absolute',
                        bottom: 0,
                        left: 0,
                        color: '#0F172A',
                        background: '#FFFFFF',
                        borderRadius: '0 4px 0 0',
                        padding: '1px 3px',
                        fontSize: 10,
                        fontWeight: 900,
                        lineHeight: 1,
                      }}>
                        ⤡
                      </div>
                    </div>
                  )}

                  {/* Rotation curved arrow indicator for Straighten matching video */}
                  {previewTool === 'straighten' && (
                    <div style={{
                      position: 'absolute',
                      right: 12,
                      top: '48%',
                      transform: 'translateY(-50%)',
                      color: '#0F172A',
                      fontSize: 22,
                      fontWeight: 900,
                    }}>
                      ↷
                    </div>
                  )}
                </div>

                {/* Title & Subtitle below preview matching exact text from user video */}
                <div style={{ marginTop: 14 }}>
                  <div style={{ fontSize: 13.5, fontWeight: 700, color: '#FFFFFF', marginBottom: 4 }}>
                    {previewTool === 'straighten' ? 'Straighten' : 'Crop'}
                  </div>
                  <div style={{ fontSize: 12, color: '#9CA3AF', lineHeight: 1.45 }}>
                    {previewTool === 'straighten'
                      ? 'Automatically rotates every photo according to its horizon'
                      : 'Crops to a 2x3 aspect ratio for optimal composition in many types of photography.'}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Universal Footer (Matches Imagen AI footer) */}
        <div style={{
          padding: '16px 24px',
          borderTop: '1px solid rgba(255, 255, 255, 0.08)',
          background: '#131417',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}>
          {/* Photos count */}
          <div style={{ fontSize: 13, color: '#9CA3AF' }}>
            {detectedPhotoCount > 0 ? `${detectedPhotoCount} Photos selected` : ''}
          </div>

          {/* Action buttons */}
          <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
            {/* Back button */}
            {step !== 'source' && (
              <button
                type="button"
                onClick={() => {
                  if (step === 'setup') setStep('source');
                  else if (step === 'workflow') setStep('setup');
                  else if (step === 'tools') setStep('workflow');
                }}
                style={{
                  height: 36,
                  padding: '0 20px',
                  background: 'transparent',
                  border: '1px solid rgba(255, 255, 255, 0.16)',
                  borderRadius: 6,
                  color: '#E5E7EB',
                  fontFamily: F,
                  fontSize: 12.5,
                  fontWeight: 600,
                  cursor: 'pointer',
                }}
              >
                Back
              </button>
            )}

            {/* Cancel on Step 1 */}
            {step === 'source' && (
              <button
                type="button"
                onClick={onClose}
                style={{
                  height: 36,
                  padding: '0 18px',
                  background: 'transparent',
                  border: '1px solid rgba(255, 255, 255, 0.16)',
                  borderRadius: 6,
                  color: '#E5E7EB',
                  fontFamily: F,
                  fontSize: 12.5,
                  fontWeight: 600,
                  cursor: 'pointer',
                }}
              >
                Cancel
              </button>
            )}

            {/* Next on Step 2 (Project setup) */}
            {step === 'setup' && (
              <button
                type="button"
                onClick={() => setStep('workflow')}
                style={{
                  height: 36,
                  padding: '0 24px',
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
              >
                Next
              </button>
            )}

            {/* Next on Step 3 (Your workflow) - Gated by selectedProfileId! */}
            {step === 'workflow' && (
              <button
                type="button"
                disabled={!selectedProfileId}
                onClick={() => {
                  if (selectedProfileId) {
                    setStep('tools');
                  }
                }}
                title={!selectedProfileId ? 'Please select an AI Profile to continue' : ''}
                style={{
                  height: 36,
                  padding: '0 24px',
                  background: selectedProfileId ? '#E05353' : 'rgba(255, 255, 255, 0.1)',
                  color: selectedProfileId ? '#FFFFFF' : '#888D96',
                  border: 'none',
                  borderRadius: 6,
                  fontFamily: F,
                  fontSize: 12.5,
                  fontWeight: 700,
                  cursor: selectedProfileId ? 'pointer' : 'not-allowed',
                  boxShadow: selectedProfileId ? '0 4px 14px rgba(224, 83, 83, 0.35)' : 'none',
                }}
              >
                Next
              </button>
            )}

            {/* Create project on Step 4 (AI tools) */}
            {step === 'tools' && (
              <button
                type="button"
                onClick={handleStart}
                style={{
                  height: 36,
                  padding: '0 24px',
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
              >
                Create project
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default NewProjectModal;
