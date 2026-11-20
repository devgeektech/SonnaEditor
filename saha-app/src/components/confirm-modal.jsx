// Confirm & Prompt Modal
// Reference: media_1794859072884_a8852405.png (In-app confirmation prompt matching Imagen reference)
// Features:
// - Dialog title: e.g. "Are you sure you want to leave?"
// - Message text: e.g. "Your progress won't be saved for later"
// - Cancel button + vibrant theme-accent Confirm button ("Yes, leave")
// - Optional input field for prompt replacements (e.g. Rename profile)
// - Full Light and Dark theme support via SONNA tokens

import { useEffect, useRef, useState } from 'react';
import SONNA from '../tokens.js';

const F = SONNA.font;

export function ConfirmModal({
  isOpen,
  title = 'Are you sure you want to leave?',
  message = "Your progress won't be saved for later",
  confirmText = 'Yes, leave',
  cancelText = 'Cancel',
  isPrompt = false,
  promptInitialValue = '',
  promptPlaceholder = '',
  onConfirm,
  onCancel,
  theme = 'dark',
}) {
  const isLight = theme === 'light';
  const [inputValue, setInputValue] = useState(promptInitialValue);
  const inputRef = useRef(null);

  useEffect(() => {
    if (isOpen) {
      setInputValue(promptInitialValue);
      if (isPrompt) {
        const t = setTimeout(() => {
          inputRef.current?.focus();
          inputRef.current?.select();
        }, 50);
        return () => clearTimeout(t);
      }
    }
  }, [isOpen, isPrompt, promptInitialValue]);

  if (!isOpen) return null;

  const modalBg = isLight ? '#FFFFFF' : '#212429';
  const modalBorder = isLight ? 'rgba(0, 0, 0, 0.12)' : 'rgba(255, 255, 255, 0.09)';
  const titleColor = isLight ? '#1c1917' : '#FFFFFF';
  const messageColor = isLight ? '#78716c' : '#9E9E9E';
  const cancelColor = isLight ? '#57534e' : '#D1D5DB';
  const inputBg = isLight ? '#FFFFFF' : '#16181B';
  const inputBorder = isLight ? 'rgba(0, 0, 0, 0.18)' : 'rgba(255, 255, 255, 0.14)';

  const handleConfirm = () => {
    if (isPrompt) {
      onConfirm?.(inputValue.trim());
    } else {
      onConfirm?.();
    }
  };

  return (
    <div
      onClick={onCancel}
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 1200,
        background: 'rgba(0, 0, 0, 0.65)',
        backdropFilter: 'blur(3px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 24,
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        style={{
          width: '100%',
          maxWidth: 420,
          background: modalBg,
          border: `1px solid ${modalBorder}`,
          borderRadius: 10,
          boxShadow: isLight
            ? '0 20px 48px rgba(0, 0, 0, 0.16)'
            : '0 24px 60px rgba(0, 0, 0, 0.72)',
          padding: '22px 24px 20px',
          display: 'flex',
          flexDirection: 'column',
          transition: 'all 0.18s ease',
        }}
      >
        {/* Title */}
        <h3 style={{
          fontSize: 16,
          fontWeight: 600,
          color: titleColor,
          margin: '0 0 10px 0',
          letterSpacing: -0.2,
          fontFamily: F,
        }}>
          {title}
        </h3>

        {/* Message */}
        {message && (
          <div style={{
            fontSize: 13,
            color: messageColor,
            lineHeight: 1.45,
            marginBottom: isPrompt ? 14 : 24,
            fontFamily: F,
          }}>
            {message}
          </div>
        )}

        {/* Optional text input for prompt mode */}
        {isPrompt && (
          <input
            ref={inputRef}
            type="text"
            value={inputValue}
            placeholder={promptPlaceholder}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault();
                handleConfirm();
              } else if (e.key === 'Escape') {
                e.preventDefault();
                onCancel?.();
              }
            }}
            style={{
              width: '100%',
              height: 36,
              padding: '0 12px',
              background: inputBg,
              border: `1px solid ${inputBorder}`,
              borderRadius: 4,
              color: titleColor,
              fontFamily: F,
              fontSize: 13,
              outline: 'none',
              marginBottom: 22,
              boxSizing: 'border-box',
            }}
          />
        )}

        {/* Action Buttons */}
        <div style={{
          display: 'flex',
          justifyContent: 'flex-end',
          alignItems: 'center',
          gap: 12,
        }}>
          <button
            type="button"
            onClick={onCancel}
            style={{
              background: 'transparent',
              border: 'none',
              color: cancelColor,
              fontFamily: F,
              fontSize: 13,
              fontWeight: 600,
              cursor: 'pointer',
              padding: '0 12px',
              height: 34,
              borderRadius: 4,
              transition: 'color 0.15s ease',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.color = titleColor;
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.color = cancelColor;
            }}
          >
            {cancelText}
          </button>

          <button
            type="button"
            onClick={handleConfirm}
            style={{
              height: 34,
              padding: '0 18px',
              background: SONNA.ochre,
              border: 'none',
              borderRadius: 6,
              color: '#FFFFFF',
              fontFamily: F,
              fontSize: 13,
              fontWeight: 600,
              cursor: 'pointer',
              boxShadow: '0 2px 8px rgba(224, 83, 83, 0.35)',
              transition: 'all 0.15s ease',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = '#CC4444';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = SONNA.ochre;
            }}
          >
            {confirmText}
          </button>
        </div>
      </div>
    </div>
  );
}

export default ConfirmModal;
