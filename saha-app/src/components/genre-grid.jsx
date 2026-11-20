// Photography Genre Selection Grid
// Reference: references/32_imagen_photography_type_selection.png
// Reusable genre picker matching Imagen AI's 9-category taxonomy.

import SONNA from '../tokens.js';

const GENRES = [
  { id: 'wedding', label: 'Weddings & Engagements', icon: '💍' },
  { id: 'portrait', label: 'Portraits & Headshots', icon: '👤' },
  { id: 'events', label: 'Events & Parties', icon: '🎉' },
  { id: 'family', label: 'Family & Newborn', icon: '👶' },
  { id: 'commercial', label: 'Commercial & Fashion', icon: '🛍️' },
  { id: 'real_estate', label: 'Real Estate & Architecture', icon: '🏠' },
  { id: 'landscape', label: 'Landscape & Nature', icon: '🌄' },
  { id: 'street', label: 'Street & Travel', icon: '✈️' },
  { id: 'other', label: 'Other / General', icon: '📷' },
];

export function PhotographyGenreGrid({ selectedGenre, onSelectGenre }) {
  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: 'repeat(3, 1fr)',
      gap: 10,
    }}>
      {GENRES.map((g) => {
        const isSelected = selectedGenre === g.id;
        return (
          <button
            key={g.id}
            type="button"
            onClick={() => onSelectGenre?.(g.id)}
            style={{
              padding: '12px 14px',
              borderRadius: 6,
              background: isSelected ? SONNA.bgHover : SONNA.bgLifted,
              border: `1.5px solid ${isSelected ? SONNA.ochre : SONNA.lineSoft}`,
              color: isSelected ? SONNA.fg : SONNA.fgMute,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: 10,
              textAlign: 'left',
              transition: 'border-color 150ms, background 150ms',
            }}
          >
            <span style={{ fontSize: 18 }}>{g.icon}</span>
            <span style={{
              fontSize: 12.5,
              fontWeight: isSelected ? 600 : 500,
              color: isSelected ? SONNA.fg : SONNA.fgMute,
              lineHeight: 1.25,
            }}>
              {g.label}
            </span>
          </button>
        );
      })}
    </div>
  );
}

export default PhotographyGenreGrid;

