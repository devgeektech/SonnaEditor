// Sonna Editor — design tokens, shared across all directions.
// Ported from "SAHA UI/tokens.js" (window-attached) to ES module default export.

const TYPE = {
  font: '-apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", sans-serif',
  mono: '"SF Mono", "JetBrains Mono", Menlo, Consolas, monospace',
};

export const THEMES = {
  dark: {
    // Backgrounds (dark mode, slightly warm — photographer-tool feel)
    bgDeep: '#16140F',
    bgPanel: '#1F1D17',
    bgLifted: '#2A2822',
    bgHover: '#34322B',

    // Hairlines
    line: '#2C2A24',
    lineSoft: '#24221C',

    // Text
    fg: '#E8E6E0',
    fgMute: '#B2ADA1',
    fgDim: '#807A6D',
    fgFaint: '#5C574D',

    // Status / accent
    ochre: '#D9864A',
    ochreSoft: '#B66E3E',
    ochreTint: 'rgba(217, 134, 74, 0.16)',
    onAccent: '#1F130B',
    green: '#6E9B7E',
    amber: '#C49A5E',
    red: '#B7655E',
    onDanger: '#FFFFFF',
  },
  light: {
    bgDeep: '#F7F3ED',
    bgPanel: '#FFFDF9',
    bgLifted: '#EFE7DC',
    bgHover: '#E7DCCF',

    line: '#D8CEC1',
    lineSoft: '#E7DED3',

    fg: '#221B15',
    fgMute: '#5F564E',
    fgDim: '#7A7067',
    fgFaint: '#A3988D',

    ochre: '#C76F35',
    ochreSoft: '#A65A2B',
    ochreTint: 'rgba(199, 111, 53, 0.14)',
    onAccent: '#FFFFFF',
    green: '#4F8464',
    amber: '#A9792C',
    red: '#A24B45',
    onDanger: '#FFFFFF',
  },
};

const SONNA = {
  ...TYPE,
  bgDeep: 'var(--saha-bg-deep)',
  bgPanel: 'var(--saha-bg-panel)',
  bgLifted: 'var(--saha-bg-lifted)',
  bgHover: 'var(--saha-bg-hover)',
  line: 'var(--saha-line)',
  lineSoft: 'var(--saha-line-soft)',
  fg: 'var(--saha-fg)',
  fgMute: 'var(--saha-fg-mute)',
  fgDim: 'var(--saha-fg-dim)',
  fgFaint: 'var(--saha-fg-faint)',
  ochre: 'var(--saha-accent)',
  ochreSoft: 'var(--saha-accent-soft)',
  ochreTint: 'var(--saha-accent-tint)',
  onAccent: 'var(--saha-on-accent)',
  cta: 'var(--saha-accent)',
  onCta: 'var(--saha-on-accent)',
  green: 'var(--saha-green)',
  amber: 'var(--saha-amber)',
  red: 'var(--saha-red)',
  onDanger: 'var(--saha-on-danger)',
  theme: 'light',
};

export function applyTheme(theme) {
  const nextTheme = THEMES[theme] ? theme : 'light';
  SONNA.theme = nextTheme;
  if (typeof document !== 'undefined') {
    document.documentElement.dataset.theme = nextTheme;
    document.body.style.background = SONNA.bgDeep;
    document.body.style.color = SONNA.fg;
  }
  return SONNA;
}

export default SONNA;
