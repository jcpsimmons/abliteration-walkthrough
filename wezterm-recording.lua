local wezterm = require 'wezterm'
return {
  font = wezterm.font_with_fallback { 'BigBlueTerm437 Nerd Font Mono', 'Menlo' },
  font_size = 22.0,
  initial_cols = 110,
  initial_rows = 35,
  enable_tab_bar = false,
  window_decorations = 'RESIZE',
  window_padding = { left = 20, right = 20, top = 20, bottom = 20 },
  colors = {
    foreground = '#87ff87', background = '#000000', cursor_bg = '#87ff87',
    ansi = { '#000000', '#ff6b6b', '#87ff87', '#ffff00', '#00008b', '#ff00ff', '#00afaf', '#bcbcbc' },
    brights = { '#444444', '#ff6b6b', '#87ff87', '#ffff00', '#0000ff', '#ff00ff', '#00ffff', '#ffffff' },
  },
}
