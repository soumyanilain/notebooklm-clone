"""Visual styles for the Gradio UI. Each style is a Gradio theme plus a small set of CSS
variables; the page markup is the same for every style. Pick one with UI_STYLE=<key>."""

from __future__ import annotations

import os
from dataclasses import dataclass

import gradio as gr

LF = gr.themes.LocalFont  # fonts bundled with Gradio, so they work offline too


@dataclass(frozen=True)
class Style:
    key: str
    name: str
    theme: gr.themes.Base
    variables: str  # CSS custom properties for light and dark mode


# Shared structural CSS. Everything visual comes from the variables of the active style.
BASE_CSS = """
.gradio-container { max-width: 1320px !important; margin: 0 auto !important; }
footer { display: none !important; }

.app-hero { background: var(--hero-bg); color: var(--hero-fg); border: var(--hero-border); border-radius: var(--hero-radius);
  padding: var(--hero-pad); margin: 4px 0 12px; box-shadow: var(--hero-shadow); }
.app-hero .app-title { font-family: var(--title-font); font-weight: 700; font-size: 30px; line-height: 1.15;
  letter-spacing: -0.01em; color: var(--hero-fg) !important; }
.app-sub { color: var(--hero-sub); font-size: 15.5px; margin-top: 6px; max-width: 72ch; }
.app-tags { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }
.app-tags span { background: var(--hero-chip-bg); color: var(--hero-chip-fg); border-radius: 999px; padding: 4px 12px;
  font-size: 12.5px; font-weight: 600; }

.side-title { font-family: var(--title-font); font-weight: 700; font-size: 16px; color: var(--strong); margin: 12px 0 2px; }
.side-note { font-size: 13px; color: var(--muted); margin: 0 0 6px; }
.status ul { margin: 4px 0 0; padding-left: 18px; font-size: 13.5px; }

.nb-head { padding: 2px 2px 8px; }
.nb-title { font-family: var(--title-font); font-weight: 700; font-size: 28px; line-height: 1.2; color: var(--strong); }
.nb-chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
.nb-chips span { background: var(--chip-bg); color: var(--chip-fg); border-radius: 999px; padding: 4px 12px;
  font-size: 13px; font-weight: 600; }
.nb-empty { color: var(--muted); font-size: 15px; margin-top: 6px; max-width: 64ch; }

mark { background: var(--cite-bg); color: var(--cite-fg); border-radius: 6px; padding: 0 0.4em; font-weight: 700;
  font-size: 0.8em; vertical-align: 0.1em; margin: 0 1px; }

#chat details { margin-top: 10px; border-top: 1px solid var(--rule); padding-top: 6px; }
#chat details summary { font-size: 13px; font-weight: 700; color: var(--muted); cursor: pointer; }
#chat details li { font-size: 13.5px; line-height: 1.5; color: var(--muted); }
#chat details li strong, #chat details li a { color: var(--strong); }

.trace { font-size: 13.5px; }
.trace .timing { font-family: var(--mono); font-size: 12.5px; color: var(--muted); }
.trace blockquote { border-left: 3px solid var(--accent); margin: 4px 0 12px; padding: 2px 10px; color: var(--muted); }

.tool-card { border: 1px solid var(--card-border) !important; border-radius: var(--card-radius) !important;
  padding: 14px !important; background: var(--card-bg) !important; }
.tool-title { font-family: var(--title-font); font-weight: 700; font-size: 16px; color: var(--strong); margin: 0; }
.tool-note { font-size: 13px; color: var(--muted); margin: 3px 0 8px; min-height: 2.6em; }

#doc { border: 1px solid var(--card-border); border-radius: var(--card-radius); padding: 10px 26px 18px; background: var(--card-bg); }
#doc .prose { font-size: 15.5px; line-height: 1.62; max-width: 80ch; }
#doc .prose h1, #doc .prose h2, #doc .prose h3 { font-family: var(--title-font); color: var(--strong); }
#tree pre, #tree .cm-content { font-size: 12.5px !important; }

button[role="tab"][aria-selected="true"] { color: var(--accent-strong) !important; }
button[role="tab"][aria-selected="true"]::after { background: var(--accent-strong) !important; }

.app-footer { margin-top: 28px; padding: 16px 4px 8px; border-top: 1px solid var(--rule); display: flex; flex-wrap: wrap;
  justify-content: space-between; gap: 10px 24px; color: var(--muted); font-size: 13.5px; }
.app-footer strong { color: var(--strong); }
.app-footer nav { display: flex; flex-wrap: wrap; gap: 6px 18px; }
.app-footer a { color: var(--accent-strong); text-decoration: none; font-weight: 600; }
.app-footer a:hover, .app-footer a:focus-visible { text-decoration: underline; }

button:focus-visible, a:focus-visible, input:focus-visible, textarea:focus-visible {
  outline: 2px solid var(--accent) !important; outline-offset: 2px; }
@media (max-width: 720px) { .app-hero .app-title { font-size: 25px; } .nb-title { font-size: 23px; } #doc { padding: 6px 14px 14px; } }
"""


# ---------------------------------------------------------------- 1. Indigo Studio
INDIGO = Style(
    key="indigo",
    name="Indigo Studio",
    theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="violet", neutral_hue="slate").set(
        block_radius="14px", button_large_radius="10px", button_small_radius="8px",
        color_accent_soft="#EEF2FF", color_accent_soft_dark="#272452",
    ),
    variables="""
:root { --hero-bg: linear-gradient(120deg, #4F46E5 0%, #7C3AED 55%, #A855F7 100%); --hero-fg: #FFFFFF;
  --hero-sub: rgba(255,255,255,.88); --hero-border: 0; --hero-radius: 18px; --hero-pad: 22px 26px; --hero-shadow: none;
  --hero-chip-bg: rgba(255,255,255,.18); --hero-chip-fg: #FFFFFF;
  --title-font: Montserrat, ui-sans-serif, sans-serif; --mono: "IBM Plex Mono", ui-monospace, monospace;
  --strong: #1E1B4B; --muted: #64748B; --rule: #E2E8F0; --accent: #6366F1; --accent-strong: #4F46E5;
  --chip-bg: #EEF2FF; --chip-fg: #4338CA; --cite-bg: #E0E7FF; --cite-fg: #3730A3;
  --card-bg: #FFFFFF; --card-border: #E0E7FF; --card-radius: 14px; }
.dark { --strong: #E0E7FF; --muted: #A5B4FC; --rule: #312E81; --accent: #818CF8; --accent-strong: #A5B4FC;
  --chip-bg: #272452; --chip-fg: #C7D2FE; --cite-bg: #3730A3; --cite-fg: #E0E7FF;
  --card-bg: #1E1B3A; --card-border: #312E81; }
""",
)

# ---------------------------------------------------------------- 2. Studio (NotebookLM-inspired)
STUDIO = Style(
    key="studio",
    name="Studio (NotebookLM-inspired)",
    theme=gr.themes.Base(
        primary_hue="blue", neutral_hue="slate",
        font=[LF("Source Sans 3"), LF("Lexend"), "ui-sans-serif", "system-ui", "sans-serif"],
        font_mono=[LF("IBM Plex Mono"), "ui-monospace", "monospace"],
    ).set(
        body_background_fill="#F0F4F9", body_background_fill_dark="#131314",
        body_text_color="#1F1F1F", body_text_color_dark="#E3E3E3",
        body_text_color_subdued="#5E6368", body_text_color_subdued_dark="#A8ABAF",
        background_fill_primary="#FFFFFF", background_fill_primary_dark="#1E1F20",
        background_fill_secondary="#F0F4F9", background_fill_secondary_dark="#131314",
        block_background_fill="#FFFFFF", block_background_fill_dark="#1E1F20",
        block_border_color="#FFFFFF", block_border_color_dark="#1E1F20",
        block_radius="20px", container_radius="20px", input_radius="14px",
        block_shadow="0 1px 2px rgba(60,64,67,.08)", block_shadow_dark="none",
        block_label_background_fill="transparent", block_label_background_fill_dark="transparent",
        block_label_border_width="0px", block_label_text_color="#5E6368", block_label_text_color_dark="#A8ABAF",
        block_title_text_color="#1F1F1F", block_title_text_color_dark="#E3E3E3",
        border_color_primary="#E1E3E1", border_color_primary_dark="#2D2F31",
        color_accent="#0B57D0", color_accent_soft="#E8F0FE", color_accent_soft_dark="#0B2A5C",
        input_background_fill="#F0F4F9", input_background_fill_dark="#282A2C",
        input_border_color="#F0F4F9", input_border_color_dark="#282A2C",
        input_border_color_focus="#0B57D0", input_border_color_focus_dark="#A8C7FA",
        button_large_radius="999px", button_small_radius="999px",
        button_primary_background_fill="#0B57D0", button_primary_background_fill_hover="#0842A0",
        button_primary_text_color="#FFFFFF", button_primary_border_color="#0B57D0",
        button_primary_background_fill_dark="#A8C7FA", button_primary_background_fill_hover_dark="#D3E3FD",
        button_primary_text_color_dark="#062E6F", button_primary_border_color_dark="#A8C7FA",
        button_secondary_background_fill="#E9EEF6", button_secondary_background_fill_hover="#DDE3EA",
        button_secondary_text_color="#1F1F1F", button_secondary_border_color="#E9EEF6",
        button_secondary_background_fill_dark="#333537", button_secondary_background_fill_hover_dark="#3C3F42",
        button_secondary_text_color_dark="#E3E3E3", button_secondary_border_color_dark="#333537",
        button_cancel_background_fill="#FCE8E6", button_cancel_text_color="#B3261E", button_cancel_border_color="#FCE8E6",
        button_cancel_background_fill_dark="#601410", button_cancel_text_color_dark="#F9DEDC",
        button_primary_shadow="none", button_secondary_shadow="none",
        checkbox_label_background_fill="#F0F4F9", checkbox_label_background_fill_selected="#D3E3FD",
        checkbox_label_border_color="#F0F4F9", checkbox_label_border_color_selected="#D3E3FD",
        checkbox_label_text_color_selected="#041E49",
        checkbox_label_background_fill_dark="#282A2C", checkbox_label_background_fill_selected_dark="#0842A0",
        checkbox_label_text_color_selected_dark="#D3E3FD",
        checkbox_background_color_selected="#0B57D0", checkbox_border_color_selected="#0B57D0",
        slider_color="#0B57D0", slider_color_dark="#A8C7FA",
    ),
    variables="""
:root { --hero-bg: transparent; --hero-fg: #1F1F1F; --hero-sub: #5E6368; --hero-border: 0; --hero-radius: 0;
  --hero-pad: 8px 2px 4px; --hero-shadow: none; --hero-chip-bg: #FFFFFF; --hero-chip-fg: #0B57D0;
  --title-font: Lexend, "Source Sans 3", sans-serif; --mono: "IBM Plex Mono", ui-monospace, monospace;
  --strong: #1F1F1F; --muted: #5E6368; --rule: #E1E3E1; --accent: #0B57D0; --accent-strong: #0B57D0;
  --chip-bg: #E8F0FE; --chip-fg: #0842A0; --cite-bg: #D3E3FD; --cite-fg: #041E49;
  --card-bg: #FFFFFF; --card-border: #E1E3E1; --card-radius: 20px; }
.dark { --hero-fg: #E3E3E3; --hero-sub: #A8ABAF; --hero-chip-bg: #1E1F20; --hero-chip-fg: #A8C7FA;
  --strong: #E3E3E3; --muted: #A8ABAF; --rule: #2D2F31; --accent: #A8C7FA; --accent-strong: #A8C7FA;
  --chip-bg: #0B2A5C; --chip-fg: #D3E3FD; --cite-bg: #0842A0; --cite-fg: #D3E3FD;
  --card-bg: #1E1F20; --card-border: #2D2F31; }
""",
)

# ---------------------------------------------------------------- 3. Midnight (dark)
_MID = dict(bg="#0E1628", panel="#152036", panel2="#1B2945", rule="#24324D", text="#E2E8F0", muted="#94A3B8",
            accent="#22D3EE", accent_hover="#67E8F9", on_accent="#04222B", input="#0F1A2E")
MIDNIGHT = Style(
    key="midnight",
    name="Midnight (dark)",
    theme=gr.themes.Base(
        primary_hue="cyan", neutral_hue="slate",
        font=[LF("IBM Plex Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
        font_mono=[LF("IBM Plex Mono"), "ui-monospace", "monospace"],
    ).set(**{f"{k}{suffix}": v for suffix in ("", "_dark") for k, v in {
        "body_background_fill": _MID["bg"], "body_text_color": _MID["text"], "body_text_color_subdued": _MID["muted"],
        "background_fill_primary": _MID["panel"], "background_fill_secondary": _MID["bg"],
        "block_background_fill": _MID["panel"], "block_border_color": _MID["rule"],
        "block_label_background_fill": "transparent", "block_label_text_color": _MID["muted"],
        "block_title_text_color": _MID["text"], "border_color_primary": _MID["rule"],
        "input_background_fill": _MID["input"], "input_border_color": _MID["rule"],
        "input_border_color_focus": _MID["accent"], "color_accent_soft": "#17304A",
        "button_primary_background_fill": _MID["accent"], "button_primary_background_fill_hover": _MID["accent_hover"],
        "button_primary_text_color": _MID["on_accent"], "button_primary_border_color": _MID["accent"],
        "button_secondary_background_fill": _MID["panel2"], "button_secondary_background_fill_hover": "#22345A",
        "button_secondary_text_color": _MID["text"], "button_secondary_border_color": _MID["rule"],
        "button_cancel_background_fill": "#2A1620", "button_cancel_text_color": "#FDA4AF",
        "button_cancel_border_color": "#4C1D2B",
        "checkbox_label_background_fill": _MID["input"], "checkbox_label_background_fill_selected": "#17304A",
        "checkbox_label_border_color": _MID["rule"], "checkbox_label_border_color_selected": "#155E75",
        "checkbox_label_text_color": _MID["text"], "checkbox_label_text_color_selected": "#CFFAFE",
        "checkbox_background_color_selected": _MID["accent"], "checkbox_border_color_selected": _MID["accent"],
        "slider_color": _MID["accent"], "code_background_fill": _MID["input"],
        "table_even_background_fill": _MID["panel"], "table_odd_background_fill": _MID["input"],
        "table_border_color": _MID["rule"],
    }.items()}, block_radius="12px", input_radius="10px", button_large_radius="10px", button_small_radius="8px",
        block_label_border_width="0px", color_accent=_MID["accent"], block_shadow="none", block_shadow_dark="none",
        button_primary_shadow="none", button_secondary_shadow="none"),
    variables="""
:root, .dark { --hero-bg: radial-gradient(1200px 300px at 0% 0%, #1D3A6B 0%, #132447 45%, #0E1628 100%);
  --hero-fg: #F1F5F9; --hero-sub: #A5B4C8; --hero-border: 1px solid #24324D; --hero-radius: 16px; --hero-pad: 22px 26px;
  --hero-shadow: none; --hero-chip-bg: rgba(34,211,238,.12); --hero-chip-fg: #67E8F9;
  --title-font: "IBM Plex Sans", ui-sans-serif, sans-serif; --mono: "IBM Plex Mono", ui-monospace, monospace;
  --strong: #F1F5F9; --muted: #94A3B8; --rule: #24324D; --accent: #22D3EE; --accent-strong: #22D3EE;
  --chip-bg: #17304A; --chip-fg: #A5F3FC; --cite-bg: #FBBF24; --cite-fg: #1A1205;
  --card-bg: #152036; --card-border: #24324D; --card-radius: 12px; }
""",
)

# ---------------------------------------------------------------- 4. Scholar (teal + serif headings)
SCHOLAR = Style(
    key="scholar",
    name="Scholar (teal)",
    theme=gr.themes.Base(
        primary_hue="teal", neutral_hue="gray",
        font=[LF("Source Sans 3"), LF("Lora"), "ui-sans-serif", "system-ui", "sans-serif"],
        font_mono=[LF("IBM Plex Mono"), "ui-monospace", "monospace"],
    ).set(
        body_background_fill="#F3F8F7", body_background_fill_dark="#0F1918",
        body_text_color="#1B2B29", body_text_color_dark="#E3EEEC",
        body_text_color_subdued="#5B6B68", body_text_color_subdued_dark="#9FB3AF",
        background_fill_primary="#FFFFFF", background_fill_primary_dark="#15211F",
        block_background_fill="#FFFFFF", block_background_fill_dark="#15211F",
        block_border_color="#D5E6E2", block_border_color_dark="#25403B",
        block_radius="12px", input_radius="10px", block_shadow="none", block_shadow_dark="none",
        block_label_background_fill="transparent", block_label_background_fill_dark="transparent",
        block_label_border_width="0px", block_label_text_color="#3F5653", block_label_text_color_dark="#9FB3AF",
        border_color_primary="#D5E6E2", border_color_primary_dark="#25403B",
        color_accent="#0F766E", color_accent_soft="#E6F4F1", color_accent_soft_dark="#173A35",
        input_border_color="#C9DDD8", input_border_color_focus="#0F766E", input_border_color_focus_dark="#2DD4BF",
        input_background_fill_dark="#101C1A", input_border_color_dark="#25403B",
        button_large_radius="10px", button_small_radius="8px",
        button_primary_background_fill="#0F766E", button_primary_background_fill_hover="#115E59",
        button_primary_text_color="#FFFFFF", button_primary_border_color="#0F766E",
        button_primary_background_fill_dark="#2DD4BF", button_primary_background_fill_hover_dark="#5EEAD4",
        button_primary_text_color_dark="#042F2B", button_primary_border_color_dark="#2DD4BF",
        button_secondary_background_fill="#FFFFFF", button_secondary_background_fill_hover="#ECF6F4",
        button_secondary_text_color="#0F766E", button_secondary_border_color="#9CCFC6",
        button_secondary_background_fill_dark="#15211F", button_secondary_text_color_dark="#5EEAD4",
        button_secondary_border_color_dark="#25403B",
        button_cancel_background_fill="#FFFFFF", button_cancel_text_color="#B42318", button_cancel_border_color="#F1C2BD",
        button_primary_shadow="none", button_secondary_shadow="none",
        checkbox_label_background_fill="#FFFFFF", checkbox_label_background_fill_selected="#E6F4F1",
        checkbox_label_border_color="#D5E6E2", checkbox_label_border_color_selected="#7BC4B8",
        checkbox_label_text_color_selected="#134E4A",
        checkbox_label_background_fill_dark="#15211F", checkbox_label_background_fill_selected_dark="#173A35",
        checkbox_label_text_color_selected_dark="#CCFBF1",
        checkbox_background_color_selected="#0F766E", checkbox_border_color_selected="#0F766E",
        slider_color="#0F766E", slider_color_dark="#2DD4BF",
    ),
    variables="""
:root { --hero-bg: #FFFFFF; --hero-fg: #134E4A; --hero-sub: #4B605C; --hero-border: 1px solid #D5E6E2;
  --hero-radius: 14px; --hero-pad: 20px 26px 20px 30px; --hero-shadow: inset 6px 0 0 #0F766E;
  --hero-chip-bg: #E6F4F1; --hero-chip-fg: #115E59;
  --title-font: Lora, Georgia, serif; --mono: "IBM Plex Mono", ui-monospace, monospace;
  --strong: #134E4A; --muted: #5B6B68; --rule: #D5E6E2; --accent: #0F766E; --accent-strong: #0F766E;
  --chip-bg: #E6F4F1; --chip-fg: #115E59; --cite-bg: #CCFBF1; --cite-fg: #134E4A;
  --card-bg: #FFFFFF; --card-border: #D5E6E2; --card-radius: 12px; }
.dark { --hero-bg: #15211F; --hero-fg: #CCFBF1; --hero-sub: #9FB3AF; --hero-border: 1px solid #25403B;
  --hero-shadow: inset 6px 0 0 #2DD4BF; --hero-chip-bg: #173A35; --hero-chip-fg: #99F6E4;
  --strong: #CCFBF1; --muted: #9FB3AF; --rule: #25403B; --accent: #2DD4BF; --accent-strong: #5EEAD4;
  --chip-bg: #173A35; --chip-fg: #99F6E4; --cite-bg: #115E59; --cite-fg: #CCFBF1;
  --card-bg: #15211F; --card-border: #25403B; }
#chat .bot-row { background: #FAFCFB !important; }
.dark #chat .bot-row, .dark #chat .message.bot { background: #15211F !important; border-color: #25403B !important; }
.dark #chat .user-row { background: #173A35 !important; }
""",
)

STYLES = {s.key: s for s in (INDIGO, STUDIO, MIDNIGHT, SCHOLAR)}
DEFAULT_STYLE = "scholar"


def active_style() -> Style:
    return STYLES.get((os.getenv("UI_STYLE") or DEFAULT_STYLE).strip().lower(), STYLES[DEFAULT_STYLE])


def style_css(style: Style) -> str:
    return style.variables + BASE_CSS
