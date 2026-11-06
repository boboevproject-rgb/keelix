"""Naval Base Watch: the desktop application.

Layout
------
A base list on the left; on the right, five views of the selected base:

* **Overview** -- the headline numbers, the activity chart and the latest
  notable changes.
* **Berths** -- which berths held a hull on each pass, as a grid.  This is
  the raw material every other view is built from.
* **Events** -- every arrival, departure and anomaly, filterable, each with
  a structured explanation: what, when, where, size reading, evidence,
  confidence.
* **Radar** -- the radar image behind any pass or event, with hulls outlined.
* **Compare** -- several bases on one time axis.

Work runs on one background thread; the window never blocks, every job
shows progress and can be cancelled, and finished analyses stay in memory.
"""

from __future__ import annotations

import json
import queue
import threading
import tkinter as tk
import tkinter.font as tkfont
import traceback
from datetime import datetime, timezone
from tkinter import filedialog, messagebox, ttk

import numpy as np

from . import charts, credentials, pursuit, radar, store
from .bases import BASES, BY_KEY, REGIONS
from .monitor import BaseReport, Event, compare, interpret, normal_range

# Display labels are the keys: the combobox shows them and looks them up here.
PERIODS = {"6 месяцев": 6, "9 месяцев": 9, "12 месяцев": 12,
           "18 месяцев": 18, "24 месяца": 24}
DEFAULT_PERIOD = "12 месяцев"
SEARCH_HINT = "Поиск баз"

# Region keys stay English (bases.py, CLI and tests use them); only the
# label shown in the sidebar is Russian.
REGION_LABEL = {"North America": "Северная Америка", "Europe": "Европа",
                "Russia": "Россия",
                "Middle East & Africa": "Ближний Восток и Африка",
                "Asia-Pacific": "Азиатско-Тихоокеанский регион"}

COUNTRY_LABEL = {"USA": "США", "Russia (Crimea)": "Россия (Крым)",
                 "Russia": "Россия", "China (Djibouti)": "Китай (Джибути)",
                 "China": "Китай", "UK": "Великобритания",
                 "France": "Франция", "Japan": "Япония",
                 "Germany": "Германия", "Italy": "Италия",
                 "Spain": "Испания", "Greece": "Греция", "Turkey": "Турция",
                 "Canada": "Канада", "Australia": "Австралия",
                 "India": "Индия", "Pakistan": "Пакистан",
                 "South Korea": "Южная Корея", "Netherlands": "Нидерланды",
                 "Syria": "Сирия", "Israel": "Израиль", "Bahrain": "Бахрейн",
                 "Iran": "Иран"}

# Event confidence codes come from monitor.py ("high"/"medium"/"low") and are
# compared in logic; only the displayed word is translated.
CONF_LABEL = {"high": "высокая", "medium": "средняя", "low": "низкая"}

# Event-type filter: label shown -> group key used by _filtered().
KIND_FILTERS = {"все": "all",
                "приходы и уходы": "moves",
                "группы и смены уровня": "groups",
                "долгие стоянки": "long",
                "единичные наблюдения": "single"}

_MONTHS_SHORT = ("янв", "фев", "мар", "апр", "мая", "июн", "июл", "авг",
                 "сен", "окт", "ноя", "дек")
_MONTHS_GEN = ("января", "февраля", "марта", "апреля", "мая", "июня", "июля",
               "августа", "сентября", "октября", "ноября", "декабря")


def _d(dt, fmt: str = "%d %b %Y") -> str:
    """strftime with Russian month names, independent of the OS locale."""
    m = dt.month - 1
    fmt = fmt.replace("%B", _MONTHS_GEN[m]).replace("%b", _MONTHS_SHORT[m])
    return dt.strftime(fmt)


def _conf(c: str) -> str:
    return CONF_LABEL.get(c, c)


def _plural(n: int, one: str, few: str, many: str) -> str:
    n = abs(int(n))
    if n % 10 == 1 and n % 100 != 11:
        return one
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return few
    return many


def _ships(n: int) -> str:
    return f"{n} {_plural(n, 'корабль', 'корабля', 'кораблей')}"

# ---------------------------------------------------------------- theme
# Quiet and expensive-looking rather than loud: a slate sidebar, a warm
# off-white workspace, white cards with hairline borders, one ink-blue
# accent, large tabular numbers, and no decoration that does not carry
# meaning.
APP_NAME = "Keelix"
BG = "#f6f5f2"            # workspace
PANEL = "#ffffff"         # cards
SIDE = "#161b22"          # sidebar
SIDE_TEXT = "#e6e8eb"
SIDE_MUTED = "#8b949e"
INK = "#1b1f24"
MUTED = "#6e7681"
LINE = "#e3e1dc"
ACCENT = "#0b5fa5"
RED = "#b42318"
GREEN = "#1a7f37"
AMBER = "#9a6700"

SEVERITY_LABEL = {3: "Главное", 2: "Заметное", 1: "Обычное", 0: "Мелкое"}


def _apply_theme(root: tk.Tk) -> dict:
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    base = tkfont.nametofont("TkDefaultFont")
    fam = base.cget("family")
    base.configure(size=12)
    fonts = {
        "body": base,
        "small": tkfont.Font(family=fam, size=10),
        "caps": tkfont.Font(family=fam, size=9, weight="bold"),
        "title": tkfont.Font(family=fam, size=17, weight="bold"),
        "brand": tkfont.Font(family=fam, size=14, weight="bold"),
        "kpi": tkfont.Font(family=fam, size=26, weight="bold"),
        "label": tkfont.Font(family=fam, size=11, weight="bold"),
        "mono": tkfont.Font(family="Menlo", size=10),
    }
    style.configure(".", background=BG, foreground=INK, font=base,
                    borderwidth=0, focuscolor=BG)
    style.configure("TFrame", background=BG)
    style.configure("Panel.TFrame", background=PANEL)
    style.configure("Side.TFrame", background=SIDE)
    style.configure("TLabel", background=BG, foreground=INK)
    style.configure("Panel.TLabel", background=PANEL)
    style.configure("Muted.TLabel", foreground=MUTED, font=fonts["small"])
    style.configure("PanelMuted.TLabel", background=PANEL, foreground=MUTED,
                    font=fonts["small"])
    style.configure("Caps.TLabel", background=PANEL, foreground=MUTED,
                    font=fonts["caps"])
    style.configure("Title.TLabel", font=fonts["title"])
    style.configure("Brand.TLabel", background=SIDE, foreground=SIDE_TEXT,
                    font=fonts["brand"])
    style.configure("SideMuted.TLabel", background=SIDE, foreground=SIDE_MUTED,
                    font=fonts["small"])
    style.configure("KPI.TLabel", background=PANEL, font=fonts["kpi"])
    style.configure("KPIcap.TLabel", background=PANEL, foreground=MUTED,
                    font=fonts["small"])
    style.configure("TButton", padding=(12, 6), background="#ffffff",
                    bordercolor=LINE, lightcolor="#ffffff", darkcolor="#ffffff",
                    relief="flat")
    style.map("TButton", background=[("active", "#f0efeb")])
    style.configure("Accent.TButton", foreground="white", background=ACCENT,
                    bordercolor=ACCENT, lightcolor=ACCENT, darkcolor=ACCENT,
                    padding=(14, 6))
    style.map("Accent.TButton", background=[("active", "#094b84"),
                                             ("disabled", "#9cbbd8")])
    style.configure("Nav.TButton", background=SIDE, foreground=SIDE_MUTED,
                    bordercolor=SIDE, lightcolor=SIDE, darkcolor=SIDE,
                    padding=(14, 8), anchor="w", font=fonts["label"])
    style.map("Nav.TButton", background=[("active", "#1f262e")],
              foreground=[("active", SIDE_TEXT)])
    style.configure("NavOn.TButton", background="#232b35",
                    foreground=SIDE_TEXT, bordercolor="#232b35",
                    lightcolor="#232b35", darkcolor="#232b35",
                    padding=(14, 8), anchor="w", font=fonts["label"])
    style.map("NavOn.TButton", background=[("active", "#232b35")])
    style.layout("Plain.TNotebook.Tab", [])
    style.configure("Plain.TNotebook", background=BG, borderwidth=0)
    style.configure("Treeview", background=PANEL, fieldbackground=PANEL,
                    rowheight=26, borderwidth=0, font=base,
                    bordercolor=PANEL, lightcolor=PANEL, darkcolor=PANEL)
    style.configure("Treeview.Heading", font=fonts["caps"], background=PANEL,
                    foreground=MUTED, relief="flat", padding=(6, 6))
    style.map("Treeview", background=[("selected", "#e8f0f9")],
              foreground=[("selected", INK)])
    style.configure("Side.Treeview", background=SIDE, fieldbackground=SIDE,
                    foreground=SIDE_TEXT, rowheight=26, borderwidth=0,
                    bordercolor=SIDE, lightcolor=SIDE, darkcolor=SIDE)
    style.configure("Side.Treeview.Heading", background=SIDE,
                    foreground=SIDE_MUTED, font=fonts["caps"], relief="flat")
    style.map("Side.Treeview", background=[("selected", "#2b3440")],
              foreground=[("selected", "#ffffff")])
    style.configure("Side.TEntry", fieldbackground="#232b35",
                    foreground=SIDE_TEXT, bordercolor="#232b35",
                    lightcolor="#232b35", darkcolor="#232b35",
                    insertcolor=SIDE_TEXT, padding=6)
    style.configure("TEntry", fieldbackground=PANEL, bordercolor=LINE,
                    lightcolor=PANEL, darkcolor=PANEL, padding=5)
    style.configure("TCombobox", fieldbackground=PANEL, bordercolor=LINE,
                    arrowcolor=MUTED, padding=4)
    style.configure("TSpinbox", fieldbackground=PANEL, bordercolor=LINE,
                    arrowcolor=MUTED, padding=4)
    style.configure("TProgressbar", troughcolor=LINE, background=ACCENT,
                    thickness=4, borderwidth=0)
    style.configure("TCheckbutton", background=BG)
    style.configure("Panel.TCheckbutton", background=PANEL)
    style.configure("Vertical.TScrollbar", background=LINE, troughcolor=PANEL,
                    bordercolor=PANEL, arrowcolor=MUTED, width=10)
    style.configure("Side.Vertical.TScrollbar", background="#2b3440",
                    troughcolor=SIDE, bordercolor=SIDE, arrowcolor=SIDE_MUTED,
                    width=10)
    root.configure(background=BG)
    return fonts


def _short_country(c: str) -> str:
    return {"USA": "US", "Russia (Crimea)": "RU", "Russia": "RU",
            "China (Djibouti)": "CN", "China": "CN", "UK": "UK",
            "France": "FR", "Japan": "JP", "Germany": "DE", "Italy": "IT",
            "Spain": "ES", "Greece": "GR", "Turkey": "TR", "Canada": "CA",
            "Australia": "AU", "India": "IN", "Pakistan": "PK",
            "South Korea": "KR", "Netherlands": "NL", "Syria": "SY",
            "Israel": "IL", "Bahrain": "BH", "Iran": "IR"}.get(c, c)


def _autowrap(label: ttk.Label, margin: int = 24) -> None:
    """Wrap a label to the width of its container.

    Measured from the parent, not the label itself: a label that wraps
    gets narrower, which would shrink its own next measurement, and so on
    down to one word per line.
    """
    parent = label.master
    last = {"w": 0}

    def on_conf(e):
        w = parent.winfo_width() - margin
        if w > 60 and abs(w - last["w"]) > 4:
            last["w"] = w
            label.configure(wraplength=w)
    parent.bind("<Configure>", on_conf, add=True)


def _scrolled_text(master, font, height=10, **kw) -> tk.Text:
    """A borderless Text with a scrollbar that appears only when needed."""
    frame = ttk.Frame(master, style="Panel.TFrame")
    frame.rowconfigure(0, weight=1)
    frame.columnconfigure(0, weight=1)
    t = tk.Text(frame, wrap="word", relief="flat", background=PANEL,
                foreground=INK, insertbackground=INK,
                selectbackground="#d6e4f5", selectforeground=INK,
                font=font, padx=6, pady=6, height=height, bd=0,
                highlightthickness=0, **kw)
    sb = ttk.Scrollbar(frame, orient="vertical", command=t.yview)

    def on_scroll(lo, hi):
        sb.set(lo, hi)
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            sb.grid_remove()
        else:
            sb.grid(row=0, column=1, sticky="ns")
    t.configure(yscrollcommand=on_scroll)
    t.grid(row=0, column=0, sticky="nsew")
    sb.grid(row=0, column=1, sticky="ns")
    t.frame = frame
    return t


def _placeholder(entry: ttk.Entry, var: tk.StringVar, text: str) -> None:
    """Grey hint text in an empty entry, cleared on focus."""
    state = {"on": False}

    def show():
        if not var.get():
            state["on"] = True
            entry.configure(foreground=SIDE_MUTED)
            var.set(text)

    def hide(_e=None):
        if state["on"]:
            state["on"] = False
            var.set("")
            entry.configure(foreground=SIDE_TEXT)

    def blur(_e=None):
        if not var.get():
            show()
    entry.bind("<FocusIn>", hide)
    entry.bind("<FocusOut>", blur)
    entry.configure(foreground=SIDE_TEXT)
    show()
    entry.placeholder_active = lambda: state["on"]


def _figure(master, size=(8, 4)):
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=size, dpi=100, facecolor="white")
    canvas = FigureCanvasTkAgg(fig, master=master)
    w = canvas.get_tk_widget()
    w.configure(highlightthickness=0, bd=0)
    return fig, canvas, w


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_NAME)
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{min(1600, int(sw * 0.94))}x{min(1000, int(sh * 0.9))}"
                      f"+{int(sw * 0.03)}+{int(sh * 0.04)}")
        self.minsize(1200, 760)
        self.fonts = _apply_theme(self)
        self.reports: dict[str, BaseReport] = {}
        self.current: str | None = None
        self.checked: set[str] = set()
        self.selected_event: Event | None = None
        self.view_pass: int = -1
        self.jobs: "queue.Queue" = queue.Queue()
        self.ui: "queue.Queue" = queue.Queue()
        self.cancel = threading.Event()
        self.busy = False
        self._photos: dict = {}
        self._build()
        threading.Thread(target=self._worker, daemon=True).start()
        self.after(100, self._poll)
        self._fill_list()
        self.after(250, self._startup)

    # =============================================================== layout
    def _build(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)
        self._build_sidebar()
        self._build_toolbar()
        self.nb = ttk.Notebook(self, style="Plain.TNotebook")
        self.nb.grid(row=1, column=1, sticky="nsew", padx=(18, 18),
                     pady=(0, 6))
        self._build_overview()
        self._build_berths()
        self._build_events()
        self._build_radar()
        self._build_compare()
        self._build_carriers()
        self._build_method()
        self._build_statusbar()
        self._build_nav()
        self._select_view(0)

    VIEWS = ("Обзор", "Причалы", "События", "Радар", "Сравнение",
             "Поиск в море", "Метод")

    def _build_nav(self) -> None:
        self.nav_btns = []
        for i, name in enumerate(self.VIEWS):
            b = ttk.Button(self.nav, text=name, style="Nav.TButton",
                           command=lambda i=i: self._select_view(i))
            b.pack(fill="x", pady=1)
            self.nav_btns.append(b)
        self.nb.bind("<<NotebookTabChanged>>", lambda e: self._sync_nav())

    def _select_view(self, i: int) -> None:
        self.nb.select(i)
        self._sync_nav()

    def _sync_nav(self) -> None:
        cur = self.nb.index(self.nb.select()) if self.nb.tabs() else 0
        for i, b in enumerate(self.nav_btns):
            b.configure(style="NavOn.TButton" if i == cur else "Nav.TButton")

    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self, padding=(18, 16, 18, 10))
        bar.grid(row=0, column=1, sticky="ew")
        bar.columnconfigure(0, weight=1)
        self.base_title = tk.StringVar(value="Выберите базу")
        self.base_sub = tk.StringVar(value="")
        ttk.Label(bar, textvariable=self.base_title, style="Title.TLabel"
                  ).grid(row=0, column=0, sticky="w")
        subf = ttk.Frame(bar)
        subf.grid(row=1, column=0, sticky="ew")
        sub = ttk.Label(subf, textvariable=self.base_sub, style="Muted.TLabel",
                        justify="left")
        sub.pack(anchor="w", fill="x")
        _autowrap(sub, margin=12)
        right = ttk.Frame(bar)
        right.grid(row=0, column=1, rowspan=2, sticky="e")
        ttk.Label(right, text="Период", style="Muted.TLabel").pack(
            side="left", padx=(0, 6))
        self.period = tk.StringVar(value=DEFAULT_PERIOD)
        cb = ttk.Combobox(right, textvariable=self.period, values=list(PERIODS),
                          state="readonly", width=12)
        cb.pack(side="left", padx=(0, 12))
        cb.bind("<<ComboboxSelected>>", lambda e: self._period_changed())
        ttk.Button(right, text="Обновить базу", style="Accent.TButton",
                   command=self._update_current).pack(side="left")
        ttk.Button(right, text="Обновить отмеченные",
                   command=self._update_checked).pack(side="left", padx=6)
        ttk.Button(right, text="Настройки", command=self._settings).pack(
            side="left")

    def _build_sidebar(self) -> None:
        side = ttk.Frame(self, style="Side.TFrame", padding=(16, 18, 16, 14))
        side.grid(row=0, column=0, rowspan=3, sticky="ns")
        side.rowconfigure(3, weight=1)
        ttk.Label(side, text=APP_NAME, style="Brand.TLabel").grid(
            row=0, column=0, sticky="w")
        ttk.Label(side, text="активность военно-морских баз по радару",
                  style="SideMuted.TLabel").grid(row=0, column=0, sticky="w",
                                                 pady=(26, 0))
        self.nav = ttk.Frame(side, style="Side.TFrame")
        self.nav.grid(row=1, column=0, sticky="ew", pady=(22, 18))
        self.search = tk.StringVar()
        self.search.trace_add("write", lambda *_: self._fill_list())
        ent = ttk.Entry(side, textvariable=self.search, width=30,
                        style="Side.TEntry")
        ent.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        _placeholder(ent, self.search, SEARCH_HINT)
        lst = ttk.Frame(side, style="Side.TFrame")
        lst.grid(row=3, column=0, sticky="nsew")
        lst.rowconfigure(0, weight=1)
        lst.columnconfigure(0, weight=1)
        self.tree = ttk.Treeview(lst, columns=("tick", "state"),
                                 show="tree headings", selectmode="browse",
                                 style="Side.Treeview")
        self.tree.heading("#0", text="БАЗА")
        self.tree.heading("tick", text="СРАВН")
        self.tree.heading("state", text="ПОСЛЕДНЕЕ")
        self.tree.column("#0", width=236, stretch=True)
        self.tree.column("tick", width=56, anchor="center", stretch=False)
        self.tree.column("state", width=130, stretch=False)
        self.tree.grid(row=0, column=0, sticky="nsew")
        tsb = ttk.Scrollbar(lst, orient="vertical", command=self.tree.yview,
                            style="Side.Vertical.TScrollbar")
        self.tree.configure(yscrollcommand=tsb.set)
        tsb.grid(row=0, column=1, sticky="ns")
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.tree.bind("<Button-1>", self._on_click_tree, add=True)
        row = ttk.Frame(side, style="Side.TFrame")
        row.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        ttk.Button(row, text="Отметить регион", style="Nav.TButton",
                   command=self._tick_region).pack(side="left")
        ttk.Button(row, text="Сбросить", style="Nav.TButton",
                   command=self._clear_ticks).pack(side="left", padx=4)
        ttk.Button(row, text="Тур", style="Nav.TButton",
                   command=lambda: Tour(self)).pack(side="right")
        self.key_status = tk.StringVar()
        ttk.Label(side, textvariable=self.key_status, style="SideMuted.TLabel"
                  ).grid(row=5, column=0, sticky="w", pady=(10, 0))
        self._refresh_key_status()

    def _panel(self, parent, **grid) -> ttk.Frame:
        """A white card with a hairline border."""
        outer = tk.Frame(parent, background=LINE, bd=0,
                         highlightthickness=0)
        outer.grid(**grid)
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)
        f = ttk.Frame(outer, style="Panel.TFrame", padding=16)
        f.grid(row=0, column=0, sticky="nsew", padx=1, pady=1)
        return f

    # --------------------------------------------------------------- overview
    def _build_overview(self) -> None:
        f = ttk.Frame(self.nb, padding=(10, 10, 0, 0))
        self.nb.add(f, text="Обзор")
        f.columnconfigure(0, weight=5)
        f.columnconfigure(1, weight=2)
        f.rowconfigure(1, weight=1)
        kpis = ttk.Frame(f)
        kpis.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        self.kpi: dict[str, tuple[tk.StringVar, tk.StringVar]] = {}
        for k, (key, cap) in enumerate((
                ("now", "КОРАБЛЕЙ\nВ ПОРТУ"), ("usual", "ОБЫЧНЫЙ\nУРОВЕНЬ"),
                ("delta", "С ПРОШЛОГО\nПРОЛЁТА"), ("carriers", "АВИА-\nНОСЦЫ"),
                ("metres", "КОРПУСА\nУ ПРИЧАЛОВ"),
                ("quality", "ПОСЛЕДНИЙ\nПРОЛЁТ"))):
            outer = tk.Frame(kpis, background=LINE)
            outer.grid(row=0, column=k, sticky="ew", padx=(0, 10))
            kpis.columnconfigure(k, weight=1)
            card = ttk.Frame(outer, style="Panel.TFrame", padding=(16, 12))
            card.pack(fill="both", expand=True, padx=1, pady=1)
            v = tk.StringVar(value="–")
            c = tk.StringVar(value=cap)
            d = tk.StringVar(value="")
            ttk.Label(card, textvariable=c, style="Caps.TLabel").pack(
                anchor="w")
            ttk.Label(card, textvariable=v, style="KPI.TLabel").pack(
                anchor="w", pady=(2, 0))
            dl = ttk.Label(card, textvariable=d, style="PanelMuted.TLabel",
                           justify="left")
            dl.pack(anchor="w", fill="x")
            _autowrap(dl, margin=34)
            self.kpi[key] = (v, c, d)
        chart = self._panel(f, row=1, column=0, sticky="nsew", padx=(0, 8))
        chart.rowconfigure(1, weight=1)
        chart.columnconfigure(0, weight=1)
        ttk.Label(chart, text="АКТИВНОСТЬ", style="Caps.TLabel").grid(
            row=0, column=0, sticky="w")
        hint = ttk.Label(chart, text="Крупные корабли в порту на каждом "
                                     "пролёте радара. Нажмите на маркер, "
                                     "чтобы узнать, что произошло: вниз — "
                                     "уход, вверх — приход, размер маркера "
                                     "соответствует кораблю.",
                         style="PanelMuted.TLabel", justify="left")
        hint.grid(row=1, column=0, sticky="ew", pady=(2, 0))
        _autowrap(hint, margin=34)
        chart.rowconfigure(2, weight=1)
        self.fig, self.canvas, w = _figure(chart, (9, 4.8))
        w.grid(row=2, column=0, sticky="nsew", pady=(8, 0))
        self.canvas.mpl_connect("pick_event", self._on_pick)
        recent = self._panel(f, row=1, column=1, sticky="nsew")
        recent.rowconfigure(1, weight=1)
        recent.columnconfigure(0, weight=1)
        ttk.Label(recent, text="ПОСЛЕДНИЕ\nИЗМЕНЕНИЯ", style="Caps.TLabel"
                  ).grid(row=0, column=0, sticky="w", pady=(0, 8))
        self.recent = _scrolled_text(recent, self.fonts["body"], height=10,
                                     cursor="arrow", width=34)
        self.recent.frame.grid(row=1, column=0, sticky="nsew")
        self.recent.tag_configure("h", font=self.fonts["label"], foreground=INK)
        self.recent.tag_configure("d", foreground=MUTED,
                                  font=self.fonts["small"])
        self.recent.tag_configure("sev3", foreground=RED)
        self.recent.tag_configure("link", foreground=ACCENT, underline=True)
        self.recent.tag_bind("link", "<Button-1>", self._recent_click)
        self.notes_var = tk.StringVar()
        nl = ttk.Label(f, textvariable=self.notes_var, style="Muted.TLabel",
                       justify="left")
        nl.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 6))
        _autowrap(nl)

    # ----------------------------------------------------------------- berths
    def _build_berths(self) -> None:
        f = ttk.Frame(self.nb, padding=(10, 10, 0, 0))
        self.nb.add(f, text="Причалы")
        f.rowconfigure(1, weight=1)
        f.columnconfigure(0, weight=1)
        head = ttk.Frame(f)
        head.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(head, text="Занятость причалов по пролётам").pack(anchor="w")
        hl = ttk.Label(head, style="Muted.TLabel", text=(
            "Каждая строка — причальное место (нумерация с севера на юг), "
            "каждый столбец — пролёт радара. Нажмите на ячейку, чтобы открыть "
            "этот пролёт в виде «Радар»."))
        hl.pack(anchor="w", fill="x")
        _autowrap(hl)
        panel = self._panel(f, row=1, column=0, sticky="nsew")
        panel.rowconfigure(0, weight=1)
        panel.columnconfigure(0, weight=1)
        self.bfig, self.bcanvas, w = _figure(panel, (10, 6))
        w.grid(row=0, column=0, sticky="nsew")
        self.bcanvas.mpl_connect("button_press_event", self._on_berth_click)
        self.berth_note = tk.StringVar()
        bl = ttk.Label(f, textvariable=self.berth_note, style="Muted.TLabel",
                       justify="left")
        bl.grid(row=2, column=0, sticky="ew", pady=(8, 6))
        _autowrap(bl)

    # ----------------------------------------------------------------- events
    def _build_events(self) -> None:
        f = ttk.Frame(self.nb, padding=(10, 10, 0, 0))
        self.nb.add(f, text="События")
        f.rowconfigure(1, weight=1)
        f.columnconfigure(0, weight=3)
        f.columnconfigure(1, weight=2)
        flt = ttk.Frame(f)
        flt.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        ttk.Label(flt, text="Показать", style="Muted.TLabel").pack(side="left")
        self.sev_vars = {}
        for sev, lab in ((3, "Главное"), (2, "Заметное"), (1, "Обычное"),
                         (0, "Мелкое")):
            v = tk.BooleanVar(value=sev >= 1)
            self.sev_vars[sev] = v
            ttk.Checkbutton(flt, text=lab, variable=v,
                            command=self._fill_events).pack(side="left",
                                                            padx=(8, 0))
        ttk.Label(flt, text="   Тип", style="Muted.TLabel").pack(side="left")
        self.kind_filter = tk.StringVar(value=next(iter(KIND_FILTERS)))
        kcb = ttk.Combobox(flt, textvariable=self.kind_filter, state="readonly",
                           width=22, values=tuple(KIND_FILTERS))
        kcb.pack(side="left", padx=(6, 0))
        kcb.bind("<<ComboboxSelected>>", lambda e: self._fill_events())
        ttk.Button(flt, text="Показать на радаре",
                   command=self._show_event_radar).pack(side="right")
        ttk.Button(flt, text="Сохранить отчёт…",
                   command=self._save_report).pack(side="right", padx=6)
        left = self._panel(f, row=1, column=0, sticky="nsew", padx=(0, 8))
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        cols = ("when", "sev", "what", "conf")
        self.ev_tree = ttk.Treeview(left, columns=cols, show="headings",
                                    selectmode="browse")
        for c, txt, wd in (("when", "Когда", 150), ("sev", "Уровень", 86),
                           ("what", "Что произошло", 230),
                           ("conf", "Уверенность", 104)):
            self.ev_tree.heading(c, text=txt)
            self.ev_tree.column(c, width=wd, anchor="w",
                                stretch=(c == "what"))
        sb = ttk.Scrollbar(left, orient="vertical", command=self.ev_tree.yview)
        self.ev_tree.configure(yscrollcommand=sb.set)
        self.ev_tree.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")
        self.ev_tree.bind("<<TreeviewSelect>>", self._on_event_select)
        self.ev_tree.tag_configure("sev3", foreground=RED)
        self.ev_tree.tag_configure("sev2", foreground=INK)
        self.ev_tree.tag_configure("sev1", foreground="#374151")
        self.ev_tree.tag_configure("sev0", foreground=MUTED)
        right = self._panel(f, row=1, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)
        self.detail = _scrolled_text(right, self.fonts["body"], height=12)
        self.detail.frame.grid(row=0, column=0, sticky="nsew")
        self.detail.tag_configure("title", font=self.fonts["title"])
        self.detail.tag_configure("field", font=self.fonts["label"],
                                  foreground=MUTED)
        self.detail.tag_configure("body", font=self.fonts["body"])
        self.detail.tag_configure("red", foreground=RED)
        self.detail.tag_configure("green", foreground=GREEN)
        self.detail.tag_configure("amber", foreground=AMBER)

    # ------------------------------------------------------------------ radar
    def _build_radar(self) -> None:
        f = ttk.Frame(self.nb, padding=(10, 10, 0, 0))
        self.nb.add(f, text="Радар")
        f.rowconfigure(1, weight=1)
        f.columnconfigure(0, weight=1)
        ctl = ttk.Frame(f)
        ctl.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Button(ctl, text="◀ Предыдущий пролёт",
                   command=lambda: self._step_pass(-1)).pack(side="left")
        ttk.Button(ctl, text="Следующий пролёт ▶",
                   command=lambda: self._step_pass(1)).pack(side="left",
                                                            padx=(4, 12))
        self.radar_title = tk.StringVar(value="")
        ttk.Label(ctl, textvariable=self.radar_title).pack(side="left")
        opts = ttk.Frame(ctl)
        opts.pack(side="right")
        self.opt_change = tk.BooleanVar(value=False)
        self.opt_struct = tk.BooleanVar(value=False)
        self.opt_berths = tk.BooleanVar(value=True)
        self.opt_photo = tk.BooleanVar(value=False)
        for txt, var in (("изменение к пустому состоянию", self.opt_change),
                         ("сооружения", self.opt_struct),
                         ("номера причалов", self.opt_berths),
                         ("спутниковый снимок (справочный, без даты)",
                          self.opt_photo)):
            ttk.Checkbutton(opts, text=txt, variable=var,
                            command=self._draw_radar).pack(side="left",
                                                           padx=(12, 0))
        panel = self._panel(f, row=1, column=0, sticky="nsew")
        panel.rowconfigure(0, weight=1)
        panel.rowconfigure(1, weight=0)
        panel.columnconfigure(0, weight=1)
        self.rfig, self.rcanvas, w = _figure(panel, (10, 6))
        w.grid(row=0, column=0, sticky="nsew")
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        tbf = ttk.Frame(panel, style="Panel.TFrame")
        tbf.grid(row=1, column=0, sticky="ew")
        tb = NavigationToolbar2Tk(self.rcanvas, tbf, pack_toolbar=False)
        tb.configure(background=PANEL)
        tb.pack(side="left")
        rl = ttk.Label(f, style="Muted.TLabel", justify="left",
                  text=("Светлое — сильное радарное эхо. Красные контуры — "
                        "корпуса, найденные на этом пролёте; зелёный контур — "
                        "корпус, к которому относится событие; оранжевый "
                        "пунктир — корабли, присутствующие на всех пролётах "
                        "записи. «Изменение к пустому состоянию» показывает "
                        "каждый пиксель относительно его самого пустого "
                        "состояния — именно на это смотрит детектор.")
                  )
        rl.grid(row=2, column=0, sticky="ew", pady=(8, 6))
        _autowrap(rl)

    # ---------------------------------------------------------------- compare
    def _build_compare(self) -> None:
        f = ttk.Frame(self.nb, padding=(10, 10, 0, 0))
        self.nb.add(f, text="Сравнение")
        f.rowconfigure(1, weight=3)
        f.rowconfigure(2, weight=1)
        f.columnconfigure(0, weight=1)
        head = ttk.Frame(f)
        head.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(head, text="Отмеченные базы на одной оси времени").pack(
            side="left")
        ttk.Button(head, text="Обновить", command=self._compare).pack(
            side="right")
        panel = self._panel(f, row=1, column=0, sticky="nsew")
        panel.rowconfigure(0, weight=1)
        panel.columnconfigure(0, weight=1)
        self.cfig, self.ccanvas, w = _figure(panel, (10, 5.5))
        w.grid(row=0, column=0, sticky="nsew")
        txt = self._panel(f, row=2, column=0, sticky="nsew", pady=(8, 6))
        txt.rowconfigure(0, weight=1)
        txt.columnconfigure(0, weight=1)
        self.ctext = _scrolled_text(txt, self.fonts["body"], height=7)
        self.ctext.frame.grid(row=0, column=0, sticky="nsew")

    # --------------------------------------------------------------- carriers
    def _build_carriers(self) -> None:
        f = ttk.Frame(self.nb, padding=(10, 10, 0, 0))
        self.nb.add(f, text="Поиск в море")
        f.rowconfigure(1, weight=1)
        f.columnconfigure(0, weight=2)
        f.columnconfigure(1, weight=3)
        head = ttk.Frame(f)
        head.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        ttk.Label(head, text="Поиск в море · экспериментальные кандидаты "
                             "по размеру").pack(anchor="w")
        sl = ttk.Label(head, style="Muted.TLabel", text=(
            "Выберите уход корабля размером с авианосец, чтобы проверить "
            "ближайшие полосы радарной съёмки на крупные широкие цели. Размер "
            "и подходящее время не устанавливают личность судна; цели рядом — "
            "не подтверждённое сопровождение. Покрытие и лимиты запросов "
            "оставляют пробелы. Поиск обычно занимает 10–40 минут."))
        sl.pack(anchor="w", fill="x")
        _autowrap(sl)
        left = self._panel(f, row=1, column=0, sticky="nsew", padx=(0, 8))
        left.rowconfigure(1, weight=1)
        left.rowconfigure(3, weight=2)
        left.columnconfigure(0, weight=1)
        ttk.Label(left, text="УХОДЫ КОРАБЛЕЙ РАЗМЕРОМ С АВИАНОСЕЦ",
                  style="Caps.TLabel").grid(row=0, column=0, sticky="w",
                                            pady=(0, 8))
        self.car_tree = ttk.Treeview(left, columns=("when", "size", "conf"),
                                     show="headings", selectmode="browse",
                                     height=6)
        for c, txt, wd in (("when", "Ушёл между", 170),
                           ("size", "Корпус на радаре", 140),
                           ("conf", "Уверенность", 104)):
            self.car_tree.heading(c, text=txt)
            self.car_tree.column(c, width=wd)
        self.car_tree.grid(row=1, column=0, sticky="nsew")
        ctl = ttk.Frame(left, style="Panel.TFrame")
        ctl.grid(row=2, column=0, sticky="ew", pady=6)
        ttk.Label(ctl, text="дней после", style="PanelMuted.TLabel").pack(
            side="left")
        self.car_days = tk.IntVar(value=10)
        ttk.Spinbox(ctl, from_=2, to=30, textvariable=self.car_days,
                    width=4).pack(side="left", padx=(4, 10))
        ttk.Label(ctl, text="радиус, км", style="PanelMuted.TLabel").pack(
            side="left")
        self.car_radius = tk.IntVar(value=700)
        ttk.Spinbox(ctl, from_=200, to=1500, increment=100,
                    textvariable=self.car_radius, width=5).pack(
            side="left", padx=(4, 10))
        ttk.Button(ctl, text="Проверить кандидатов", style="Accent.TButton",
                   command=self._pursue).pack(side="left")
        self.car_text = _scrolled_text(left, self.fonts["body"], height=12)
        self.car_text.frame.grid(row=3, column=0, sticky="nsew")
        right = self._panel(f, row=1, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)
        self.pfig, self.pcanvas, w = _figure(right, (8, 6))
        w.grid(row=0, column=0, sticky="nsew")
        charts.blank(self.pfig, "Выберите уход корабля размером с авианосец "
                                "и нажмите «Проверить кандидатов».")
        self.pursuits: dict[str, pursuit.Pursuit] = {}

    def _fill_carriers(self, rep: BaseReport) -> None:
        self.car_tree.delete(*self.car_tree.get_children())
        self._car_ids = {}
        for k, e in enumerate(sorted(pursuit.carrier_departures(rep),
                                     key=lambda e: e.t1, reverse=True)):
            iid = f"c{k}"
            self.car_tree.insert("", "end", iid=iid, values=(
                f"{_d(e.t0, '%d %b')} – {_d(e.t1, '%d %b %Y')}",
                f"{e.length_m:.0f} × {(e.width_m or 0):.0f} м",
                _conf(e.confidence)))
            self._car_ids[iid] = e
        if not self._car_ids:
            self.car_text.delete("1.0", "end")
            self.car_text.insert("1.0", "За этот период на этой базе не было "
                                        "уходов кораблей размером с "
                                        "авианосец.")

    def _pursue(self) -> None:
        rep = self.reports.get(self.current or "")
        sel = self.car_tree.selection()
        if rep is None or not sel:
            messagebox.showinfo("Выберите уход",
                                "Сначала выберите уход корабля размером с "
                                "авианосец.")
            return
        if not radar.have_credentials():
            self._settings()
            return
        e = self._car_ids[sel[0]]
        b = BY_KEY[rep.key]
        days, radius = float(self.car_days.get()), float(self.car_radius.get())
        key = f"{rep.key}:{e.t1:%Y%m%d}"
        self._select_view(5)
        self.car_text.delete("1.0", "end")
        self.car_text.insert("1.0", "Идёт поиск… каждая полоса радарной "
                                    "съёмки загружается с шагом 50 м плитками "
                                    "по 60 км и просматривается на корпус "
                                    "длиной от 200 м; лучшие кандидаты "
                                    "перемеряются с шагом 10 м по той же "
                                    "съёмке. Это не подтверждает личность "
                                    "судна. Copernicus ограничивает скорость "
                                    "для бесплатных аккаунтов, поэтому поиск "
                                    "занимает 10–40 минут. Его можно отменить "
                                    "и продолжить позже: загруженные плитки "
                                    "сохраняются.")

        def job():
            P = pursuit.search(rep.key, b.name, b.centre, e.t0, e.t1,
                               e.length_m, e.width_m or 80.0, days, radius,
                               self._progress, self.cancel)
            pursuit.link_arrivals(P, self.reports)
            try:
                from .basemap import fetch_basemap
                bm = fetch_basemap(P.region, "ocean", 1000)
            except Exception:
                bm = None
            return P, bm

        def done(out):
            P, bm = out
            self.pursuits[key] = P
            self.car_text.delete("1.0", "end")
            self.car_text.insert("1.0", P.summary())
            charts.draw_pursuit(self.pfig, P, bm)
            self.pcanvas.draw_idle()
            self.status.set(f"Поиск в море завершён: предварительных связей — "
                            f"{len(P.chain)}, проверено пролётов — "
                            f"{len(P.searched)}")

        self._submit("Проверка экспериментальных кандидатов поиска в море…",
                     job, done)

    def _build_method(self) -> None:
        f = ttk.Frame(self.nb, padding=(10, 10, 0, 0))
        self.nb.add(f, text="Метод и ограничения")
        f.rowconfigure(0, weight=1)
        f.columnconfigure(0, weight=1)
        panel = self._panel(f, row=0, column=0, sticky="nsew")
        panel.rowconfigure(0, weight=1)
        panel.columnconfigure(0, weight=1)
        t = _scrolled_text(panel, self.fonts["body"], height=20)
        t.configure(padx=14, pady=10)
        t.frame.grid(row=0, column=0, sticky="nsew")
        t.tag_configure("h", font=self.fonts["title"])
        for block in METHOD.split("\n\n"):
            if block.startswith("# "):
                t.insert("end", block[2:] + "\n\n", "h")
            else:
                t.insert("end", block + "\n\n")
        t.configure(state="disabled")

    def _build_statusbar(self) -> None:
        bar = ttk.Frame(self, padding=(18, 4, 18, 10))
        bar.grid(row=2, column=1, sticky="ew")
        bar.columnconfigure(1, weight=1)
        self.bar = ttk.Progressbar(bar, length=220, maximum=1.0)
        self.bar.grid(row=0, column=0, sticky="w")
        self.status = tk.StringVar(value="Готово")
        ttk.Label(bar, textvariable=self.status, style="Muted.TLabel").grid(
            row=0, column=1, sticky="w", padx=10)
        self.cancel_btn = ttk.Button(bar, text="Отмена", command=self._cancel,
                                     state="disabled")
        self.cancel_btn.grid(row=0, column=2, sticky="e")

    # ============================================================ base list
    def _fill_list(self) -> None:
        if not hasattr(self, "tree"):
            return                      # placeholder fires before the list
        q = self.search.get().strip().lower()
        if q == SEARCH_HINT.lower():
            q = ""
        self.tree.delete(*self.tree.get_children())
        for region in REGIONS:
            items = [b for b in BASES if b.region == region and (
                not q or q in b.name.lower() or q in b.country.lower()
                or q in COUNTRY_LABEL.get(b.country, "").lower()
                or q in b.key)]
            if not items:
                continue
            rid = f"region:{region}"
            self.tree.insert("", "end", iid=rid,
                             text=REGION_LABEL.get(region, region), open=True)
            for b in items:
                self.tree.insert(rid, "end", iid=b.key,
                                 text=f"{b.name}  ·  {_short_country(b.country)}",
                                 values=("☑" if b.key in self.checked else "☐",
                                         self._state_text(b.key)))
        if self.current and self.tree.exists(self.current):
            self.tree.selection_set(self.current)
            self.tree.see(self.current)

    def _state_text(self, key: str) -> str:
        rep = self.reports.get(key)
        if rep is not None and rep.passes:
            last = rep.passes[-1]
            pct = (f" · {last.in_port / rep.usual_in_port:.0%}"
                   if rep.usual_in_port else "")
            return f"{_ships(last.in_port)}{pct}"
        meta = store.load_meta(BY_KEY[key])
        if meta.get("summary"):
            return meta["summary"].replace("~", "")
        n = len(meta.get("passes", []))
        return (f"{n} {_plural(n, 'пролёт', 'пролёта', 'пролётов')}" if n
                else "не загружено")

    def _update_row(self, key: str) -> None:
        if self.tree.exists(key):
            self.tree.item(key, values=("☑" if key in self.checked else "☐",
                                        self._state_text(key)))

    def _on_click_tree(self, ev) -> None:
        col = self.tree.identify_column(ev.x)
        iid = self.tree.identify_row(ev.y)
        if col == "#1" and iid in BY_KEY:
            self.checked.symmetric_difference_update({iid})
            self._update_row(iid)
            return "break"

    def _on_select(self, _ev=None) -> None:
        sel = self.tree.selection()
        if not sel or sel[0] not in BY_KEY:
            return
        key = sel[0]
        if key == self.current and key in self.reports:
            return
        self.current = key
        self.selected_event = None
        self.view_pass = -1
        self._open_base(key)

    def _tick_region(self) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        iid = sel[0]
        region = (iid.split(":", 1)[1] if iid.startswith("region:")
                  else BY_KEY[iid].region)
        self.checked.update(b.key for b in BASES if b.region == region)
        self._fill_list()

    def _clear_ticks(self) -> None:
        self.checked.clear()
        self._fill_list()

    # ================================================================= jobs
    def _months(self) -> float:
        return float(PERIODS.get(self.period.get(), 12))

    def _submit(self, label: str, fn, done=None) -> None:
        if self.busy:
            messagebox.showinfo("Занято", "Выполняется другая задача. "
                                          "Дождитесь её завершения или "
                                          "нажмите «Отмена».")
            return
        self.busy = True
        self.cancel.clear()
        self.cancel_btn.configure(state="normal")
        self.status.set(label)
        self.bar["value"] = 0.0
        self.jobs.put((fn, done))

    def _worker(self) -> None:
        while True:
            fn, done = self.jobs.get()
            try:
                out = fn()
                self.ui.put(("done", done, out))
            except store.Cancelled:
                self.ui.put(("cancelled", None, None))
            except Exception as exc:
                traceback.print_exc()
                self.ui.put(("error", None, exc))

    def _progress(self, frac: float, msg: str) -> None:
        self.ui.put(("progress", frac, msg))

    def _poll(self) -> None:
        try:
            while True:
                kind, a, b = self.ui.get_nowait()
                if kind == "progress":
                    self.bar["value"] = max(0.0, min(1.0, a))
                    self.status.set(b)
                elif kind == "call":
                    a()
                else:
                    self.busy = False
                    self.cancel_btn.configure(state="disabled")
                    if kind == "done":
                        self.bar["value"] = 0.0
                        if a is not None:
                            a(b)
                    elif kind == "cancelled":
                        self.status.set("Отменено")
                        self.bar["value"] = 0.0
                    else:
                        self.status.set(f"Ошибка: {b}")
                        messagebox.showerror("Что-то пошло не так", str(b))
        except queue.Empty:
            pass
        self.after(100, self._poll)

    def _cancel(self) -> None:
        self.cancel.set()
        self.status.set("Отмена…")

    # =============================================================== actions
    def _startup(self) -> None:
        if not _ui_state().get("tour_done"):
            self.after(600, lambda: Tour(self))
        if not radar.have_credentials():
            self.status.set("Добавьте бесплатный ключ Copernicus в "
                            "«Настройках», чтобы загружать пролёты радара.")
        first = next((b.key for b in BASES
                      if store.load_meta(b).get("passes")), "norfolk")
        if self.tree.exists(first):
            self.tree.selection_set(first)
            self.tree.see(first)

    def _period_changed(self) -> None:
        self.reports.clear()
        if self.current:
            key = self.current
            self.current = None
            self.tree.selection_set(key)
            self._on_select()

    def _open_base(self, key: str) -> None:
        b = BY_KEY[key]
        self.base_title.set(f"{b.name}")
        self.base_sub.set(f"{COUNTRY_LABEL.get(b.country, b.country)}  ·  "
                          f"{b.note}")
        if key in self.reports:
            self._show_report(self.reports[key])
            return
        self._clear_views(f"Для базы {b.name} ещё нет анализа.")
        if len(store.known_passes(b)) < 4:
            self._clear_views(
                f"Для базы {b.name} ещё не загружены пролёты радара.\n\n"
                f"Нажмите «Обновить базу», чтобы загрузить их (бесплатно, "
                f"от одной до трёх минут).")
            return
        months = self._months()
        self._submit(f"Анализ: {b.name}…",
                     lambda: store.analyse_base(b, months, self._progress,
                                                self.cancel),
                     self._analysed)

    def _analysed(self, rep: BaseReport) -> None:
        self.reports[rep.key] = rep
        b = BY_KEY[rep.key]
        meta = store.load_meta(b)
        last = rep.passes[-1]
        pct = (f" · {last.in_port / rep.usual_in_port:.0%}"
               if rep.usual_in_port else "")
        meta["summary"] = f"{_ships(last.in_port)}{pct}"
        store.save_meta(b, meta)
        self._update_row(rep.key)
        self.status.set(rep.headline)
        if rep.key == self.current:
            self._show_report(rep)

    def _update_current(self) -> None:
        if not self.current:
            messagebox.showinfo("Выберите базу", "Выберите базу в списке.")
            return
        self._update_keys([self.current])

    def _update_checked(self) -> None:
        keys = [b.key for b in BASES if b.key in self.checked]
        if not keys:
            messagebox.showinfo("Ничего не отмечено",
                                "Сначала отметьте базы в столбце «СРАВН».")
            return
        self._update_keys(keys)

    def _update_keys(self, keys: list[str]) -> None:
        if not radar.have_credentials():
            self._settings()
            return
        months = self._months()

        def job():
            out = []
            for k, key in enumerate(keys):
                b = BY_KEY[key]
                store.update(b, months,
                             lambda f, m, k=k: self._progress(
                                 (k + 0.8 * f) / len(keys), m), self.cancel)
                rep = store.analyse_base(
                    b, months, lambda f, m, k=k: self._progress(
                        (k + 0.8 + 0.2 * f) / len(keys), m), self.cancel)
                self.ui.put(("call", (lambda r=rep: self._analysed(r)), None))
                out.append(rep)
            return out

        def done(reps):
            self.status.set(f"Обновлено баз: {len(reps)}. "
                            + (reps[-1].headline if reps else ""))
            if len(keys) > 1:
                self._compare()

        self._submit(f"Обновление баз: {len(keys)}…", job, done)

    # ================================================================ views
    def _clear_views(self, msg: str) -> None:
        for fig, canvas in ((self.fig, self.canvas), (self.bfig, self.bcanvas),
                            (self.rfig, self.rcanvas)):
            charts.blank(fig, msg)
            canvas.draw_idle()
        for key, (v, _c, d) in self.kpi.items():
            v.set("–")
            d.set("")
        self.recent.configure(state="normal")
        self.recent.delete("1.0", "end")
        self.recent.configure(state="disabled")
        self.ev_tree.delete(*self.ev_tree.get_children())
        self.detail.delete("1.0", "end")
        self.notes_var.set("")
        self.berth_note.set("")

    def _show_report(self, rep: BaseReport) -> None:
        self._fill_kpis(rep)
        self._pick_map = charts.draw_activity(self.fig, rep,
                                              self._filtered(rep),
                                              self.selected_event)
        self.canvas.draw_idle()
        self._fill_recent(rep)
        self.notes_var.set("  ·  ".join(rep.notes))
        charts.draw_berths(self.bfig, rep)
        self.bcanvas.draw_idle()
        st = rep.static_hulls
        nb = len(rep.berths)
        self.berth_note.set(
            f"{nb} " + _plural(nb, "причальное место", "причальных места",
                               "причальных мест") + ". "
            + (f"Объекты в форме корабля, яркие на каждом пролёте "
               f"({len(st)}: {', '.join(f'{h.length_m:.0f} м' for h in st)}), "
               f"считаются присутствующими всё время; в этой сетке их нет, "
               f"потому что они не менялись." if st else "")
            + " Причальное место — участок у пирса или причальной стенки, "
              "который на одних пролётах был пуст, а на других занят "
              "корпусом.")
        self._fill_events()
        self._fill_carriers(rep)
        self.view_pass = len(rep.passes) - 1
        self._draw_radar()

    def _fill_kpis(self, rep: BaseReport) -> None:
        P = rep.passes
        last = P[-1]
        prev = next((p for p in reversed(P[:-1]) if p.good), None)
        usual = rep.usual_in_port
        lo, hi = normal_range(rep)
        k = self.kpi
        k["now"][0].set(f"{last.in_port}")
        k["now"][2].set(f"на {_d(last.when, '%d %b %Y')}"
                        + (f" · всё время на месте: {last.static_n}"
                           if last.static_n else ""))
        k["usual"][0].set(f"{usual:.0f}")
        k["usual"][2].set(f"норма {lo:.0f}–{hi:.0f}")
        if prev is not None:
            d = last.in_port - prev.in_port
            days = (last.when - prev.when).days
            k["delta"][0].set(f"{d:+d}")
            k["delta"][2].set(f"с {_d(prev.when, '%d %b')} "
                              f"({days} {_plural(days, 'день', 'дня', 'дней')})")
        else:
            k["delta"][0].set("–")
            k["delta"][2].set("")
        k["carriers"][0].set(f"{last.carriers}")
        k["carriers"][2].set("корпуса и длинные, и широкие")
        k["metres"][0].set(f"{last.occupied_m / 1000:.1f} км")
        k["metres"][2].set("измерено, стабильнее подсчёта")
        q = ("хороший" if last.good else
             "волнение" if last.rough else
             "плохой снимок" if last.poor else "частичный обзор")
        k["quality"][0].set(q)
        k["quality"][2].set(f"{_d(last.when, '%d %b %H:%M')} UTC · ракурс "
                            f"{last.p.geometry}")

    def _fill_recent(self, rep: BaseReport) -> None:
        t = self.recent
        t.configure(state="normal")
        t.delete("1.0", "end")
        evs = [e for e in rep.events if e.severity >= 2 and e.kind != "visit"]
        evs.sort(key=lambda e: (e.t1, e.severity), reverse=True)
        self._recent_map = {}
        if not evs:
            t.insert("end", "За этот период ничего заметного.\n", "d")
        for k, e in enumerate(evs[:12]):
            tag = f"ev{k}"
            self._recent_map[tag] = e
            t.insert("end", f"{_d(e.t1, '%d %b')}  ", "d")
            t.insert("end", e.title + "\n", ("h", "link", tag)
                     + (("sev3",) if e.severity >= 3 else ()))
            t.insert("end", f"{charts.kind_label(e.kind)} · "
                            f"уверенность {_conf(e.confidence)}\n\n", "d")
        t.configure(state="disabled")

    def _recent_click(self, ev) -> None:
        idx = self.recent.index(f"@{ev.x},{ev.y}")
        tags = self.recent.tag_names(idx)
        e = next((self._recent_map[t] for t in tags
                  if t in getattr(self, "_recent_map", {})), None)
        if e is not None:
            self._select_event(e)
            self._select_view(2)

    # --------------------------------------------------------------- events
    def _filtered(self, rep: BaseReport) -> list[Event]:
        sev_ok = {s for s, v in self.sev_vars.items() if v.get()}
        kf = KIND_FILTERS.get(self.kind_filter.get(), "all")
        groups = {
            "all": None,
            "moves": ("arrived", "left", "left?",
                      "part-left", "part-arrived"),
            "groups": charts.HEADLINE_KINDS,
            "long": ("long-stay",),
            "single": ("visit",),
        }[kf]
        out = []
        for e in rep.events:
            if max(0, min(3, e.severity)) not in sev_ok:
                continue
            if groups is not None and e.kind not in groups:
                continue
            out.append(e)
        return out

    def _fill_events(self) -> None:
        self.ev_tree.delete(*self.ev_tree.get_children())
        rep = self.reports.get(self.current or "")
        if rep is None:
            return
        self._ev_ids = {}
        evs = self._filtered(rep)
        for k, e in enumerate(sorted(evs, key=lambda e: e.t1, reverse=True)):
            iid = f"e{k}"
            when = (_d(e.t1, "%d %b %Y") if e.t0 == e.t1 else
                    f"{_d(e.t0, '%d %b')} – {_d(e.t1, '%d %b %Y')}")
            sev = max(0, min(3, e.severity))
            self.ev_tree.insert("", "end", iid=iid, values=(
                when, SEVERITY_LABEL[sev], e.title, _conf(e.confidence)),
                tags=(f"sev{sev}",))
            self._ev_ids[iid] = e
        self._pick_map = charts.draw_activity(self.fig, rep, evs,
                                              self.selected_event)
        self.canvas.draw_idle()

    def _select_event(self, e: Event) -> None:
        self.selected_event = e
        for iid, ee in getattr(self, "_ev_ids", {}).items():
            if ee is e:
                self.ev_tree.selection_set(iid)
                self.ev_tree.see(iid)
                break
        self._show_detail(e)
        rep = self.reports.get(self.current or "")
        if rep is not None:
            self._pick_map = charts.draw_activity(self.fig, rep,
                                                  self._filtered(rep), e)
            self.canvas.draw_idle()

    def _on_event_select(self, _ev=None) -> None:
        sel = self.ev_tree.selection()
        if not sel:
            return
        e = self._ev_ids.get(sel[0])
        if e is not None and e is not self.selected_event:
            self._select_event(e)

    def _on_pick(self, ev) -> None:
        e = getattr(self, "_pick_map", {}).get(ev.artist)
        if e is not None:
            self._select_event(e)
            self._select_view(2)

    def _show_detail(self, e: Event) -> None:
        rep = self.reports.get(self.current or "")
        d = self.detail
        d.delete("1.0", "end")
        d.insert("end", e.title + "\n\n", "title")

        def field(name, text, tag="body"):
            d.insert("end", name.upper() + "\n", "field")
            d.insert("end", text + "\n\n", tag)

        when = (_d(e.t1, "%d %B %Y") if e.t0 == e.t1 else
                f"между пролётами {_d(e.t0, '%d %B')} и "
                f"{_d(e.t1, '%d %B %Y')}")
        field("Что", e.text)
        field("Когда", when + ". Пролёты радара идут с интервалом 6–12 "
                       "дней, поэтому точный день неизвестен.")
        if e.lon is not None:
            field("Где", f"{abs(e.lat):.4f}° {'с. ш.' if e.lat >= 0 else 'ю. ш.'}, "
                         f"{abs(e.lon):.4f}° {'в. д.' if e.lon >= 0 else 'з. д.'}"
                         + (f" · причал {e.berth}" if e.berth else
                            " · открытая вода"))
        if e.length_m:
            what = interpret(e.length_m, e.width_m or 40.0)
            field("Размер", f"длина {e.length_m:.0f} м"
                  + (f", ширина {e.width_m:.0f} м" if e.width_m else "")
                  + f" по радару (±{0.12 * e.length_m + 15:.0f} м). "
                  + what["text"][:1].upper() + what["text"][1:] + ".")
        if rep is not None and e.before >= 0 and e.after >= 0:
            field("Доказательства",
                  f"Пролёт {_d(rep.passes[e.before].when, '%d %b')} "
                  f"показывает состояние «до», а "
                  f"{_d(rep.passes[e.after].when, '%d %b')} — «после». "
                  f"Нажмите «Показать на радаре», чтобы увидеть оба "
                  f"с обведённым корпусом.")
        tag = {"high": "green", "medium": "amber", "low": "red"}.get(
            e.confidence, "body")
        field("Уверенность", _conf(e.confidence)
              + (f" — {e.why}" if e.why else ""), tag)

    # ---------------------------------------------------------------- berths
    def _on_berth_click(self, ev) -> None:
        rep = self.reports.get(self.current or "")
        if rep is None or ev.inaxes is None or ev.xdata is None:
            return
        j = int(round(ev.xdata))
        if 0 <= j < len(rep.passes):
            self.selected_event = None
            self.view_pass = j
            self._draw_radar()
            self._select_view(3)

    # ----------------------------------------------------------------- radar
    def _show_event_radar(self) -> None:
        if self.selected_event is None:
            messagebox.showinfo("Выберите событие", "Сначала выберите событие.")
            return
        self._select_view(3)
        self._draw_radar()

    def _step_pass(self, d: int) -> None:
        rep = self.reports.get(self.current or "")
        if rep is None:
            return
        self.selected_event = None
        self.view_pass = int(np.clip(self.view_pass + d, 0,
                                     len(rep.passes) - 1))
        self._draw_radar()

    def _photo(self, rep: BaseReport):
        if rep.key in self._photos:
            return self._photos[rep.key]
        try:
            from .basemap import fetch_basemap
            from .radar import to_lonlat
            H, W = rep.shape
            a, _b, c0, _d, e, f0 = rep.transform[:6]
            f = to_lonlat(rep.crs)
            lon0, lat0 = f(c0, f0 + e * H)
            lon1, lat1 = f(c0 + a * W, f0)
            bm = fetch_basemap((lon0, lat0, lon1, lat1), "satellite", 1200)
        except Exception:
            bm = None
        self._photos[rep.key] = bm
        return bm

    def _draw_radar(self) -> None:
        rep = self.reports.get(self.current or "")
        if rep is None:
            return
        e = self.selected_event
        if e is not None and e.before >= 0 and e.after >= 0:
            pb, pa = rep.passes[e.before], rep.passes[e.after]
            panels = [(e.before, f"до · {_d(pb.when, '%d %b %Y')}"
                                 f" · {_ships(pb.in_port)}"),
                      (e.after, f"после · {_d(pa.when, '%d %b %Y')}"
                                f" · {_ships(pa.in_port)}")]
            self.radar_title.set(f"{e.title}  ({_d(e.t0, '%d %b')} → "
                                 f"{_d(e.t1, '%d %b')})")
        else:
            i = self.view_pass if self.view_pass >= 0 else len(rep.passes) - 1
            p = rep.passes[i]
            panels = [(i, f"{_d(p.when, '%d %b %Y %H:%M')} UTC · ракурс "
                          f"{p.p.geometry} · {_ships(p.in_port)}"
                          + (" · волнение" if p.rough else ""))]
            self.radar_title.set(f"Пролёт {i + 1} из {len(rep.passes)}")
        charts.draw_radar(self.rfig, rep, panels,
                          photo=self._photo(rep) if self.opt_photo.get()
                          else None,
                          event=e, show_berths=self.opt_berths.get(),
                          show_structure=self.opt_struct.get(),
                          show_change=self.opt_change.get())
        self.rcanvas.draw_idle()

    # --------------------------------------------------------------- compare
    def _compare(self) -> None:
        keys = [b.key for b in BASES if b.key in self.checked]
        missing = [k for k in keys if k not in self.reports
                   and len(store.known_passes(BY_KEY[k])) >= 4]
        months = self._months()

        def job():
            for n, k in enumerate(missing):
                rep = store.analyse_base(
                    BY_KEY[k], months, lambda f, m, n=n: self._progress(
                        (n + f) / max(len(missing), 1), m), self.cancel)
                self.ui.put(("call", (lambda r=rep: self._analysed(r)), None))
            return keys

        self._select_view(4)
        if missing:
            self._submit("Анализ отмеченных баз…", job,
                         lambda _k: self._draw_compare(keys))
        else:
            self._draw_compare(keys)

    def _draw_compare(self, keys: list[str]) -> None:
        reps = [self.reports[k] for k in keys if k in self.reports]
        charts.draw_compare(self.cfig, reps)
        self.ccanvas.draw_idle()
        self.ctext.delete("1.0", "end")
        if not reps:
            self.ctext.insert("1.0", "Отметьте базы в столбце «СРАВН» и "
                                     "нажмите «Обновить». Для баз без "
                                     "загруженных пролётов сначала нажмите "
                                     "«Обновить отмеченные».")
            return
        lines = compare(reps)
        self.ctext.insert("1.0", "Красные линии: групповые уходы, падения "
                                 "уровня или необычно пустые периоды. "
                                 "Зелёные: обратное.\n\n"
                          + ("\n\n".join(lines) if lines else
                             "Ни на одной неделе ничего не происходило "
                             "сразу на нескольких базах."))

    # ------------------------------------------------------------------ misc
    def _save_report(self) -> None:
        rep = self.reports.get(self.current or "")
        if rep is None:
            return
        path = filedialog.asksaveasfilename(
            title="Сохранить отчёт", defaultextension=".txt",
            filetypes=[("Текстовые файлы", "*.txt"), ("Все файлы", "*.*")],
            initialfile=f"{rep.key}_{datetime.now(timezone.utc):%Y%m%d}.txt")
        if not path:
            return
        try:
            write_report(path, rep)
        except OSError as exc:
            messagebox.showerror("Не удалось сохранить отчёт",
                                 f"{path}\n\n{exc}")
            return
        self.status.set(f"Сохранено: {path}")

    def _refresh_key_status(self) -> None:
        self.key_status.set("Ключ Copernicus: "
                            + ("задан" if radar.have_credentials()
                               else "не задан — см. «Настройки»"))

    def _settings(self) -> None:
        d = tk.Toplevel(self)
        d.title("Настройки")
        d.transient(self)
        d.configure(background=BG)
        fr = ttk.Frame(d, padding=16)
        fr.pack(fill="both", expand=True)
        ttk.Label(fr, text="Copernicus Data Space (бесплатно)",
                  style="Title.TLabel").grid(row=0, column=0, columnspan=2,
                                             sticky="w")
        ttk.Label(fr, wraplength=520, style="Muted.TLabel", text=(
            "Зарегистрируйтесь на dataspace.copernicus.eu, откройте "
            "Sentinel Hub → User settings → OAuth clients, создайте клиента "
            "и вставьте сюда его id и secret. Данные хранятся только на этом "
            "компьютере, в вашем профиле пользователя.")).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(4, 10))
        cid = tk.StringVar()
        sec = tk.StringVar()
        ttk.Label(fr, text="ID клиента").grid(row=2, column=0, sticky="w",
                                              padx=(0, 8))
        ttk.Entry(fr, textvariable=cid, width=50).grid(row=2, column=1,
                                                       pady=2)
        ttk.Label(fr, text="Секрет клиента").grid(row=3, column=0, sticky="w",
                                                  padx=(0, 8))
        ttk.Entry(fr, textvariable=sec, width=50, show="•").grid(row=3,
                                                                 column=1,
                                                                 pady=2)
        cur = tk.StringVar()

        def refresh():
            cur.set(f"Сейчас: id "
                    f"{credentials.masked('copernicus', 'client_id')}, secret "
                    f"{credentials.masked('copernicus', 'client_secret')}")
        refresh()
        ttk.Label(fr, textvariable=cur, style="Muted.TLabel").grid(
            row=4, column=0, columnspan=2, sticky="w", pady=6)
        msg = tk.StringVar()

        def save():
            fields = {}
            if cid.get().strip():
                fields["client_id"] = cid.get().strip()
            if sec.get().strip():
                fields["client_secret"] = sec.get().strip()
            if fields:
                try:
                    credentials.set_credentials("copernicus", **fields)
                except OSError as exc:
                    msg.set(str(exc))
                    return
                radar._TOKEN["value"] = None
            refresh()
            self._refresh_key_status()
            msg.set(f"Сохранено в {credentials.CREDENTIALS_PATH}.")

        def test():
            try:
                radar.get_token(force=True)
                msg.set("Подключение к Copernicus есть: ключ работает.")
            except Exception as exc:
                msg.set(str(exc))

        bf = ttk.Frame(fr)
        bf.grid(row=5, column=0, columnspan=2, sticky="w")
        ttk.Button(bf, text="Сохранить", style="Accent.TButton",
                   command=save).pack(side="left")
        ttk.Button(bf, text="Проверить подключение", command=test).pack(
            side="left", padx=6)
        ttk.Button(bf, text="Закрыть", command=d.destroy).pack(side="left")
        ttk.Label(fr, textvariable=msg, wraplength=520).grid(
            row=6, column=0, columnspan=2, sticky="w", pady=6)


# ==========================================================================
# text report
# ==========================================================================
def write_report(path, rep: BaseReport) -> None:
    """Save the report as UTF-8.

    Explicit, because Windows otherwise writes in the locale's code page
    (cp1252), which cannot encode the arrows in the text: the file was
    created and the write then failed, leaving it empty.  The text is built
    before the file is opened so a failure never leaves a half-written file.
    """
    text = report_text(rep)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def report_text(rep: BaseReport) -> str:
    out = [rep.name, "=" * len(rep.name), rep.headline, ""]
    out += ["Примечания:"] + [f"  - {n}" for n in rep.notes] + [""]
    if rep.regimes:
        out.append("Периоды уровня:")
        for r in rep.regimes:
            out.append(f"  {rep.passes[r['start']].when:%Y-%m-%d} — "
                       f"{rep.passes[r['end']].when:%Y-%m-%d}: около "
                       f"{r['level']:.0f} крупных кораблей "
                       f"(пролётов: {r['n']})")
        out.append("")
    out.append("События, сначала новые:")
    for e in sorted(rep.events, key=lambda e: e.t1, reverse=True):
        if e.kind == "visit":
            continue
        out.append(f"\n[{e.t0:%Y-%m-%d} -> {e.t1:%Y-%m-%d}] {e.title}  "
                   f"(уверенность {_conf(e.confidence)})")
        out.append(f"  {e.text}")
        if e.why:
            out.append(f"  Почему: {e.why}")
    out.append("\nПо пролётам:")
    for p in rep.passes:
        out.append(f"  {p.when:%Y-%m-%d %H:%M}Z  ракурс {p.p.geometry:8s} "
                   f"крупных кораблей: {p.in_port:3d}, авианосцев: "
                   f"{p.carriers}, корпусов: {p.occupied_m / 1000:.1f} км"
                   + ("  волнение" if p.rough else ""))
    return "\n".join(out) + "\n"


METHOD = """# Что делает эта программа

Она следит за военно-морскими базами с помощью Sentinel-1 — бесплатного европейского радарного спутника. Каждые 6–12 дней спутник пролетает над базой в одно и то же время суток, под тем же углом, и делает радарный снимок. Радар видит сквозь облака и работает ночью, поэтому ни один пролёт не теряется из-за погоды.

# Как находят корабли

На радаре спокойная вода тёмная, а стальной корпус очень яркий, но вода у пирса никогда не бывает тёмной: собственное эхо пирса поднимает её яркость на 5–12 дБ даже при пустом причале. Поэтому ничего не сравнивается с открытой водой. Вместо этого программа запоминает для каждого 10-метрового пикселя самое «пустое» его состояние за все пролёты (3-й процентиль), предварительно убрав общий сдвиг яркости каждого угла обзора и общее усиление каждого пролёта. Корабль — это участок пикселей минимум на 6 дБ ярче собственного пустого состояния, с ядром ярче минимум на 10 дБ и пиком минимум на 18 дБ выше открытой воды. Соседние корабли разделяются: слабый ореол вокруг каждого яркого ядра отдаётся ближайшему ядру, поэтому два корабля у одного пирса не сливаются в один.

Каждое из этих чисел подобрано по снимкам PlanetScope базы Норфолк, сделанным в пределах нескольких часов от пролёта радара (четыре ясные пары в один день, 60 кораблей): полнота обнаружения корпусов 0,65 при точности 0,95 и корреляция 0,73 между занятостью причалов по радару и по фото. Первая версия, сравнивавшая пиксели с открытой водой, на тех же снимках давала 0,21 / 0,41 / 0,10.

# Что такое причальное место

Причальное место — участок у пирса или причальной стенки, который на одних пролётах был пуст, а на других занят корпусом. Пирсы, причальные стенки, краны и суша — это «сооружения»: они яркие даже в самом пустом состоянии и никогда не считаются кораблями. Корабль, ни разу не уходивший за всю запись, тоже ярок в своём пустом состоянии, поэтому для этого метода он выглядит как сооружение. Если такой участок ярок как корпус (от 18 дБ; пирсы — 9–15) и имеет форму корпуса, он отмечается как «на месте всё время» и считается кораблём — с оговоркой, что кран или ангар могут выглядеть так же.

# Как оценивают размер

Радар измеряет длину с точностью около 10–15 %, поэтому программа никогда не называет точный класс. Она опирается на размер и форму: авианосец длинный И широкий (его лётная палуба шириной ~75 м); два эсминца, стоящие нос к корме, длинные, но узкие; два корабля борт к борту короткие, но широкие. Каждое пояснение говорит, какую оценку программа выбрала и почему.

# Как находят аномалии

Периоды уровня: ряд «кораблей в порту» разбивается на периоды устойчивого уровня точным оптимальным разбиением (оцениваются все возможные наборы разрезов; стоимость — разброс каждого периода вокруг его медианы плюс штраф за каждый разрез, масштабированный по собственному шуму ряда). Падение с 24 до 15, державшееся двенадцать пролётов, — это смена уровня; один странный пролёт — нет. Периоды вне нормы (обычный уровень ± 2,5 робастных стандартных отклонения) помечаются как необычно пустые или полные.

Групповые движения: между двумя пролётами освободилось больше причальных мест, чем обычно бывает на этой базе. Уход корпуса размером с авианосец вместе с двумя или более кораблями сопровождения в том же интервале отмечается как выход авианосной группы.

Долгие стоянки: корабль стоит у причала вдвое дольше типичной стоянки на этом же причальном месте.

Уверенность: каждое событие указывает, на чём оно основано. Уход — «высокая», если корабль был виден на двух и более пролётах, а затем причал дважды оказался пустым; «низкая», если пустым его показал только один пролёт. Приход, замеченный один раз, — «средняя» до следующего пролёта.

# Экспериментальный поиск в море

После ухода корабля размером с авианосец поиск в море просматривает доступные радарные продукты вокруг базы в пределах своих ограничений по времени, площади и числу запросов. Он загружает плитки 60 км с шагом 50 м и сравнивает яркие цели с местным морским фоном. Кандидаты длиной от 200 м могут быть перемерены с шагом 10 м по той же съёмке. Успешное повторное измерение — это проверка размера, а не независимое наблюдение или опознание: исходное разрешение IW GRD около 20 на 22 м.

Крупные широкие цели — лишь кандидаты по размеру. Текущее правило связывания принимает первого кандидата по размеру с подходящим временем, а затем сверяет с ним последующих; размеры с исходным уходом оно не сопоставляет. Похожие размеры, меньшие цели поблизости и подходящее время не доказывают, что два обнаружения — одно и то же судно или что суда рядом — сопровождение. Пунктирные связи на карте — предварительные связи по времени, а не подтверждённый маршрут. Покрытие поиска разреженное, особенно вдали от берега; пропуски данных, лимиты запросов и недоступные проверки указываются в отчёте. Белые рамки на карте ограничивают пригодные пиксели загруженных плиток и могут содержать пробелы. Приведённые выше показатели точности по причалам к этому экспериментальному поиску не относятся.

# Где корабль может быть на плаву

Радар не отличает пришвартованный корабль от штабеля морских контейнеров. По году пролётов над двумя портами они неразличимы по всем признакам, доступным детектору: медианная яркость 13,2 против 13,4 дБ, самое пустое состояние 1,4 против 0,7 дБ, размах 18,0 против 18,7 дБ. И то и другое — металлические коробки на ровной поверхности. Модели рельефа тоже не помогают: намывные контейнерные терминалы во всех бесплатных моделях имеют высоту 0 м и отмечены как открытое море.

Поэтому Keelix опирается на карту земного покрова: ESA WorldCover, бесплатную, с разрешением 10 м и отдельным классом постоянной воды. Обнаружение засчитывается, только если оно перекрывает воду на карте. Это сохраняет 100 % настоящих кораблей в проверенном месте и убирает 63 % блоков контейнерного терминала. WorldCover — продукт 2021 года, поэтому суша, намытая позже, всё ещё считается водой; в примечаниях указывается, если карту покрова загрузить не удалось.

# Чего программа не умеет

Она не может назвать корабль или различить близкие классы. Корабли, стоящие борт к борту или друг за другом, сливаются в один корпус; число кораблей — оценка (метры корпусов у причалов — более стабильный показатель). Корабль, пришедший и ушедший между двумя пролётами, не виден вовсе. Даты приходов и уходов известны лишь с точностью до интервала между пролётами. Подводные лодки сидят низко в воде и дают слабое эхо. За пределами США бесплатные данные о рельефе грубее, поэтому маска суши менее строгая, а проверка глубины может сообщить, что не удаётся проследить путь выхода.

# Данные

Радар: Copernicus Sentinel-1, бесплатно (нужен бесплатный аккаунт Copernicus — см. «Настройки»). Суша и глубины: глобальная мозаика рельефа NOAA, бесплатно, без аккаунта. Спутниковый снимок под радарным видом — Esri World Imagery, только для ориентации; он сделан не в ту же дату.
"""


# ==========================================================================
# first-run tour
# ==========================================================================
UI_STATE = credentials.CONFIG_DIR / "ui.json"

TOUR_STEPS = [
    ("Добро пожаловать", None,
     "Эта программа следит за военно-морскими базами по бесплатным данным "
     "радара Sentinel-1. Каждые 6–12 дней спутник снимает каждую базу под "
     "одним и тем же углом; стальные корпуса яркие, вода тёмная, поэтому "
     "программа может считать крупные корабли в порту, следить за каждым "
     "причальным местом и объяснять, что изменилось.\n\n"
     "Этот тур займёт минуту и покажет все виды. Его можно повторить в любой "
     "момент кнопкой «Тур» внизу боковой панели."),
    ("1 · Ваш бесплатный ключ", None,
     "Радарные данные берутся из Copernicus Data Space. Это бесплатно: "
     "зарегистрируйтесь на dataspace.copernicus.eu, откройте Sentinel Hub → "
     "User settings → OAuth clients, создайте клиента и вставьте его id и "
     "secret в «Настройки» (справа вверху). Ключ хранится только на этом "
     "компьютере.\n\n"
     "Без ключа программа всё равно открывает базы, уже сохранённые на "
     "диске."),
    ("2 · Выбор базы", None,
     "На боковой панели 45 баз, сгруппированных по регионам; поле над "
     "списком фильтрует их. Нажмите на базу, чтобы открыть её. В первый раз "
     "нажмите «Обновить базу»: двенадцать месяцев радарных данных "
     "загружаются от одной до трёх минут, а потом база открывается сразу. "
     "Столбец «ПОСЛЕДНЕЕ» показывает число кораблей в порту и долю от "
     "обычного уровня базы.\n\n"
     "Столбец «СРАВН» отмечает базу для вида «Сравнение» и для кнопки "
     "«Обновить отмеченные». «Отметить регион» отмечает все базы выбранного "
     "региона."),
    ("3 · Обзор", 0,
     "Сначала шесть чисел: кораблей в порту сейчас, обычный уровень, "
     "изменение с прошлого пролёта, авианосцы, метры корпусов у причалов и "
     "качество последнего пролёта.\n\n"
     "График показывает каждый пролёт радара. Полоса — норма, чёрная линия — "
     "тренд, красные треугольники — уходы, зелёные — приходы; размер маркера "
     "соответствует кораблю. Затенённые периоды — необычно пустые или "
     "полные. Нажмите на любой маркер, чтобы прочитать, что произошло. "
     "Справа — последние заметные изменения; нажмите на любое, чтобы "
     "открыть его."),
    ("4 · Причалы", 1,
     "Исходный материал. Каждая строка — причальное место у пирса или "
     "причальной стенки, каждый столбец — пролёт радара, а цвет показывает, "
     "какая часть места была занята корпусом. Жёлтые столбцы — пролёты с "
     "волнением или частичным обзором. Нажмите на ячейку, чтобы открыть этот "
     "пролёт в виде «Радар».\n\n"
     "Корабли, не двигавшиеся за весь период, перечислены под сеткой как "
     "«на месте всё время»."),
    ("5 · События", 2,
     "Все приходы, уходы, групповые движения, смены уровня и долгие "
     "стоянки, сначала новые. Фильтр по уровню (от «Главное» до «Мелкое») и "
     "по типу. Каждое событие открывает структурированное пояснение: что, "
     "когда, где, оценка размера, пролёты-доказательства и уверенность с "
     "причиной.\n\n"
     "Нажмите «Показать на радаре», чтобы увидеть снимки «до» и «после» с "
     "обведённым корпусом. «Сохранить отчёт» записывает всё в текстовый "
     "файл."),
    ("6 · Радар", 3,
     "Снимок, на котором основано любое утверждение. Листайте пролёты или "
     "перейдите сюда из события, чтобы увидеть «до» и «после» рядом. "
     "Красные контуры — корпуса, найденные на этом пролёте; зелёный — "
     "корпус, к которому относится событие; оранжевый пунктир — корабли, "
     "присутствовавшие на всех пролётах.\n\n"
     "«Изменение к пустому состоянию» показывает то, на что действительно "
     "смотрит детектор: каждый пиксель относительно его самого пустого "
     "состояния. Панель инструментов позволяет масштабировать и сдвигать."),
    ("7 · Сравнение", 4,
     "Отмеченные базы на одной оси времени, каждая в процентах от своего "
     "обычного уровня, поэтому небольшую базу и Норфолк можно читать вместе. "
     "Красные линии отмечают групповые уходы, падения уровня и необычно "
     "пустые периоды; зелёные — обратное. Под графиком выделены недели, "
     "когда несколько баз менялись одновременно."),
    ("8 · Поиск в море", 5,
     "Выберите уход корабля размером с авианосец, чтобы проверить доступные "
     "полосы радарной съёмки вокруг базы. Крупные широкие цели — кандидаты "
     "по размеру; их личность и связь с судами поблизости остаются "
     "неподтверждёнными.\n\n"
     "Пунктирные связи — предварительные связи по времени, а не "
     "подтверждённый маршрут. Белые рамки ограничивают пригодные пиксели и "
     "могут содержать пробелы. Из-за разреженного покрытия и лимитов "
     "запросов часть кандидатов может остаться непроверенной. Поиск обычно "
     "занимает 10–40 минут, его можно отменить."),
    ("9 · Чему доверять", 6,
     "По сравнению со снимками Норфолка: находится около двух третей "
     "кораблей, девять из десяти обнаружений настоящие, а число кораблей на "
     "пролёт совпадает с фото с точностью до одного-двух. Метод не называет "
     "корабли, не различает близкие классы, сливает корабли, стоящие вплотную, "
     "и датирует события лишь интервалом между пролётами.\n\n"
     "Полное описание — в виде «Метод». Каждое событие говорит, на чём "
     "основана его уверенность; прочитайте эту строку, прежде чем "
     "действовать."),
]


def _ui_state() -> dict:
    try:
        return json.loads(UI_STATE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_ui_state(d: dict) -> None:
    try:
        UI_STATE.parent.mkdir(parents=True, exist_ok=True)
        UI_STATE.write_text(json.dumps(d), encoding="utf-8")
    except OSError:
        pass


class Tour(tk.Toplevel):
    """A small guided walk through every view; switches the view as it goes."""

    def __init__(self, app: "App") -> None:
        super().__init__(app)
        self.app = app
        self.step = 0
        self.title("Тур")
        self.configure(background=PANEL)
        self.resizable(False, False)
        if app.winfo_viewable():
            self.transient(app)
        fr = ttk.Frame(self, style="Panel.TFrame", padding=(22, 18))
        fr.pack(fill="both", expand=True)
        self.v_step = tk.StringVar()
        self.v_title = tk.StringVar()
        ttk.Label(fr, textvariable=self.v_step, style="Caps.TLabel").pack(
            anchor="w")
        ttk.Label(fr, textvariable=self.v_title, style="Panel.TLabel",
                  font=app.fonts["title"]).pack(anchor="w", pady=(4, 10))
        self.body = tk.Text(fr, wrap="word", relief="flat", background=PANEL,
                            foreground=INK, insertbackground=INK,
                            selectbackground="#d6e4f5", selectforeground=INK,
                            font=app.fonts["body"], width=62, height=14,
                            bd=0, highlightthickness=0)
        self.body.pack(fill="both", expand=True)
        row = ttk.Frame(fr, style="Panel.TFrame")
        row.pack(fill="x", pady=(14, 0))
        self.b_back = ttk.Button(row, text="Назад", command=self.back)
        self.b_back.pack(side="left")
        ttk.Button(row, text="Пропустить тур", command=self.finish).pack(
            side="left", padx=8)
        self.b_next = ttk.Button(row, text="Далее", style="Accent.TButton",
                                 command=self.next)
        self.b_next.pack(side="right")
        self.protocol("WM_DELETE_WINDOW", self.finish)
        self.bind("<Return>", lambda e: self.next())
        self.bind("<Escape>", lambda e: self.finish())
        self.show()
        if app.winfo_viewable():
            self.after(50, self._place)

    def _place(self) -> None:
        self.update_idletasks()
        ax, ay = self.app.winfo_rootx(), self.app.winfo_rooty()
        aw, ah = self.app.winfo_width(), self.app.winfo_height()
        w, h = self.winfo_width(), self.winfo_height()
        self.geometry(f"+{ax + aw - w - 40}+{ay + ah - h - 60}")
        self.lift()

    def show(self) -> None:
        title, view, text = TOUR_STEPS[self.step]
        self.v_step.set(f"ТУР · {self.step + 1} ИЗ {len(TOUR_STEPS)}")
        self.v_title.set(title)
        self.body.configure(state="normal")
        self.body.delete("1.0", "end")
        self.body.insert("1.0", text)
        self.body.configure(state="disabled")
        if view is not None:
            self.app._select_view(view)
        self.b_back.configure(state="normal" if self.step else "disabled")
        self.b_next.configure(text="Готово" if self.step == len(TOUR_STEPS) - 1
                              else "Далее")

    def next(self) -> None:
        if self.step >= len(TOUR_STEPS) - 1:
            self.finish()
            return
        self.step += 1
        self.show()

    def back(self) -> None:
        if self.step:
            self.step -= 1
            self.show()

    def finish(self) -> None:
        d = _ui_state()
        d["tour_done"] = True
        _save_ui_state(d)
        self.app._select_view(0)
        self.destroy()


def main() -> None:
    App().mainloop()


if __name__ == "__main__":
    main()
