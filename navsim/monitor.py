"""Naval base activity from Sentinel-1 radar.

What this does, in one paragraph
--------------------------------
Every few days Sentinel-1 photographs a base with radar, from the same angle
at the same time of day.  Open water is dark on radar and a steel hull is very
bright.  So for every pixel the program learns what that spot looks like when
it is *empty* (its darkest typical state across many passes), and on each pass
marks the spots that are far brighter than their own empty state.  Joined up,
those spots are hulls.  Following each hull from pass to pass gives arrivals,
departures and how long each ship stayed; counting them gives how full the
base is, and comparing that with its usual level gives the anomalies.

Why each rule exists (all measured on 45 passes over Norfolk, 2026)
-------------------------------------------------------------------
* **Compare with the pixel's own history**, not a fixed threshold: piers,
  cranes and buildings are bright on every pass, berths only when occupied.
* **Normalise each pass to its own open water**: wind roughens the sea and
  raises its brightness by up to 12 dB between passes.
* **Average VV and VH after a 3x3 smoothing**: VH shows ships with the most
  contrast but is speckly; the average is steadier than either.
* **Only "water when empty" pixels** (emptiest state within 4 dB of open
  water): car parks and lawns also change between passes, and without this
  rule land produced most of the hits.
* **Peak at least 18 dB above open water**: every ship at the Norfolk piers
  peaked at 19-41 dB, while the floating security barrier and changing land
  peaked at 11-16 dB.

What it cannot do
-----------------
Name ships or tell close classes apart (radar length is +/- ~15%); separate
ships moored side by side (they merge and count once); see a ship that
arrives and leaves between two passes; or notice a ship that sits at one
berth for nearly the whole record (that berth never looks empty).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Sequence

import numpy as np

from .radar import GSD_M, Pass, Patch


# ==========================================================================
# Configuration
# ==========================================================================
@dataclass
class MonitorConfig:
    # Every number below was set against PlanetScope truth at Norfolk (four
    # same-day radar/photo pairs, 60 ships): hull recall 0.65 at precision
    # 0.95, berth-occupancy correlation 0.73.  The first version, with
    # thresholds against open water, scored 0.21 / 0.41 / 0.10.
    smooth_px: int = 3
    ref_percentile: float = 3.0       # "emptiest state" of a pixel: the 3rd
    #                                   percentile over ALL passes, so a
    #                                   berth empty on one pass in thirty is
    #                                   still learned
    busy_percentile: float = 85.0
    own_ref_passes: int = 40          # a viewing angle with this many passes
    #                                   gets its own empty-state map
    group_min_days: float = 2.0       # passes closer than this cannot show
    #                                   a group movement, only a change of
    #                                   viewing angle
    change_db: float = 6.0            # hull halo: brighter than own empty
    #                                   state by this much ...
    seed_db: float = 10.0             # ... with a core at least this much
    level_db: float = 8.0             # and brighter than open water by
    peak_db: float = 18.0             # and somewhere inside, at least this
    structure_db: float = 11.0        # emptiest state this bright = pier,
    #                                   quay, crane, or a never-moving ship
    static_db: float = 18.0           # a structure this bright and ship-
    #                                   shaped is probably a moored ship
    zone_px: int = 7                  # berths lie this close to structure
    static_min_m: float = 90.0
    static_max_width_m: float = 110.0
    min_length_m: float = 60.0
    large_length_m: float = 100.0
    min_coverage: float = 0.5         # skip passes that imaged less of the box
    good_coverage: float = 0.6        # of berth pixels, to count a pass
    match_dilate_px: int = 2
    min_overlap: float = 0.3
    berth_change_db: float = 7.0      # p85 of change, to mark a berth
    min_berth_px: int = 15
    rough_sea_db: float = 4.0         # water this much brighter than usual
    min_water_frac: float = 0.15      # share of a hull that must lie over
    #                                   mapped water.  Measured on a year of
    #                                   passes at two ports: this keeps 100%
    #                                   of Norfolk's real ships and rejects
    #                                   63% of a container yard's blocks.
    poor_pass_ratio: float = 0.35     # hulls vs neighbouring passes: below
    #                                   this the pass itself is bad


# ==========================================================================
# Size classes: what a hull of a given radar length most probably is
# ==========================================================================
#: (name, shortest, longest, typical draft m, plain-language note).  Lengths
#: are of real warship classes worldwide; the ranges overlap on purpose.
#: Names and notes are display text (Russian); logic refers to the carrier
#: class through :data:`CARRIER_CLASS`, never by its wording.
SIZE_CLASSES: tuple[tuple[str, float, float, float, str], ...] = (
    ("авианосец", 285, 345, 11.5,
     "среди военных кораблей такую длину имеют только авианосцы"),
    ("универсальный десантный корабль или средний авианосец", 230, 290, 8.5,
     "напр., универсальные десантные корабли (~250 м) или авианосцы вроде "
     "«Шарля де Голля» (261 м)"),
    ("десантный транспорт-док или крупное судно снабжения", 180, 235, 7.5,
     "напр., десантные транспорты-доки (~208 м) и суда комплексного "
     "снабжения флота"),
    ("крейсер или большой эсминец", 150, 185, 9.5,
     "напр., крейсеры (~173 м) и крупнейшие эсминцы"),
    ("эсминец, фрегат или подводная лодка", 110, 160, 8.0,
     "напр., эсминцы (~155 м), фрегаты (~135 м), большие подводные лодки"),
    ("фрегат, корвет или подводная лодка", 70, 120, 5.5,
     "малые надводные боевые корабли и многоцелевые подводные лодки"),
    ("патрульный корабль, тральщик, буксир или баржа", 40, 75, 3.5,
     "малые суда"),
)
#: Name of the aircraft-carrier class in :data:`SIZE_CLASSES`.
CARRIER_CLASS = SIZE_CLASSES[0][0]


# ==========================================================================
# Russian wording helpers
# ==========================================================================
_MONTHS_GEN = ("января", "февраля", "марта", "апреля", "мая", "июня",
               "июля", "августа", "сентября", "октября", "ноября", "декабря")


def plural(n: float, one: str, few: str, many: str) -> str:
    """Russian noun form for a count: 1 пролёт, 2 пролёта, 5 пролётов."""
    k = abs(int(round(float(n))))
    if k % 10 == 1 and k % 100 != 11:
        return one
    if 2 <= k % 10 <= 4 and not 12 <= k % 100 <= 14:
        return few
    return many


def count(n: float, one: str, few: str, many: str) -> str:
    """``n`` (rounded) followed by the matching Russian noun form."""
    return f"{int(round(float(n)))} {plural(n, one, few, many)}"


#: Noun forms used with counts.
PASSES = ("пролёт", "пролёта", "пролётов")
PASSES_LOC = ("пролёте", "пролётах", "пролётах")     # "на N пролётах"
SHIPS = ("корабль", "корабля", "кораблей")
LARGE_SHIPS = ("крупный корабль", "крупных корабля", "крупных кораблей")
LARGE_HULLS = ("крупный корпус", "крупных корпуса", "крупных корпусов")
DAYS = ("день", "дня", "дней")


def fmt_date(t: datetime, year: bool = False, time: bool = False) -> str:
    """Russian date: «5 апреля», «5 апреля 2026», «5 апреля 14:30»."""
    s = f"{t.day} {_MONTHS_GEN[t.month - 1]}"
    if year:
        s += f" {t.year}"
    if time:
        s += f" {t:%H:%M}"
    return s


def fmt_lat_lon(lat: float, lon: float, digits: int = 4) -> str:
    """Position as «36.9500 с.ш. 76.3300 з.д.»."""
    return (f"{abs(lat):.{digits}f} {'с.ш.' if lat >= 0 else 'ю.ш.'} "
            f"{abs(lon):.{digits}f} {'в.д.' if lon >= 0 else 'з.д.'}")


#: What separates a carrier from ships moored end to end is width, not
#: length.  Measured at Norfolk: hulls at the carrier pier came out 81-89 m
#: wide on radar (a flight deck is ~75 m across), while pairs of destroyers
#: berthed bow to stern along one pier came out 290-420 m long but only
#: 46-61 m wide.
CARRIER_MIN_WIDTH_M = 72.0
CARRIER_MIN_LENGTH_M = 270.0
#: Two ships nested side by side come out about twice a single beam.
NESTED_MIN_WIDTH_M = 80.0
#: Typical berth length taken by one destroyer-size ship, bow gap included.
ONE_SHIP_BERTH_M = 165.0


def length_sigma(length_m: float) -> float:
    """One-sigma radar length error: 10 m pixels, smoothing and sidelobes."""
    return 15.0 + 0.12 * float(length_m)


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def size_class(length_m: float, exclude_carrier: bool = False
               ) -> list[tuple[str, float]]:
    """Ship types for one hull of this radar length, best first.

    Each class gets the probability that a true length inside its range
    would be measured as ``length_m``, with equal priors.
    """
    L = float(length_m)
    sd = length_sigma(L)
    probs = []
    for name, lo, hi, _d, _n in SIZE_CLASSES:
        if exclude_carrier and name == CARRIER_CLASS:
            continue
        probs.append((name, max(_phi((hi - L) / sd) - _phi((lo - L) / sd),
                                0.0)))
    tot = sum(p for _, p in probs) or 1.0
    return sorted(((n, p / tot) for n, p in probs), key=lambda t: -t[1])


def _class_text(length_m: float, exclude_carrier: bool = False,
                bare: bool = False) -> str:
    """Most probable class in words.  ``bare`` drops «скорее всего» from a
    confident reading, for use inside a longer sentence."""
    probs = size_class(length_m, exclude_carrier)
    (n1, p1), (n2, p2) = probs[0], probs[1]
    note1 = next(nt for n, *_r, nt in SIZE_CLASSES if n == n1)
    if p1 >= 0.6:
        return f"{n1} ({note1})" if bare else f"скорее всего, {n1} ({note1})"
    return f"вероятно, {n1} ({p1:.0%}), возможно, {n2} ({p2:.0%})"


def interpret(length_m: float, width_m: float) -> dict:
    """What a radar hull of this size most probably is, in words.

    Returns ``ships`` (estimated number of ships in the hull), ``kind``
    (carrier / single / end-to-end / side-by-side), ``draft`` (typical, m)
    and ``text``.
    """
    L, Wd = float(length_m), float(width_m)
    if L >= CARRIER_MIN_LENGTH_M and Wd >= CARRIER_MIN_WIDTH_M:
        txt = ("скорее всего, авианосец: корпус и длинный, и широкий "
               f"({Wd:.0f} м на радаре; полётная палуба авианосца шириной "
               "~75 м, а корабли, стоящие друг за другом, длинные, но узкие)")
        if L > 360:
            txt += ("; длиннее любого авианосца, так что у его носа или "
                    "кормы, вероятно, стоит ещё что-то")
        return {"ships": 1, "kind": "carrier", "draft": 11.5, "text": txt}
    if Wd >= NESTED_MIN_WIDTH_M:
        n = max(1, round(L / ONE_SHIP_BERTH_M))
        each = _class_text(L / n, True, bare=True)
        return {"ships": 2 * n, "kind": "side-by-side",
                "draft": typical_draft(L / n),
                "text": (f"ширина {Wd:.0f} м — примерно две ширины военного "
                         f"корабля: скорее всего, два корабля стоят борт к "
                         f"борту, каждый — {each}")}
    if L >= 250:
        n = max(2, round(L / ONE_SHIP_BERTH_M))
        single = _class_text(L, True, bare=True)
        return {"ships": n, "kind": "end-to-end", "draft": 9.0,
                "text": (f"{L:.0f} м занятого пирса, но ширина лишь {Wd:.0f} м"
                         f" — слишком узко для авианосца: скорее всего, "
                         f"{count(n, *SHIPS)} размером с эсминец или фрегат "
                         f"стоят друг за другом; менее вероятно — один "
                         f"крупный корабль ({single})")}
    return {"ships": 1, "kind": "single", "draft": typical_draft(L),
            "text": _class_text(L, True)}


def typical_draft(length_m: float) -> float:
    best = size_class(length_m, exclude_carrier=True)[0][0]
    return next(d for n, lo, hi, d, _ in SIZE_CLASSES if n == best)


def describe_size(length_m: float, width_m: float | None = None) -> str:
    """Plain-language reading of a hull's size."""
    return interpret(length_m, width_m if width_m is not None else 40.0)["text"]


# ==========================================================================
# Data structures
# ==========================================================================
@dataclass
class Hull:
    pass_index: int
    pixels: np.ndarray                  # flat indices into the grid
    x: float                            # UTM centroid
    y: float
    lon: float
    lat: float
    length_m: float
    width_m: float
    angle_deg: float
    area_m2: float
    peak_db: float
    berth: int = 0                      # 0 = not on a known berth
    track: int = -1


@dataclass
class PassResult:
    index: int
    p: Pass
    coverage: float                     # of the whole box
    berth_coverage: float               # of berth pixels
    water_db: float                     # open-water brightness (VV)
    rough: bool
    poor: bool = False                  # found far fewer hulls than its
    #                                     neighbours: a bad pass, not an
    #                                     empty base
    hulls: list[Hull] = field(default_factory=list)
    in_port: int = 0                    # estimated large ships present
    stretches: int = 0                  # occupied hull stretches (tracks)
    carriers: int = 0                   # of which carrier-shaped
    hull_m2: float = 0.0
    biggest_m: float = 0.0
    trend: float = 0.0                  # 3-pass median of in_port
    occupied_m: float = 0.0             # metres of hull alongside (measured)
    static_n: int = 0                   # ships present throughout the record
    berth_occ: dict = field(default_factory=dict)   # berth id -> fraction

    @property
    def when(self) -> datetime:
        return self.p.when

    @property
    def good(self) -> bool:
        return self.berth_coverage >= 0.6 and not self.rough and not self.poor


@dataclass
class Track:
    """One hull followed across passes: one ship (or one nested group)."""

    id: int
    presences: list[int] = field(default_factory=list)   # pass indices
    hulls: list[Hull] = field(default_factory=list)
    missed: list[int] = field(default_factory=list)
    misses_since: int = 0
    closed: bool = False
    geos: set = field(default_factory=set)      # viewing angles that saw it

    @property
    def length_m(self) -> float:
        return float(np.median([h.length_m for h in self.hulls]))

    @property
    def width_m(self) -> float:
        return float(np.median([h.width_m for h in self.hulls]))

    @property
    def lon(self) -> float:
        return float(np.mean([h.lon for h in self.hulls]))

    @property
    def lat(self) -> float:
        return float(np.mean([h.lat for h in self.hulls]))

    @property
    def berth(self) -> int:
        b = [h.berth for h in self.hulls if h.berth]
        return max(set(b), key=b.count) if b else 0

    @property
    def first(self) -> int:
        return self.presences[0]

    @property
    def last(self) -> int:
        return self.presences[-1]


@dataclass
class Event:
    kind: str            # arrived | left | visit | left? | emptier | fuller
    #                      | group-left | group-arrived
    t0: datetime         # happened after this pass ...
    t1: datetime         # ... and before (or at) this one
    title: str
    text: str
    severity: int = 1    # 3 = headline, 2 = notable, 1 = routine
    length_m: float | None = None
    width_m: float | None = None
    lon: float | None = None
    lat: float | None = None
    berth: int = 0
    track: int = -1
    confirmed: bool = True
    before: int = -1     # pass index showing the "before" state
    after: int = -1      # pass index showing the "after" state
    confidence: str = "medium"     # high | medium | low
    why: str = ""                  # what the confidence rests on
    evidence: dict = field(default_factory=dict)

    @property
    def when(self) -> datetime:
        return self.t1


@dataclass
class BaseReport:
    key: str
    name: str
    passes: list[PassResult]
    tracks: list[Track]
    events: list[Event]
    usual_in_port: float
    spread_in_port: float
    berth_map: np.ndarray               # int labels, 0 = none
    change_map: np.ndarray              # busy - empty, dB
    empty_map: np.ndarray               # emptiest state, dB over water
    contrast: np.ndarray                # (P, H, W) per-pass contrast, dB
    change: np.ndarray                  # (P, H, W) contrast - own empty state
    structure: np.ndarray               # piers, quays, land: never a ship
    zone: np.ndarray                    # where a berth can be
    transform: tuple[float, ...]
    crs: str
    shape: tuple[int, int]
    berths: list[dict]
    notes: list[str]
    exit_route: dict | None = None
    static_hulls: list = field(default_factory=list)   # present throughout
    regimes: list = field(default_factory=list)        # level periods
    no_water_mask: bool = False       # ran without a land-cover map

    # -------------------------------------------------------------- series
    def series(self, only_good: bool = False):
        ps = [p for p in self.passes if p.good or not only_good]
        return ([p.when for p in ps], np.array([p.in_port for p in ps]))

    @property
    def headline(self) -> str:
        if not self.passes:
            return "нет радарных пролётов"
        last = self.passes[-1]
        usual = self.usual_in_port
        rel = (f"{last.in_port / usual:.0%} от обычного" if usual > 0
               else "обычный уровень пока неизвестен")
        car = (f", из них {count(last.carriers, 'авианосец', 'авианосца', 'авианосцев')}"
               if last.carriers else "")
        return (f"{fmt_date(last.when, year=True)}: в порту примерно "
                f"{count(last.in_port, *LARGE_SHIPS)}{car} ({rel}; обычно "
                f"{usual:.0f})")


# ==========================================================================
# Image preparation
# ==========================================================================
#: Linear VH given to pixels that came back as exactly zero (about -42 dB,
#: below the darkest non-zero values seen).
VH_FLOOR = 10 ** -4.2


def _db(x):
    return 10.0 * np.log10(np.maximum(x, 1e-7))


def _smooth(a: np.ndarray, valid: np.ndarray, k: int) -> np.ndarray:
    from scipy.ndimage import uniform_filter
    num = uniform_filter(np.where(valid, a, 0.0), k, mode="nearest")
    den = uniform_filter(valid.astype(np.float64), k, mode="nearest")
    return np.where(den > 0.3, num / np.maximum(den, 1e-9), np.nan)


def contrast_stack(patches: Sequence[Patch], cfg: MonitorConfig
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-pass brightness above that pass's own open water, in dB.

    Returns ``(contrast, valid, water_vv_db, open_water_mask)``.
    """
    vv = np.stack([p.data[0] for p in patches]).astype(np.float64)
    vh = np.stack([p.data[1] for p in patches]).astype(np.float64)
    valid = np.stack([p.valid for p in patches])
    # VH below the noise floor comes back as exactly 0: very dark water, not
    # missing data.  Give it the floor value, and keep the zeros out of the
    # open-water level so that level means the same thing on every pass.
    vh_zero = valid & ~(vh > 0)
    vh = np.where(vh_zero, VH_FLOOR, np.nan_to_num(vh, nan=VH_FLOOR))
    dv = np.stack([_db(_smooth(a, v, cfg.smooth_px)) for a, v in zip(vv, valid)])
    dh = np.stack([_db(_smooth(a, v, cfg.smooth_px)) for a, v in zip(vh, valid)])
    dv[~valid] = np.nan
    dh[~valid] = np.nan
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(dh, axis=0)
    finite = np.isfinite(med)
    if finite.sum() < 50:
        raise ValueError("слишком мало снятых пикселей для анализа")
    # Open water: the darkest tenth of the box in cross-pol.  Every base box
    # contains harbour water, and nothing else is that dark in VH.
    water = finite & (med <= np.nanpercentile(med[finite], 10))
    wv = np.array([np.nanmedian(d[water & np.isfinite(d)])
                   if np.isfinite(d[water]).any() else np.nan for d in dv])
    wh = np.array([np.nanmedian(d[water & np.isfinite(d) & ~z])
                   if (water & np.isfinite(d) & ~z).any() else np.nan
                   for d, z in zip(dh, vh_zero)])
    c = 0.5 * ((dv - wv[:, None, None]) + (dh - wh[:, None, None]))
    return c.astype(np.float32), valid, wv, water


def _pct(stack: np.ndarray, q: float) -> np.ndarray:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanpercentile(stack, q, axis=0)


def _angle_offsets(C: np.ndarray, geo: np.ndarray, land: np.ndarray) -> dict:
    """How much brighter the static scene is from each viewing angle.

    Measured at Norfolk: the same piers are 4 dB brighter from relative
    orbit 4 than from orbit 106, and occupied berths 4.4 dB.  Removing the
    angle's offset lets every pass share one set of thresholds.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ma = np.nanmedian(C, axis=0)
        out = {}
        for g in np.unique(geo):
            mg = np.nanmedian(C[geo == g], axis=0)
            ok = np.isfinite(mg) & np.isfinite(ma) & ~land & (ma > 3.0)
            out[g] = float(np.nanmedian(mg[ok] - ma[ok])) if ok.sum() > 50 \
                else 0.0
    return out


def _references(Cn: np.ndarray, geo: np.ndarray, cfg: MonitorConfig
                ) -> tuple[dict, np.ndarray]:
    """The emptiest state of every pixel, on angle-corrected contrast.

    Pooled across angles by default: the angles fall on different days, so
    with ~30 passes each a berth occupied on every pass from one orbit but
    empty on a few from the other was invisible to a per-angle reference
    (its "empty" state was a ship).  A base seen from many orbits -- four
    at Sevastopol, ~50 passes each -- gets a map per angle instead, because
    ascending and descending views of the same pier differ pixel by pixel
    (layover and shadow fall the other way) and a pooled map then mistakes
    the view for a ship.  Returns per-angle maps and the pooled one.
    """
    pooled = _pct(Cn, cfg.ref_percentile)
    out = {}
    for g in np.unique(geo):
        sel = geo == g
        out[g] = (_pct(Cn[sel], cfg.ref_percentile)
                  if sel.sum() >= cfg.own_ref_passes else pooled)
    return out, pooled


def _static_hulls(ref_all, structure, land, W, transform, tolonlat, cfg
                  ) -> list:
    """Ship-shaped things that were bright on every pass.

    A ship that never left during the record is, to a change detector, part
    of the pier.  But piers and quays are 9-15 dB over water while hulls are
    18-30 dB, so a stretch of "structure" that is that bright and has a
    hull's proportions is most probably a ship in long maintenance, and is
    reported as present throughout rather than dropped.  Cranes and sheds
    can pass this test too, which is why it is labelled "probably".
    """
    from scipy.ndimage import label
    bright = structure & ~land & (np.nan_to_num(ref_all, nan=0.0)
                                  >= cfg.static_db)
    lab, n = label(bright, np.ones((3, 3)))
    out = []
    for pix in _component_pixels(lab, n) if n else []:
        h = _measure(-1, pix, np.nan_to_num(ref_all, nan=0.0), W, transform,
                     tolonlat, np.zeros_like(lab))
        if h is None or h.length_m < cfg.static_min_m \
                or h.length_m > 400 or h.width_m > cfg.static_max_width_m \
                or h.width_m > 0.6 * h.length_m:
            continue
        out.append(h)
    return out


def _split_hulls(d: np.ndarray, x: np.ndarray, allowed: np.ndarray,
                 cfg: MonitorConfig) -> tuple[np.ndarray, int]:
    """Label hulls so that neighbours do not merge through their halos.

    A ship's radar return has a bright core and a halo of sidelobes several
    pixels wide; two ships at the same pier touch through their halos and
    came out as one 950 m "hull" before this.  Cores (change >= seed_db)
    define the objects; halo pixels (change >= change_db) are given to the
    nearest core.
    """
    from scipy.ndimage import distance_transform_edt, label
    with np.errstate(invalid="ignore"):
        halo = (d >= cfg.change_db) & (x >= cfg.level_db) & allowed
        core = halo & (d >= cfg.seed_db)
    lab, n = label(core, np.ones((3, 3)))
    if n == 0:
        return lab, 0
    sizes = np.bincount(lab.ravel())
    tiny = np.nonzero(sizes < 3)[0]
    tiny = tiny[tiny > 0]
    if tiny.size:
        lab[np.isin(lab, tiny)] = 0
        core = lab > 0
        if not core.any():
            return lab, 0
    _dist, (ir, ic) = distance_transform_edt(~core, return_indices=True)
    out = np.where(halo & (_dist <= 3.0), lab[ir, ic], 0)
    return out, int(out.max())


# ==========================================================================
# Analysis
# ==========================================================================
def analyse(key: str, name: str, passes: Sequence[Pass],
            patches: Sequence[Patch], cfg: MonitorConfig | None = None,
            exit_route: dict | None = None,
            land: np.ndarray | None = None,
            land_cell_m: float = 30.0,
            water: np.ndarray | None = None) -> BaseReport:
    """Turn a stack of passes over one base into tracks, events and a series.

    ``passes`` and ``patches`` are parallel and in time order.  Patches must
    share one grid -- they do when fetched with the same box.  ``land`` is a
    boolean mask on that grid (see :func:`land_mask`); without it, car parks
    -- as dark as water on radar, with cars coming and going -- produce
    phantom ships.
    """
    cfg = cfg or MonitorConfig()
    from scipy.ndimage import binary_closing, label

    keep = [i for i, pt in enumerate(patches)
            if pt is not None and pt.shape == patches[0].shape
            and pt.valid.mean() >= cfg.min_coverage]
    passes = [passes[i] for i in keep]
    patches = [patches[i] for i in keep]
    if len(patches) < 4:
        raise ValueError(
            f"пригодных радарных пролётов: {len(patches)}; нужно не менее 4, "
            f"чтобы узнать, как выглядит пустой причал")
    H, W = patches[0].shape
    transform, crs = patches[0].transform, patches[0].crs
    C, valid, water_vv, open_water = contrast_stack(patches, cfg)

    geo = np.array([p.geometry for p in passes])
    if land is not None and land.shape != (H, W):
        land = None
    on_land = land if land is not None else np.zeros((H, W), bool)
    if water is not None and water.shape == (H, W):
        # Anything the land-cover map does not call water is land, even where
        # the elevation model says sea level: reclaimed terminals read as 0 m
        # in every free elevation model, which is how container yards became
        # "ships".
        on_land = on_land | ~water
    else:
        water = None
    land_reject = 0.5 if land_cell_m <= 15 else (0.7 if land_cell_m <= 60
                                                 else 0.9)

    # The empty state of every pixel, per viewing angle.  Measured at Norfolk
    # against PlanetScope truth: the same occupied berth is ~4 dB brighter
    # from one orbit than the other, and water alongside a pier is never
    # dark (pier sidelobes lift it 5-12 dB even when empty).  So nothing is
    # compared with open water; a ship is a pixel far brighter than *that
    # pixel* normally is *from that angle*.
    offsets = _angle_offsets(C, geo, on_land)
    Cn = np.stack([C[i] - offsets[g] for i, g in enumerate(geo)])
    refs, ref_all = _references(Cn, geo, cfg)
    Dc = np.stack([Cn[i] - refs[g] for i, g in enumerate(geo)])
    # Per-pass baseline: a low-percentile reference sits a few dB below a
    # typical pass, and individual passes come out brighter or darker as a
    # whole (one Norfolk pass was +8 dB on the piers).  Zero the change on
    # the parts of the scene that do not move.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i in range(len(passes)):
            ok = valid[i] & ~on_land & np.isfinite(Dc[i])
            if ok.sum() > 100:
                Dc[i] -= np.nanmedian(Dc[i][ok])
    Dc = Dc.astype(np.float32)

    # Structure: land, plus anything bright even in its emptiest state (a
    # pier, a quay, a crane -- or a ship that never left during the record,
    # which this method cannot tell from a pier and says so in the notes).
    structure = on_land | (np.nan_to_num(ref_all, nan=99.0) >= cfg.structure_db)
    from scipy.ndimage import binary_dilation
    near = binary_dilation(structure, np.ones((2 * cfg.zone_px + 1,) * 2))
    zone = near & ~structure & ~on_land

    static_hulls = _static_hulls(ref_all, structure, on_land, W, transform,
                                 tolonlat_fn := _lonlat_fn(crs), cfg)

    busy = _pct(Dc, cfg.busy_percentile)
    change = np.nan_to_num(busy, nan=0.0).astype(np.float32)
    berth_mask = zone & (change >= cfg.berth_change_db)
    berth_mask = binary_closing(berth_mask, np.ones((3, 3))) & ~structure
    blab, nb = label(berth_mask, np.ones((3, 3)))
    sizes = np.bincount(blab.ravel(), minlength=nb + 1)
    small = np.nonzero(sizes < cfg.min_berth_px)[0]
    blab[np.isin(blab, small[small > 0])] = 0
    blab = _renumber_north_to_south(blab)
    tolonlat = tolonlat_fn
    berths = _berth_info(blab, transform, tolonlat)
    berth_px = blab > 0
    zone_wide = binary_dilation(berth_px, np.ones((5, 5)))

    wv_med = float(np.nanmedian(water_vv))
    results: list[PassResult] = []
    for i, (ps, pt) in enumerate(zip(passes, patches)):
        v = valid[i]
        cov = float(v.mean())
        bcov = float(v[berth_px].mean()) if berth_px.any() else cov
        rough = bool(np.isfinite(water_vv[i])
                     and water_vv[i] - wv_med > cfg.rough_sea_db)
        pr = PassResult(i, ps, cov, bcov, float(water_vv[i]), rough)
        x = Cn[i]
        d = Dc[i]
        allowed = v & ~structure
        lab, n = _split_hulls(d, x, allowed, cfg)
        occ = lab > 0
        if n:
            for pix in _component_pixels(lab, n):
                h = _measure(i, pix, x, W, transform, tolonlat, blab)
                if h is None or h.length_m < cfg.min_length_m \
                        or h.peak_db < cfg.peak_db:
                    continue
                if on_land.ravel()[h.pixels].mean() > land_reject:
                    continue
                if water is not None and \
                        water.ravel()[h.pixels].mean() < cfg.min_water_frac:
                    continue        # not enough water under it to float
                pr.hulls.append(h)
        occ_flat = occ.ravel()
        for b in berths:
            m = blab.ravel() == b["id"]
            vv_ = v.ravel()[m]
            pr.berth_occ[b["id"]] = (float(occ_flat[m][vv_].mean())
                                     if vv_.mean() >= 0.5 else float("nan"))
        results.append(pr)

    # Quality gate.  A pass that finds a third of the hulls its neighbours
    # find did not see an empty base: it saw a bad image (storm clutter,
    # a swath edge through the piers, a calibration fault).  Such a pass is
    # kept for display but treated as "not imaged" by the tracker, so it
    # neither closes tracks nor dips the series.  Seen at Sevastopol, where
    # single passes dropped from 20 hulls to 0 and back.
    counts = np.array([sum(h.length_m >= cfg.min_length_m for h in pr.hulls)
                       for pr in results], float)
    link_valid = valid.copy()
    for i, pr in enumerate(results):
        nb = [counts[j] for j in range(max(0, i - 2), min(len(results), i + 3))
              if j != i]
        ref = float(np.median(nb)) if nb else 0.0
        if ref >= 4 and counts[i] < cfg.poor_pass_ratio * ref:
            pr.poor = True
            link_valid[i] = False
            pr.hulls = []
    tracks = _link(results, link_valid, (H, W), cfg)
    # A hull seen on one pass, nowhere near a berth, is a ship passing
    # through the harbour, not one using the base.
    tracks = [t for t in tracks if t.berth or len(t.presences) >= 2]
    _fill_series(results, tracks, valid, cfg, static_hulls)
    usual, spread = _usual(results)
    rep = BaseReport(key, name, results, tracks, [], usual, spread, blab,
                     change, ref_all.astype(np.float32), C, Dc, structure,
                     zone, transform, crs, (H, W), berths, [], exit_route,
                     static_hulls=static_hulls,
                     no_water_mask=water is None)
    rep.notes = _notes(rep, geo)
    rep.events = _events(rep, link_valid, cfg)
    return rep


def _lonlat_fn(crs: str) -> Callable:
    try:
        from .radar import to_lonlat
        return to_lonlat(crs)
    except Exception:
        return lambda x, y: (x, y)


def _component_pixels(lab: np.ndarray, n: int) -> list[np.ndarray]:
    flat = lab.ravel()
    order = np.argsort(flat, kind="stable")
    counts = np.bincount(flat, minlength=n + 1)
    starts = np.concatenate([[0], np.cumsum(counts)])
    return [order[starts[k]:starts[k + 1]] for k in range(1, n + 1)]


def _measure(i, pix, img, W, transform, tolonlat, blab) -> Hull | None:
    if pix.size < 4:
        return None
    rr, cc = np.divmod(pix, W)
    a, b, c0, d, e, f0 = transform[:6]
    X = c0 + a * (cc + 0.5) + b * (rr + 0.5)
    Y = f0 + d * (cc + 0.5) + e * (rr + 0.5)
    cov = np.cov(np.vstack([X, Y]))
    ev, evec = np.linalg.eigh(cov)
    px = abs(a) if a else GSD_M
    # A uniform bar of length L has variance L^2/12 along it; one pixel is
    # added back for the discretisation of the ends.
    L = math.sqrt(12.0 * max(ev[1], 0.0)) + px
    Wd = math.sqrt(12.0 * max(ev[0], 0.0)) + px
    ang = (math.degrees(math.atan2(evec[1, 1], evec[0, 1])) + 180.0) % 180.0
    vals = img.ravel()[pix]
    xm, ym = float(X.mean()), float(Y.mean())
    lon, lat = tolonlat(xm, ym)
    bl = blab.ravel()[pix]
    bl = bl[bl > 0]
    berth = int(np.bincount(bl).argmax()) if bl.size else 0
    return Hull(i, pix.astype(np.int64), xm, ym, float(lon), float(lat),
                float(L), float(Wd), float(ang), float(pix.size * px * px),
                float(np.nanmax(vals)), berth)


def _renumber_north_to_south(lab: np.ndarray) -> np.ndarray:
    """Berth numbers that read like a map: north first, then west to east."""
    ids = [k for k in np.unique(lab) if k > 0]
    if not ids:
        return lab
    from scipy.ndimage import center_of_mass
    cms = center_of_mass(lab > 0, lab, ids)
    order = sorted(zip(ids, cms), key=lambda t: (round(t[1][0] / 8), t[1][1]))
    out = np.zeros_like(lab)
    for new, (old, _cm) in enumerate(order, start=1):
        out[lab == old] = new
    return out


def _berth_info(blab, transform, tolonlat) -> list[dict]:
    out = []
    a, b, c0, d, e, f0 = transform[:6]
    for k in range(1, int(blab.max()) + 1):
        rr, cc = np.nonzero(blab == k)
        if rr.size == 0:
            continue
        x = c0 + a * (cc.mean() + 0.5)
        y = f0 + e * (rr.mean() + 0.5)
        lon, lat = tolonlat(x, y)
        out.append({"id": k, "lon": float(lon), "lat": float(lat),
                    "row": float(rr.mean()), "col": float(cc.mean()),
                    "area_m2": float(rr.size * abs(a * e))})
    return out


# ==========================================================================
# Linking hulls across passes
# ==========================================================================
def _dilated_index(pix: np.ndarray, shape, k: int) -> np.ndarray:
    """Flat indices of a pixel set grown by ``k`` pixels, computed locally."""
    from scipy.ndimage import binary_dilation
    H, W = shape
    rr, cc = np.divmod(pix, W)
    r0, r1 = max(rr.min() - k, 0), min(rr.max() + k + 1, H)
    c0, c1 = max(cc.min() - k, 0), min(cc.max() + k + 1, W)
    m = np.zeros((r1 - r0, c1 - c0), bool)
    m[rr - r0, cc - c0] = True
    if k > 0:
        m = binary_dilation(m, np.ones((2 * k + 1, 2 * k + 1)))
    lr, lc = np.nonzero(m)
    return np.sort((lr + r0) * W + (lc + c0))


def _covered(pix: np.ndarray, valid_i: np.ndarray) -> bool:
    return bool(valid_i.ravel()[pix].mean() >= 0.5)


def _link(results: list[PassResult], valid: np.ndarray, shape,
          cfg: MonitorConfig) -> list[Track]:
    """Follow hulls from pass to pass.

    Three things make radar hulls awkward to follow, and each has a rule:

    * **Misses.**  A target is occasionally lost to geometry or speckle, so a
      track survives one miss and closes only after two passes in a row that
      imaged the spot without seeing it.  Passes that did not image the spot
      neither confirm nor deny.
    * **Splits.**  One ship can come out as two pieces on one pass.  Every
      piece that overlaps a track belongs to it; none starts a new track.
    * **Merges.**  Two ships moored together can come out as one blob.  The
      blob counts as a sighting of every track it overlaps, but its size is
      not used to estimate either ship's length.

    Tracks are compared against the union of their last three sightings,
    which absorbs the small shifts between viewing angles.
    """
    k = cfg.match_dilate_px
    tracks: list[Track] = []
    active: list[Track] = []
    foot: dict[int, np.ndarray] = {}
    recent: dict[int, list[np.ndarray]] = {}
    for pr in results:
        i = pr.index
        dil = [_dilated_index(h.pixels, shape, k) for h in pr.hulls]
        # overlap of hull h with track t, as a share of the smaller of the two
        links: list[list[tuple[float, int]]] = [[] for _ in pr.hulls]
        for h_i, h in enumerate(pr.hulls):
            for t in active:
                last = t.hulls[-1]
                if math.hypot(h.x - last.x, h.y - last.y) > \
                        0.5 * (h.length_m + t.length_m) + 80.0:
                    continue
                a = np.intersect1d(h.pixels, foot[t.id],
                                   assume_unique=True).size / h.pixels.size
                b = np.intersect1d(last.pixels, dil[h_i],
                                   assume_unique=True).size / last.pixels.size
                ov = max(a, b)
                if ov >= cfg.min_overlap:
                    links[h_i].append((ov, t.id))
        by_id = {t.id: t for t in active}
        seen: set[int] = set()
        own: dict[int, tuple[float, int]] = {}      # track -> best own hull
        for h_i, h in enumerate(pr.hulls):
            if not links[h_i]:
                t = Track(len(tracks), [i], [h])
                t.geos.add(pr.p.geometry)
                h.track = t.id
                tracks.append(t)
                active.append(t)
                by_id[t.id] = t
                seen.add(t.id)
                recent[t.id] = [dil[h_i]]
                foot[t.id] = dil[h_i]
                continue
            links[h_i].sort(key=lambda c: -c[0])
            best_ov, best = links[h_i][0]
            h.track = best
            for _ov, tid in links[h_i]:
                seen.add(tid)
            if len(links[h_i]) == 1:           # not a merge: usable for size
                if best not in own or h.pixels.size > \
                        pr.hulls[own[best][1]].pixels.size:
                    own[best] = (best_ov, h_i)
            recent.setdefault(best, []).append(dil[h_i])
        for tid in seen:
            t = by_id[tid]
            t.geos.add(pr.p.geometry)
            if t.presences[-1] != i:
                t.presences.append(i)
                t.misses_since = 0
                if tid in own:
                    t.hulls.append(pr.hulls[own[tid][1]])
            r = recent.get(tid, [])[-3:]
            recent[tid] = r
            if r:
                foot[tid] = np.unique(np.concatenate(r))
        still = []
        for t in active:
            if t.id in seen:
                still.append(t)
                continue
            # A miss only counts from a viewing angle that has seen this
            # hull before: ascending and descending passes show the same
            # pier differently, and a ship visible from one angle but not
            # the other is a geometry effect, not a departure.
            g_now = results[i].p.geometry
            comparable = g_now in t.geos or len(t.geos) >= 3
            if comparable and _covered(t.hulls[-1].pixels, valid[i]):
                t.misses_since += 1
                t.missed.append(i)
            if t.misses_since >= 2:
                t.closed = True
            else:
                still.append(t)
        active = still
    for t in tracks:
        # misses bridged by a later sighting were just misses; the ones after
        # the last sighting are the departure evidence.
        t.missed = [m for m in t.missed if m < t.last]
    return tracks


def smoothed_sizes(t: Track) -> list[tuple[int, float, float]]:
    """(pass index, length, width) per own sighting, median of 3 neighbours.

    One sighting's size is noisy (+/-15%), and read raw it would flip a
    hull between "one ship" and "two ships end to end" from pass to pass.
    """
    hs = t.hulls
    out = []
    for k, h in enumerate(hs):
        win = hs[max(k - 1, 0):k + 2]
        out.append((h.pass_index, float(np.median([x.length_m for x in win])),
                    float(np.median([x.width_m for x in win]))))
    return out


def _fill_series(results, tracks, valid, cfg, static_hulls=()) -> None:
    """Per pass: estimated large ships, carriers and metres of hull alongside.

    Each track counts from its first to its last sighting, so a single
    missed pass does not dip the line.  A hull is converted to ships with
    :func:`interpret` -- a carrier is one ship, ~330 m of narrow hull along a
    pier is two destroyers end to end -- using its smoothed size at the time.
    Metres of hull is the more direct measurement: it does not depend on
    how a stretch is split into ships, so it is the steadier of the two.
    Ships present throughout the record are added as a constant.
    """
    n = len(results)
    ships = np.zeros(n)
    metres = np.zeros(n)
    carriers = np.zeros(n, int)
    area = np.zeros(n)
    big = np.zeros(n)
    for t in tracks:
        if t.length_m < cfg.min_length_m:
            continue
        large = t.length_m >= cfg.large_length_m
        sizes = {i: (L, Wd) for i, L, Wd in smoothed_sizes(t)}
        cur = sizes[t.hulls[0].pass_index]
        for i in range(t.first, t.last + 1):
            cur = sizes.get(i, cur)
            metres[i] += cur[0]
            if large:
                what = interpret(*cur)
                ships[i] += what["ships"]
                carriers[i] += what["kind"] == "carrier"
                big[i] = max(big[i], cur[0])
        for h in t.hulls:
            area[h.pass_index] += h.area_m2
    st_ships = sum(interpret(h.length_m, h.width_m)["ships"]
                   for h in static_hulls if h.length_m >= cfg.large_length_m)
    st_m = sum(h.length_m for h in static_hulls)
    st_car = sum(interpret(h.length_m, h.width_m)["kind"] == "carrier"
                 for h in static_hulls)
    for pr in results:
        i = pr.index
        pr.in_port = int(round(ships[i])) + st_ships
        pr.static_n = st_ships
        pr.carriers = int(carriers[i]) + st_car
        pr.occupied_m = float(metres[i] + st_m)
        pr.hull_m2 = float(area[i])
        pr.biggest_m = float(big[i])


def _usual(results) -> tuple[float, float]:
    """Usual level and spread, from the 3-pass trend of the good passes."""
    y = np.array([p.in_port for p in results], float)
    tr = np.array([np.median(y[max(0, i - 1):i + 2]) for i in range(len(y))])
    for p, v in zip(results, tr):
        p.trend = float(v)
    good = [p.trend for p in results if p.good] or [p.trend for p in results]
    if not good:
        return 0.0, 0.0
    a = np.array(good, float)
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med))) * 1.4826
    return med, max(mad, 0.5)


def normal_range(rep: "BaseReport") -> tuple[float, float]:
    """The band outside which the base counts as unusually empty or full."""
    u, sp = rep.usual_in_port, rep.spread_in_port
    thr = max(2.0, 3.0 * sp, 0.15 * u)
    return max(u - thr, 0.0), u + thr


# ==========================================================================
# Level periods: change-point detection on the in-port series
# ==========================================================================
def change_points(y: np.ndarray, min_seg: int = 3, min_shift: float = 2.0,
                  min_rel: float = 0.2) -> list[int]:
    """Indices where the level of ``y`` shifts: exact optimal partitioning.

    Dynamic programming over every possible set of cuts (the series is
    short, so this is instant), with a robust cost -- absolute deviation
    from each segment's median -- and a penalty per cut scaled to the
    series' own noise.  Binary segmentation, tried first, could not see a
    dip in the middle of a flat series: no single cut improves things, so
    it never made the second.  Afterwards, neighbouring segments whose
    levels differ by less than ``min_shift`` ships and ``min_rel`` of the
    overall level are merged back.
    """
    y = np.asarray(y, float)
    n = len(y)
    if n < 2 * min_seg:
        return []
    med_all = float(np.median(y))
    mad = float(np.median(np.abs(y - med_all))) * 1.4826
    pen = max(mad, 0.75) * 2.0 * math.log(max(n, 2))

    cost = np.full((n + 1, n + 1), np.inf)
    for a in range(n):
        for b in range(a + min_seg, n + 1):
            seg = y[a:b]
            cost[a, b] = float(np.abs(seg - np.median(seg)).sum())
    F = np.full(n + 1, np.inf)
    F[0] = -pen
    back = np.zeros(n + 1, int)
    for t in range(min_seg, n + 1):
        for s_ in range(0, t - min_seg + 1):
            if not np.isfinite(F[s_]):
                continue
            v = F[s_] + cost[s_, t] + pen
            if v < F[t]:
                F[t] = v
                back[t] = s_
    cuts = []
    t = n
    while t > 0:
        s_ = back[t]
        if s_ > 0:
            cuts.append(s_)
        t = s_
    cuts = sorted(cuts)
    # merge weak shifts
    thr = max(min_shift, min_rel * max(med_all, 1.0))
    changed = True
    while changed and cuts:
        changed = False
        bounds = [0] + cuts + [n]
        for k in range(1, len(bounds) - 1):
            lo = np.median(y[bounds[k - 1]:bounds[k]])
            hi = np.median(y[bounds[k]:bounds[k + 1]])
            if abs(hi - lo) < thr:
                cuts.pop(k - 1)
                changed = True
                break
    return cuts


def _regimes(rep: "BaseReport") -> list[dict]:
    """Periods of steady level in the in-port series (good passes only)."""
    P = rep.passes
    idx = [p.index for p in P if p.good]
    if len(idx) < 6:
        return [{"start": 0, "end": len(P) - 1,
                 "level": float(np.median([p.in_port for p in P]) if P else 0),
                 "n": len(P)}]
    y = np.array([P[i].in_port for i in idx], float)
    cuts = change_points(y)
    bounds = [0] + cuts + [len(idx)]
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        out.append({"start": idx[a], "end": idx[b - 1],
                    "level": float(np.median(y[a:b])), "n": b - a,
                    "metres": float(np.median([P[i].occupied_m
                                               for i in idx[a:b]]))})
    return out


# ==========================================================================
# Events and the text that explains them
# ==========================================================================
def _fmt(t: datetime) -> str:
    return fmt_date(t)


def _span(t0: datetime, t1: datetime) -> str:
    return f"между {_fmt(t0)} и {_fmt(t1)}"


def _cap(s: str) -> str:
    """Capitalise the first letter only (str.capitalize lowercases the rest)."""
    return s[:1].upper() + s[1:]


def _where(rep: BaseReport, t: Track, case: str = "loc") -> str:
    """Where a track is, as a phrase: ``loc`` = «у причала 3 (…)» (where it
    stands), ``from`` = «от причала 3 (…)» (where it left from)."""
    b = t.berth
    pos = fmt_lat_lon(t.lat, t.lon)
    if case == "from":
        return f"от причала {b} ({pos})" if b else f"из открытой воды ({pos})"
    return f"у причала {b} ({pos})" if b else f"в открытой воде ({pos})"


def _route_text(rep: BaseReport, length_m: float, width_m: float) -> str:
    r = rep.exit_route
    if not r or length_m < 150:
        return ""
    draft = interpret(length_m, width_m)["draft"]
    if not r.get("ok"):
        return (" Проверка глубин: бесплатная карта глубин здесь слишком "
                f"грубая (ячейки {r.get('cell_m', 0):.0f} м), чтобы "
                f"проследить путь выхода.")
    need = draft * 1.1
    km = r["distance_km"]
    hours = km / (15 * 1.852)
    txt = (f" Проверка глубин: осадка корабля такого размера — около "
           f"{draft:.0f} м. Самый глубокий путь в открытое море длиной "
           f"{km:.0f} км (около {hours:.1f} ч на 15 узлах)")
    if r["bottleneck_m"] < need:
        txt += (f", и бесплатная карта глубин показывает в самом мелком месте "
                f"лишь {r['bottleneck_m']:.0f} м — реальный фарватер углублён "
                f"сильнее, чем знает карта")
    else:
        txt += f", глубина не менее {r['bottleneck_m']:.0f} м на всём пути"
    return txt + "."


def _conf(level: str, why: str) -> dict:
    return {"confidence": level, "why": why}


def _events(rep: BaseReport, valid: np.ndarray, cfg: MonitorConfig
            ) -> list[Event]:
    P = rep.passes
    n = len(P)
    ev: list[Event] = []

    def covered_before(t: Track, i: int) -> int:
        for j in range(i - 1, -1, -1):
            if _covered(t.hulls[0].pixels, valid[j]):
                return j
        return -1

    def covered_after(t: Track, i: int) -> int:
        for j in range(i + 1, n):
            if _covered(t.hulls[-1].pixels, valid[j]):
                return j
        return -1

    # ---------------------------------------------------- per-ship events
    for t in rep.tracks:
        L, Wd = t.length_m, t.width_m
        if L < cfg.min_length_m:
            continue
        what = interpret(L, Wd)
        size = what["text"]
        sev = 3 if what["kind"] == "carrier" else (2 if L >= 180 else 1)
        if L < cfg.large_length_m:
            sev = 0
        peak = float(np.median([h.peak_db for h in t.hulls]))
        # "на 3 пролётах": the count in the prepositional case
        stay = count(len(t.presences), *PASSES_LOC)
        j0 = covered_before(t, t.first)
        j1 = covered_after(t, t.last)
        arrived = j0 >= 0
        left = j1 >= 0 and t.closed
        maybe_left = j1 >= 0 and not t.closed
        where = _where(rep, t)
        spot_empty = "причал пуст" if t.berth else "место пусто"
        spot_imaged = ("причал снят пустым" if t.berth else
                       "место снято пустым")
        L_txt = (f"длиной около {L:.0f} м (оценка по радару, "
                 f"±{length_sigma(L):.0f} м)")
        strength = ("сильным" if peak >= 24 else "чётким" if peak >= 20
                    else "умеренным")
        if len(t.presences) == 1 and t.last < n - 1:
            gone = (f", а к {_fmt(P[j1].when)} его уже не было"
                    if j1 >= 0 else "")
            ev.append(Event(
                "visit", P[t.first].when, P[t.first].when,
                f"Корпус {L:.0f} м замечен один раз"
                + (f" у причала {t.berth}" if t.berth else " в открытой воде"),
                f"Корпус {L_txt} замечен {where} {_fmt(P[t.first].when)}"
                f"{gone}. Он виден только на одном пролёте: это может быть "
                f"проходящий корабль, короткая остановка или разовый "
                f"артефакт радара. По размеру: {size}.",
                min(sev, 1), L, Wd, t.lon, t.lat, t.berth, t.id, False,
                j0 if j0 >= 0 else t.first, j1 if j1 >= 0 else t.first,
                **_conf("low", "одно наблюдение; ни до, ни после него "
                        "корпус не виден")))
            continue
        if arrived:
            once = len(t.presences) == 1
            ev.append(Event(
                "arrived", P[j0].when, P[t.first].when,
                (f"Корпус {L:.0f} м: приход к причалу {t.berth}"
                 if t.berth else f"Корпус {L:.0f} м появился в открытой воде")
                + (" (пока замечен один раз)" if once else ""),
                f"{_cap(_span(P[j0].when, P[t.first].when))} корпус "
                f"{L_txt} появился {where}. По размеру: {size}. "
                + (f"На последнем пролёте ({_fmt(P[t.last].when)}) он всё "
                   f"ещё на месте." if t.last == n - 1 else
                   f"Виден на {stay}."),
                min(sev, 1) if once else sev, L, Wd, t.lon, t.lat, t.berth,
                t.id, not once, j0, t.first,
                **_conf("medium" if once else "high",
                        f"{_fmt(P[j0].when)} {spot_empty}, затем корпус с "
                        f"{strength} отражением ({peak:.0f} дБ) на "
                        f"{stay}" + ("; следующий пролёт это подтвердит"
                                     if once else ""))))
        if left or maybe_left:
            since = (f"в порту как минимум с {_fmt(P[t.first].when)}"
                     if not arrived else
                     f"он пришёл {_span(P[j0].when, P[t.first].when)}")
            conf = ("" if left else
                    f" Пока не подтверждено: {spot_empty} только на одном "
                    f"пролёте; следующий пролёт подтвердит или отменит это.")
            empties = sum(1 for j in range(t.last + 1, n)
                          if _covered(t.hulls[-1].pixels, valid[j]))
            ev.append(Event(
                "left" if left else "left?", P[t.last].when, P[j1].when,
                (f"Корпус {L:.0f} м: уход от причала {t.berth}"
                 if t.berth else f"Корпус {L:.0f} м ушёл из открытой воды")
                + ("" if left else " (не подтверждено)"),
                f"{_cap(_span(P[t.last].when, P[j1].when))} корпус "
                f"{L_txt} ушёл {_where(rep, t, 'from')}; {since} (виден "
                f"на {stay}). По размеру: {size}."
                + _route_text(rep, L, Wd) + conf,
                sev, L, Wd, t.lon, t.lat, t.berth, t.id, left,
                t.last, j1,
                **_conf("high" if left and len(t.presences) >= 2 else
                        "medium" if left else "low",
                        f"виден на {stay}, затем {spot_imaged} на "
                        f"{count(empties, *PASSES_LOC)}")))

    # ------------------------------------------- partial changes in a stretch
    for t in rep.tracks:
        sz = smoothed_sizes(t)
        if len(sz) < 4 or t.length_m < cfg.large_length_m:
            continue
        k = 2
        while k <= len(sz) - 2:
            before = float(np.median([x[1] for x in sz[k - 2:k]]))
            after = float(np.median([x[1] for x in sz[k:k + 2]]))
            wb = float(np.median([x[2] for x in sz[k - 2:k]]))
            wa = float(np.median([x[2] for x in sz[k:k + 2]]))
            nb = interpret(before, wb)["ships"]
            na = interpret(after, wa)["ships"]
            if nb != na and abs(after - before) >= max(90.0, 0.35 * before):
                i0, i1 = sz[k - 1][0], sz[k][0]
                diff = abs(before - after)
                if na < nb:
                    title = f"Один корабль из группы ушёл от причала {t.berth}"
                    text = (f"{_cap(_span(P[i0].when, P[i1].when))} занятый "
                            f"участок {_where(rep, t)} сократился примерно с "
                            f"{before:.0f} до {after:.0f} м, а остальные "
                            f"корабли остались: ушёл корабль длиной около "
                            f"{diff:.0f} м ({_class_text(diff, True)}).")
                    kind = "part-left"
                else:
                    title = f"К причалу {t.berth} подошёл ещё один корабль"
                    text = (f"{_cap(_span(P[i0].when, P[i1].when))} занятый "
                            f"участок {_where(rep, t)} вырос примерно с "
                            f"{before:.0f} до {after:.0f} м: корабль длиной "
                            f"около {diff:.0f} м встал борт к борту или за "
                            f"кормой ({_class_text(diff, True)}).")
                    kind = "part-arrived"
                ev.append(Event(kind, P[i0].when, P[i1].when, title, text,
                                1 if diff < 180 else 2, diff, None, t.lon,
                                t.lat, t.berth, t.id, True, i0, i1,
                                **_conf("medium", "изменение измеренной "
                                        "длины одного участка на радаре, "
                                        "державшееся по два пролёта с каждой "
                                        "стороны")))
                k += 2
            else:
                k += 1

    # ------------------------------------------------------- group movements
    # More large hulls leaving (or arriving) between two passes than this
    # base normally sees.  The bar comes from the base's own churn: at a
    # base the size of Norfolk three berths changing in six days is routine.
    churn_note = ""
    for kind, word in (("left", "ушли"), ("arrived", "пришли")):
        by_gap: dict[tuple, list[Event]] = {}
        for e in ev:
            if e.kind == kind and (e.length_m or 0) >= cfg.large_length_m:
                by_gap.setdefault((e.t0, e.t1), []).append(e)
        counts = np.zeros(max(n - 1, 1))
        for (t0, _t1), es in by_gap.items():
            k = next((i for i, p in enumerate(P) if p.when == t0), None)
            if k is not None and k < len(counts):
                counts[k] = len(es)
        med = float(np.median(counts))
        mad = float(np.median(np.abs(counts - med))) * 1.4826
        need = max(3, int(math.ceil(med + 3.0 * max(mad, 0.5))),
                   int(math.ceil(0.25 * rep.usual_in_port)))
        if kind == "left":
            churn_note = (f"Между двумя пролётами с этой базы обычно уходит "
                          f"примерно {count(med, *LARGE_HULLS)}; {need} и "
                          f"более считаются групповым уходом.")
        for (t0, t1), es in by_gap.items():
            if (t1 - t0).total_seconds() < cfg.group_min_days * 86400:
                continue
            carriers = [e for e in es if interpret(e.length_m, e.width_m
                                                   or 40.0)["kind"] == "carrier"]
            escorts = [e for e in es if e not in carriers]
            nships = sum(interpret(e.length_m, e.width_m or 40.0)["ships"]
                         for e in es)
            sizes = ", ".join(f"{e.length_m:.0f}" for e in
                              sorted(es, key=lambda e: -(e.length_m or 0)))
            n_high = sum(e.confidence == "high" for e in es)
            if carriers and len(escorts) >= 2:
                verb = "вышла в море" if kind == "left" else "вернулась"
                esc_n = count(len(escorts), "корабль сопровождения",
                              "корабля сопровождения",
                              "кораблей сопровождения")
                ev.append(Event(
                    f"group-{kind}", t0, t1,
                    f"Авианосная группа {verb}: авианосец + {esc_n}",
                    f"{_cap(_span(t0, t1))} корпус размером с авианосец "
                    f"({carriers[0].length_m:.0f} м) и ещё "
                    f"{count(len(escorts), *LARGE_HULLS)} "
                    f"({', '.join(f'{e.length_m:.0f}' for e in escorts)} м) "
                    f"{word} за один и тот же интервал. "
                    + ("Уход авианосца вместе с кораблями сопровождения за "
                       "несколько дней — признак выхода авианосной группы в "
                       "море: развёртывание, крупные учения или уход от "
                       "шторма."
                       if kind == "left" else
                       "Приход авианосца вместе с кораблями сопровождения за "
                       "несколько дней — признак возвращения авианосной "
                       "группы домой."),
                    3, None, None, None, None, 0, -1, True,
                    es[0].before, es[0].after,
                    **_conf("high" if all(e.confidence == "high" for e in es)
                            else "medium",
                            f"{count(len(es), 'отдельное изменение', 'отдельных изменения', 'отдельных изменений')}"
                            f" у причалов за один интервал; подтверждено "
                            f"вторым пролётом: {n_high}")))
                continue
            if len(es) < need:
                continue
            ev.append(Event(
                f"group-{kind}", t0, t1,
                f"Примерно {count(nships, *SHIPS)} {word} вместе",
                f"{_cap(_span(t0, t1))} сразу "
                f"{'опустели' if kind == 'left' else 'заполнились'} "
                f"{count(len(es), 'причал', 'причала', 'причалов')} (длины "
                f"корпусов {sizes} м) — это примерно "
                f"{count(nships, *SHIPS)}. Между двумя пролётами здесь обычно "
                f"меняется примерно {med:.0f}. "
                + ("Если несколько кораблей уходят за несколько дней, обычно "
                   "это группа, выходящая в море вместе: учения, "
                   "развёртывание или уход от шторма."
                   if kind == "left" else
                   "Если несколько кораблей возвращаются за несколько дней, "
                   "обычно это группа, вернувшаяся с учений или из "
                   "развёртывания."),
                3 if len(es) >= need + 2 else 2, None, None, None, None, 0,
                -1, True, es[0].before, es[0].after,
                **_conf("high" if n_high >= need else "medium",
                        f"{count(len(es), 'изменение', 'изменения', 'изменений')}"
                        f" у причалов за один интервал при обычных "
                        f"{med:.0f}; подтверждено вторым пролётом: "
                        f"{n_high}")))

    # ------------------------------------------------- unusually long stays
    by_berth: dict[int, list[int]] = {}
    for t in rep.tracks:
        if t.berth and t.closed and len(t.presences) >= 2:
            by_berth.setdefault(t.berth, []).append(len(t.presences))
    for t in rep.tracks:
        if not t.berth or t.closed or t.length_m < cfg.large_length_m:
            continue
        hist = by_berth.get(t.berth, [])
        if len(hist) < 3:
            continue
        typical = float(np.median(hist))
        if len(t.presences) >= max(2 * typical, typical + 4):
            days = (P[t.last].when - P[t.first].when).days
            ev.append(Event(
                "long-stay", P[t.first].when, P[t.last].when,
                f"Корпус {t.length_m:.0f} м у причала {t.berth} уже "
                f"{count(days, *DAYS)}",
                f"Корпус длиной около {t.length_m:.0f} м стоит "
                f"{_where(rep, t)} с {_fmt(P[t.first].when)}: "
                f"{count(len(t.presences), *PASSES)} при обычной стоянке в "
                f"{count(typical, *PASSES)} для кораблей у этого причала. "
                f"Долгая стоянка обычно означает обслуживание или ремонт, а "
                f"не готовность к выходу. По размеру: "
                f"{interpret(t.length_m, t.width_m)['text']}.",
                1 if t.length_m < 180 else 2, t.length_m, t.width_m, t.lon,
                t.lat, t.berth, t.id, True, t.first, t.last,
                **_conf("medium", "сравнение с "
                        f"{count(len(hist), 'прежней стоянкой', 'прежними стоянками', 'прежними стоянками')}"
                        " у того же причала")))

    # ------------------------------------------------ level periods / shifts
    rep.regimes = _regimes(rep)
    # The spread that defines "normal" is measured within level periods, so
    # a real two-month drop does not widen the normal band until it hides
    # itself.
    resid = []
    for r in rep.regimes:
        for i in range(r["start"], r["end"] + 1):
            if P[i].good:
                resid.append(P[i].trend - r["level"])
    if len(resid) >= 4:
        rep.spread_in_port = max(float(np.median(np.abs(resid))) * 1.4826,
                                 0.5)
    lo, hi = normal_range(rep)
    usual = rep.usual_in_port
    R = rep.regimes
    for k, r in enumerate(R):
        a, b = P[r["start"]], P[r["end"]]
        if k > 0:
            prev = R[k - 1]
            d = r["level"] - prev["level"]
            pct = d / max(prev["level"], 1.0)
            direction = "упал" if d < 0 else "вырос"
            number = "снизилось" if d < 0 else "выросло"
            held = ("и с тех пор держится" if r["end"] == n - 1 else
                    f"и держалось до {_fmt(b.when)}")
            ev.append(Event(
                "shift-down" if d < 0 else "shift-up",
                P[prev["end"]].when, a.when,
                f"Уровень {direction}: {prev['level']:.0f} → "
                f"{count(r['level'], *SHIPS)}",
                f"{_cap(_span(P[prev['end']].when, a.when))} число крупных "
                f"кораблей в порту {number} примерно с {prev['level']:.0f} "
                f"до {r['level']:.0f} ({pct:+.0%}) {held} "
                f"({count(r['n'], *PASSES)}). "
                + ("Устойчивое снижение означает, что корабли ушли на "
                   "развёртывание или учения либо рассредоточились, а не "
                   "просто один корабль вышел в море."
                   if d < 0 else
                   "Устойчивый рост означает, что корабли вернулись или "
                   "собрались: конец развёртывания, период обслуживания или "
                   "сбор перед совместным выходом."),
                3 if abs(pct) >= 0.4 else 2, None, None, None, None, 0, -1,
                True, prev["end"], r["start"],
                **_conf("high" if r["n"] >= 4 else "medium",
                        f"точка смены уровня в ряду, державшаяся "
                        f"{count(r['n'], *PASSES)}; медианы "
                        f"{prev['level']:.0f} → {r['level']:.0f}")))
        if r["level"] <= lo or r["level"] >= hi:
            kind = "emptier" if r["level"] <= lo else "fuller"
            span = (f"{_fmt(a.when)}" if r["start"] == r["end"]
                    else f"с {_fmt(a.when)} по {_fmt(b.when)}")
            ev.append(Event(
                kind, a.when, b.when,
                f"Необычно {'мало' if kind == 'emptier' else 'много'} "
                f"кораблей: ~{r['level']:.0f} (обычно {usual:.0f})",
                f"{_cap(span)} на базе было примерно "
                f"{count(r['level'], *LARGE_SHIPS)} при обычных "
                f"{usual:.0f} (нормальный диапазон {lo:.0f}–{hi:.0f}). "
                + ("Больше кораблей, чем обычно, одновременно отсутствовали: "
                   "развёртывание, крупные учения или уход от шторма."
                   if kind == "emptier" else
                   "Больше кораблей, чем обычно, одновременно были дома: "
                   "период обслуживания, праздничный перерыв или сбор перед "
                   "совместным выходом."),
                3 if r["n"] >= 4 else 2, None, None, None, None, 0, -1, True,
                r["start"], r["end"],
                **_conf("high" if r["n"] >= 4 else "medium",
                        f"{count(r['n'], 'хороший пролёт', 'хороших пролёта', 'хороших пролётов')}"
                        f" подряд на этом уровне")))

    rep.notes.append(churn_note) if churn_note else None
    ev.sort(key=lambda e: (e.t1, -e.severity))
    return ev


def berth_matrix(rep: BaseReport) -> tuple[list[int], np.ndarray]:
    """Berth ids and an (berths x passes) array of occupied fraction."""
    ids = [b["id"] for b in rep.berths]
    M = np.full((len(ids), len(rep.passes)), np.nan)
    for j, p in enumerate(rep.passes):
        for i, b in enumerate(ids):
            M[i, j] = p.berth_occ.get(b, np.nan)
    return ids, M


def _notes(rep: BaseReport, geo: np.ndarray) -> list[str]:
    P = rep.passes
    notes = []
    if not P:
        return notes
    days = (P[-1].when - P[0].when).total_seconds() / 86400
    gaps = np.diff([p.when.timestamp() / 86400 for p in P]) if len(P) > 1 \
        else np.array([0.0])
    notes.append(f"{count(len(P), 'радарный пролёт', 'радарных пролёта', 'радарных пролётов')}"
                 f" с {fmt_date(P[0].when, year=True)} по "
                 f"{fmt_date(P[-1].when, year=True)}: обычно один раз в "
                 f"{count(np.median(gaps), *DAYS)}, самый длинный перерыв — "
                 f"{count(gaps.max(), *DAYS)}.")
    g = sorted(set(geo))
    notes.append(f"Углы съёмки: {len(g)} ({', '.join(g)}); перед сравнением "
                 f"у каждого угла убирается его сдвиг яркости.")
    rough = sum(p.rough for p in P)
    poor = sum(p.poor for p in P)
    partial = sum(p.berth_coverage < 0.6 for p in P)
    if poor:
        notes.append(f"Пролётов, где найдено намного меньше корпусов, чем на "
                     f"соседних: {poor}. Они считаются плохими снимками, а не "
                     f"пустыми причалами (рисуются полыми; не используются "
                     f"для событий и обычного уровня).")
    if rough:
        notes.append(f"Пролётов с бурным морем: {rough}. Они рисуются полыми "
                     f"и не учитываются в «обычном» уровне.")
    if partial:
        notes.append(f"Пролётов, снявших только часть причалов: {partial}. "
                     f"Они тоже рисуются полыми.")
    notes.append(f"Найдено причалов: {len(rep.berths)} (места у пирса или "
                 f"стенки, которые то пустуют, то заняты корпусом).")
    if rep.no_water_mask:
        notes.append("Карта земного покрова недоступна, поэтому контейнеры в "
                     "штабелях и другой металл на причале могут считаться "
                     "кораблями. Вне проверенных мест числа стоит считать "
                     "верхней оценкой.")
    if rep.static_hulls:
        notes.append(f"Объектов в форме корабля, ярких на каждом пролёте: "
                     f"{len(rep.static_hulls)}. Они считаются "
                     f"присутствовавшими всё время (скорее всего, это "
                     f"корабли на долгом обслуживании; кран или ангар могут "
                     f"выглядеть так же).")
    if days < 120:
        notes.append("История короче четырёх месяцев: «обычный» уровень и "
                     "эталон пустого причала ещё уточняются.")
    if rep.usual_in_port < 1 or sum(t.length_m >= 100 for t in rep.tracks) < 2:
        notes.append("Приходов и уходов почти не замечено. Либо причалы "
                     "были в основном пусты, либо корабли простояли весь "
                     "период — для этого метода корабль, который не уходит, "
                     "выглядит частью пирса. Может помочь более длинный "
                     "период.")
    return notes


# ==========================================================================
# Comparing bases
# ==========================================================================
def compare(reports: Sequence[BaseReport], window_days: float = 7.0
            ) -> list[str]:
    """Plain-language links between bases: things that happened together."""
    out = []
    flagged: list[tuple[datetime, str, str]] = []
    for r in reports:
        for e in r.events:
            if e.kind in ("emptier", "group-left", "shift-down") or (
                    e.kind == "left" and e.severity >= 3):
                flagged.append((e.t1, r.name, e.title))
    flagged.sort()
    used = set()
    for i, (t, name, title) in enumerate(flagged):
        if i in used:
            continue
        group = [(t, name, title)]
        for j in range(i + 1, len(flagged)):
            t2, n2, ti2 = flagged[j]
            if (t2 - t).total_seconds() / 86400 > window_days:
                break
            if n2 != name and all(n2 != g[1] for g in group):
                group.append(flagged[j])
                used.add(j)
        if len(group) >= 2:
            out.append(f"Неделя с {fmt_date(t, year=True)}: " + "; ".join(
                f"{g[1]} — {g[2]}" for g in group)
                + ". Если несколько баз теряют корабли в одну неделю, это "
                  "указывает на согласованное перемещение, а не на обычные "
                  "выходы в море.")
    for r in reports:
        if r.passes:
            last = r.passes[-1]
            u = r.usual_in_port
            if u > 0:
                out.append(f"{r.name}: на {fmt_date(last.when)} примерно "
                           f"{count(last.in_port, *LARGE_SHIPS)} "
                           f"({last.in_port / u:.0%} от обычных {u:.0f}).")
    return out


# ==========================================================================
def land_mask(shape: tuple[int, int], transform: Sequence[float], crs: str,
              min_elev_m: float = 1.5) -> tuple[np.ndarray, float]:
    """Pixels clearly above sea level, on the radar grid.

    Uses NOAA's keyless global DEM mosaic: metre-scale coastal models in the
    US, ~30-90 m land elevation elsewhere.  Returns ``(mask, cell_m)`` so the
    caller can be more forgiving where the grid is coarse.  Piers are open
    structures and are not in elevation models, so they stay "water" -- which
    is what keeps a ship at a pier from being rejected.
    """
    from .bathymetry import fetch_noaa_dem
    H, W = shape
    a, b, c0, d, e, f0 = transform[:6]
    cc, rr = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    lon, lat = _lonlat_fn(crs)(c0 + a * cc + b * rr, f0 + d * cc + e * rr)
    lon, lat = np.asarray(lon), np.asarray(lat)
    pad = 0.002
    box = (float(lon.min()) - pad, float(lat.min()) - pad,
           float(lon.max()) + pad, float(lat.max()) + pad)
    bt = fetch_noaa_dem(box, target_px=max(H, W) * 2)
    dh, dw = bt.depth_m.shape
    w, s, ee, n = bt.bbox
    ci = np.clip(((lon - w) / (ee - w) * dw).astype(int), 0, dw - 1)
    ri = np.clip(((n - lat) / (n - s) * dh).astype(int), 0, dh - 1)
    elev = -np.asarray(bt.depth_m, float)[ri, ci]
    mask = np.nan_to_num(elev, nan=-1.0) > min_elev_m
    # The resampled grid is as fine as requested; the data under it is not.
    # Estimate the true cell from how blocky the coastline is.
    cell = _native_cell_m(elev, abs(a))
    return mask, cell


def _native_cell_m(elev: np.ndarray, px_m: float) -> float:
    """Rough native resolution: length of runs of identical elevation."""
    row = elev[elev.shape[0] // 2]
    if row.size < 4:
        return 30.0
    same = np.isclose(row[1:], row[:-1], atol=1e-3)
    runs, cur = [], 1
    for s_ in same:
        if s_:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    return float(np.median(runs) * px_m)
