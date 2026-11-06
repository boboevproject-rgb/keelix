"""Command line.  ``python -m navsim.cli --help``.

    gui                       открыть приложение (как ``python -m navsim``)
    bases                     список готовых баз
    update  KEY [KEY ...]     скачать новые пролёты радара и проанализировать
    report  KEY               отчёт об активности по данным на диске
    compare KEY KEY [...]     что происходило на нескольких базах вместе
    setup                     сохранить бесплатный ключ Copernicus
"""

from __future__ import annotations

import argparse
import getpass
import sys


def _gui(args) -> int:
    from .app import main
    main()
    return 0


def _bases(args) -> int:
    from . import store
    from .bases import BASES, REGIONS
    for region in REGIONS:
        print(f"\n{region}")
        for b in BASES:
            if b.region != region:
                continue
            n = len(store.known_passes(b))
            print(f"  {b.key:16s} {b.name:38s} {b.country:18s} "
                  f"пролётов на диске: {n:3d}")
    return 0


def _print_progress(frac: float, msg: str) -> None:
    sys.stdout.write(f"\r[{frac:4.0%}] {msg[:90]:90s}")
    sys.stdout.flush()


def _update(args) -> int:
    from . import store
    from .app import report_text
    from .bases import get
    for key in args.keys:
        b = get(key)
        s = store.update(b, args.months, _print_progress)
        print(f"\n{b.name}: пролётов {s['found']} (в кэше {s['cached']}, "
              f"скачано {s['downloaded']}, пустых {s['failed']})")
        rep = store.analyse_base(b, args.months, _print_progress)
        print("\n" + (report_text(rep) if args.verbose else rep.headline))
    return 0


def _report(args) -> int:
    from . import store
    from .app import report_text
    from .bases import get
    rep = store.analyse_base(get(args.key), args.months)
    print(report_text(rep))
    return 0


def _compare(args) -> int:
    from . import store
    from .bases import get
    from .monitor import compare
    reps = [store.analyse_base(get(k), args.months) for k in args.keys]
    for r in reps:
        print(f"{r.name}: {r.headline}")
    print()
    for line in compare(reps):
        print("-", line)
    return 0


def _setup(args) -> int:
    from . import credentials, radar
    print("Copernicus Data Space (бесплатно): зарегистрируйтесь на "
          "dataspace.copernicus.eu, затем Sentinel Hub > User settings > "
          "OAuth clients > create.")
    cid = input("Client id: ").strip()
    sec = getpass.getpass("Client secret (не отображается): ").strip()
    if cid or sec:
        credentials.set_credentials("copernicus", client_id=cid,
                                    client_secret=sec)
    try:
        radar.get_token(force=True)
        print("Подключено: ключ работает.")
    except Exception as exc:
        print(f"Сохранено, но проверка не прошла: {exc}")
        return 1
    return 0


def _ru_help(p: argparse.ArgumentParser) -> None:
    """Russian headings for argparse's own English boilerplate."""
    p._positionals.title = "позиционные аргументы"
    p._optionals.title = "параметры"
    for a in p._actions:
        if isinstance(a, argparse._HelpAction):
            a.help = "показать эту справку и выйти"


class _RuParser(argparse.ArgumentParser):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        _ru_help(self)

    def format_usage(self):
        return super().format_usage().replace("usage:", "использование:", 1)

    def format_help(self):
        return super().format_help().replace("usage:", "использование:", 1)


def main(argv=None) -> int:
    # Output is Russian text with arrows.  A redirect to a file (or an old
    # console) uses the locale's code page (cp1252/cp866), which cannot encode
    # all of it: write UTF-8 and replace what still cannot be shown instead
    # of crashing.  Done before parsing so --help is covered too.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass
    p = _RuParser(prog="navsim", description=(
        "Keelix: активность военно-морских баз по бесплатному радару "
        "Sentinel-1."))
    sub = p.add_subparsers(dest="cmd", title="команды", metavar="КОМАНДА",
                           parser_class=_RuParser)
    sub.add_parser("gui", help="открыть приложение").set_defaults(fn=_gui)
    sub.add_parser("bases", help="список готовых баз").set_defaults(fn=_bases)
    months_help = "за сколько месяцев назад (по умолчанию 9)"
    u = sub.add_parser("update", help="скачать новые пролёты и проанализировать",
                       description="Скачать новые пролёты радара и "
                                   "проанализировать базы.")
    u.add_argument("keys", nargs="+", metavar="KEY", help="ключи баз (см. bases)")
    u.add_argument("--months", type=float, default=9.0, help=months_help)
    u.add_argument("-v", "--verbose", action="store_true",
                   help="полный отчёт вместо краткой сводки")
    u.set_defaults(fn=_update)
    r = sub.add_parser("report", help="вывести отчёт об активности базы",
                       description="Отчёт об активности базы по данным на "
                                   "диске (без скачивания).")
    r.add_argument("key", metavar="KEY", help="ключ базы (см. bases)")
    r.add_argument("--months", type=float, default=9.0, help=months_help)
    r.set_defaults(fn=_report)
    c = sub.add_parser("compare", help="сравнить несколько баз",
                       description="Что происходило на нескольких базах "
                                   "вместе.")
    c.add_argument("keys", nargs="+", metavar="KEY", help="ключи баз (см. bases)")
    c.add_argument("--months", type=float, default=9.0, help=months_help)
    c.set_defaults(fn=_compare)
    sub.add_parser("setup", help="сохранить ключ Copernicus").set_defaults(
        fn=_setup)
    args = p.parse_args(argv)
    if not getattr(args, "fn", None):
        return _gui(args)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
