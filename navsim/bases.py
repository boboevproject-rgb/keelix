"""Naval bases the monitor knows, with the box that covers their berths.

Each box was checked twice: on satellite imagery (the piers are inside) and
on Sentinel-1 itself (berths light up as places that are sometimes empty and
sometimes hold a hull).  Boxes are kept to the berthing area -- a few km --
because everything outside it is either land or open water and only adds
false alarms and download time.

``sea`` is a point in open water outside the harbour, used for the depth
check: how far, and how deep, the way out to sea is.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Base:
    key: str
    name: str
    country: str
    region: str
    bbox: tuple[float, float, float, float]          # west, south, east, north
    sea: tuple[float, float] | None = None
    note: str = ""

    @property
    def centre(self) -> tuple[float, float]:
        w, s, e, n = self.bbox
        return (0.5 * (w + e), 0.5 * (s + n))

    @property
    def label(self) -> str:
        return f"{self.name} ({self.country})"


REGIONS = ("North America", "Europe", "Russia", "Middle East & Africa",
           "Asia-Pacific")

BASES: tuple[Base, ...] = (
    # ---------------------------------------------------------- North America
    Base("norfolk", "Naval Station Norfolk", "USA", "North America",
         (-76.345, 36.930, -76.318, 36.967), (-75.85, 36.92),
         "Крупнейшая в мире военно-морская база: авианосцы, десантные "
         "корабли, крейсеры и эсминцы Атлантического флота США."),
    Base("san_diego", "Naval Base San Diego", "USA", "North America",
         (-117.140, 32.662, -117.105, 32.695), (-117.30, 32.62),
         "Главная база надводных сил Тихоокеанского флота США: десантные "
         "корабли, крейсеры, эсминцы, корабли прибрежной зоны."),
    Base("north_island", "NAS North Island", "USA", "North America",
         (-117.195, 32.698, -117.170, 32.716), (-117.30, 32.62),
         "Авианосные причалы Тихоокеанского флота США в бухте Сан-Диего."),
    Base("mayport", "Naval Station Mayport", "USA", "North America",
         (-81.420, 30.385, -81.395, 30.402), (-81.30, 30.40),
         "Эсминцы и корабли прибрежной зоны; разворотный бассейн у реки "
         "Сент-Джонс."),
    Base("pearl_harbor", "Pearl Harbor", "USA", "North America",
         (-157.975, 21.340, -157.935, 21.372), (-157.97, 21.29),
         "Тихоокеанский флот США: эсминцы, подводные лодки, заходящие "
         "корабли."),
    Base("bremerton", "Puget Sound Naval Shipyard", "USA", "North America",
         (-122.650, 47.550, -122.615, 47.567), (-122.45, 47.65),
         "Порт приписки авианосцев и верфь; корабли стоят на долгих "
         "ремонтах."),
    Base("everett", "Naval Station Everett", "USA", "North America",
         (-122.235, 47.978, -122.212, 47.997), (-122.30, 48.00),
         "Эсминцы Тихоокеанского флота США."),
    Base("kings_bay", "Naval Submarine Base Kings Bay", "USA",
         "North America", (-81.530, 30.780, -81.500, 30.808), (-81.38, 30.70),
         "Подводные лодки с баллистическими ракетами: сидят низко в воде, "
         "поэтому слабо видны на радаре."),
    Base("halifax", "CFB Halifax", "Canada", "North America",
         (-63.595, 44.655, -63.570, 44.675), (-63.50, 44.55),
         "Атлантический флот ВМС Канады."),
    Base("esquimalt", "CFB Esquimalt", "Canada", "North America",
         (-123.440, 48.425, -123.415, 48.440), (-123.45, 48.38),
         "Тихоокеанский флот ВМС Канады."),
    # ----------------------------------------------------------------- Europe
    Base("portsmouth_uk", "HMNB Portsmouth", "UK", "Europe",
         (-1.118, 50.793, -1.095, 50.815), (-1.10, 50.75),
         "База авианосцев и эсминцев Type 45 Королевского флота."),
    Base("devonport", "HMNB Devonport", "UK", "Europe",
         (-4.200, 50.370, -4.170, 50.395), (-4.15, 50.30),
         "Фрегаты, десантные корабли и верфь."),
    Base("faslane", "HMNB Clyde (Faslane)", "UK", "Europe",
         (-4.835, 56.055, -4.810, 56.075), (-4.90, 55.95),
         "Подводные лодки Королевского флота: сидят низко в воде, поэтому "
         "слабо видны на радаре."),
    Base("toulon", "Toulon naval base", "France", "Europe",
         (5.900, 43.100, 5.940, 43.125), (5.95, 43.05),
         "Главная база ВМС Франции: авианосец «Шарль де Голль», десантные "
         "корабли, фрегаты."),
    Base("brest", "Brest naval base", "France", "Europe",
         (-4.510, 48.372, -4.475, 48.390), (-4.65, 48.30),
         "Атлантический флот Франции: фрегаты и корабли обеспечения."),
    Base("rota", "Naval Station Rota", "Spain", "Europe",
         (-6.345, 36.612, -6.320, 36.630), (-6.40, 36.58),
         "Эсминцы США передового базирования в Европе; ВМС Испании."),
    Base("wilhelmshaven", "Wilhelmshaven", "Germany", "Europe",
         (8.120, 53.505, 8.170, 53.530), (8.10, 53.70),
         "Фрегаты и суда снабжения ВМС Германии."),
    Base("kiel", "Kiel naval base", "Germany", "Europe",
         (10.140, 54.325, 10.170, 54.345), (10.25, 54.45),
         "Корветы, подводные лодки и тральщики ВМС Германии."),
    Base("den_helder", "Den Helder", "Netherlands", "Europe",
         (4.770, 52.950, 4.800, 52.970), (4.70, 52.97),
         "Главная база ВМС Нидерландов."),
    Base("la_spezia", "La Spezia", "Italy", "Europe",
         (9.830, 44.095, 9.860, 44.110), (9.85, 44.03),
         "Фрегаты и эсминцы ВМС Италии."),
    Base("taranto", "Taranto", "Italy", "Europe",
         (17.200, 40.460, 17.240, 40.485), (17.15, 40.40),
         "Главная база ВМС Италии, включая её авианосцы."),
    Base("souda", "Souda Bay", "Greece", "Europe",
         (24.140, 35.480, 24.165, 35.495), (24.25, 35.52),
         "Пирс НАТО на Крите для заходящих кораблей США и союзников."),
    Base("gibraltar", "HM Naval Base Gibraltar", "UK", "Europe",
         (-5.365, 36.135, -5.345, 36.150), (-5.40, 36.10),
         "Заходящие корабли Королевского флота, США и союзников."),
    Base("golcuk", "Golcuk naval base", "Turkey", "Europe",
         (29.810, 40.715, 29.850, 40.735), (29.30, 40.70),
         "Главная база ВМС Турции в заливе Измит."),
    # ----------------------------------------------------------------- Russia
    Base("sevastopol", "Sevastopol", "Russia (Crimea)", "Russia",
         (33.495, 44.600, 33.570, 44.630), (33.40, 44.62),
         "Черноморский флот: фрегаты, десантные корабли, подводные лодки."),
    Base("novorossiysk", "Novorossiysk", "Russia", "Russia",
         (37.770, 44.695, 37.815, 44.728), (37.85, 44.62),
         "Корабли Черноморского флота, перебазированные из Крыма."),
    Base("baltiysk", "Baltiysk", "Russia", "Russia",
         (19.890, 54.625, 19.930, 54.650), (19.85, 54.65),
         "Главная база Балтийского флота."),
    Base("severomorsk", "Severomorsk", "Russia", "Russia",
         (33.390, 69.055, 33.460, 69.085), (33.70, 69.35),
         "Штаб Северного флота: крейсеры и эсминцы."),
    Base("vladivostok", "Vladivostok", "Russia", "Russia",
         (131.870, 43.095, 131.930, 43.120), (131.95, 43.00),
         "Тихоокеанский флот: крейсеры, эсминцы, десантные корабли."),
    # --------------------------------------------------- Middle East & Africa
    Base("tartus", "Tartus", "Syria", "Middle East & Africa",
         (35.855, 34.893, 35.880, 34.915), (35.80, 34.90),
         "Военно-морской пункт в северной части порта."),
    Base("haifa", "Haifa", "Israel", "Middle East & Africa",
         (34.990, 32.810, 35.020, 32.830), (34.95, 32.85),
         "Главная база ВМС Израиля."),
    Base("bahrain", "NSA Bahrain / Mina Salman", "Bahrain",
         "Middle East & Africa", (50.600, 26.195, 50.630, 26.215),
         (50.70, 26.25), "Пятый флот США."),
    Base("bandar_abbas", "Bandar Abbas", "Iran", "Middle East & Africa",
         (56.180, 27.130, 56.230, 27.160), (56.30, 27.05),
         "Главная база ВМС Ирана у Ормузского пролива."),
    Base("djibouti_pla", "PLA Support Base Djibouti", "China (Djibouti)",
         "Middle East & Africa", (43.050, 11.585, 43.080, 11.600),
         (43.10, 11.65), "Зарубежная военно-морская база Китая; пирс "
                         "принимает крупные корабли."),
    # ----------------------------------------------------------- Asia-Pacific
    Base("yokosuka", "Yokosuka", "Japan", "Asia-Pacific",
         (139.645, 35.278, 139.680, 35.300), (139.75, 35.25),
         "Авианосная база Седьмого флота США и Морских сил самообороны "
         "Японии."),
    Base("sasebo", "Sasebo", "Japan", "Asia-Pacific",
         (129.700, 33.150, 129.735, 33.172), (129.60, 33.10),
         "Десантные корабли США и Морские силы самообороны Японии."),
    Base("kure", "Kure", "Japan", "Asia-Pacific",
         (132.540, 34.228, 132.570, 34.250), (132.45, 34.18),
         "Эсминцы и подводные лодки Морских сил самообороны Японии, верфь."),
    Base("guam", "Naval Base Guam (Apra Harbor)", "USA", "Asia-Pacific",
         (144.640, 13.425, 144.675, 13.455), (144.60, 13.46),
         "Подводные лодки США и заходящие корабли."),
    Base("sanya", "Yulin naval base (Sanya)", "China", "Asia-Pacific",
         (109.535, 18.205, 109.575, 18.235), (109.55, 18.15),
         "База Южного флота ВМС НОАК, включая авианосные пирсы."),
    Base("busan", "Busan", "South Korea", "Asia-Pacific",
         (129.070, 35.085, 129.100, 35.105), (129.10, 35.03),
         "База флота ВМС Республики Корея."),
    Base("sydney", "Fleet Base East (Sydney)", "Australia", "Asia-Pacific",
         (151.218, -33.872, 151.235, -33.858), (151.30, -33.83),
         "Главная база флота ВМС Австралии на острове Гарден-Айленд."),
    Base("stirling", "Fleet Base West (HMAS Stirling)", "Australia",
         "Asia-Pacific", (115.670, -32.250, 115.695, -32.225),
         (115.65, -32.20), "Подводные лодки и фрегаты ВМС Австралии."),
    Base("mumbai", "Mumbai naval dockyard", "India", "Asia-Pacific",
         (72.830, 18.920, 72.850, 18.940), (72.75, 18.90),
         "Западное военно-морское командование ВМС Индии."),
    Base("visakhapatnam", "Visakhapatnam", "India", "Asia-Pacific",
         (83.270, 17.680, 83.300, 17.700), (83.35, 17.65),
         "Восточное военно-морское командование ВМС Индии."),
    Base("karachi", "Karachi naval dockyard", "Pakistan", "Asia-Pacific",
         (66.960, 24.820, 66.990, 24.845), (66.90, 24.75),
         "Главная база ВМС Пакистана."),
)

BY_KEY: dict[str, Base] = {b.key: b for b in BASES}


def get(key: str) -> Base:
    try:
        return BY_KEY[key]
    except KeyError:
        raise KeyError(f"unknown base {key!r}; known: "
                       f"{', '.join(sorted(BY_KEY))}") from None


def by_region() -> dict[str, list[Base]]:
    out: dict[str, list[Base]] = {r: [] for r in REGIONS}
    for b in BASES:
        out.setdefault(b.region, []).append(b)
    return out
