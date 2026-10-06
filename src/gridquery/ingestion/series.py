"""Catalog of the SMARD data series GridQuery collects.

Filter IDs come from the SMARD API specification
(https://github.com/bundesAPI/smard-api/blob/main/openapi.yaml) and were checked
against the live endpoints. All series are hourly values for Germany.
"""

from dataclasses import dataclass
from enum import StrEnum


class Category(StrEnum):
    """Kind of quantity a series measures."""

    GENERATION = "generation"
    CONSUMPTION = "consumption"
    PRICE = "price"


@dataclass(frozen=True)
class SeriesSpec:
    """Static description of one SMARD series.

    Attributes:
        key: Stable identifier used in file paths, the CLI, and the database.
        filter_id: SMARD filter ID.
        region: SMARD region code (country or bidding zone).
        category: Kind of quantity.
        unit: Unit of the values as delivered by SMARD.
        label_de: Original German SMARD label, for traceability.
    """

    key: str
    filter_id: int
    region: str
    category: Category
    unit: str
    label_de: str


def _generation(key: str, filter_id: int, label_de: str) -> SeriesSpec:
    """Build a generation series for Germany (values in MWh per hour)."""
    return SeriesSpec(key, filter_id, "DE", Category.GENERATION, "MWh", label_de)


SERIES: tuple[SeriesSpec, ...] = (
    _generation("lignite", 1223, "Stromerzeugung: Braunkohle"),
    _generation("hard_coal", 4069, "Stromerzeugung: Steinkohle"),
    _generation("natural_gas", 4071, "Stromerzeugung: Erdgas"),
    _generation("nuclear", 1224, "Stromerzeugung: Kernenergie"),
    _generation("wind_onshore", 4067, "Stromerzeugung: Wind Onshore"),
    _generation("wind_offshore", 1225, "Stromerzeugung: Wind Offshore"),
    _generation("solar", 4068, "Stromerzeugung: Photovoltaik"),
    _generation("biomass", 4066, "Stromerzeugung: Biomasse"),
    _generation("hydro", 1226, "Stromerzeugung: Wasserkraft"),
    _generation("other_renewables", 1228, "Stromerzeugung: Sonstige Erneuerbare"),
    _generation("other_conventional", 1227, "Stromerzeugung: Sonstige Konventionelle"),
    SeriesSpec(
        "consumption", 410, "DE", Category.CONSUMPTION, "MWh", "Stromverbrauch: Gesamt (Netzlast)"
    ),
    # Day-ahead prices are published per bidding zone; Germany shares one with Luxembourg.
    SeriesSpec(
        "day_ahead_price",
        4169,
        "DE-LU",
        Category.PRICE,
        "EUR/MWh",
        "Marktpreis: Deutschland/Luxemburg",
    ),
)

SERIES_BY_KEY: dict[str, SeriesSpec] = {spec.key: spec for spec in SERIES}


def get_series(keys: list[str] | None = None) -> list[SeriesSpec]:
    """Return series specs by key, or all series when no keys are given.

    Args:
        keys: Series keys to look up.

    Raises:
        ValueError: If a key is unknown.
    """
    if not keys:
        return list(SERIES)
    unknown = [key for key in keys if key not in SERIES_BY_KEY]
    if unknown:
        raise ValueError(f"Unknown series: {', '.join(unknown)}. Known: {', '.join(SERIES_BY_KEY)}")
    return [SERIES_BY_KEY[key] for key in keys]
