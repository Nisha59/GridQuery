"""Reference data for the energy_sources table."""

from dataclasses import dataclass

from gridquery.ingestion.series import SERIES_BY_KEY


@dataclass(frozen=True, kw_only=True)
class EnergySource:
    """Description of one generation source, as stored in energy_sources."""

    source: str
    name_en: str
    name_de: str
    is_renewable: bool
    is_fossil: bool
    description: str

    @property
    def smard_filter_id(self) -> int:
        """SMARD filter ID of the matching generation series."""
        return SERIES_BY_KEY[self.source].filter_id


# fmt: off
ENERGY_SOURCES: tuple[EnergySource, ...] = (
    EnergySource(source="lignite", name_en="Lignite", name_de="Braunkohle",
                 is_renewable=False, is_fossil=True,
                 description="Brown coal, mined in Germany. The most CO2-intensive source."),
    EnergySource(source="hard_coal", name_en="Hard coal", name_de="Steinkohle",
                 is_renewable=False, is_fossil=True,
                 description="Black coal, mostly imported."),
    EnergySource(source="natural_gas", name_en="Natural gas", name_de="Erdgas",
                 is_renewable=False, is_fossil=True,
                 description="Gas-fired power plants, often used flexibly at peak times."),
    EnergySource(source="nuclear", name_en="Nuclear", name_de="Kernenergie",
                 is_renewable=False, is_fossil=False,
                 description="Nuclear power. Germany's last three reactors shut down on "
                             "15 April 2023; SMARD reports zero or no data afterwards."),
    EnergySource(source="wind_onshore", name_en="Wind onshore", name_de="Wind Onshore",
                 is_renewable=True, is_fossil=False,
                 description="Wind turbines on land."),
    EnergySource(source="wind_offshore", name_en="Wind offshore", name_de="Wind Offshore",
                 is_renewable=True, is_fossil=False,
                 description="Wind turbines at sea (North Sea and Baltic Sea)."),
    EnergySource(source="solar", name_en="Solar (photovoltaics)", name_de="Photovoltaik",
                 is_renewable=True, is_fossil=False,
                 description="Solar panels. Near zero at night."),
    EnergySource(source="biomass", name_en="Biomass", name_de="Biomasse",
                 is_renewable=True, is_fossil=False,
                 description="Biogas, wood and other biomass."),
    EnergySource(source="hydro", name_en="Hydropower", name_de="Wasserkraft",
                 is_renewable=True, is_fossil=False,
                 description="Run-of-river and storage hydropower, excluding pumped storage."),
    EnergySource(source="other_renewables", name_en="Other renewables",
                 name_de="Sonstige Erneuerbare", is_renewable=True, is_fossil=False,
                 description="Small renewable sources such as geothermal energy, landfill gas "
                             "and sewage gas."),
    EnergySource(source="other_conventional", name_en="Other conventional",
                 name_de="Sonstige Konventionelle", is_renewable=False, is_fossil=True,
                 description="Other non-renewable sources such as oil and non-biogenic waste."),
)
# fmt: on
