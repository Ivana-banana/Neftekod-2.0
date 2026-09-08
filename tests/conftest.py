import pytest

from neftekod import parse_lims_pak


@pytest.fixture(scope="session")
def lims_long():
    """Реальный lims_export_2023.xlsx целиком (разбор ~0.5 с)."""
    return parse_lims_pak.parse_lims()


@pytest.fixture(scope="session")
def pak_long():
    """Реальный pak_export_2023.xlsx целиком (разбор ~4 с)."""
    return parse_lims_pak.parse_pak()
