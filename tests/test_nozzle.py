import numpy as np
import pytest

from aerotvc.nozzle import (
    Nozzle, area_ratio, mach_from_area_ratio, pressure_ratio, temperature_ratio,
)


def test_textbook_values_and_branches():
    assert mach_from_area_ratio(2) == pytest.approx(2.197, abs=5e-4)
    assert mach_from_area_ratio(2, branch="subsonic") == pytest.approx(
        0.3059, abs=5e-5)
    assert pressure_ratio(2) == pytest.approx(0.1278, abs=5e-5)
    assert area_ratio(3) == pytest.approx(4.2346, abs=5e-5)
    for branch in ("subsonic", "supersonic"):
        assert mach_from_area_ratio(1, branch=branch) == 1
        for ratio in (1.001, 2, 10, 100):
            assert area_ratio(mach_from_area_ratio(ratio, branch=branch)) == (
                pytest.approx(ratio, rel=2e-11))


def test_mass_thrust_isp_and_enthalpy():
    nozzle = Nozzle(0.01, 0.02, g=1.4, R=287)
    result = nozzle.solve(2e6, 1000, 101325)
    assert result.mdot == pytest.approx(
        0.0404 * 2e6 * 0.01 / np.sqrt(1000), rel=5e-4)
    assert result.thrust == pytest.approx(
        result.mdot * result.ve + (result.pe - 101325) * 0.02)
    assert result.thrust == pytest.approx(result.cf * 2e6 * 0.01)
    assert result.isp == pytest.approx(result.thrust / (result.mdot * 9.80665))
    cp = 1.4 * 287 / 0.4
    assert cp * result.te + result.ve**2 / 2 == pytest.approx(cp * 1000)
    assert result.te / 1000 == pytest.approx(temperature_ratio(result.mach))
    assert result.pe / 2e6 == pytest.approx(pressure_ratio(result.mach))


@pytest.mark.parametrize("g,expected", [(1.4, 1.812), (1.2, 2.247)])
def test_vacuum_cf_limit(g, expected):
    result = Nozzle(1, 1e18, g=g).solve(1e6, 3000, 0)
    assert result.cf == pytest.approx(expected, abs=6e-4)
    exact = np.sqrt(2 * g**2 / (g - 1)
                    * (2 / (g + 1)) ** ((g + 1) / (g - 1)))
    assert result.cf == pytest.approx(exact, rel=3e-4)
    assert not result.separation_risk


def test_separation_flag():
    result = Nozzle(0.01, 0.06).solve(1e6, 1000, 101325)
    assert result.pe / 101325 < 0.4
    assert result.separation_risk
    assert not Nozzle(0.01, 0.02).solve(1e6, 1000, 0).separation_risk


@pytest.mark.parametrize("call", [
    lambda: Nozzle(0.02, 0.01),
    lambda: Nozzle(0, 1),
    lambda: Nozzle(1, 2, g=1),
    lambda: Nozzle(1, 2, g=0.9),
    lambda: Nozzle(1, 2, R=0),
    lambda: area_ratio(0),
    lambda: pressure_ratio(-1),
    lambda: temperature_ratio(0),
    lambda: area_ratio(2, g=1),
    lambda: mach_from_area_ratio(0.9),
    lambda: mach_from_area_ratio(2, branch="invalid"),
    lambda: Nozzle(1, 2).solve(-1, 300, 0),
    lambda: Nozzle(1, 2).solve(1, 0, 0),
    lambda: Nozzle(1, 2).solve(1, 300, -1),
    lambda: area_ratio(float("nan")),
])
def test_invalid_inputs(call):
    with pytest.raises(ValueError):
        call()
