import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tunnelscreen import screening as sc  # noqa: E402


@pytest.fixture(scope="session")
def scn():
    return sc.load_scenario()


@pytest.fixture(scope="session")
def buildings():
    return sc.load_buildings()


@pytest.fixture(scope="session")
def by_id(buildings):
    return {b.id: b for b in buildings}
