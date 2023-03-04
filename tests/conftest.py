import pytest
import pandas as pd


@pytest.fixture(scope="module")
def test_values():
    df = pd.read_csv('./tests/data/synthetics.csv')
    values = df.values.tolist()
    return values
