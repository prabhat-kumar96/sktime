#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for NaiveForecaster update functionality."""

import pandas as pd
import pytest

from sktime.datasets import load_airline
from sktime.forecasting.naive import NaiveForecaster
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(NaiveForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_naive_forecaster_update_basic():
    """Basic fit -> update -> predict workflow."""
    y = load_airline()

    train = y.iloc[:100]
    new_data = y.iloc[100:110]

    forecaster = NaiveForecaster()
    forecaster.fit(train)

    # update with new observations
    forecaster.update(new_data)

    fh = [1, 2, 3]
    y_pred = forecaster.predict(fh)

    assert len(y_pred) == len(fh)
    # index should be a pandas Index of length len(fh)
    assert isinstance(y_pred.index, pd.Index)


@pytest.mark.skipif(
    not run_test_for_class(NaiveForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_naive_forecaster_update_increases_series_length():
    """Ensure internal stored series `_y` grows after update."""
    y = load_airline()

    train = y.iloc[:100]
    new_data = y.iloc[100:105]

    forecaster = NaiveForecaster()
    forecaster.fit(train)

    original_len = len(forecaster._y)

    forecaster.update(new_data)

    assert len(forecaster._y) == original_len + len(new_data)


@pytest.mark.skipif(
    not run_test_for_class(NaiveForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_naive_forecaster_multiple_updates():
    """Ensure repeated updates work and predictions still produced."""
    y = load_airline()

    train = y.iloc[:100]
    part1 = y.iloc[100:105]
    part2 = y.iloc[105:110]

    forecaster = NaiveForecaster()
    forecaster.fit(train)

    forecaster.update(part1)
    forecaster.update(part2)

    fh = [1, 2, 3, 4]
    y_pred = forecaster.predict(fh)

    assert len(y_pred) == len(fh)


@pytest.mark.skipif(
    not run_test_for_class(NaiveForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_naive_forecaster_update_consistency():
    """Check predictions remain valid (no NaNs) after updates."""
    y = load_airline()

    train = y.iloc[:100]
    new_data = y.iloc[100:110]

    forecaster = NaiveForecaster()
    forecaster.fit(train)
    forecaster.update(new_data)

    fh = [1, 2, 3, 4, 5]
    y_pred = forecaster.predict(fh)

    # predictions should not be all NaN
    assert not y_pred.isna().all()
