"""Tests for metrics calculation functions."""

import numpy as np
import pandas as pd
import pytest

from fitanalyzer.metrics import (
    calculate_intensity_factor,
    calculate_tss,
    np_power,
    trimp_from_hr,
)


class TestNormalizedPower:
    """Tests for normalized power calculation."""

    def test_normalized_power_valid_data(self):
        """Test NP calculation with valid power data."""
        power = np.array([200, 210, 190, 205, 195])
        result = np_power(power)
        assert np.isfinite(result)
        assert result > 0

    def test_normalized_power_empty_array(self):
        """Test NP with empty array returns nan."""
        result = np_power(np.array([]))
        assert np.isnan(result)

    def test_normalized_power_all_nan(self):
        """Test NP with all NaN values returns nan."""
        power = np.array([np.nan, np.nan, np.nan])
        result = np_power(power)
        assert np.isnan(result)

    def test_normalized_power_with_nans(self):
        """Test NP with some NaN values."""
        power = np.array([200, np.nan, 210, np.nan, 190])
        result = np_power(power)
        assert np.isfinite(result)
        assert result > 0

    def test_normalized_power_all_zeros(self):
        """Test NP with all zero values."""
        power = np.zeros(100)
        result = np_power(power)
        assert result == 0.0


class TestTRIMP:
    """Tests for TRIMP calculation."""

    def test_trimp_valid_data(self):
        """Test TRIMP calculation with valid HR data."""
        hr = [120, 130, 140, 150, 160]
        result = trimp_from_hr(hr, hr_rest=60, hr_max=190)
        assert result > 0
        assert np.isfinite(result)

    def test_trimp_empty_list(self):
        """Test TRIMP with empty list returns 0."""
        result = trimp_from_hr([], hr_rest=60, hr_max=190)
        assert result == 0.0

    def test_trimp_all_nan(self):
        """Test TRIMP with all NaN values returns 0."""
        hr = [np.nan, np.nan, np.nan]
        result = trimp_from_hr(hr, hr_rest=60, hr_max=190)
        assert result == 0.0

    def test_trimp_with_nans(self):
        """Test TRIMP with some NaN values."""
        hr = [120, np.nan, 140, np.nan, 160]
        result = trimp_from_hr(hr, hr_rest=60, hr_max=190)
        assert result > 0
        assert np.isfinite(result)

    def test_trimp_below_resting(self):
        """Test TRIMP with HR below resting (should clip to 0)."""
        hr = [50, 55, 58]  # All below hr_rest=60
        result = trimp_from_hr(hr, hr_rest=60, hr_max=190)
        assert result == 0.0

    def test_trimp_at_max(self):
        """Test TRIMP with HR at maximum."""
        hr = [190] * 60  # 1 minute at max HR
        result = trimp_from_hr(hr, hr_rest=60, hr_max=190)
        assert result > 0
        assert np.isfinite(result)


class TestTSS:
    """Tests for TSS calculation."""

    def test_tss_valid_data(self):
        """Test TSS calculation with valid inputs."""
        result = calculate_tss(
            normalized_power=250.0,
            intensity_factor=1.0,
            duration_hours=1.0,
            ftp=250.0
        )
        assert np.isfinite(result)
        assert result > 0

    def test_tss_with_nan_normalized_power(self):
        """Test TSS with NaN normalized power returns nan."""
        result = calculate_tss(
            normalized_power=np.nan,
            intensity_factor=1.0,
            duration_hours=1.0,
            ftp=250.0
        )
        assert np.isnan(result)

    def test_tss_with_nan_intensity_factor(self):
        """Test TSS with NaN intensity factor returns nan."""
        result = calculate_tss(
            normalized_power=250.0,
            intensity_factor=np.nan,
            duration_hours=1.0,
            ftp=250.0
        )
        assert np.isnan(result)

    def test_tss_with_nan_duration(self):
        """Test TSS with NaN duration returns nan."""
        result = calculate_tss(
            normalized_power=250.0,
            intensity_factor=1.0,
            duration_hours=np.nan,
            ftp=250.0
        )
        assert np.isnan(result)

    def test_tss_with_zero_ftp(self):
        """Test TSS with zero FTP returns nan."""
        result = calculate_tss(
            normalized_power=250.0,
            intensity_factor=1.0,
            duration_hours=1.0,
            ftp=0.0
        )
        assert np.isnan(result)

    def test_tss_with_negative_ftp(self):
        """Test TSS with negative FTP returns nan."""
        result = calculate_tss(
            normalized_power=250.0,
            intensity_factor=1.0,
            duration_hours=1.0,
            ftp=-250.0
        )
        assert np.isnan(result)

    def test_tss_calculation_correctness(self):
        """Test TSS calculation formula."""
        # TSS = (duration_hours * NP * IF) / FTP * 100
        # With NP=250, IF=1.0, duration=1h, FTP=250:
        # TSS = (1 * 250 * 1.0) / 250 * 100 = 100
        result = calculate_tss(
            normalized_power=250.0,
            intensity_factor=1.0,
            duration_hours=1.0,
            ftp=250.0
        )
        assert abs(result - 100.0) < 0.01


class TestIntensityFactor:
    """Tests for Intensity Factor calculation."""

    def test_intensity_factor_valid_data(self):
        """Test IF calculation with valid inputs."""
        result = calculate_intensity_factor(normalized_power=250.0, ftp=250.0)
        assert np.isfinite(result)
        assert result == 1.0

    def test_intensity_factor_with_nan_power(self):
        """Test IF with NaN normalized power returns nan."""
        result = calculate_intensity_factor(normalized_power=np.nan, ftp=250.0)
        assert np.isnan(result)

    def test_intensity_factor_with_zero_ftp(self):
        """Test IF with zero FTP returns nan."""
        result = calculate_intensity_factor(normalized_power=250.0, ftp=0.0)
        assert np.isnan(result)

    def test_intensity_factor_with_negative_ftp(self):
        """Test IF with negative FTP returns nan."""
        result = calculate_intensity_factor(normalized_power=250.0, ftp=-250.0)
        assert np.isnan(result)

    def test_intensity_factor_calculation_correctness(self):
        """Test IF calculation formula."""
        # IF = NP / FTP
        result = calculate_intensity_factor(normalized_power=200.0, ftp=250.0)
        assert abs(result - 0.8) < 0.01
        
        result = calculate_intensity_factor(normalized_power=300.0, ftp=250.0)
        assert abs(result - 1.2) < 0.01

    def test_intensity_factor_with_inf(self):
        """Test IF with infinite normalized power returns nan."""
        result = calculate_intensity_factor(normalized_power=np.inf, ftp=250.0)
        assert np.isnan(result)
