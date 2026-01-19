"""
Tests for demographic integration.

Verifies that population-dyna demographic data correctly enriches
agent profiles and calibrates behavioral parameters.
"""

import pytest
import numpy as np


class TestDemographicLoader:
    """Test the DemographicLoader utility."""

    def test_get_zcta(self):
        """Test single ZCTA lookup."""
        from src.data.demographics import DemographicLoader

        with DemographicLoader("warehouse.duckdb") as loader:
            # Nashville downtown
            demo = loader.get_zcta("37203")

            assert demo is not None
            assert demo.zcta_code == "37203"
            assert 0 <= demo.poverty_rate <= 1
            assert 1 <= demo.income_quintile <= 5
            assert demo.median_household_income_est > 0

    def test_sample_zctas(self):
        """Test ZCTA sampling."""
        from src.data.demographics import DemographicLoader

        with DemographicLoader("warehouse.duckdb") as loader:
            samples = loader.sample_zctas(n=50, state_prefix="37")

            assert len(samples) == 50
            assert all(s.zcta_code.startswith("37") for s in samples)
            assert all(1 <= s.income_quintile <= 5 for s in samples)

            # Check diversity in income quintiles
            quintiles = [s.income_quintile for s in samples]
            unique_quintiles = set(quintiles)
            assert len(unique_quintiles) >= 3  # At least 3 different quintiles

    def test_get_summary_stats(self):
        """Test summary statistics."""
        from src.data.demographics import DemographicLoader

        with DemographicLoader("warehouse.duckdb") as loader:
            stats = loader.get_summary_stats()

            assert stats["n_zctas"] > 0
            assert 0 <= stats["poverty"]["min"] <= stats["poverty"]["max"] <= 1
            assert stats["income"]["min"] < stats["income"]["avg"] < stats["income"]["max"]


class TestDemographicAwarePreferences:
    """Test demographic-aware preference generation."""

    def test_vot_scales_with_income(self):
        """Verify VOT increases with income."""
        from src.data.demographics import ZCTADemographics
        from src.agents.commuter import create_demographic_aware_preferences

        rng = np.random.default_rng(42)

        # Low income ZCTA
        low_income_demo = ZCTADemographics(
            zcta_code="37000",
            poverty_rate=0.30,
            median_household_income_est=30000,
            income_quintile=1,
        )

        # High income ZCTA
        high_income_demo = ZCTADemographics(
            zcta_code="37001",
            poverty_rate=0.05,
            median_household_income_est=90000,
            income_quintile=5,
        )

        low_prefs = create_demographic_aware_preferences(low_income_demo, rng=rng)
        high_prefs = create_demographic_aware_preferences(high_income_demo, rng=rng)

        # VOT should be higher for high income
        assert high_prefs.vot > low_prefs.vot

        # Low income should be more incentive-sensitive
        assert low_prefs.beta_incentive > high_prefs.beta_incentive

        # Low income should be more cost-sensitive (more negative)
        assert low_prefs.beta_cost < high_prefs.beta_cost

    def test_preferences_have_noise(self):
        """Verify agents from same ZCTA aren't identical."""
        from src.data.demographics import ZCTADemographics
        from src.agents.commuter import create_demographic_aware_preferences

        demo = ZCTADemographics(
            zcta_code="37203",
            poverty_rate=0.15,
            median_household_income_est=60000,
            income_quintile=3,
        )

        # Generate 10 agents from same ZCTA
        prefs_list = [
            create_demographic_aware_preferences(demo, rng=np.random.default_rng(i))
            for i in range(10)
        ]

        # VOT is deterministic (based on income), but beta coefficients have noise
        vots = [p.vot for p in prefs_list]
        assert len(set(vots)) == 1  # All identical (deterministic from income)

        # Beta coefficients should vary (have noise)
        beta_incentives = [p.beta_incentive for p in prefs_list]
        assert len(set(beta_incentives)) > 1  # Not all identical

        # Mean should be reasonable
        assert 0.05 < np.mean(beta_incentives) < 0.30


class TestDemographicPopulationGeneration:
    """Test full population generation with demographics."""

    def test_create_demographic_population(self):
        """Test creating agents with demographic data."""
        from src.agents.commuter import create_demographic_commuter_population

        # Nashville bounding box (approximate)
        home_region = ((36.0, -87.0), (36.4, -86.5))
        work_region = ((36.1, -86.9), (36.3, -86.6))

        agents = create_demographic_commuter_population(
            n_agents=100,
            home_region=home_region,
            work_region=work_region,
            warehouse_path="warehouse.duckdb",
            rng=np.random.default_rng(42),
        )

        assert len(agents) == 100

        # Verify all agents have demographics
        assert all(a.profile.home_zcta is not None for a in agents)
        assert all(a.profile.household_income is not None for a in agents)
        assert all(a.profile.income_quintile is not None for a in agents)
        assert all(1 <= a.profile.income_quintile <= 5 for a in agents)

    def test_income_distribution(self):
        """Verify income distribution across agents."""
        from src.agents.commuter import create_demographic_commuter_population

        home_region = ((36.0, -87.0), (36.4, -86.5))
        work_region = ((36.1, -86.9), (36.3, -86.6))

        agents = create_demographic_commuter_population(
            n_agents=200,
            home_region=home_region,
            work_region=work_region,
            warehouse_path="warehouse.duckdb",
            rng=np.random.default_rng(42),
        )

        incomes = [a.profile.household_income for a in agents]

        # Should have meaningful variation
        assert max(incomes) > min(incomes) * 1.5

        # All quintiles should be represented
        quintiles = [a.profile.income_quintile for a in agents]
        unique_quintiles = set(quintiles)
        assert len(unique_quintiles) >= 4  # At least 4 of 5 quintiles

    def test_vot_correlates_with_income(self):
        """Verify VOT increases with household income."""
        from src.agents.commuter import create_demographic_commuter_population

        home_region = ((36.0, -87.0), (36.4, -86.5))
        work_region = ((36.1, -86.9), (36.3, -86.6))

        agents = create_demographic_commuter_population(
            n_agents=100,
            home_region=home_region,
            work_region=work_region,
            warehouse_path="warehouse.duckdb",
            rng=np.random.default_rng(42),
        )

        # Group by quintile
        quintile_vots = {q: [] for q in range(1, 6)}
        for agent in agents:
            quintile_vots[agent.profile.income_quintile].append(agent.preferences.vot)

        # Average VOT should increase with quintile
        avg_vots = {q: np.mean(vots) for q, vots in quintile_vots.items() if vots}

        # Q1 < Q3 < Q5
        if 1 in avg_vots and 5 in avg_vots:
            assert avg_vots[1] < avg_vots[5]

    def test_car_ownership_increases_with_income(self):
        """Verify car ownership probability increases with income."""
        from src.agents.commuter import create_demographic_commuter_population

        home_region = ((36.0, -87.0), (36.4, -86.5))
        work_region = ((36.1, -86.9), (36.3, -86.6))

        agents = create_demographic_commuter_population(
            n_agents=200,
            home_region=home_region,
            work_region=work_region,
            warehouse_path="warehouse.duckdb",
            rng=np.random.default_rng(42),
        )

        # Group by quintile
        quintile_car_rates = {}
        for q in range(1, 6):
            q_agents = [a for a in agents if a.profile.income_quintile == q]
            if q_agents:
                car_rate = sum(1 for a in q_agents if a.profile.has_car) / len(q_agents)
                quintile_car_rates[q] = car_rate

        # Car ownership should generally increase with quintile
        if 1 in quintile_car_rates and 5 in quintile_car_rates:
            assert quintile_car_rates[5] >= quintile_car_rates[1]


class TestEndToEndIntegration:
    """End-to-end integration tests."""

    def test_full_pipeline(self):
        """Test complete pipeline from data load to agent creation."""
        from src.data.demographics import DemographicLoader
        from src.agents.commuter import create_demographic_commuter_population

        # 1. Verify data loaded
        with DemographicLoader("warehouse.duckdb") as loader:
            stats = loader.get_summary_stats()
            assert stats["n_zctas"] > 0

        # 2. Create population
        agents = create_demographic_commuter_population(
            n_agents=50,
            home_region=((36.0, -87.0), (36.4, -86.5)),
            work_region=((36.1, -86.9), (36.3, -86.6)),
            warehouse_path="warehouse.duckdb",
            rng=np.random.default_rng(42),
        )

        # 3. Verify agent properties
        assert len(agents) == 50
        assert all(hasattr(a, "profile") for a in agents)
        assert all(hasattr(a, "preferences") for a in agents)

        # 4. Verify demographic attributes present
        assert all(a.profile.home_zcta is not None for a in agents)
        assert all(a.profile.household_income is not None for a in agents)

        # 5. Verify behavioral heterogeneity
        vots = [a.preferences.vot for a in agents]
        assert len(set(vots)) > 10  # At least 10 unique values

        # 6. Verify income-VOT relationship
        low_income_agents = [a for a in agents if a.profile.income_quintile <= 2]
        high_income_agents = [a for a in agents if a.profile.income_quintile >= 4]

        if low_income_agents and high_income_agents:
            avg_vot_low = np.mean([a.preferences.vot for a in low_income_agents])
            avg_vot_high = np.mean([a.preferences.vot for a in high_income_agents])
            assert avg_vot_high > avg_vot_low

    def test_demographic_summary_report(self):
        """Generate and verify demographic summary report."""
        from src.agents.commuter import create_demographic_commuter_population

        agents = create_demographic_commuter_population(
            n_agents=100,
            home_region=((36.0, -87.0), (36.4, -86.5)),
            work_region=((36.1, -86.9), (36.3, -86.6)),
            warehouse_path="warehouse.duckdb",
            rng=np.random.default_rng(42),
        )

        # Generate summary
        incomes = [a.profile.household_income for a in agents]
        vots = [a.preferences.vot for a in agents]
        quintiles = [a.profile.income_quintile for a in agents]

        report = {
            "n_agents": len(agents),
            "income": {
                "min": min(incomes),
                "max": max(incomes),
                "mean": np.mean(incomes),
            },
            "vot": {
                "min": min(vots),
                "max": max(vots),
                "mean": np.mean(vots),
            },
            "quintile_distribution": {
                q: quintiles.count(q) for q in range(1, 6)
            },
        }

        # Verify report structure
        assert report["n_agents"] == 100
        assert report["income"]["min"] < report["income"]["mean"] < report["income"]["max"]
        assert report["vot"]["min"] < report["vot"]["mean"] < report["vot"]["max"]
        assert sum(report["quintile_distribution"].values()) == 100
