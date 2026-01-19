#!/usr/bin/env python3
"""
Demonstration of demographic integration.

Shows how population-dyna data enriches agent behavioral models
with realistic income/poverty-based heterogeneity.
"""

import numpy as np
import sys

sys.path.insert(0, "src")

from agents.commuter import create_demographic_commuter_population


def main():
    print("=" * 70)
    print("DEMOGRAPHIC INTEGRATION DEMONSTRATION")
    print("=" * 70)
    print("\nCreating 100 agents with demographic profiles from Nashville ZCTAs...")

    # Nashville region (approximate)
    home_region = ((36.0, -87.0), (36.4, -86.5))
    work_region = ((36.1, -86.9), (36.3, -86.6))

    agents = create_demographic_commuter_population(
        n_agents=100,
        home_region=home_region,
        work_region=work_region,
        warehouse_path="warehouse.duckdb",
        rng=np.random.default_rng(42),
    )

    print(f"✓ Created {len(agents)} agents with demographic profiles\n")

    # Analyze by income quintile
    print("=" * 70)
    print("ANALYSIS BY INCOME QUINTILE")
    print("=" * 70)

    for quintile in range(1, 6):
        q_agents = [a for a in agents if a.profile.income_quintile == quintile]

        if not q_agents:
            continue

        incomes = [a.profile.household_income for a in q_agents]
        vots = [a.preferences.vot for a in q_agents]
        beta_incentives = [a.preferences.beta_incentive for a in q_agents]
        car_ownership = sum(1 for a in q_agents if a.profile.has_car) / len(q_agents)

        print(f"\nQuintile {quintile} ({'Lowest' if quintile == 1 else 'Highest' if quintile == 5 else 'Middle'} Income)")
        print(f"  Agents: {len(q_agents)}")
        print(f"  Avg Income: ${np.mean(incomes):,.0f}")
        print(f"  Avg VOT: ${np.mean(vots):.2f}/hr")
        print(f"  Avg β_incentive: {np.mean(beta_incentives):.3f}")
        print(f"  Car Ownership: {car_ownership:.1%}")

    # Show income-VOT correlation
    print("\n" + "=" * 70)
    print("INCOME-VOT CORRELATION")
    print("=" * 70)

    low_income = [a for a in agents if a.profile.income_quintile <= 2]
    high_income = [a for a in agents if a.profile.income_quintile >= 4]

    if low_income and high_income:
        avg_vot_low = np.mean([a.preferences.vot for a in low_income])
        avg_vot_high = np.mean([a.preferences.vot for a in high_income])

        print(f"\nLow Income (Q1-Q2):")
        print(f"  Avg VOT: ${avg_vot_low:.2f}/hr")
        print(f"\nHigh Income (Q4-Q5):")
        print(f"  Avg VOT: ${avg_vot_high:.2f}/hr")
        print(f"\nRatio: {avg_vot_high / avg_vot_low:.2f}x")

    # Show incentive sensitivity
    print("\n" + "=" * 70)
    print("INCENTIVE SENSITIVITY")
    print("=" * 70)

    if low_income and high_income:
        avg_beta_low = np.mean([a.preferences.beta_incentive for a in low_income])
        avg_beta_high = np.mean([a.preferences.beta_incentive for a in high_income])

        print(f"\nLow Income (Q1-Q2):")
        print(f"  Avg β_incentive: {avg_beta_low:.3f}")
        print(f"\nHigh Income (Q4-Q5):")
        print(f"  Avg β_incentive: {avg_beta_high:.3f}")
        print(f"\nRatio: {avg_beta_low / avg_beta_high:.2f}x")
        print("\n(Higher value = more responsive to incentives)")

    # Sample agents
    print("\n" + "=" * 70)
    print("SAMPLE AGENTS")
    print("=" * 70)

    for i in [0, 25, 50, 75, 99]:
        agent = agents[i]
        print(f"\nAgent {i}:")
        print(f"  ZCTA: {agent.profile.home_zcta}")
        print(f"  Income: ${agent.profile.household_income:,.0f}")
        print(f"  Quintile: Q{agent.profile.income_quintile}")
        print(f"  VOT: ${agent.preferences.vot:.2f}/hr")
        print(f"  β_incentive: {agent.preferences.beta_incentive:.3f}")
        print(f"  Has Car: {agent.profile.has_car}")

    print("\n" + "=" * 70)
    print("✅ DEMONSTRATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
