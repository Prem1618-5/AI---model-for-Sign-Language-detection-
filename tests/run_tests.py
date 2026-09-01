"""
Unified Tier-by-Tier Test Runner for Sign Language Detection ML System
Executes tests categorized into 5 execution tiers with SLA timing diagnostics and exit status reporting.
"""

import os
import sys
import time
import argparse
import unittest

# Ensure src and tests are in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
TESTS_DIR = os.path.join(PROJECT_ROOT, 'tests')

for d in [SRC_DIR, TESTS_DIR, PROJECT_ROOT]:
    if d not in sys.path:
        sys.path.insert(0, d)

# Import test cases
from test_preprocessing import (
    TestLandmarkNormalization,
    TestLandmarkFlatteningAndAugmentation,
    TestSchemaParserAndDatasetPreparation
)
from test_model import (
    TestModelArchitectureAndTensors,
    TestModelTrainingAndPersistence
)
from test_temporal import (
    TestTemporalFiltering,
    TestRealtimeRecognizerBufferMethods
)
from test_ui import (
    TestHUDStateAndDecoupledRenderer,
    TestRealtimeRecognizerHUDMethods
)
from test_camera import (
    TestCameraDiagnostics
)
from test_cli import (
    TestCLIParsing
)


TIERS = {
    1: {
        'name': 'Tier 1: Fast Invariants & Schema Normalization',
        'sla_budget_ms': 100,
        'test_classes': [
            TestLandmarkNormalization,
            TestLandmarkFlatteningAndAugmentation,
            TestHUDStateAndDecoupledRenderer
        ]
    },
    2: {
        'name': 'Tier 2: Algorithmic & ML Architecture Unit Tests',
        'sla_budget_ms': 500,
        'test_classes': [
            TestModelArchitectureAndTensors,
            TestTemporalFiltering,
            TestRealtimeRecognizerBufferMethods
        ]
    },
    3: {
        'name': 'Tier 3: Mock-Driven UI, Hardware & CLI Integration',
        'sla_budget_ms': 1500,
        'test_classes': [
            TestCLIParsing,
            TestCameraDiagnostics,
            TestRealtimeRecognizerHUDMethods
        ]
    },
    4: {
        'name': 'Tier 4: Pipeline E2E & Dataset Persistence',
        'sla_budget_ms': 6000,
        'test_classes': [
            TestSchemaParserAndDatasetPreparation,
            TestModelTrainingAndPersistence
        ]
    },
    5: {
        'name': 'Tier 5: Adversarial Boundary & Defensive Stress Invariants',
        'sla_budget_ms': 2000,
        'test_classes': [
            # Selected adversarial tests from suites
            TestLandmarkNormalization,
            TestCLIParsing
        ]
    }
}


def create_suite_for_classes(classes: list) -> unittest.TestSuite:
    """Build a TestSuite from a list of TestCase classes."""
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    for cls in classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    return suite


def run_tier(tier_num: int, tier_info: dict, verbose: bool = False, failfast: bool = False) -> tuple[int, int, int, float]:
    """
    Runs a single test tier and outputs detailed timing diagnostics.
    Returns: (total_tests, failures, errors, elapsed_time_sec)
    """
    print(f"\n{'='*70}")
    print(f"  RUNNING {tier_info['name'].upper()}")
    print(f"  Target SLA Budget: < {tier_info['sla_budget_ms']}ms")
    print(f"{'='*70}")

    suite = create_suite_for_classes(tier_info['test_classes'])
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1, failfast=failfast)

    start_time = time.perf_counter()
    result = runner.run(suite)
    elapsed_time = time.perf_counter() - start_time
    elapsed_ms = elapsed_time * 1000.0

    total = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = total - failures - errors

    sla_status = "PASSED" if elapsed_ms <= tier_info['sla_budget_ms'] else "EXCEEDED (WARN)"
    print(f"  --> Tier {tier_num} Summary: {passed}/{total} Passed | "
          f"Duration: {elapsed_ms:.1f}ms | SLA: {sla_status}")

    return total, failures, errors, elapsed_time


def main():
    parser = argparse.ArgumentParser(description="Unified 4-Tier Test Runner for Sign Language ML")
    parser.add_argument('--tier', type=int, choices=[1, 2, 3, 4, 5], default=None,
                        help="Run only a specific test tier (1 to 5)")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Show verbose test outputs")
    parser.add_argument('--failfast', action='store_true',
                        help="Stop immediately on first failure")
    args = parser.parse_args()

    print("\n" + "#"*70)
    print("  SIGN LANGUAGE DETECTION ML SYSTEM - 4-TIER TEST SUITE")
    print("  Framework: Standard Library unittest (Ponytail zero-dependency)")
    print("#"*70)

    total_run = 0
    total_failures = 0
    total_errors = 0
    total_time = 0.0

    tiers_to_run = [args.tier] if args.tier is not None else sorted(TIERS.keys())

    for t_num in tiers_to_run:
        t_info = TIERS[t_num]
        t_run, t_fail, t_err, t_time = run_tier(t_num, t_info, verbose=args.verbose, failfast=args.failfast)
        total_run += t_run
        total_failures += t_fail
        total_errors += t_err
        total_time += t_time

        if args.failfast and (t_fail > 0 or t_err > 0):
            break

    print("\n" + "#"*70)
    print("  TEST EXECUTION COMPLETED")
    print(f"  Total Test Cases Executed : {total_run}")
    print(f"  Total Passed              : {total_run - total_failures - total_errors}")
    print(f"  Total Failures            : {total_failures}")
    print(f"  Total Errors              : {total_errors}")
    print(f"  Total Suite Wall Time     : {total_time:.3f}s")
    print("#"*70)

    if total_failures == 0 and total_errors == 0:
        print("\n>>> ALL TESTS PASSED SUCCESSFULLY! [EXIT CODE 0] <<<\n")
        sys.exit(0)
    else:
        print(f"\n>>> TEST SUITE FAILED WITH {total_failures + total_errors} ISSUES [EXIT CODE 1] <<<\n")
        sys.exit(1)


if __name__ == '__main__':
    main()
