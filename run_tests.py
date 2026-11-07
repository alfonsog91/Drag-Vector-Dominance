"""Run pytest when available, with explicit per-test PASS/SKIP/FAIL output."""

from pathlib import Path
import sys


class Reporter:
    def __init__(self):
        self.counts = dict(PASS=0, SKIP=0, FAIL=0)

    def _report(self, status, report):
        self.counts[status] += 1
        print(f"{status}: {report.nodeid}", flush=True)

    def pytest_runtest_logreport(self, report):
        if report.failed:
            self._report("FAIL", report)
        elif report.skipped:
            self._report("SKIP", report)
        elif report.when == "call" and report.passed:
            self._report("PASS", report)

    def pytest_collectreport(self, report):
        if report.failed:
            self._report("FAIL", report)
        elif report.skipped:
            self._report("SKIP", report)


def main():
    try:
        import pytest
    except ImportError:
        print("SKIP: pytest is not installed; no tests were run")
        return 2
    reporter = Reporter()
    args = sys.argv[1:] or [str(Path(__file__).resolve().parent / "tests"), "-q"]
    status = pytest.main(args, plugins=[reporter])
    print(" ".join(f"{key}={value}" for key, value in reporter.counts.items()))
    if status:
        print(f"FAIL: pytest exited with status {int(status)}")
    return int(status)


if __name__ == "__main__":
    sys.exit(main())
