import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.propagation import propagate_delay

def test_slack_absorbs_delay():
    assert propagate_delay(10, 3, 0, 6) == 7

def test_recovery_reduces_delay():
    assert propagate_delay(10, 2, 4, 0) == 8

def test_delay_never_negative_from_recovery():
    assert propagate_delay(2, 0, 10, 0) == 0
