import pytest
from easywakeword import WakeWord

# Dummy test to check import and class instantiation
def test_wakeword_init():
    ww = WakeWord(
        textword="ok google",
        wavword="okgoogle.wav",
        numberofwords=2,
        timeout=1
    )
    assert ww.textword == "ok google"
    assert ww.numberofwords == 2
    assert ww.timeout == 1
