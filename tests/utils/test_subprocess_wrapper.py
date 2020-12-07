import pytest
import time

from halite_rl.utils import SubProcessWrapper, SubProcessWrapperException

class MockObject():
    def add(self, x, y):
        return x + y

    def raise_exception(self):
        raise ValueError("test")


def test_exception_in_constructor():
    def mock_constructor():
        raise ValueError("test")

    with pytest.raises(ValueError):
        w = SubProcessWrapper(mock_constructor)

def test_async_call_success():
    w = SubProcessWrapper(MockObject)
    w.call("add", 1, y=2)
    result = w.get_result()
    assert result == 1 + 2
    w.close()

def test_async_call_already_in_progress():
    w = SubProcessWrapper(MockObject)
    w.call("add", 1, 2) # Call function that will sleep for 1 second.

    with pytest.raises(SubProcessWrapperException):
        # Attempting to make another call before the previous result
        # has been restrieved should raise an exception.
        w.call("add", 3, 4)

    result = w.get_result()
    assert result == 1 + 2
    w.close()

def test_async_call_already_closed():
    w = SubProcessWrapper(MockObject)
    w.close()

    with pytest.raises(SubProcessWrapperException):
        # Attempting to make a call after the wrapper has been closed
        # should raise an exception.
        w.call("add", 1, 2)

def test_async_call_exception():
    w = SubProcessWrapper(MockObject)
    w.call("raise_exception") # Call function that will raise an exception.
    with pytest.raises(SubProcessWrapperException):
        w.get_result()
    w.close()
