"""
Lightweight utilities for easywakeword
"""
_last_debug_msg = None

def debug_print(msg, debug=True, dedup=True):
    """Print the debug message unless dedup is true and message repeated."""
    global _last_debug_msg
    if not debug:
        return
    if dedup and msg == _last_debug_msg:
        return
    print(msg)
    _last_debug_msg = msg

def reset_debug_state():
    """Reset the debug print dedup cache."""
    global _last_debug_msg
    _last_debug_msg = None

__all__ = ["debug_print", "reset_debug_state"]
"""
Lightweight utilities for easywakeword
"""
_last_debug_msg = None

def debug_print(msg, debug=True, dedup=True):
    """Print the debug message unless dedup is true and message repeated."""
    global _last_debug_msg
    if not debug:
        return
    if dedup and msg == _last_debug_msg:
        return
    print(msg)
    _last_debug_msg = msg

def reset_debug_state():
    """Reset the debug print dedup cache."""
    global _last_debug_msg
    _last_debug_msg = None

__all__ = ["debug_print", "reset_debug_state"]
"""
Lightweight utilities for easywakeword
"""
_last_debug_msg = None

def debug_print(msg, debug=True, dedup=True):
    """Print the debug message unless dedup is true and message repeated."""
    global _last_debug_msg
    if not debug:
        return
    if dedup and msg == _last_debug_msg:
        return
    print(msg)
    _last_debug_msg = msg

def reset_debug_state():
    """Reset the debug print dedup cache."""
    global _last_debug_msg
    _last_debug_msg = None

__all__ = ["debug_print", "reset_debug_state"]
"""
Lightweight utilities for easywakeword
"""
_last_debug_msg = None

def debug_print(msg, debug=True, dedup=True):
    """Print the debug message unless dedup is true and message repeated."""
    global _last_debug_msg
    if not debug:
        return
    if dedup and msg == _last_debug_msg:
        return
    print(msg)
    _last_debug_msg = msg

def reset_debug_state():
    """Reset the debug print dedup cache."""
    global _last_debug_msg
    _last_debug_msg = None

__all__ = ["debug_print", "reset_debug_state"]
"""
Lightweight utilities (migrated from easywakeword.utils)
"""
_last_debug_msg = None

def debug_print(msg, debug=True, dedup=True):
    global _last_debug_msg
    if not debug:
        return
    if dedup and msg == _last_debug_msg:
        return
    print(msg)
    _last_debug_msg = msg

def reset_debug_state():
    global _last_debug_msg
    _last_debug_msg = None

__all__ = ["debug_print", "reset_debug_state"]
"""
Lightweight utilities (migrated from easywakeword.utils)
"""
_last_debug_msg = None

def debug_print(msg, debug=True, dedup=True):
    global _last_debug_msg
    if not debug:
        return
    if dedup and msg == _last_debug_msg:
        return
    print(msg)
    _last_debug_msg = msg

def reset_debug_state():
    global _last_debug_msg
    _last_debug_msg = None

__all__ = ["debug_print", "reset_debug_state"]
"""
Lightweight utilities (migrated from easywakeword.utils)
"""
_last_debug_msg = None

def debug_print(msg, debug=True, dedup=True):
    global _last_debug_msg
    if not debug:
        return
    if dedup and msg == _last_debug_msg:
        return
    print(msg)
    _last_debug_msg = msg

def reset_debug_state():
    global _last_debug_msg
    _last_debug_msg = None
"""
Proxy wrapper to expose the debug utilities from easywakeword.utils
"""
from easywakeword.utils import debug_print, reset_debug_state

__all__ = ["debug_print", "reset_debug_state"]
