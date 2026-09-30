from .base import Problem
from .heat import Heat
from .pendulum import Pendulum
from .wave import Wave

PROBLEMS = {"pendulum": Pendulum, "heat": Heat, "wave": Wave}


def make_problem(cfg: dict) -> Problem:
    return PROBLEMS[cfg["name"]](cfg["problem"])


__all__ = ["Problem", "Pendulum", "Heat", "Wave", "PROBLEMS", "make_problem"]
