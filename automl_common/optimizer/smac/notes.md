# Scenario params

* [docs](https://automl.github.io/SMAC3/master/pages/details/scenario.html)

* `crash_cost : float` (cost_for_crash)
    The cost for a crash when `objective=="quality"`

* `time : int` (wallclock_limit)
    The maximum amount of wallclock time allowed

* `run_cutoff: int` (cutoff)
    The time allowed per run, required for `objective=="runtime"`

* `runs : int` (ta_run_limit)
    The maximum amount of runs allowed

* `save_instantly: bool` UNLISTED
    Whether to save runs instantly

*  `objective` : `str = "quality"` UNLISTED (run_obj)
    TODO Not sure what the others are

* `same_seed: bool`
    Whether the target being optimized is deterministic


