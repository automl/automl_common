import multiprocessing
import sys
import typing


def preload_modules(
    context: multiprocessing.context.BaseContext, module_list: typing.List[str]
) -> None:
    all_loaded_modules = sys.modules.keys()
    preload = [
        loaded_module
        for loaded_module in all_loaded_modules
        if loaded_module.split(".")[0] in module_list and "logging" not in loaded_module
    ]
    context.set_forkserver_preload(preload)
