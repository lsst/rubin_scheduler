from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("rubin_scheduler")
except PackageNotFoundError:
    # package is not installed
    pass
