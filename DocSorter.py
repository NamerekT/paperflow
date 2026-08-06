"""Transitional shim.

The Task Scheduler entry for this machine still launches `pythonw.exe DocSorter.py`;
switching it to `pythonw.exe -m paperflow.apps.watcher` is the last step of this stage
and this file is deleted once that switch is verified (see the migration plan, stage 10).
Until then, this keeps the existing scheduled task working unchanged.
"""

from paperflow.apps.watcher.main import main

main()
