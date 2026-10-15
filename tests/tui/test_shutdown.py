import sys
from types import ModuleType

import pytest


@pytest.mark.parametrize("database_state", ["absent", "importing", "loaded"])
async def test_ctrl_c_quits_with_database_at_each_startup_stage(
    app_factory, monkeypatch, database_state
):
    module = ModuleType("src.db")
    if database_state == "absent":
        monkeypatch.delitem(sys.modules, "src.db", raising=False)
    else:
        if database_state == "loaded":
            module._QUERY_MODEL = object()
            module._INGEST_MODEL = None
        monkeypatch.setitem(sys.modules, "src.db", module)

    app = app_factory()
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.press("ctrl+c")
        assert not app._exit
        await pilot.press("ctrl+c")
        assert app._exit
        if database_state == "loaded":
            assert module._QUERY_MODEL is None
        elif database_state == "importing":
            assert "_QUERY_MODEL" not in vars(module)
