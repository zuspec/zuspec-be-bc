"""``zuspec-bc run`` (zuspec.be.bc.cli; docs/spec/run-api.md "Command line")."""
from zuspec.be.bc import cli

from .test_run import _model


def _image(tmp_path):
    path = tmp_path / "m.zbc"
    path.write_bytes(_model().to_bytes())
    return str(path)


def test_function_entry_passes(tmp_path, capsys):
    rc = cli.main(["run", _image(tmp_path), "--entry", "C::add",
                   "--arg", "5", "--arg", "0x7", "--seed", "3"])
    assert rc == cli.EXIT_OK
    assert capsys.readouterr().out.splitlines()[-1] == "PASS C::add seed=3"


def test_bad_entry_and_missing_image(tmp_path, capsys):
    assert cli.main(["run", _image(tmp_path), "--entry", "nope"]) == \
        cli.EXIT_USAGE
    assert "no entry 'nope'" in capsys.readouterr().err
    assert cli.main(["run", str(tmp_path / "none.zbc")]) == cli.EXIT_USAGE


def test_missing_platform_module(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["run", _image(tmp_path), "--entry", "main",
                     "--platform", "py:no_such_bc_platform"]) == \
        cli.EXIT_PLATFORM
    assert "failed to import" in capsys.readouterr().err
    assert cli.main(["run", _image(tmp_path), "--platform", "fake"]) == \
        cli.EXIT_PLATFORM


def test_py_platform_supplies_memory_and_imports(tmp_path, capsys,
                                                 monkeypatch):
    (tmp_path / "bc_cli_plat.py").write_text(
        "from zuspec.be.bc import run\n"
        "from zuspec.be.bc.interp.extern import Memory, "
        "RecordingImportProvider\n"
        "def platform():\n"
        "    mem = Memory()\n"
        "    mem.write(0x100, 4, 0x41)\n"
        "    return run.SimplePlatform(mem, RecordingImportProvider("
        "{0: lambda a: a[0] + 1}))\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    rc = cli.main(["run", _image(tmp_path), "--entry", "main", "--seed", "1",
                   "--platform", "py:bc_cli_plat"])
    assert rc == cli.EXIT_OK, capsys.readouterr()
    assert capsys.readouterr().out.splitlines()[-1] == "PASS main seed=1"


def test_failed_run_exits_1(tmp_path, capsys, monkeypatch):
    (tmp_path / "bc_cli_bad.py").write_text(
        "from zuspec.be.bc import run\n"
        "from zuspec.be.bc.interp.extern import Memory\n"
        "from zuspec.be.bc.interp.ops_proc import VMError\n"
        "class Imports:\n"
        "    calls = []\n"
        "    def call(self, fn_id, args):\n"
        "        raise VMError('boom')\n"
        "platform = run.SimplePlatform(Memory(), Imports())\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    assert cli.main(["run", _image(tmp_path), "--entry", "main",
                     "--seed", "2", "--platform", "py:bc_cli_bad"]) == \
        cli.EXIT_FAILED
    assert capsys.readouterr().out.splitlines()[-1] == \
        "FAIL main seed=2: runtime_error: boom"
