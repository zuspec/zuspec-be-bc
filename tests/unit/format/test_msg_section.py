"""The MSG section: ``ZbcModel.messages`` and ``strings`` survive write and
read in every profile, so the native engine can format message text (pssc P4.2g)."""

import pytest

from zuspec.be.bc.format import spec
from zuspec.be.bc.model import CoroDescriptor, ZbcModel

ENUM = {"kind": "enum", "width": 32, "signed": True, "items": [["A", -1], ["B", 1]]}
I8 = {"kind": "int", "width": 8, "signed": True}
STR = {"kind": "string", "width": 64, "signed": False}
BOOL = {"kind": "bool", "width": 1, "signed": False}


def _model(profile):
    m = ZbcModel(coros=[CoroDescriptor(name="")], profile=profile)
    m.strings = ["hello", "", "w"]
    m.messages = [
        {"fmt": "a=%d e=%n s=%s b=%n", "args": [I8, ENUM, STR, BOOL], "slots": [3, 4, 5, 6, 7]},
        {"fmt": "plain", "args": [], "slots": [0]},
    ]
    return m


@pytest.mark.parametrize("profile", ["runtime", "codegen", "debug"])
def test_messages_round_trip(profile):
    m = _model(profile)
    back = ZbcModel.from_bytes(m.to_bytes())
    assert back.messages == m.messages
    assert back.strings == m.strings


def test_one_msg_section():
    img = _model("runtime").to_container()
    sec = [s for s in img.sections if s.kind == spec.sec_kind("ZBC_SEC_MSG")]
    assert len(sec) == 1 and sec[0].count == 1
    assert sec[0].elem_size == spec.RECORDS_BY_NAME["zbc_msgtab"].size() == 32
    for name in ("zbc_msg", "zbc_msgarg", "zbc_msgitem"):
        assert spec.RECORDS_BY_NAME[name].size() == 16


def test_no_messages_no_section():
    img = ZbcModel(coros=[CoroDescriptor(name="t_a")]).to_container()
    assert spec.sec_kind("ZBC_SEC_MSG") not in [s.kind for s in img.sections]
