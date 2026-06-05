import pytest

from ecg_adv_gen.runner.matrix_parallel import assign_gpus_to_commands, parse_gpu_list


def _command(name, center):
    return {"name": name, "matrix": {"center": center}, "argv": ["python", "script.py"]}


def test_parse_gpu_list_rejects_empty_and_duplicates():
    assert parse_gpu_list("0,1,2,3") == ["0", "1", "2", "3"]

    with pytest.raises(ValueError, match="at least one GPU"):
        parse_gpu_list("")
    with pytest.raises(ValueError, match="duplicate GPU"):
        parse_gpu_list("0,1,0")


def test_assign_gpus_to_commands_uses_center_order():
    commands = [
        _command("run_ningbo", "ningbo"),
        _command("run_chapman", "chapman_shaoxing"),
        _command("run_cpsc", "cpsc_2018"),
        _command("run_georgia", "georgia"),
    ]

    assigned = assign_gpus_to_commands(commands, ["0", "1", "2", "3"], matrix_key="center")

    assert [(item.command_index, item.matrix_value, item.gpu) for item in assigned] == [
        (0, "ningbo", "0"),
        (1, "chapman_shaoxing", "1"),
        (2, "cpsc_2018", "2"),
        (3, "georgia", "3"),
    ]


def test_assign_gpus_to_commands_refuses_more_commands_than_gpus_without_queueing():
    commands = [_command("a", "a"), _command("b", "b")]

    with pytest.raises(ValueError, match="2 commands but only 1 GPUs"):
        assign_gpus_to_commands(commands, ["0"], matrix_key="center")
