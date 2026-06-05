from ecg_adv_gen.training.online_at_records import build_checkpoint_selection_record


def test_build_checkpoint_selection_record_marks_target_val_as_internal():
    record = build_checkpoint_selection_record(
        center="ningbo",
        best_epoch=4,
        metric_name="target_macro_auprc",
        metric_value=0.51,
        selection_source="target_real_val",
        heldout_target_labels_used=False,
    )

    assert record["center"] == "ningbo"
    assert record["best_epoch"] == 4
    assert record["metric_name"] == "target_macro_auprc"
    assert record["metric_value"] == 0.51
    assert record["selection_source"] == "target_real_val"
    assert record["heldout_target_labels_used_for_selection"] is False
    assert record["paper_safe_selection"] is True
