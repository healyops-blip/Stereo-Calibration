import numpy as np

from src.deeplearning.holdout_support import pose_transforms, summarize_holdout


def test_pose_transforms_use_board_to_left_then_left_to_right() -> None:
    model = dict(R_RL=np.eye(3), t_RL=np.array([-60.0, 0, 0]))
    transforms = pose_transforms(np.array([0.0, 0, 0, 10, 20, 500]), model)
    assert np.allclose(transforms["T_board_to_left"][:3, 3], [10, 20, 500])
    assert np.allclose(transforms["T_board_to_right"][:3, 3], [-50, 20, 500])


def test_failed_pairs_remain_in_summary_denominator() -> None:
    rows = [
        {
            "pair_id": "1",
            "methods": {
                "log3": {"pose": {"joint_rms_px": 0.2}},
                "sb": {"pose": {"joint_rms_px": 0.3}},
            },
        },
        {
            "pair_id": "2",
            "methods": {"log3": {"error": "missing"}, "sb": {"pose": {"joint_rms_px": 0.4}}},
        },
    ]
    summary = summarize_holdout(rows)
    assert summary["total_pairs"] == 2
    assert summary["methods"]["log3"]["successful_pairs"] == 1
    assert summary["common_pair_ids"] == ["1"]
    assert summary["methods"]["sb"]["common_joint_rms_mean_px"] == 0.3
