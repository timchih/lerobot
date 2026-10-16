"""Run lerobot-eval with recording enabled.

Eval recording maps env feature keys (e.g. ``robot_state/eef/pos``) to valid
LeRobot dataset keys via ``env.features_map`` and reads nested LIBERO
observations when building frames.
"""

from lerobot.scripts.lerobot_eval import main


if __name__ == "__main__":
    main()
