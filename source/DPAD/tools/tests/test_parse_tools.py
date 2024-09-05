""" Omid Sani, Shanechi Lab, University of Southern California, 2020 """

# pylint: disable=C0103, C0111

"Tests the module"

import copy
import os
import pickle
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from DPAD.tools.parse_tools import parseMethodCodeArgOptimizer


class TestParseTools(unittest.TestCase):

    def test_optimizers(self):
        codes = [
            "DPAD_LR1e-3",
            "DPAD_LR1e-3_optAdamW_ErSV128",
            "DPAD_LR1e-3_WD1e-3_optAdamW_scCDR2000_ErSV128",
            "DPAD_LR1e-3_optAdamW_scCD2000_ErSV128",
            "DPAD_LR1e-3_optAdamW_scED2000_0.96_ErSV128",
            "DPAD_LR1e-3_optAdamW_scED2000_0.96_strT_ErSV128",
            "DPAD_LR1e-3_optAdamW_scITD2000_0.96_ErSV128",
            "DPAD_LR1e-3_optAdamW_scPD2000_1e-4_ErSV128",
            "DPAD_LR1e-3_optAdamW_scPD2000_1e-4_cycT_ErSV128",
            "DPAD_LR1e-3_optAdamW_scPD2000_1e-4_2_cycT_ErSV128",
        ]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(1, 3, (1, 2))
        lineStyles = ["-", "--", "-.", ":"]

        for mi, methodCode in enumerate(codes):
            lr_scheduler_name = None
            lr_scheduler_args = None

            optimizer_name = "Adam"  # default
            optimizer_args = None
            optimizer_infos, matches = parseMethodCodeArgOptimizer(methodCode)
            if len(optimizer_infos) > 0:
                optimizer_info = optimizer_infos[0]
                if "optimizer_name" in optimizer_info:
                    optimizer_name = optimizer_info["optimizer_name"]
                if "optimizer_args" in optimizer_info:
                    optimizer_args = optimizer_info["optimizer_args"]
                if "scheduler_name" in optimizer_info:
                    lr_scheduler_name = optimizer_info["scheduler_name"]
                if "scheduler_args" in optimizer_info:
                    lr_scheduler_args = optimizer_info["scheduler_args"]

            if lr_scheduler_args is None:
                lr_scheduler_args = {}
            if optimizer_args is None:
                optimizer_args = {}
            optimizer_args_BU = copy.deepcopy(optimizer_args)

            if isinstance(lr_scheduler_name, str):
                if hasattr(tf.keras.optimizers.schedules, lr_scheduler_name):
                    lr_scheduler_constructor = getattr(
                        tf.keras.optimizers.schedules, lr_scheduler_name
                    )
                else:
                    raise Exception(
                        "Learning rate scheduler {lr_scheduler_name} not supported as string, pass actual class for the optimizer (e.g. tf.keras.optimizers.Adam)"
                    )
            else:
                lr_scheduler_constructor = lr_scheduler_name
            if isinstance(optimizer_name, str):
                if hasattr(tf.keras.optimizers, optimizer_name):
                    optimizer_constructor = getattr(tf.keras.optimizers, optimizer_name)
                else:
                    raise Exception(
                        "optimizer not supported as string, pass actual class for the optimizer (e.g. tf.keras.optimizers.Adam)"
                    )
            else:
                optimizer_constructor = optimizer_name
            if lr_scheduler_constructor is not None:
                if (
                    "learning_rate" in optimizer_args
                    and "initial_learning_rate" not in lr_scheduler_args
                ):
                    lr_scheduler_args["initial_learning_rate"] = optimizer_args[
                        "learning_rate"
                    ]
                lr_scheduler = lr_scheduler_constructor(**lr_scheduler_args)
                optimizer_args["learning_rate"] = lr_scheduler
            else:
                lr_scheduler = lambda steps: optimizer_args[
                    "learning_rate"
                ] * np.ones_like(steps)
            optimizer = optimizer_constructor(**optimizer_args)

            epochs = 2000
            batches = 20
            steps = np.arange(epochs * batches)
            lr = np.array(lr_scheduler(steps))

            ax.plot(
                steps,
                lr,
                label=f"{methodCode}\nOptimizer: {optimizer_name}, {optimizer_args_BU}, Scheduler: {lr_scheduler_name}\n{lr_scheduler_args}",
                linestyle=lineStyles[mi % len(lineStyles)],
            )

            print(f"Done with {lr_scheduler_name}")

        ax.set_xlabel(f"Training steps")
        ax.set_ylabel(f"Learning rate")
        ax.legend(
            bbox_to_anchor=(1.04, 0.5),
            loc="center left",
            borderaxespad=0,
            fontsize="x-small",
        )
        plt.show()

        print("Test!")


if __name__ == "__main__":
    unittest.main()
