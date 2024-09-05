""" 
Copyright (c) 2024 University of Southern California
See full notice in LICENSE.md
Omid G. Sani and Maryam M. Shanechi
Shanechi Lab, University of Southern California
"""

""" Tools for parsing method description strings """

import copy
import re

import numpy as np


def extractNumberFromRegex(
    saveCode, regex=r"([-+]?[\d]+\.?[\d]*)([Ee][-+]?[\d]+)?", prefix=None
):
    if prefix is not None:
        regex = re.compile(prefix + str(regex))
    out = []
    out_matches = []
    if len(re.findall(regex, saveCode)):
        matches = re.finditer(regex, saveCode)
        for matchNum, match in enumerate(matches, start=1):
            num = float(match.groups()[0])
            if match.groups()[1] is not None:
                num *= 10 ** float(match.groups()[1][1:])
            out_matches.append(match)
            out.append(num)
    return out, out_matches


def extractPowRangesFromRegex(regex, saveCode, base_type=int):
    out = []
    out_matches = []
    if len(re.findall(regex, saveCode)):
        matches = re.finditer(regex, saveCode)
        for matchNum, match in enumerate(matches, start=1):
            base, min_val, step_val, max_val = match.groups()
            out_matches.append(match)
            pows = np.arange(int(min_val), 1 + int(max_val), int(step_val))
            steps = [base_type(base) ** p for p in pows]
            out.extend(steps)
    return out, out_matches


def extractLinearRangesFromRegex(regex, saveCode):
    out = []
    out_matches = []
    if len(re.findall(regex, saveCode)):
        matches = re.finditer(regex, saveCode)
        for matchNum, match in enumerate(matches, start=1):
            min_val, step_val, max_val = match.groups()
            out_matches.append(match)

            out.extend(np.arange(float(min_val), 1 + float(max_val), float(step_val)))
    return out, out_matches


def extractIntRangesFromRegex(regex, saveCode):
    out = []
    out_matches = []
    if len(re.findall(regex, saveCode)):
        matches = re.finditer(regex, saveCode)
        for matchNum, match in enumerate(matches, start=1):
            min_val, step_val, max_val = match.groups()
            out_matches.append(match)

            out.extend(np.arange(int(min_val), 1 + int(max_val), int(step_val)))
    return out, out_matches


def extractStrsFromRegex(regex, saveCode):
    out = []
    out_matches = []
    if len(re.findall(regex, saveCode)):
        matches = re.finditer(regex, saveCode)
        for matchNum, match in enumerate(matches, start=1):
            this_val = match.groups()[0]
            out_matches.append(match)
            out.extend([this_val])
    return out, out_matches


def extractIntsFromRegex(regex, saveCode):
    out = []
    out_matches = []
    if len(re.findall(regex, saveCode)):
        matches = re.finditer(regex, saveCode)
        for matchNum, match in enumerate(matches, start=1):
            this_val = match.groups()[0]
            out_matches.append(match)
            out.extend([int(this_val)])
    return out, out_matches


def extractFloatsFromRegex(regex, saveCode):
    out = []
    out_matches = []
    if len(re.findall(regex, saveCode)):
        matches = re.finditer(regex, saveCode)
        for matchNum, match in enumerate(matches, start=1):
            this_val = match.groups()[0]
            out_matches.append(match)
            out.extend([float(this_val)])
    return out, out_matches


def parseMethodCodeArg_kpp(saveCode):
    if (
        "kpp" in saveCode
    ):  # Keeps this portion of the whole preprocessed data for all analyses
        regex = r"(tr)?kpp(\d+\.?\d*|\d*\.?\d+)_(\d+\.?\d*|\d*\.?\d+)"  # _kpp0.5_1 or _trkpp0.5_1
        if len(re.findall(regex, saveCode)):
            matches = re.finditer(regex, saveCode)
            for matchNum, match in enumerate(matches, start=1):
                dataPeriod = (
                    "train"
                    if len(match.groups()) > 2 or match.groups()[0] == "tr"
                    else "all"
                )
                keptPortion = (float(match.groups()[-2]), float(match.groups()[-1]))
        else:
            regex = r"(tr)kpp(\d+\.?\d*|\d*\.?\d+)"  # _kpp0.5 or _trkpp0.5
            matches = re.finditer(regex, saveCode)
            for matchNum, match in enumerate(matches, start=1):
                dataPeriod = (
                    "train"
                    if len(match.groups()) > 1 or match.groups()[0] == "tr"
                    else "all"
                )
                keptPortion = (0, float(match.groups()[-1]))
        return dataPeriod, keptPortion, match
    else:
        return None, None, None


def parseMethodCodeArgStepsAhead(saveCode):
    out_matches = []
    steps_ahead, matches1 = extractIntRangesFromRegex(
        r"sta(\d+);(\d+);(\d+)", saveCode
    )  # sta1;1;5
    out_matches.extend(matches1)

    steps_ahead2, matches2 = extractPowRangesFromRegex(
        r"sta(\d+)\^(\d+);(\d+);(\d+)", saveCode
    )  # sta2^1;1;5
    out_matches.extend(matches2)
    steps_ahead.extend(steps_ahead2)

    steps_ahead3, matches3 = extractIntsFromRegex(
        r"sta(\d+)(?![;\^])", saveCode
    )  # sta10 (but not sta10;)
    out_matches.extend(matches3)
    steps_ahead.extend(steps_ahead3)

    if len(steps_ahead) == 0:
        steps_ahead = None

    steps_ahead_loss_weights = None

    if steps_ahead is not None:
        zeroWeightList, matches1 = extractIntRangesFromRegex(
            r"staZW(\d+);(\d+);(\d+)", saveCode
        )  # staZW2;1;5
        out_matches.extend(matches1)
        zeroWeightList2, matches2 = extractPowRangesFromRegex(
            r"staZW(\d+)\^(\d+);(\d+);(\d+)", saveCode
        )  # staZW2^2;1;5
        zeroWeightList2.extend(zeroWeightList2)
        out_matches.extend(matches2)
        zeroWeightList3, matches3 = extractIntsFromRegex(
            r"staZW(\d+)(?!;)", saveCode
        )  # staZW10 (but not staZW10;)
        zeroWeightList.extend(zeroWeightList3)
        out_matches.extend(matches3)

        if len(zeroWeightList) > 0:
            steps_ahead_loss_weights = [
                0.0 if step_ahead in zeroWeightList else 1.0
                for step_ahead in steps_ahead
            ]

    return steps_ahead, steps_ahead_loss_weights, out_matches


def parseMethodCodeArgEnsemble(saveCode):
    out_matches = []
    ensemble_cnt, matches1 = extractIntsFromRegex(r"ensm(\d+)", saveCode)  # ensm10
    out_matches.extend(matches1)
    return ensemble_cnt, out_matches


def parseMethodCodeArgOptimizer(saveCode):
    """Parses the optimizer settings from methodCode

    Args:
        saveCode (str): the string specifying the method settings

    Returns:
        _type_: _description_
    """
    outs = []
    out_matches = []

    optimizer_args = None
    learning_rates, matches = extractNumberFromRegex(
        saveCode, prefix="LR"
    )  # LR1e-05 or LR0.01
    if len(learning_rates) > 0:
        if optimizer_args is None:
            optimizer_args = {}
        optimizer_args.update(
            {"learning_rate": learning_rates[0]}  # Default for Adam 0.001
        )

    weight_decays, matches = extractNumberFromRegex(
        saveCode, prefix="WD"
    )  # WD1e-05 or WD0.01
    if len(weight_decays) > 0:
        if optimizer_args is None:
            optimizer_args = {}
        optimizer_args.update(
            {"weight_decay": weight_decays[0]}  # Default for AdamW is 1
        )

    regex = r"opt(AdamW|Adam)(_sc)?(CDR|CD|ED|ITD|PCD|PD)?"
    if len(re.findall(regex, saveCode)) == 0:
        # Revert to default
        outs = [{"optimizer_args": optimizer_args}]
    else:
        matches = re.finditer(regex, saveCode)
        for matchNum, match in enumerate(matches, start=1):
            groups = match.groups()
            optimizer_name = groups[0]
            out = {"optimizer_name": optimizer_name, "optimizer_args": optimizer_args}
            if len(groups) > 0 and groups[1] == "_sc":
                scheduler_code = groups[2]
                scheduler_options = {
                    "CD": {  # https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/CosineDecay
                        "name": "CosineDecay",
                        "args": [
                            "initial_learning_rate",
                            "decay_steps",
                            "alpha",
                            "warmup_target",
                            "warmup_steps",
                        ],
                    },
                    "CDR": {  # https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/CosineDecayRestarts
                        "name": "CosineDecayRestarts",
                        "args": [
                            "initial_learning_rate",
                            "first_decay_steps",
                            "t_mul",
                            "m_mul",
                            "alpha",
                        ],
                    },
                    "ED": {  # https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/ExponentialDecay
                        "name": "ExponentialDecay",
                        "args": [
                            "initial_learning_rate",
                            "decay_steps",
                            "decay_rate",
                            "staircase",
                        ],
                    },
                    "ITD": {  # https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/InverseTimeDecay
                        "name": "InverseTimeDecay",
                        "args": [
                            "initial_learning_rate",
                            "decay_steps",
                            "decay_rate",
                            "staircase",
                        ],
                    },
                    "PCD": {  # https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/PiecewiseConstantDecay
                        "name": "PiecewiseConstantDecay",
                        "args": [
                            "boundaries",
                            "values",
                        ],  # boundaries and values are each lists
                    },
                    "PD": {  # https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/PolynomialDecay
                        "name": "PolynomialDecay",
                        "args": [
                            "initial_learning_rate",
                            "decay_steps",
                            "end_learning_rate",
                            "power",
                            "cycle",
                        ],
                    },
                }
                scheduler_name = scheduler_options[scheduler_code]["name"]
                out["scheduler_name"] = scheduler_name
                regex2 = re.compile(
                    f"opt{optimizer_name}_sc{scheduler_code}"
                    + r"(_?(?:_?[-+]?[\d]+\.?[\d]*)(?:[Ee][-+]?[\d]+)?)*"
                )
                if len(re.findall(regex2, saveCode)):
                    matches2 = re.finditer(regex2, saveCode)
                    for matchNum2, match2 in enumerate(matches2, start=1):
                        matchstr = match2.group().replace(
                            f"opt{optimizer_name}_sc{scheduler_code}", ""
                        )
                        numbers = matchstr.split("_")
                        args = {}
                        for param_ind, num_str in enumerate(numbers):
                            out_this, out_matches_this = extractNumberFromRegex(num_str)
                            num = out_this[0]
                            if float(num) == int(num):
                                num = int(num)
                            arg_name = scheduler_options[scheduler_code]["args"][
                                1 + param_ind
                            ]
                            args[arg_name] = num
                        out["scheduler_args"] = args
                regex2 = re.compile(
                    f"opt{optimizer_name}_sc{scheduler_code}"
                    + r"(?:_?(?:_?[-+]?[\d]+\.?[\d]*)(?:[Ee][-+]?[\d]+)?)*(_str(T|F))"
                )  # Check for staircase in ExponentialDecay and InverseTimeDecay
                if len(re.findall(regex2, saveCode)):
                    matches2 = re.finditer(regex2, saveCode)
                    for matchNum2, match2 in enumerate(matches2, start=1):
                        out["scheduler_args"]["staircase"] = (
                            True if match2.groups()[-1] == "T" else False
                        )
                regex2 = re.compile(
                    f"opt{optimizer_name}_sc{scheduler_code}"
                    + r"(?:_?(?:_?[-+]?[\d]+\.?[\d]*)(?:[Ee][-+]?[\d]+)?)*(_cyc(T|F))"
                )  # Check for cycle in PolynomialDecay
                if len(re.findall(regex2, saveCode)):
                    matches2 = re.finditer(regex2, saveCode)
                    for matchNum2, match2 in enumerate(matches2, start=1):
                        out["scheduler_args"]["cycle"] = (
                            True if match2.groups()[-1] == "T" else False
                        )
            out_matches.append(match)
            outs.append(out)

    return outs, out_matches


def parseInnerCVFoldSettings(saveCode):
    out = []
    out_matches = []
    regex = r"iCVF(\d+)o?(\d+)?"  # iCVF2, or iCVF5o5
    if len(re.findall(regex, saveCode)):
        matches = re.finditer(regex, saveCode)
        for matchNum, match in enumerate(matches, start=1):
            vals = match.groups()
            numFolds = int(vals[0])
            foldsToRun = (
                [int(vals[1])] if len(vals) > 1 and vals[1] is not None else None
            )
            out_matches.append(match)
            out.extend([{"folds": numFolds, "foldsToRun": foldsToRun}])
    return out, out_matches


def extractValueRanges(methodCode, prefix="L"):
    methodCodeCpy = copy.copy(methodCode)
    lambdaVals = []
    lambdaValStrs = []
    # Find lambda vals provided as linear ranges
    regex = f"{prefix}" + r"(\d+)e([-+])?(\d+);(\d+);(\d+)e([-+])?(\d+)"  # L1e-2:-2:-8
    matches = re.finditer(regex, methodCodeCpy)
    for matchNum, match in enumerate(matches, start=1):
        m, sgn, power, count, m2, sgn2, power2 = match.groups()
        power = -float(power) if sgn is not None and sgn == "-" else float(power)
        power2 = -float(power2) if sgn2 is not None and sgn2 == "-" else float(power2)
        lVals = np.linspace(float(m) * 10**power, float(m2) * 10**power2, int(count))
        lValsC = np.array([float(f"{l:.5f}") for l in lVals])
        if np.max(np.abs(lVals - lValsC)) < np.min(lVals) * 1e-3:
            lVals = lValsC
        lambdaVals.extend(list(lVals))
        strSpan = match.span()
        lambdaValStrs.extend([methodCodeCpy[strSpan[0] : strSpan[1]]] * int(count))
    for ls in lambdaValStrs:
        methodCodeCpy = methodCodeCpy.replace(ls, "")
    # Find lambda vals provided as ranges of exponents
    regex = f"{prefix}" + r"(\d+)e([-+])?(\d+);([-+])?(\d+);([-+])?(\d+)"  # L1e-2:-2:-8
    matches = re.finditer(regex, methodCodeCpy)
    for matchNum, match in enumerate(matches, start=1):
        m, sgn, power, step_sgn, step_val, sgn2, power2 = match.groups()
        power = -float(power) if sgn is not None and sgn == "-" else float(power)
        power2 = -float(power2) if sgn2 is not None and sgn2 == "-" else float(power2)
        step_val = (
            -float(step_val)
            if step_sgn is not None and step_sgn == "-"
            else float(step_val)
        )
        pow_vals = np.array(np.arange(power, power2, step_val))
        lVals = float(m) * 10**pow_vals
        lValsC = np.array([float(f"{l:.5f}") for l in lVals])
        if np.max(np.abs(lVals - lValsC)) < np.min(lVals) * 1e-3:
            lVals = lValsC
        lambdaVals.extend(list(lVals))
        strSpan = match.span()
        lambdaValStrs.extend([methodCodeCpy[strSpan[0] : strSpan[1]]] * pow_vals.size)
    for ls in lambdaValStrs:
        methodCodeCpy = methodCodeCpy.replace(ls, "")
    # Find individual lambda vals
    regex = f"{prefix}" + r"(\d+)+e([-+])?(\d+)+"  # L1e-2
    matches = re.finditer(regex, methodCodeCpy)
    for matchNum, match in enumerate(matches, start=1):
        m, sgn, power = match.groups()
        if sgn is not None and sgn == "-":
            power = -float(power)
        lVals = np.array([float(m) * 10 ** float(power)])
        lValsC = np.array([float(f"{l:.5f}") for l in lVals])
        if np.max(np.abs(lVals - lValsC)) < np.min(lVals) * 1e-3:
            lVals = lValsC
        lambdaVals.append(lVals[0])
        strSpan = match.span()
        lambdaValStrs.append(methodCodeCpy[strSpan[0] : strSpan[1]])
    for ls in lambdaValStrs:
        methodCodeCpy = methodCodeCpy.replace(ls, "")
    return lambdaVals, lambdaValStrs
