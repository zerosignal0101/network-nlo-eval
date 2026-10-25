"""重捕 tests/test_carena2014.py 里的单跨黄金值（MCI 积分域/积分口径修复之后）。

只做两件事：算出新值、按原有排版重写数组字面量。不改任何断言形式。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from network_nlo_eval.benchmarks.cases import build_case, build_path, launch_profile
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.models.base import MultiSpanOpticalPath

TEST = Path(__file__).resolve().parents[1] / "tests" / "test_carena2014.py"

case = build_case("c_57ch")
path = build_path(case)
model = make_model("carena2014", case)
power = launch_profile(case, 0.0)

one_span = MultiSpanOpticalPath([path.fiber_configs[0]], [path.edfa_configs[0]])
result = model.evaluate_path(one_span, power)

kernel = model.kernel_for(path.fiber_configs[0])
out = kernel.evaluate(power, 100e3, 1)

values = {
    "GOLDEN_ONE_SPAN_SNR_DB": np.asarray(result.snr_db),
    "GOLDEN_ONE_SPAN_SPM_W": np.asarray(result.spm_noise_w),
    "GOLDEN_ONE_SPAN_XPM_W": np.asarray(result.xpm_noise_w),
    "GOLDEN_ONE_SPAN_KERNEL_SPM_W": np.asarray(out.spm_noise_w),
    "GOLDEN_ONE_SPAN_KERNEL_XPM_W": np.asarray(out.xpm_noise_w),
}


def render(name: str, arr: np.ndarray) -> list[str]:
    """把数组渲染成测试文件里的 ``NAME = np.array([...])`` 字面量行。"""
    lines = [f"{name} = np.array(", "    ["]
    row: list[str] = []
    for value in arr:
        row.append(repr(float(value)))
        if len(row) == 4:
            lines.append("        " + ", ".join(row) + ",")
            row = []
    if row:
        lines.append("        " + ", ".join(row) + ",")
    lines += ["    ]", ")", ""]
    return lines


text = TEST.read_text(encoding="utf-8").splitlines()
for name, arr in values.items():
    start = next(i for i, line in enumerate(text) if line.startswith(f"{name} = np.array("))
    end = next(i for i in range(start + 1, len(text)) if text[i].strip() == ")")
    text[start : end + 1] = render(name, arr)
    print(f"{name}: {len(arr)} values rewritten")

TEST.write_text("\n".join(text) + "\n", encoding="utf-8")
print("done")
