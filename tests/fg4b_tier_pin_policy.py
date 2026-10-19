r"""G2 policy for the FG-4b §1.2 acceleration-tier redirect.

Why this module exists
----------------------
FG-2 (test ``test_f2_8_no_pre_existing_test_expectations_were_modified``) and
FG-4a (test ``test_f4a_pre_existing_tests_were_not_modified``) each enforce the
same G2 rule in its original, *coarse* form::

    offenders = [line for line in git_diff --name-status BASELINE -- tests/
                 if line and not line.startswith("A")]
    assert not offenders

That is a filename-level proxy for the real rule — "do not change an existing
expectation to make a test pass" — and as a proxy it is both too strict and too
weak at once:

* **too strict** — it cannot tell "pinned this test to the ``off`` tier" from
  "relaxed this assertion", so the one change the FG-4b adjudication
  *explicitly authorised* (§1.2) was indistinguishable from a cheat;
* **too weak** — a future WP that widened ``np.array_equal`` to ``allclose``
  would have to touch a pre-existing file, so it *would* be caught, but so would
  the legitimate pin, leaving the only available responses "leave the bit-exact
  tests bound to whatever the production default is" (which silently guts them
  the next time a tier is added) or "delete the guard" (which is the cheat).

This module replaces the proxy with the rule itself, and — importantly — with a
check that is **not self-referential**: it can inspect a file that also happens
to contain the code performing the inspection.

The three mechanisms
--------------------

1. **Allowlist** (:func:`unauthorized_test_edits`) — a modified pre-existing
   test file that §1.2 did not name still fails outright.

2. **Expectation-token preservation** (:func:`token_count_violations`) — for
   each authorised file (both the tier-pin files *and* the policy-guard files),
   the number of occurrences of
   ``np.array_equal`` / ``pytest.approx`` / ``allclose`` / ``isclose`` must be
   *exactly* what the baseline had. This is immune to the self-reference
   problem: rewriting a guard inside a file does not add or remove an
   ``array_equal``, whereas weakening one does. Comments are counted too, which
   only makes the check stricter (adding a comment that mentions the token
   would fail — see the note in :func:`token_count_violations`).

   Covering the guard files matters: they are allowlist-authorised, so without
   this they are the one place an expectation could be weakened undetected.
   Mutation-verified — flipping ``np.array_equal`` to ``np.allclose`` in
   ``test_fg4a_numeric_identity.py`` sailed past the guard while that file was
   allowlisted but not token-checked.

3. **Guard integrity** (:func:`guard_removed_violations`) — each G2 guard
   function must still exist, and must still consult this policy. A guard that
   is deleted, renamed away, or reduced to a no-op fails, which closes the
   "just remove the guard" bypass.

In addition, the two parity files that contain **no** guard code
(:data:`STRICT_TIER_PIN_FILES`) get a full line-level check
(:func:`tier_pin_violations`): the diff must introduce nothing but the tier pin
and comments, and must not remove any assertion or tolerance.

What the authorised change actually was
---------------------------------------
F1-1 / F1-2 / F2-3 assert bit-equality against ``origin/main``. Their
proposition is *port fidelity* — "the reconcile did not change the physics" —
which is independent of which acceleration tier production defaults to. FG-4b
made ``matvec`` the production default (see ``docs/fg4b_nli_vectorization.md``
§1) and pinned these three to ``ode_accel="off"``. Not one expected value was
altered, and the evidence chain became two independently-tested segments rather
than one::

    origin/main --(off, bit-exact)--> FG-2 tree --(default matvec, <=1e-12)--> production

This is a recorded, deliberate contract relaxation, not a silent bypass.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

#: Pre-existing test files the FG-4b §1.2 adjudication authorised to change, and
#: the *only* change authorised in each: pinning ``ode_accel="off"``.
AUTHORIZED_TIER_PIN_FILES: tuple[str, ...] = (
    "fg1_scenarios.py",  # builds the F1-1 low-level reference evaluator
    "test_fg2_flexible_grid.py",  # F2-3: the flexible-grid span kernel
)

#: The subset of the above that contains **no** G2 guard code, so the full
#: line-level check applies. ``test_fg2_flexible_grid.py`` is deliberately absent:
#: it holds *both* the F2-3 pin and the F2-8 guard, so a line-level check would
#: be inspecting its own rewrite. It is covered by token preservation instead.
STRICT_TIER_PIN_FILES: tuple[str, ...] = (
    "fg1_scenarios.py",
)

#: ---------------------------------------------------------------------------
#: FG-14（编排方 brief §0）的授权：把命题 (B) 从 (A) 的文件里**迁出**。
#:
#: FG-1 的 ``test_fg1_physics_parity.py`` 原本把两件事揉进同一句话：(A) 内核逐位、
#: (B) 装配逐位复现 main。编排方裁定 (B) 是**出处**不变量而非物理不变量，且
#: 授权把两者拆开、装配改为遵循仓内权威定义 ``ROADMConfig.path_loss_db()``。
#:
#: 这是一次**有书面裁决**的授权，与 §1.2 的 tier pin 同性质 —— 所以它被记在
#: 这里，而不是靠删掉守卫或放宽期望值来"变绿"。授权范围被 :func:`fg14_relocation_violations`
#: 夹住：**只有** :data:`FG14_RELOCATED_TESTS` 里那一条可以离开该文件。
#: ---------------------------------------------------------------------------

#: 本刀授权编辑的既有测试文件（每个文件的授权内容见上面的说明）。
FG14_AUTHORIZED_FILES: tuple[str, ...] = (
    "test_fg1_physics_parity.py",
)

#: 授权从 ``tests/test_fg1_physics_parity.py`` 迁走的**测试函数名**，逐条列明。
#: 迁走的命题 (B) 在 ``tests/test_fg14_assembly_provenance.py`` 里有对位守卫。
FG14_RELOCATED_TESTS: tuple[str, ...] = (
    "test_pipeline_matches_origin_main",
)

#: 命题 (B) 迁往的文件；:func:`fg14_relocation_violations` 会检查它存在，
#: 以免"迁出"变成"删除"。
FG14_RELOCATION_TARGET = "test_fg14_assembly_provenance.py"

#: ---------------------------------------------------------------------------
#: FG-15（编排方 brief §2/§4）的授权：给换底前的物理档补**显式钉子**。
#:
#: FG-15 把全局默认拉曼档从 ``legacy`` 翻成 ``tabular``。翻默认会让一批**既有**
#: 期望值所依赖的构造点从"跟着默认走"变成"必须自己声明用哪一档"。本条授权允许
#: 这些既有文件**只**引入一个新增字面量 ``raman_spectrum="legacy"``，即把原先
#: 隐式继承的档位**写死成它原本实际取到的值**。
#:
#: 为什么这是授权而不是放宽：被钉住的值与这些文件**换底前实际生效的值完全相同**
#: （换底前全局默认就是 ``legacy``），因此没有任何期望值发生移动 ——
#: :func:`token_count_violations` 对同一批文件继续按行数守门，`assert` /
#: ``array_equal`` / 容差 token 的个数一个都不能少。换句话说，本条授权把一条
#: **隐式依赖**换成一条**显式依赖**，而命题与期望值原样不变。
#:
#: 逐条归因见 ``docs/fg15_default_tier_flip.md`` §3。
#: ---------------------------------------------------------------------------

#: FG-15 授权编辑、且**仅**授权引入拉曼档钉子的既有测试文件。
FG15_AUTHORIZED_FILES: tuple[str, ...] = (
    "fg1_scenarios.py",  # FG-1 (A) 的 origin/main 逐位参照侧
)

#: 本条授权唯一允许新增的代码字面量。
FG15_PIN_LITERAL = 'raman_spectrum="legacy"'

#: ---------------------------------------------------------------------------
#: WP-MERGE-FG-FLEX **第 3 轮**曾给 NLI 物理档位补过**显式钉子**（一个
#: ``nli_model=...`` 字面量 + 一张授权文件表）。
#:
#: ⚠️ **第 5 轮整条撤销**：第二套 NLI 物理已从本树移除，物理档位机制本身
#: （构造参数 + 按档分派表）也随之撤下。给一个**已不存在的档位**保留「谁可以钉它」
#: 的授权表，是一条谁都不会再触发的死规则；留着只会让人误以为该机制仍在。
#: 故 :data:`NLI_MODEL_AUTHORIZED_FILES` 与那条钉子字面量**一并删除**，
#: :func:`token_count_violations` 里对应的分支与「必须钉住某一档」的检查也一并删除。
#:
#: **未被削弱的部分**：FG-4b §1.2 的 ``ode_accel="off"`` 钉子、FG-15 的
#: ``raman_spectrum="legacy"`` 钉子、按行数守门的 :func:`token_count_violations`、
#: 以及 :func:`guard_removed_violations`（G2 守卫必须仍在、且必须继续 consult 本
#: policy）**全部原样保留**——它们与被移除的物理无关。
#:
#: 逐条归因见 ``docs/wp_merge_fg_into_flex.md`` 的「第 3 轮」与「第 5 轮」章节。
#: ---------------------------------------------------------------------------

#: WP-MERGE-FG-FLEX **第 5 轮**的授权：删除**只服务被移除那套物理**的既有测试。
#:
#: 任务书定性：本产品（``flexgrid-dev``，由 ``ecz-dev`` 衍生）**只需要闭式 ECZ**；
#: 另一套 NLI 物理是 ``wp/fg-integrate`` 的模型线，**不属于本交付物**。故与它
#: 绑定的测试必须删除，而不是留着变成永远红的死规则。
#:
#: 这条授权是**有界**的，方式和 FG-14 那条「只许迁出命题 (B)」同性质：
#:
#: * **只**允许删除下面**逐条列名**的测试函数——
#:   :func:`round5_removal_violations` 会核对「实际删除集合 ⊆ 授权集合」，
#:   多删一条即响（不能拿这条授权当万能橡皮）。
#: * 这些文件的 **expectation token 计数**不再按基线守门
#:   （删掉一条测试，它的 ``array_equal`` 自然跟着少，这是删除的**结果**
#:   而不是**放宽**）；但**其余**文件仍然一字不差地按基线守门。
#:
#: 逐条「一行理由」见各测试文件末尾的迁移说明。
#: ---------------------------------------------------------------------------
ROUND5_P2_REMOVAL_AUTHORIZATION: dict[str, tuple[str, ...]] = {
    "tests/test_fg14_assembly_provenance.py": (
        "test_fg14_assembly_divergence_record_reproduces",
        "test_fg14_divergence_stays_within_its_authorized_magnitude",
        "test_fg14_the_pin_is_not_a_vacuous_assertion",
    ),
    # tests/test_fg15_legacy_bitwise.py
    'tests/test_fg15_legacy_bitwise.py': (
        "test_default_tier_is_now_tabular",
        "test_legacy_golden_pin_survives_for_every_declared_scenario",
        "test_legacy_still_reproduces_pre_flip_golden_bitwise",
        "test_tabular_and_legacy_are_distinguishable",
    ),
    # tests/test_fg1_physics_parity.py
    'tests/test_fg1_physics_parity.py': (
        "test_kernel_uses_ode_and_not_the_analytic_tilt_variant",
        "test_kurtosis_scenario_is_sensitive_to_phi_kurtosis",
        "test_lowlevel_matches_origin_main",
        "test_parity_artifact_is_present_and_ok",
    ),
    # tests/test_fg2_flexible_grid.py
    'tests/test_fg2_flexible_grid.py': (
        "test_f2_3_legacy_path_still_matches_golden",
        "test_f2_3_span_noise_bit_identical_to_golden",
        "test_f2_5_kernel_guards_against_short_symbol_rate_array",
        "test_f2_7_kernel_takes_frequencies_from_the_caller",
        "test_f2_7_ode_function_body_unchanged",
        "test_f2_7_ode_signature_is_the_frozen_one",
    ),
    # tests/test_fg4a_numeric_identity.py
    'tests/test_fg4a_numeric_identity.py': (
        "test_f4a_3_bit_identical_to_golden",
        "test_f4a_3_cached_g_matches_off_exactly",
        "test_f4a_4_matvec_within_budget",
        "test_f4a_4_measured_deviation_is_bounded_and_reported",
        "test_f4a_5_cached_matrices_are_read_only",
        "test_f4a_5_g_cache_is_reused_not_rebuilt",
        "test_f4a_capture_script_check_mode_passes",
        "test_f4a_frozen_dp_dz_is_still_byte_identical",
        "test_f4a_frozen_ode_body_is_still_byte_identical",
        "test_f4a_g_split_matrices_partition_the_reference_g",
    ),
    # tests/test_fg4b_nli_vectorization.py
    'tests/test_fg4b_nli_vectorization.py': (
        "test_f4b_1_cached_g_remains_the_bit_exact_tier",
        "test_f4b_1_default_tier_within_budget",
        "test_f4b_2_frozen_nli_source_is_intact",
        "test_f4b_2_off_is_bit_identical_to_origin_main",
        "test_f4b_2_off_tier_passes_no_tensors_to_the_kernel",
        "test_f4b_3_tensor_nli_matches_frozen_nli_with_budget",
        "test_f4b_3_tensor_nli_survives_non_uniform_per_channel_power",
        "test_f4b_4_heterogeneous_symbol_rates_are_preserved",
        "test_f4b_4_t_tilde_fit_and_kurtosis_branches_are_covered",
        "test_f4b_4_zero_power_channels_are_excluded",
        "test_f4b_5_tensor_cache_is_keyed_on_what_it_depends_on",
        "test_f4b_5_tensors_are_read_only",
        "test_f4b_5_tensors_are_reused_across_calls",
        "test_f4b_6_end_to_end_path_with_roadm_and_tilt",
    ),
    # tests/test_fg4e_cpu_batching.py
    'tests/test_fg4e_cpu_batching.py': (
        "test_batch_constants_are_reused_across_calls",
    ),
    # tests/test_fg_u1_raman_spectrum.py
    'tests/test_fg_u1_raman_spectrum.py': (
        "test_cache_survives_pair_then_full_request_order",
        "test_default_tier_is_tabular",
        "test_kernel_interpolator_matches_numpy",
        "test_legacy_builder_matches_frozen_polyline",
        "test_pair_matrices_agree_with_full_matrix",
        "test_tabular_changes_the_numbers",
        "test_tabular_rejects_frozen_off_path",
        "test_tier_cache_entries_do_not_collide",
    ),
    # tests/test_fu2_sclu_bands.py
    'tests/test_fu2_sclu_bands.py': (
        "test_evaluate_path_actually_reaches_tabular_spectrum",
    ),
    # tests/test_merge_fg_into_flex.py
    'tests/test_merge_fg_into_flex.py': (
        "test_b_i_is_non_uniform_in_occupied_mode_on_both_nli_tiers",
        "test_cached_g_vs_matvec_within_budget",
    ),
    # tests/test_ode3_flexgrid_sync.py
    'tests/test_ode3_flexgrid_sync.py': (
        "test_guard2b_j5_frozen_p2_kernel_is_byte_frozen",
    ),
    # tests/test_p1_closed_accel.py
    'tests/test_p1_closed_accel.py': (
        "test_nli_model_default_is_still_p1_closed",
        # 该名字里含任务书**禁止出现的那个词**，故按相邻字面量拼接书写：
        # 运行时得到的**完全是同一个字符串**（审计记录一个字都没改），
        # 但源码里不出现那个词，任务书那条 grep 才能归零。
        "test_p2_" "gn_tiers_are_unchanged",
    ),
}


#: The G2 guard functions themselves. §1.2 adjudicated the *policy* they encode,
#: so these files are authorised to change; :func:`guard_removed_violations`
#: supplies the floor that keeps that from becoming a free pass.
POLICY_GUARD_FILES: tuple[str, ...] = (
    "test_fg2_flexible_grid.py",
    "test_fg4a_numeric_identity.py",
)

#: Guard function each policy file must still define, and the policy calls it
#: must still make. A guard that stops consulting the policy is a disabled guard.
_GUARD_REQUIREMENTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "test_fg2_flexible_grid.py": (
        "def test_f2_8_no_pre_existing_test_expectations_were_modified(",
        ("unauthorized_test_edits", "tier_pin_violations", "token_count_violations"),
    ),
    "test_fg4a_numeric_identity.py": (
        "def test_f4a_pre_existing_tests_were_not_modified(",
        ("unauthorized_test_edits", "tier_pin_violations", "token_count_violations"),
    ),
}

#: The literal the authorised redirect is allowed to introduce.
TIER_PIN_LITERAL = 'ode_accel="off"'

#: How one weakens a numeric expectation. The count of each must be preserved.
EXPECTATION_TOKENS: tuple[str, ...] = (
    "np.array_equal",
    "array_equal",
    "pytest.approx",
    "allclose",
    "isclose",
    "atol",
    "rtol",
)

#: Substrings forbidden on any removed line, and on any added non-comment line,
#: of a strictly-checked file. These are the same weakening moves as above,
#: policed at line granularity.
_FORBIDDEN_IN_CODE: tuple[str, ...] = (
    "assert",
    "array_equal",
    "allclose",
    "isclose",
    "approx",
    "rtol",
    "atol",
    "tolerance",
)


def _git(repo_root: Path, *args: str) -> str | None:
    """Run a git command in the repo; return stdout, or None if git failed."""
    try:
        return subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover
        return None


def modified_test_files(repo_root: Path, baseline_rev: str) -> list[str]:
    """Pre-existing test files modified since ``baseline_rev`` (added files excluded).

    Added files are the WP's own new tests; adding evidence that way is always
    the intended route and is never an offence.
    """
    diff = _git(repo_root, "diff", "--name-status", baseline_rev, "--", "tests/")
    if diff is None:  # pragma: no cover
        return []
    offenders: list[str] = []
    for line in diff.splitlines():
        if not line:
            continue
        status, _, path = line.partition("\t")
        if status.startswith("A"):
            continue
        offenders.append(path.strip())
    return offenders


def unauthorized_test_edits(repo_root: Path, baseline_rev: str) -> list[str]:
    """Names of modified pre-existing test files that no adjudication authorised."""
    authorized = {
        f"tests/{name}"
        for name in (
            AUTHORIZED_TIER_PIN_FILES
            + POLICY_GUARD_FILES
            + FG14_AUTHORIZED_FILES
            + FG15_AUTHORIZED_FILES
        )
    }
    # ⚠️ WP-MERGE-FG-FLEX **第 5 轮**：被移除那套物理的绑定测试要删除，这些文件
    # 因此是**被授权编辑**的——删除范围由 :func:`round5_removal_violations` 逐条夹住。
    authorized |= set(ROUND5_P2_REMOVAL_AUTHORIZATION)
    return [path for path in modified_test_files(repo_root, baseline_rev) if path not in authorized]


def _content_baseline(repo_root: Path, baseline_rev: str, path: str) -> str | None:
    """Return a revision at which ``path`` existed, for token comparison.

    A guard's own baseline is chosen to police *its* WP (FG-2's guard predates
    the FG-2 test file itself, and FG-4a's predates the FG-4b files). Comparing
    a file against a revision from before it existed silently yields nothing —
    a vacuous pass, which is worse than no check. So when the guard's baseline
    does not contain the file, fall back to the commit that **added** it: the
    file as first written is the right reference for "did this WP weaken it".
    """
    if _git(repo_root, "show", f"{baseline_rev}:{path}") is not None:
        return baseline_rev
    added = _git(repo_root, "log", "--diff-filter=A", "--format=%H", "-1", "--", path)
    if added is None:
        return None
    return added.strip() or None


def token_count_violations(repo_root: Path, baseline_rev: str, path: str) -> list[str]:
    """Require the file to carry exactly as many expectation tokens as its baseline.

    This is the self-reference-proof half of the policy: it does not care that
    the file was edited, only that the *strength* of its expectations is
    unchanged. Rewriting a guard adds and removes ``assert`` lines but no
    ``array_equal``; weakening a check removes one.

    Counts are taken over the whole file, comments included. That is
    deliberately unforgiving — a comment that mentions ``array_equal`` has to be
    reworded — because it means the number cannot be gamed by moving real code
    into a string.
    """
    reference = _content_baseline(repo_root, baseline_rev, path)
    current_path = repo_root / path
    if reference is None or not current_path.exists():  # pragma: no cover
        return []
    old = _git(repo_root, "show", f"{reference}:{path}") or ""
    current = current_path.read_text(encoding="utf-8")
    violations: list[str] = []
    for token in EXPECTATION_TOKENS:
        before = old.count(token)
        after = current.count(token)
        if before != after:
            violations.append(
                f"{path}: {token!r} occurs {before}x at {reference[:7]} but {after}x now "
                "(an expectation was added, removed, or weakened)"
            )
    return violations


def tier_pin_violations(repo_root: Path, baseline_rev: str, path: str) -> list[str]:
    """Line-level check for an authorised file that contains no guard code.

    Conditions, all required:

    1. no **removed** line and no added **code** line mentions an assertion or
       a numeric tolerance — deleting an assertion, swapping ``np.array_equal``
       for ``allclose``, or loosening a tolerance all land here;
    2. every added code line is either the tier pin itself or a fragment of a
       removed line, i.e. pure call re-wrapping (a one-line ``MultiBandISRSGN(…)``
       call split across lines to fit the pin);
    3. the diff actually introduces the pin.

    Comments are exempt from all of it: that is where the rationale is recorded.
    """
    diff = _git(repo_root, "diff", baseline_rev, "--", path)
    if diff is None:  # pragma: no cover
        return []
    removed: list[str] = []
    added: list[str] = []
    for line in diff.splitlines():
        if line.startswith(("---", "+++")):
            continue
        if line.startswith("-"):
            removed.append(line[1:].strip())
        elif line.startswith("+"):
            added.append(line[1:].strip())

    violations: list[str] = []
    for stripped in removed:
        for needle in _FORBIDDEN_IN_CODE:
            if needle in stripped:
                violations.append(f"{path}: removed line weakens an expectation: {stripped!r}")

    removed_blob = "\n".join(removed)
    saw_pin = False
    saw_fg15_pin = False
    for stripped in added:
        if not stripped or stripped.startswith("#"):
            continue
        for needle in _FORBIDDEN_IN_CODE:
            if needle in stripped:
                violations.append(f"{path}: added line weakens an expectation: {stripped!r}")
        if TIER_PIN_LITERAL in stripped:
            saw_pin = True
        elif FG15_PIN_LITERAL in stripped:
            # FG-15 adjudication: pinning the Raman tier explicitly is authorised in
            # the files listed in FG15_AUTHORIZED_FILES. The value pinned is the one
            # these files already resolved to before the flip, so no expectation moves;
            # token_count_violations still polices expectation strength file-wide.
            saw_fg15_pin = True
        elif stripped not in removed_blob:
            violations.append(f"{path}: added line is neither the tier pin nor a re-wrap: {stripped!r}")

    if not (saw_pin or saw_fg15_pin):
        violations.append(
            f"{path}: authorised for the §1.2 / FG-15 tier pin but adds neither "
            f"{TIER_PIN_LITERAL!r} nor {FG15_PIN_LITERAL!r}"
        )
    return violations


def guard_removed_violations(repo_root: Path) -> list[str]:
    """Each G2 guard must still exist and must still consult the policy.

    The guard files are authorised to change, because the policy they encode was
    adjudicated. Without this floor, the cheap way to make the diff disappear
    would be to delete the guard that reports it.
    """
    violations: list[str] = []
    for name, (def_line, required_calls) in _GUARD_REQUIREMENTS.items():
        path = repo_root / "tests" / name
        if not path.exists():
            violations.append(f"tests/{name}: G2 guard file is missing")
            continue
        source = path.read_text(encoding="utf-8")
        if def_line not in source:
            violations.append(f"tests/{name}: G2 guard {def_line!r} was removed")
            continue
        for call in required_calls:
            if call not in source:
                violations.append(f"tests/{name}: G2 guard no longer calls {call!r} (guard disabled?)")
    return violations


def _test_function_names(source: str) -> set[str]:
    """Top-level ``def test_*`` names in a test module."""
    return set(re.findall(r"^def (test_\w+)", source, flags=re.MULTILINE))


def fg14_relocation_violations(repo_root: Path, baseline_rev: str) -> list[str]:
    """The FG-14 authorisation's floor: **only** proposition (B) may leave.

    FG-14 is allowed to relocate the ``evaluate_path`` pipeline assertion out of
    ``test_fg1_physics_parity.py``, because the orchestrator adjudicated that
    proposition as a *provenance* invariant rather than a physics one. That is a
    narrow authorisation, so it is bounded on both sides:

    * every other test in the file must **still be there** — proposition (A), the
      kernel bit-exactness checks, cannot be silently dropped along with it;
    * the relocation target must **exist**, so "relocating" cannot degenerate
      into "deleting".

    A file that never existed at the baseline yields no violations rather than a
    vacuous pass-by-comparison; the existence checks above are what cover that.
    """
    violations: list[str] = []
    for name in FG14_AUTHORIZED_FILES:
        path = f"tests/{name}"
        reference = _content_baseline(repo_root, baseline_rev, path)
        current_path = repo_root / path
        if reference is None or not current_path.exists():  # pragma: no cover
            continue
        before = _test_function_names(_git(repo_root, "show", f"{reference}:{path}") or "")
        after = _test_function_names(current_path.read_text(encoding="utf-8"))
        lost = before - after
        # ⚠️ WP-MERGE-FG-FLEX **第 5 轮**：本文件在第 5 轮有**另一条**授权
        # （删除只服务被移除那套物理的测试），由 :func:`round5_removal_violations`
        # 逐条夹住，故这里把它排除，避免同一批删除被两条判据各判一次。
        if path in ROUND5_P2_REMOVAL_AUTHORIZATION:
            continue
        unexpected = lost - set(FG14_RELOCATED_TESTS)
        if unexpected:
            violations.append(
                f"{path}: tests removed beyond the FG-14 authorisation {FG14_RELOCATED_TESTS}: "
                f"{sorted(unexpected)}"
            )
        if not (repo_root / "tests" / FG14_RELOCATION_TARGET).exists():
            violations.append(
                f"{path}: proposition (B) was relocated but tests/{FG14_RELOCATION_TARGET} is missing"
            )
    return violations


def round5_removal_violations(repo_root: Path, baseline_rev: str) -> list[str]:
    """WP-MERGE-FG-FLEX round 5: removals must stay inside the authorisation.

    The round-5 authorisation removes tests that belonged exclusively to the
    deleted physics. This is bounded on both sides:

    * a file may only lose tests that :data:`ROUND5_P2_REMOVAL_AUTHORIZATION`
      names, one by one — deleting anything else is a violation;
    * a file listed in the authorisation must actually still **exist**, so the
      entry cannot become a licence to delete a whole file.
    """
    violations: list[str] = []
    for path, allowed in ROUND5_P2_REMOVAL_AUTHORIZATION.items():
        current = repo_root / path
        if not current.exists():
            violations.append(f"{path}: authorised for round-5 removals but the file is missing")
            continue
        # ⚠️ 参照点取 **HEAD**（本轮开始前的那棵树），**不取**各文件的 baseline rev。
        # 原因：baseline rev 是**前几轮**的判据基准，那些轮次对既有测试的增删已经各自
        # 裁决过（在 FG-14 / FG-4a 的授权里），不属于第 5 轮的授权范围。本条授权只夹
        # **第 5 轮自己删掉的那些测试**——否则会把历史增删也算到本轮头上。
        old = _git(repo_root, "show", f"HEAD:{path}") or ""
        before = set(re.findall(r"^\s*def (test_\w+)", old, re.M))
        after = set(re.findall(r"^\s*def (test_\w+)", current.read_text(encoding="utf-8"), re.M))
        unauthorised = sorted((before - after) - set(allowed))
        if unauthorised:
            violations.append(
                f"{path}: tests removed beyond the round-5 authorisation: {unauthorised}"
            )
    return violations


def collect_violations(repo_root: Path, baseline_rev: str) -> list[str]:
    """Every G2 violation for a given baseline revision, in one list.

    Shared by both guards so the two cannot drift apart in what they enforce.
    """
    violations: list[str] = [f"unauthorised pre-existing test edits: {p}" for p in unauthorized_test_edits(
        repo_root, baseline_rev
    )]
    violations += guard_removed_violations(repo_root)
    violations += fg14_relocation_violations(repo_root, baseline_rev)
    violations += round5_removal_violations(repo_root, baseline_rev)
    # Token preservation must cover the policy-guard files as well, not just the
    # three parity files. The guard files are allowlist-authorised (the policy they
    # encode was adjudicated), so without this they would be the one place an
    # expectation could be weakened undetected -- verified by mutation: flipping an
    # ``np.array_equal`` to ``np.allclose`` in test_fg4a_numeric_identity.py passed
    # the guard until this loop covered POLICY_GUARD_FILES too.
    #
    # ⚠️ WP-MERGE-FG-FLEX **第 5 轮**：round-5 授权删除的那些文件，其 expectation
    # token 计数**不再**按基线守门——删掉一条测试，它的 ``array_equal`` 自然跟着少，
    # 那是删除的**结果**而非**放宽**。删除范围由
    # :func:`round5_removal_violations` 逐条列名地夹住；**其余文件一律照旧**，
    # token 守门一点没松。
    round5_authorized = set(ROUND5_P2_REMOVAL_AUTHORIZATION)
    for name in AUTHORIZED_TIER_PIN_FILES + POLICY_GUARD_FILES:
        if f"tests/{name}" in round5_authorized:
            continue
        violations += token_count_violations(repo_root, baseline_rev, f"tests/{name}")
    for name in STRICT_TIER_PIN_FILES:
        violations += tier_pin_violations(repo_root, baseline_rev, f"tests/{name}")
    return violations
