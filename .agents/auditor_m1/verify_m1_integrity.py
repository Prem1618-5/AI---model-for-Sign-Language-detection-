"""
Forensic Integrity Verification Script for Milestone M1
Analyzes AST, executes mathematical invariant checks, checks dependencies, and verifies CLI/camera tools.
"""

import ast
import os
import sys
import json
import glob
import subprocess
import numpy as np

PROJECT_ROOT = r"d:\Development Project\Sign Language"
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
TESTS_DIR = os.path.join(PROJECT_ROOT, "tests")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

results = {
    "ast_checks": {},
    "math_invariants": {},
    "parser_checks": {},
    "dependency_checks": {},
    "execution_checks": {},
    "unit_tests": {},
    "overall_clean": True
}


def log_check(category, check_name, passed, details=""):
    if category not in results:
        results[category] = {}
    results[category][check_name] = {
        "passed": bool(passed),
        "details": str(details)
    }
    if not passed:
        results["overall_clean"] = False
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] [{category}] {check_name}: {details}")


# ==========================================
# 1. AST & Static Code Analysis
# ==========================================
print("\n=== 1. AST & STATIC CODE ANALYSIS ===")

for filename in ["data_preprocessing.py", "main.py", "data_collection.py", "camera_test.py"]:
    filepath = os.path.join(SRC_DIR, filename)
    if not os.path.exists(filepath):
        log_check("ast_checks", f"file_exists_{filename}", False, f"File not found: {filepath}")
        continue
    
    with open(filepath, "r", encoding="utf-8") as f:
        source_code = f.read()
    
    try:
        tree = ast.parse(source_code, filename=filename)
        log_check("ast_checks", f"parse_ast_{filename}", True, "AST parsed successfully")
    except Exception as e:
        log_check("ast_checks", f"parse_ast_{filename}", False, f"SyntaxError/AST parse failed: {e}")
        continue

    # Check for forbidden imports (pandas, seaborn)
    forbidden_imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in ["pandas", "seaborn"]:
                    forbidden_imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module in ["pandas", "seaborn"]:
                forbidden_imports.append(node.module)
        elif isinstance(node, ast.Call):
            # Check dynamic imports: __import__('pandas') or importlib.import_module('pandas')
            if isinstance(node.func, ast.Name) and node.func.id == "__import__":
                if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value in ["pandas", "seaborn"]:
                    forbidden_imports.append(f"__import__({node.args[0].value})")
            elif isinstance(node.func, ast.Attribute) and node.func.attr == "import_module":
                if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value in ["pandas", "seaborn"]:
                    forbidden_imports.append(f"import_module({node.args[0].value})")

    log_check(
        "ast_checks",
        f"forbidden_imports_{filename}",
        len(forbidden_imports) == 0,
        f"Forbidden imports detected: {forbidden_imports}" if forbidden_imports else "No forbidden imports"
    )

    # Check for facade implementations (functions with only docstring and constant return)
    facades = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Ignore empty/trivial dunder or abstract methods if any
            body_statements = [s for s in node.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
            if len(body_statements) == 1 and isinstance(body_statements[0], ast.Return):
                ret_val = body_statements[0].value
                # If return value is a simple constant or literal without computation
                if isinstance(ret_val, (ast.Constant, ast.Dict, ast.List)) and not isinstance(ret_val, ast.Call):
                    # Flag if function looks like a real business logic function
                    if node.name not in ["__repr__", "__str__"]:
                        facades.append((node.name, ast.unparse(body_statements[0])))

    log_check(
        "ast_checks",
        f"facade_functions_{filename}",
        len(facades) == 0,
        f"Potential facade functions: {facades}" if facades else "No facade functions"
    )


# ==========================================
# 2. Mathematical Invariants & Preprocessing
# ==========================================
print("\n=== 2. MATHEMATICAL INVARIANTS & PREPROCESSING ===")

from data_preprocessing import parse_raw_sample, GestureDataProcessor

# Test parse_raw_sample on multiple schema variations
sample_empty = []
sample_single_flat = [{"x": float(i), "y": float(i*2), "z": 0.1, "visibility": 1.0} for i in range(21)]
sample_single_nested = [[{"x": float(i), "y": float(i*2), "z": 0.1, "visibility": 1.0} for i in range(21)]]
sample_two_nested = [
    [{"x": float(i), "y": float(i*2), "z": 0.1, "visibility": 1.0} for i in range(21)],
    [{"x": float(i+5), "y": float(i*3), "z": 0.2, "visibility": 1.0} for i in range(21)]
]

out_empty = parse_raw_sample(sample_empty)
log_check("parser_checks", "empty_sample", out_empty == [], f"Output: {out_empty}")

out_single_flat = parse_raw_sample(sample_single_flat)
log_check(
    "parser_checks",
    "single_flat_sample",
    len(out_single_flat) == 1 and len(out_single_flat[0]) == 21,
    f"Normalized 21 dicts to 1 hand of 21 dicts: shape=({len(out_single_flat)}, {len(out_single_flat[0]) if out_single_flat else 0})"
)

out_single_nested = parse_raw_sample(sample_single_nested)
log_check(
    "parser_checks",
    "single_nested_sample",
    len(out_single_nested) == 1 and len(out_single_nested[0]) == 21,
    f"Preserved 1-hand nested list: shape=({len(out_single_nested)}, {len(out_single_nested[0]) if out_single_nested else 0})"
)

out_two_nested = parse_raw_sample(sample_two_nested)
log_check(
    "parser_checks",
    "two_nested_sample",
    len(out_two_nested) == 2 and len(out_two_nested[0]) == 21 and len(out_two_nested[1]) == 21,
    f"Preserved 2-hand nested list: shape=({len(out_two_nested)}, 21)"
)

# Test mathematical normalization invariants
processor = GestureDataProcessor(data_dir=os.path.join(PROJECT_ROOT, "data", "raw"))

# Generate realistic hand landmarks
np.random.seed(42)
raw_hand = []
for i in range(21):
    raw_hand.append({
        "x": float(np.random.uniform(0.2, 0.8)),
        "y": float(np.random.uniform(0.2, 0.8)),
        "z": float(np.random.uniform(-0.1, 0.1)),
        "visibility": 1.0
    })

norm_hand = processor.normalize_landmarks(raw_hand)

# Invariant 1: Palm center (P0 + P9)/2 must equal (0, 0, 0)
p0 = np.array([norm_hand[0]["x"], norm_hand[0]["y"], norm_hand[0]["z"]])
p9 = np.array([norm_hand[9]["x"], norm_hand[9]["y"], norm_hand[9]["z"]])
palm_center = (p0 + p9) / 2.0
center_error = np.linalg.norm(palm_center)
log_check(
    "math_invariants",
    "palm_center_is_origin",
    center_error < 1e-6,
    f"Palm center residual norm: {center_error:.2e}"
)

# Invariant 2: Scale reference distance ||P9 - P0|| must equal 1.0
scale_dist = np.linalg.norm(p9 - p0)
log_check(
    "math_invariants",
    "unit_scale_reference",
    abs(scale_dist - 1.0) < 1e-5,
    f"Distance ||P9 - P0||: {scale_dist:.6f}"
)

# Invariant 3: Translation Invariance
offset_x, offset_y, offset_z = 10.5, -25.3, 7.8
translated_hand = [
    {
        "x": pt["x"] + offset_x,
        "y": pt["y"] + offset_y,
        "z": pt["z"] + offset_z,
        "visibility": pt["visibility"]
    }
    for pt in raw_hand
]
norm_translated = processor.normalize_landmarks(translated_hand)
trans_diffs = [
    max(abs(n["x"] - t["x"]), abs(n["y"] - t["y"]), abs(n["z"] - t["z"]))
    for n, t in zip(norm_hand, norm_translated)
]
max_trans_diff = max(trans_diffs)
log_check(
    "math_invariants",
    "translation_invariance",
    max_trans_diff < 1e-5,
    f"Max coordinate difference under translation: {max_trans_diff:.2e}"
)

# Invariant 4: Scale Invariance
scale_factor = 4.75
scaled_hand = [
    {
        "x": pt["x"] * scale_factor,
        "y": pt["y"] * scale_factor,
        "z": pt["z"] * scale_factor,
        "visibility": pt["visibility"]
    }
    for pt in raw_hand
]
norm_scaled = processor.normalize_landmarks(scaled_hand)
scale_diffs = [
    max(abs(n["x"] - s["x"]), abs(n["y"] - s["y"]), abs(n["z"] - s["z"]))
    for n, s in zip(norm_hand, norm_scaled)
]
max_scale_diff = max(scale_diffs)
log_check(
    "math_invariants",
    "scale_invariance",
    max_scale_diff < 1e-5,
    f"Max coordinate difference under scaling: {max_scale_diff:.2e}"
)

# Invariant 5: Zero-division defense
zero_hand = [{"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 1.0} for _ in range(21)]
norm_zero = processor.normalize_landmarks(zero_hand)
has_nan_inf = any(
    np.isnan(pt["x"]) or np.isinf(pt["x"]) or np.isnan(pt["y"]) or np.isinf(pt["y"])
    for pt in norm_zero
)
log_check(
    "math_invariants",
    "zero_division_guard",
    not has_nan_inf,
    "Normalized degenerate hand without NaN/Inf" if not has_nan_inf else "Found NaN/Inf in degenerate hand"
)


# ==========================================
# 3. Dependency Verification
# ==========================================
print("\n=== 3. DEPENDENCY VERIFICATION ===")

req_path = os.path.join(PROJECT_ROOT, "requirements.txt")
with open(req_path, "r", encoding="utf-8") as f:
    req_lines = [line.strip().lower() for line in f.readlines() if line.strip() and not line.startswith("#")]

has_pandas_in_req = any("pandas" in line for line in req_lines)
has_seaborn_in_req = any("seaborn" in line for line in req_lines)

log_check(
    "dependency_checks",
    "pandas_removed_from_requirements",
    not has_pandas_in_req,
    f"pandas in requirements.txt: {has_pandas_in_req}"
)
log_check(
    "dependency_checks",
    "seaborn_removed_from_requirements",
    not has_seaborn_in_req,
    f"seaborn in requirements.txt: {has_seaborn_in_req}"
)


# ==========================================
# 4. CLI & Camera Execution Checks
# ==========================================
print("\n=== 4. CLI & CAMERA EXECUTION CHECKS ===")

py_exe = sys.executable

# Test main.py --help
res = subprocess.run([py_exe, os.path.join(SRC_DIR, "main.py"), "--help"], capture_output=True, text=True, cwd=PROJECT_ROOT)
log_check(
    "execution_checks",
    "cli_main_help",
    res.returncode == 0 and "Sign Language Detection ML System" in res.stdout,
    f"Exit code: {res.returncode}"
)

# Test camera_test.py --headless --duration 0.5
res = subprocess.run([py_exe, os.path.join(SRC_DIR, "camera_test.py"), "--headless", "--duration", "0.5"], capture_output=True, text=True, cwd=PROJECT_ROOT)
# camera_test returns 0 on success or 1 if no camera device is connected, but should execute without unhandled exception/crash
is_camera_valid = (res.returncode in (0, 1)) and ("Testing camera access" in res.stdout)
log_check(
    "execution_checks",
    "camera_test_execution",
    is_camera_valid,
    f"Exit code: {res.returncode}, Output: {res.stdout.strip()[:100]}"
)


# ==========================================
# 5. Unit Tests
# ==========================================
print("\n=== 5. UNIT TEST EXECUTION ===")

res = subprocess.run([py_exe, os.path.join(TESTS_DIR, "run_tests.py"), "--tier", "1"], capture_output=True, text=True, cwd=PROJECT_ROOT)
log_check("unit_tests", "tier_1_tests", res.returncode == 0, f"Exit code: {res.returncode}")

res = subprocess.run([py_exe, os.path.join(TESTS_DIR, "run_tests.py"), "--tier", "3"], capture_output=True, text=True, cwd=PROJECT_ROOT)
log_check("unit_tests", "tier_3_tests", res.returncode == 0, f"Exit code: {res.returncode}")

res = subprocess.run([py_exe, os.path.join(TESTS_DIR, "run_tests.py"), "--tier", "4"], capture_output=True, text=True, cwd=PROJECT_ROOT)
log_check("unit_tests", "tier_4_tests", res.returncode == 0, f"Exit code: {res.returncode}")


print(f"\n==========================================")
print(f"VERDICT: {'CLEAN' if results['overall_clean'] else 'INTEGRITY VIOLATION'}")
print(f"==========================================")

with open(os.path.join(r"d:\Development Project\Sign Language\.agents\auditor_m1", "audit_evidence.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
