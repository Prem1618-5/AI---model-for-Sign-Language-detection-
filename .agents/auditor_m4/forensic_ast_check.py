import ast
import os
import sys

src_dir = r"d:\Development Project\Sign Language\src"
py_files = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if f.endswith('.py')]

print(f"Auditing {len(py_files)} files in {src_dir}...\n")

total_functions = 0
total_classes = 0
flagged_items = []

for path in sorted(py_files):
    fname = os.path.basename(path)
    with open(path, 'r', encoding='utf-8') as f:
        source = f.read()
    try:
        tree = ast.parse(source, filename=path)
        
        functions = []
        classes = []
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                else:
                    module = node.module or ""
                    imports.append(module)
                    
            elif isinstance(node, ast.FunctionDef):
                # check body length (excluding docstrings)
                body = [n for n in node.body if not (isinstance(n, ast.Expr) and isinstance(n.value, (ast.Str, ast.Constant)))]
                is_empty = len(body) == 0
                is_constant_return = False
                is_pass_only = len(body) == 1 and isinstance(body[0], ast.Pass)
                
                if len(body) == 1 and isinstance(body[0], ast.Return):
                    if isinstance(body[0].value, (ast.Constant, ast.Str, ast.Num)):
                        is_constant_return = True
                        
                functions.append({
                    'name': node.name,
                    'line': node.lineno,
                    'body_stmts': len(body),
                    'is_empty': is_empty,
                    'is_pass_only': is_pass_only,
                    'is_constant_return': is_constant_return
                })
                
            elif isinstance(node, ast.ClassDef):
                classes.append({
                    'name': node.name,
                    'line': node.lineno,
                    'methods': [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                })
                
        total_functions += len(functions)
        total_classes += len(classes)
        
        print(f"=== {fname} ===")
        print(f"  Classes ({len(classes)}): {[c['name'] for c in classes]}")
        print(f"  Functions/Methods ({len(functions)}): {[f['name'] for f in functions]}")
        print(f"  Imports: {sorted(set(imports))}")
        
        # Check suspicious patterns
        for fn in functions:
            if fn['is_empty']:
                print(f"  [FLAG] Empty function: {fn['name']} (line {fn['line']})")
                flagged_items.append((fname, fn['name'], "Empty function"))
            elif fn['is_pass_only']:
                print(f"  [FLAG] Pass-only function: {fn['name']} (line {fn['line']})")
                flagged_items.append((fname, fn['name'], "Pass-only function"))
            elif fn['is_constant_return']:
                print(f"  [NOTE] Constant return function: {fn['name']} (line {fn['line']})")
                
        print()
        
    except SyntaxError as e:
        print(f"  [SYNTAX ERROR] {fname}: {e}")
        flagged_items.append((fname, "FILE", f"SyntaxError: {e}"))

print(f"Audit Summary: Total files: {len(py_files)}, Classes: {total_classes}, Functions: {total_functions}")
print(f"Total Flags: {len(flagged_items)}")
