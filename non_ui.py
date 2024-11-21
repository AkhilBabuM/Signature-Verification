import os
import ast
from pathlib import Path

BASE_DIR = Path(__file__).parent
SCRIPT_PATH = BASE_DIR / 'non_uifiles.py'

def get_imports(file_path):
    """Extract imports from a Python file."""
    imports = set()
    try:
        with open(file_path, "r") as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
    return imports

def resolve_dependencies(file_path, base_dir):
    """Recursively resolve all dependencies of a Python file."""
    resolved = set()
    to_process = [file_path]

    while to_process:
        current_file = to_process.pop()
        if current_file in resolved:
            continue
        resolved.add(current_file)

        # Get imports from the current file
        imports = get_imports(current_file)
        for imp in imports:
            imp_path = base_dir / f"{imp.replace('.', '/')}.py"
            if imp_path.exists():
                to_process.append(imp_path)
            else:
                # Handle directories or __init__.py files
                module_dir = base_dir / imp.replace('.', '/')
                if (module_dir / "__init__.py").exists():
                    to_process.append(module_dir / "__init__.py")

    return resolved

def find_python_files(base_dir):
    """Find all Python files in the project."""
    return {p for p in base_dir.rglob("*.py")}

if __name__ == "__main__":
    # Step 1: Resolve all dependencies of `non_uifiles.py`
    used_files = resolve_dependencies(SCRIPT_PATH, BASE_DIR)
    print(f"Used Python files: {len(used_files)}")

    # Step 2: Find all Python files in the project
    all_files = find_python_files(BASE_DIR)
    print(f"Total Python files in project: {len(all_files)}")

    # Step 3: Determine unused files
    unused_files = all_files - used_files
    print(f"Unused Python files: {len(unused_files)}")
    for file in unused_files:
        print(file)

    # Step 4: Optionally delete unused files
    delete = input("Delete unused files? (y/n): ").strip().lower()
    if delete == 'y':
        for file in unused_files:
            os.remove(file)
            print(f"Deleted: {file}")
