import ast
import os
from typing import List, Dict, Any


def _normalize(path: str) -> str:
    path = path.replace("\\", "/").strip()
    while path.startswith(("./", "../")):
        path = path.split("/", 1)[1] if "/" in path else ""
    return path


def resolve_path(files: Dict[str, Any], path: str):
    """Maps a path an LLM asked for onto a real key of the genome JSON, or None."""
    if not path:
        return None
    if path in files:
        return path

    target = _normalize(path)
    if not target:
        return None

    for key in files:
        normalized = _normalize(key)
        if normalized == target:
            return key
    for key in files:
        normalized = _normalize(key)
        if normalized.endswith("/" + target) or target.endswith("/" + normalized):
            return key

    matches = [k for k in files if os.path.basename(_normalize(k)) == os.path.basename(target)]
    return matches[0] if len(matches) == 1 else None


class RepoInspector:
    """Read-only, on-demand navigation of a genome's source files."""

    def __init__(self, repo_json: Dict[str, Any]):
        self.repo = repo_json
        self.files = repo_json.get("files", {})

    def _resolve(self, path: str):
        return resolve_path(self.files, path)

    def _not_found(self, path: str) -> str:
        return (f"File '{path}' not found. Valid paths: {list(self.files.keys())}. "
                f"Use one of these exactly.")

    def _get_ast(self, path: str) -> ast.Module:
        key = self._resolve(path)
        content = self.files.get(key, {}).get("content", "") if key else ""
        if not content:
            raise ValueError(self._not_found(path))
        return ast.parse(content)

    def list_repository_structure(self) -> List[str]:
        """
        USE THIS FIRST. Returns a comprehensive list of all file paths in the archive.
        Use this to identify relevant files based on their names and directory structure
        (e.g., searching for 'models/', 'utils/', or 'train.py').
        """
        return list(self.files.keys())

    def get_file_outline(self, path: str) -> Dict[str, List[str]]:
        """
        USE THIS TO BROWSE. Returns a flat list of all Class names and Function/Method names
        within a specific file.
        Use this tool to verify if a specific logic component exists in a file
        without retrieving code or signatures. It is the most token-efficient way
        to map the contents of a script.
        """
        tree = self._get_ast(path)
        outline = {"classes": [], "functions": []}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                outline["classes"].append(node.name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                outline["functions"].append(node.name)
        return outline

    def get_entity_info(self, path: str, entity_name: str) -> Dict[str, str]:
        """
        USE THIS FOR SPECIFIC DETAILS. Returns the technical signature (arguments and return types)
        and the docstring for a specific Class, Method, or Function.
        Use this tool when you have identified a likely source of an error and need to
        understand its intended contract (API) and documentation without reading the implementation body.
        """
        tree = self._get_ast(path)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                if node.name == entity_name:
                    doc = ast.get_docstring(node) or "No docstring provided."

                    if isinstance(node, ast.ClassDef):
                        signature = f"class {node.name}"
                    else:
                        args = [f"{a.arg}: {ast.unparse(a.annotation)}" if a.annotation else a.arg for a in node.args.args]
                        ret = ast.unparse(node.returns) if node.returns else "Any"
                        signature = f"def {node.name}({', '.join(args)}) -> {ret}"

                    return {"signature": signature, "docstring": doc}
        return {"error": f"Entity '{entity_name}' not found in {path}."}

    def get_method_body(self, path: str, entity_name: str) -> str:
        """
        USE THIS TO INSPECT LOGIC. Returns the raw Python source code for a specific
        Function, Method, or Class.
        WARNING: This is token-heavy. Only use this once you have identified the
        exact entity responsible for the incorrect performance through get_entity_info
        or get_call_graph.
        """
        tree = self._get_ast(path)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                if node.name == entity_name:
                    return ast.unparse(node)
        return f"Entity '{entity_name}' not found in {path}."

    def get_full_file(self, path: str) -> str:
        """
        USE ONLY AS A LAST RESORT. Returns the complete content of a file.
        Only use this if the file is very small or if you need to see top-level
        variable assignments and imports that are not encapsulated in functions.
        """
        key = self._resolve(path)
        if key is None:
            return self._not_found(path)
        return self.files[key].get("content", "")

    def get_imports(self, path: str) -> List[str]:
        """
        USE TO CHECK DEPENDENCIES. Returns a list of all external libraries and
        internal modules imported by the file.
        Use this to troubleshoot missing dependencies or to see if the script
        is importing the correct internal utilities.
        """
        tree = self._get_ast(path)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for n in node.names:
                    imports.append(n.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        return list(set(imports))

    def get_call_graph(self, path: str) -> Dict[str, List[str]]:
        """
        USE TO TRACE EXECUTION. Analyzes which functions call which other methods
        within the same file.
        Use this to understand the 'ripple effect' of a bug—if you change one function,
        this tool tells you which other functions in the file will be affected.
        """
        tree = self._get_ast(path)
        graph = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                calls = []
                for subnode in ast.walk(node):
                    if isinstance(subnode, ast.Call):
                        if isinstance(subnode.func, ast.Attribute):
                            calls.append(subnode.func.attr)
                        elif isinstance(subnode.func, ast.Name):
                            calls.append(subnode.func.id)
                graph[node.name] = list(set(calls))
        return graph


INSPECTION_TOOLS = (
    ("list_repository_structure", "Returns every file path in the agent repository. Call this first."),
    ("get_file_outline", "Returns the classes and functions defined in a file."),
    ("get_entity_info", "Returns the signature and docstring of one class or function."),
    ("get_method_body", "Returns the source code of one class or function."),
    ("get_full_file", "Returns the entire content of a file."),
    ("get_imports", "Lists what a file imports."),
    ("get_call_graph", "Maps which functions call which inside one file."),
)


def inspection_tools(genome: dict) -> list:
    """The code-inspection toolset shared by the understanding and proposer agents."""
    from langchain_core.tools import StructuredTool

    inspector = RepoInspector(genome)
    return [
        StructuredTool.from_function(func=getattr(inspector, name), name=name, description=description)
        for name, description in INSPECTION_TOOLS
    ]
