from src.tools.repo_inspector import resolve_path


class RepoEditor:
    def __init__(self, repo_json):
        self.repo = repo_json
        self.files = repo_json.get("files", {})

    def _not_found(self, path: str) -> str:
        return (f"Error: file '{path}' not found. Valid paths: {list(self.files.keys())}. "
                f"Use one of these exactly.")

    def read_file(self, path: str) -> str:
        key = resolve_path(self.files, path)
        if key is None:
            return self._not_found(path)
        return self.files[key].get("content", "")

    def write_file(self, path: str, content: str) -> str:
        key = resolve_path(self.files, path)
        if key is not None:
            self.files[key]["content"] = content
            return f"File {key} updated successfully."

        # No existing file matches, so this creates a new one. Only plain
        # relative paths are accepted.
        normalized = (path or "").strip().replace("\\", "/")
        if not normalized or normalized.startswith("/") or ".." in normalized.split("/"):
            return f"Error: '{path}' is not a usable file path."
        self.files[normalized] = {"content": content}
        return f"File {normalized} created."
