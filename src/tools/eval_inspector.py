import json
import os

OBS_PREVIEW_CHARS = 200
MAX_EPISODE_LOG_CHARS = 12000


class EvalInspector:
    """
    Staged access to one evaluation: the summary log, the predictions JSONL, and
    the raw turn-by-turn log of a single episode. No tool returns the whole
    evaluation at once.
    """

    def __init__(self, log_path: str, predictions_path: str):
        self.log_path = log_path
        self.predictions_path = predictions_path

    def _predictions(self):
        with open(self.predictions_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

    def list_prediction_ids(self) -> str:
        """Returns every episode id with its task type, outcome and step count. Call this first."""
        if not os.path.exists(self.predictions_path):
            return f"Error: predictions file not found at {self.predictions_path}"
        try:
            return "\n".join(
                f"{p.get('instance_id', 'unknown')}: "
                f"{'SUCCESS' if p.get('success') else 'FAIL'} "
                f"[{p.get('task_type', 'unknown')}] {p.get('num_steps', 0)} steps — "
                f"{p.get('task_instruction', '')}"
                for p in self._predictions()
            )
        except Exception as e:
            return f"Error reading predictions: {e}"

    def get_prediction(self, instance_id: str = None) -> str:
        """
        Returns one episode's record, including its interaction history.

        Observations are shortened to a preview; use read_episode_log for the
        untruncated turn-by-turn text of a single episode.
        """
        if not os.path.exists(self.predictions_path):
            return f"Error: predictions file not found at {self.predictions_path}"
        try:
            for prediction in self._predictions():
                if instance_id is None or prediction.get("instance_id") == instance_id:
                    for entry in prediction.get("interaction_history", []):
                        obs = entry.get("obs", "")
                        if len(obs) > OBS_PREVIEW_CHARS:
                            entry["obs"] = obs[:OBS_PREVIEW_CHARS] + " ...[truncated]"
                    return json.dumps(prediction, indent=2)
            return f"Prediction for episode '{instance_id}' not found."
        except Exception as e:
            return f"Error reading predictions: {e}"

    def read_episode_log(self, instance_id: str) -> str:
        """
        Returns the raw log of one episode: what the model actually wrote each turn,
        the action parsed out of it, and the environment's reply.
        """
        try:
            for prediction in self._predictions():
                if prediction.get("instance_id") != instance_id:
                    continue
                path = prediction.get("episode_log")
                if not path or not os.path.exists(path):
                    return f"No episode log recorded for '{instance_id}'."
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                if len(text) > MAX_EPISODE_LOG_CHARS:
                    return text[:MAX_EPISODE_LOG_CHARS] + "\n[truncated]"
                return text
            return f"Episode '{instance_id}' not found."
        except Exception as e:
            return f"Error reading episode log: {e}"

    def read_log_tail(self, lines: int = 100) -> str:
        """Returns the last `lines` lines of the evaluation log."""
        if not os.path.exists(self.log_path):
            return f"Error: log file not found at {self.log_path}"
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                return "".join(f.readlines()[-lines:])
        except Exception as e:
            return f"Error reading log: {e}"

    def search_log(self, query: str, context_lines: int = 5) -> str:
        """Searches the evaluation log and returns matches with surrounding lines."""
        if not os.path.exists(self.log_path):
            return f"Error: log file not found at {self.log_path}"
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            results = []
            for i, line in enumerate(lines):
                if query.lower() not in line.lower():
                    continue
                start, end = max(0, i - context_lines), min(len(lines), i + context_lines + 1)
                results.append(f"--- match at line {i + 1} ---\n" + "".join(lines[start:end]))
                if len(results) >= 5:
                    results.append("... (further matches omitted)")
                    break
            return "\n".join(results) if results else f"No matches found for '{query}'"
        except Exception as e:
            return f"Error searching log: {e}"


EVAL_TOOLS = (
    ("list_prediction_ids", "Every episode id with its task type, outcome and step count. Call this first."),
    ("get_prediction", "One episode's record and its interaction history (observations shortened)."),
    ("read_episode_log", "The raw turn-by-turn log of one episode, including the model's own output."),
    ("read_log_tail", "The last lines of the evaluation log."),
    ("search_log", "Search the evaluation log for a keyword."),
)


def evaluation_tools(log_path: str, predictions_path: str) -> list:
    from langchain_core.tools import StructuredTool

    inspector = EvalInspector(log_path=log_path, predictions_path=predictions_path)
    return [
        StructuredTool.from_function(func=getattr(inspector, name), name=name, description=description)
        for name, description in EVAL_TOOLS
    ]
