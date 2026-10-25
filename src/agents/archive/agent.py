from langchain_core.messages import HumanMessage, SystemMessage


class ArchiveAgent:
    """
    The agent under evolution. It receives a ready-made LLM client and answers
    one ALFWorld turn at a time.

    generate_response(self, user_input: str) -> str is the fixed contract with
    the episode runner and must keep its signature.

    Two things make this a CoT+ReAct baseline rather than a bare prompt-and-parse
    loop:
    - Chain-of-Thought (Wei et al., 2022): the system prompt requires an explicit
      Thought line reasoning about the current state before the Action line.
    - ReAct (Yao et al., 2023): the episode runner feeds every prior turn's
      Thought, Action and Observation back into this prompt (see
      src/eval/alfworld_policy.py's build_alfworld_prompt), so each new Thought
      is conditioned on the full Thought/Action/Observation trace so far, not
      just the latest observation.
    """

    def __init__(self, llm):
        # llm exposes only invoke(messages) — it is not a LangChain chat model,
        # so it has no .bind(), .bind_tools(), or other such methods.
        self.llm = llm
        self.system_prompt = (
            "You are playing a text-based household task in the ALFWorld environment.\n"
            "\n"
            "VALID ACTIONS (the engine accepts only these phrasings):\n"
            "  go to <location>\n"
            "  take <object> from <location>\n"
            "  move <object> to <receptacle>     placement is 'move ... to ...', never 'put'\n"
            "  open <container>\n"
            "  clean <object> with <sink>\n"
            "  heat <object> with <microwave>\n"
            "  cool <object> with <fridge>\n"
            "  use <object>\n"
            "  examine <object>\n"
            "  look\n"
            "\n"
            "RULES:\n"
            "- You cannot see an object until you have gone to the location holding it.\n"
            "- 'Nothing happens' means the action's precondition was not met. Change what you\n"
            "  are doing; repeating the same action will not work.\n"
            "- 'You won!' means the task is complete.\n"
            "\n"
            "HOW TO PLAY:\n"
            "1. Identify the objects and the receptacle the task names.\n"
            "2. Visit each plausible location once until you find the object.\n"
            "3. Once you hold it, go straight to the receptacle or appliance the task needs.\n"
            "4. Do not revisit a location you have already searched.\n"
            "\n"
            "REASONING: your prompt shows every prior turn's Thought, Action and\n"
            "Observation from this episode. Use them — do not repeat a Thought or an\n"
            "Action that already failed.\n"
            "\n"
            "OUTPUT FORMAT — always both lines, in this order:\n"
            "  Thought: one or two sentences on the current state and what to do next.\n"
            "  Action: <one command from the list above>\n"
        )

    def generate_response(self, user_input: str) -> str:
        messages = [SystemMessage(content=self.system_prompt),
                    HumanMessage(content=user_input)]
        content = self.llm.invoke(messages).content

        if isinstance(content, list):
            content = "".join(
                part if isinstance(part, str) else part.get("text", "")
                for part in content
                if isinstance(part, str) or part.get("type") == "text")

        return (content or "").strip()
