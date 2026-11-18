from pathlib import Path

# Path to module-local templates directory
_TEMPLATES_DIR = Path(__file__).parent / "templates"

# Default fallback if file is not found
_FALLBACK_SYSTEM_PROMPT = (
    "You are a concise, accurate, and reliable AI assistant.\n"
    "Guidelines:\n"
    "- Answer the user's question directly and clearly without unnecessary filler.\n"
    "- Be strictly truthful: if you are unsure or lack sufficient context, say so explicitly rather than guessing or fabricating details.\n"
    "- Use clean formatting (lists, code blocks, or tables) when it improves readability.\n"
    "- Match the depth of your response to the complexity of the request."
)


class PromptService:
    """
    Manages versioned prompt templates and system prompts for LLM completions.
    All prompt templates live in app/modules/prompts/templates/ (e.g. customer_support/v1.txt).
    """

    def __init__(
        self,
        templates_dir: Path | None = None,
        prompts_dir: Path | None = None,
    ) -> None:
        self.templates_dir = templates_dir or prompts_dir or _TEMPLATES_DIR
        self._cache: dict[str, str] = {}

    def load_prompt(self, category: str, version: str) -> str:
        """
        Dynamically load a specific versioned prompt from the prompt library.
        Example: load_prompt("customer_support", "v1") -> loads templates/customer_support/v1.txt
        """
        cache_key = f"{category}/{version}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        target_file = self.templates_dir / category / f"{version}.txt"
        if not target_file.is_file():
            raise FileNotFoundError(
                f"Versioned prompt '{version}' for category '{category}' not found at {target_file}"
            )

        content = target_file.read_text(encoding="utf-8").strip()
        self._cache[cache_key] = content
        return content

    def list_versions(self, category: str) -> list[str]:
        """List all available versions for a given prompt category."""
        target_dir = self.templates_dir / category
        if not target_dir.is_dir():
            return []

        return sorted([f.stem for f in target_dir.glob("*.txt")])

    def list_categories(self) -> list[str]:
        """List all available prompt categories."""
        categories = set()
        if self.templates_dir.is_dir():
            for item in self.templates_dir.iterdir():
                if item.is_dir() and not item.name.startswith((".", "_", "__pycache__")):
                    categories.add(item.name)
        return sorted(list(categories))

    def list_all(self) -> dict[str, list[str]]:
        """List all categories with their available versions."""
        result = {}
        for cat in self.list_categories():
            versions = self.list_versions(cat)
            if versions:
                result[cat] = versions
        return result

    def get_system_prompt(self, template_name: str = "default") -> str:
        """Load a system prompt template by name or path."""
        if "/" in template_name:
            cat, ver = template_name.split("/", 1)
            return self.load_prompt(cat, ver)

        if template_name in self._cache:
            return self._cache[template_name]

        template_file = self.templates_dir / f"{template_name}.txt"
        if template_file.is_file():
            content = template_file.read_text(encoding="utf-8").strip()
            self._cache[template_name] = content
            return content

        if template_name == "default":
            return _FALLBACK_SYSTEM_PROMPT

        raise FileNotFoundError(f"Prompt template '{template_name}' not found at {template_file}")

    def format_prompt(self, template_name: str, **kwargs) -> str:
        template = self.get_system_prompt(template_name)
        return template.format(**kwargs)

    def clear_cache(self) -> None:
        """Clear cached prompt templates."""
        self._cache.clear()


default_prompt_service = PromptService()
SYSTEM_PROMPT = default_prompt_service.get_system_prompt()

