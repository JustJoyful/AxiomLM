"""One registry for typed commands, completion and native palette providers."""

import shlex
from dataclasses import dataclass
from functools import partial

from textual.command import DiscoveryHit, Hit, Provider
from textual.widgets import OptionList
from textual.widgets.option_list import Option


@dataclass(frozen=True)
class CommandSpec:
    name: str
    usage: str
    description: str
    available_phase: int = 1


COMMANDS = {
    name: CommandSpec(name, usage, description, phase)
    for name, usage, description, phase in [
        ("/help", "", "List commands", 1),
        (
            "/parse",
            "<path> [--mode auto|pymupdf4llm|mineru|marker] [--cpu-embeddings]",
            "Parse and index a PDF",
            1,
        ),
        ("/books", "[filter]", "List indexed books", 1),
        ("/book", "<name>", "Switch book", 1),
        ("/remove", "<book>", "Remove a book after confirmation", 1),
        ("/models", "", "Refresh installed models", 1),
        ("/model", "<name>", "Switch model", 1),
        ("/k", "<n>", "Set retrieval count (3–15)", 1),
        ("/d", "<value>", "Set distance cutoff", 1),
        ("/thinking", "on|off|show", "Thinking controls", 3),
        ("/web", "on|off", "Web fallback", 3),
        ("/sources", "[n]", "Expand retrieved evidence", 1),
        ("/details", "[n]", "Expand pipeline steps", 1),
        ("/copy", "[n] | code [n]", "Copy answer or fenced code", 1),
        ("/export", "[n]", "Typeset to PDF", 5),
        ("/retry", "", "Retry the last question", 1),
        ("/clear", "", "Clear displayed history", 1),
        ("/new", "", "Start a new conversation", 1),
        ("/theme", "[name]", "Preview or set theme", 1),
        ("/accent", "<color>", "Set terminal ANSI accent", 1),
        ("/settings", "", "Show settings commands", 1),
        ("/cancel", "", "Cancel the active operation", 1),
        ("/quit", "", "Exit", 1),
        ("/delete", "<book>", "Alias for /remove", 1),
        ("/gpu-parsing", "", "Toggle explicit CPU embedding ingest", 1),
        ("/gpu-status", "", "Show ingest device choice", 1),
    ]
}


def parse_command(text: str) -> tuple[str, list[str]]:
    parts = shlex.split(text)
    if not parts:
        raise ValueError("empty command")
    return parts[0].lower(), parts[1:]


async def dispatch_command(app, text: str) -> None:
    try:
        name, arguments = parse_command(text)
    except ValueError as exc:
        await app.write_line(f"✗ {exc} · /help")
        return
    if name not in COMMANDS:
        await app.write_line(f"✗ unknown command {name} · /help")
        return
    await app.handle_command(name, arguments)


class AxiomCommandProvider(Provider):
    async def search(self, query):
        matcher = self.matcher(query)
        for spec in COMMANDS.values():
            score = matcher.match(spec.name)
            if score:
                yield Hit(
                    score,
                    spec.name,
                    partial(self.app.fill_command, spec.name),
                    help=spec.description,
                )

    async def discover(self):
        for spec in COMMANDS.values():
            yield DiscoveryHit(
                spec.name,
                partial(self.app.fill_command, spec.name),
                help=spec.description,
            )


class CommandOverlay(OptionList):
    def __init__(self, names=None, *, kind="commands", labels=None):
        self.kind = kind
        self.names = list(COMMANDS if names is None else names)
        super().__init__(
            *(Option((labels or {}).get(name, name), id=name) for name in self.names),
            id="overlay",
        )
        self.highlighted = 0 if self.names else None
