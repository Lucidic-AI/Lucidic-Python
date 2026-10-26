"""``Run``: one case of an eval run, and the session the agent runs in."""
import json
import logging
import os
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Union

from ..api.documents import DownloadSummary
from ..core.errors import ConflictError, InvalidOperationError, NotFoundError
from ..sdk.tools.context import MockContext, _current_mock_context, bind_mock_context, clear_mock_context
from ..session_obj import Session
from .tools import RunTools

if TYPE_CHECKING:
    from ..client import LucidicAI

logger = logging.getLogger("Lucidic")

RUN_ENV_VAR = "LUCIDIC_RUN"
# reserved payload key carrying the frozen config to a subprocess (to_json)
SNAPSHOT_KEY = "_lucidic"


@dataclass(frozen=True)
class RunIds:
    """The ids that tie a run to its case. Canonical lowercase uuid strings."""

    session_id: str
    datasetitem_id: Optional[str] = None
    experiment_id: Optional[str] = None
    checkpoint_id: Optional[str] = None
    eval_run_id: Optional[str] = None


class Run:
    """One case: its input, ids, params, frozen config and session.

    Build it with ``lucidicai.load()``. ``with run:`` opens the case's session
    (fixtures bound, so ``@mockable`` tools are answered from the case data),
    then on exit ends the session, which first waits for the session's pending
    ``evals.emit`` posts. A Run is entered once. Prompts and tools are resolved
    once per run and reused, also by subprocesses given ``to_env()`` or
    ``to_json()``. A child that joined an open run's session doesn't end it, so
    on exit it waits for that session's posts itself.

    ``ids.session_id`` is the idempotency key: deliveries are at-least-once.
    The backend may give the session a different id; ``session_id`` is that one.
    """

    def __init__(
        self,
        ids: RunIds,
        input: Any = None,
        params: Optional[Mapping[str, Any]] = None,
        *,
        client: Optional["LucidicAI"] = None,
        snapshot: Optional[Mapping[str, Any]] = None,
    ):
        snapshot = snapshot or {}
        self._ids = ids
        self._input = input
        self._params = MappingProxyType(dict(params or {}))
        self._client = client
        self._owns_client = client is None
        self._prompts: Dict[str, Optional[str]] = dict(snapshot.get("prompts") or {})
        self._tools = RunTools(self, snapshot.get("tools"))
        # a child of an entered run binds to the parent's session instead of creating one
        self._attached = bool(snapshot.get("attached"))
        self._parent_mock = bool(snapshot.get("mock"))
        self._parent_session_id: Optional[str] = snapshot.get("session_id")
        self._server_session_id: Optional[str] = None
        self._entered = False
        self._session: Optional[Session] = None

    @property
    def input(self) -> Any:
        return self._input

    @property
    def ids(self) -> RunIds:
        return self._ids

    @property
    def session_id(self) -> Optional[str]:
        """The server's id for the run's session; None until ``with run:`` opens it.

        Pass this, not ``ids.session_id``, to calls that target the session
        (``evals.emit``, ``sessions.update``): the backend may store the
        dispatched id as the session's custom id under a fresh primary key.
        """
        return self._server_session_id

    def _target_session_id(self) -> str:
        return self._server_session_id or self._ids.session_id

    @property
    def params(self) -> Mapping[str, Any]:
        """Extra payload keys (e.g. tuner params), read-only."""
        return self._params

    @property
    def tools(self) -> RunTools:
        return self._tools

    @property
    def client(self) -> "LucidicAI":
        """The client passed to ``load()``, else one built from env and closed on exit."""
        if self._client is None:
            from ..client import LucidicAI

            self._client = LucidicAI()
        return self._client

    # ==================== Prompts ====================

    def prompt(
        self, name: str, *, default: Optional[str] = None,
        register: bool = False, label: str = "production",
    ) -> str:
        """The run's version of prompt ``name``: checkpoint-pinned > ``label`` > ``default``.

        Resolved once per run, so ``label`` is fixed by the first call for a
        name. Only a missing prompt falls back to ``default``;
        other errors raise, so a checkpoint run never silently grades the
        baseline. ``register=True`` creates a missing prompt from ``default``
        (best effort, needs ``prompt:write``).

        Raises:
            NotFoundError: the prompt does not exist and no default was given.
        """
        if name not in self._prompts:
            try:
                content = self.client.prompts.get(name, **self._prompt_kwargs(label)).content
            except NotFoundError:
                if default is None:
                    raise
                content = None
                if register:
                    self._register(name, default)
            self._prompts[name] = self._content(name, content)
        return self._resolve(name, default)

    async def aprompt(
        self, name: str, *, default: Optional[str] = None,
        register: bool = False, label: str = "production",
    ) -> str:
        """Async sibling of ``prompt``."""
        if name not in self._prompts:
            try:
                content = (await self.client.prompts.aget(name, **self._prompt_kwargs(label))).content
            except NotFoundError:
                if default is None:
                    raise
                content = None
                if register:
                    await self._aregister(name, default)
            self._prompts[name] = self._content(name, content)
        return self._resolve(name, default)

    def _prompt_kwargs(self, label: str) -> Dict[str, Any]:
        # explicit ids pin the checkpoint version even before the session exists
        return {"label": label, "checkpoint_id": self._ids.checkpoint_id, "session_id": self._target_session_id()}

    @staticmethod
    def _content(name: str, content: Optional[str]) -> Optional[str]:
        if not content:
            # missing, or a production client that swallowed an error into an empty prompt
            logger.info(f"[Run] prompt {name!r} unresolved; using the local default")
        return content or None

    def _resolve(self, name: str, default: Optional[str]) -> str:
        content = self._prompts[name]
        return content if content is not None else (default or "")

    def _register(self, name: str, content: str) -> None:
        try:
            self.client.prompts.create(name, content)
        except ConflictError:
            pass  # another worker registered it first
        except Exception as e:  # best effort: the default is already in hand
            logger.warning(f"[Run] could not register prompt {name!r}: {e}")

    async def _aregister(self, name: str, content: str) -> None:
        try:
            await self.client.prompts.acreate(name, content)
        except ConflictError:
            pass
        except Exception as e:  # best effort: the default is already in hand
            logger.warning(f"[Run] could not register prompt {name!r}: {e}")

    # ==================== Documents ====================

    def download(self, dir: Union[str, "os.PathLike[str]"]) -> DownloadSummary:
        """The case's documents into ``dir``; see ``client.resources.download``."""
        return self.client.resources.download(self._target_session_id(), dir)

    async def adownload(self, dir: Union[str, "os.PathLike[str]"]) -> DownloadSummary:
        """Async sibling of ``download``."""
        return await self.client.resources.adownload(self._target_session_id(), dir)

    # ==================== Session lifecycle ====================

    def __enter__(self) -> "Run":
        self._begin()
        try:
            self._tools._ensure()
            self._session = self._attach() if self._attached else self.client.sessions.create(**self._session_kwargs())
            self._server_session_id = self._session.session_id
        except BaseException:
            self._close_client()
            raise
        return self

    async def __aenter__(self) -> "Run":
        self._begin()
        try:
            await self._tools._aensure()
            self._session = self._attach() if self._attached else await self.client.sessions.acreate(**self._session_kwargs())
            self._server_session_id = self._session.session_id
        except BaseException:
            await self._aclose_client()
            raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        session, self._session = self._session, None
        if self._attached:
            # the parent ends this session; deliver the child's self-scores now
            self.client._flush_evals(session_id=session.session_id)
        session.__exit__(exc_type, exc_val, exc_tb)  # end() waits for the session's posts
        self._unbind_mock()
        self._close_client()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        session, self._session = self._session, None
        if self._attached:
            # the parent ends this session; deliver the child's self-scores now
            await self.client._aflush_evals(session_id=session.session_id)
        await session.__aexit__(exc_type, exc_val, exc_tb)  # aend() waits for the session's posts
        self._unbind_mock()
        await self._aclose_client()

    def _close_client(self) -> None:
        if self._owns_client and self._client is not None:
            try:
                self._client.close()
            except Exception as e:
                logger.debug(f"[Run] error closing the run's client: {e}")

    async def _aclose_client(self) -> None:
        if self._owns_client and self._client is not None:
            try:
                await self._client.aclose()
            except Exception as e:
                logger.debug(f"[Run] error closing the run's client: {e}")

    def _begin(self) -> None:
        if self._entered:
            raise InvalidOperationError("a Run is entered once; load() a new one per case")
        self._entered = True

    def _session_kwargs(self) -> Dict[str, Any]:
        ids = self._ids
        return {
            "session_id": ids.session_id,
            "datasetitem_id": ids.datasetitem_id,
            "experiment_id": ids.experiment_id,
            "checkpoint_id": ids.checkpoint_id,
            "tags": [f"eval_run:{ids.eval_run_id}"] if ids.eval_run_id else None,
            "auto_end": True,
        }

    def _attach(self) -> Session:
        session_id = self._parent_session_id or self._ids.session_id
        session = Session(self.client, session_id, auto_end=False)
        session._bind_context()
        if self._parent_mock:
            bind_mock_context(MockContext(session_id=session_id, client=self.client))
        return session

    def _mock_bound(self) -> bool:
        ctx = _current_mock_context()
        return ctx is not None and ctx.session_id == self._target_session_id() and ctx.client is self._client

    def _unbind_mock(self) -> None:
        # nothing else clears it on session end; a reused pool thread would inherit it
        if self._mock_bound():
            clear_mock_context()

    def _require_open(self, what: str) -> None:
        if self._session is None:
            raise InvalidOperationError(f"{what} needs an open run: call it inside `with run:`")

    # ==================== Hand-off ====================

    def to_json(self) -> str:
        """The run as a ``LUCIDIC_RUN`` value, with its resolved prompts and tools.

        From inside ``with run:``, a child that loads it joins this session
        instead of creating its own; the parent still ends it.
        """
        is_open = self._session is not None
        payload = {**self._params, **asdict(self._ids), "input": self._input}
        payload[SNAPSHOT_KEY] = {
            "prompts": dict(self._prompts),
            "tools": self._tools._snapshot,
            "attached": is_open,
            "session_id": self._server_session_id if is_open else None,
            "mock": is_open and self._mock_bound(),
        }
        return json.dumps(payload)

    def to_env(self) -> Dict[str, str]:
        """``{"LUCIDIC_RUN": to_json()}``, to merge into a subprocess env.

        ``subprocess.run(cmd, env={**os.environ, **run.to_env()})``

        For inputs near the 128 KiB env-string limit, write ``to_json()`` to a
        file and set ``LUCIDIC_RUN`` to its path instead.
        """
        return {RUN_ENV_VAR: self.to_json()}

    def __repr__(self) -> str:
        return f"<Run(session_id={self._ids.session_id[:8]}..., checkpoint_id={self._ids.checkpoint_id})>"
