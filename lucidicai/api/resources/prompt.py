"""Prompt resource API operations."""
import logging
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Tuple, TYPE_CHECKING

from ..client import HttpClient
from ..models.base import CursorPage
from ..models.prompt import PromptInfo, PromptVersion
from ..pagination import apaginate, paginate
from ...core.errors import require_agent_id

if TYPE_CHECKING:
    from ...core.config import SDKConfig

logger = logging.getLogger("Lucidic")


def _current_session_id() -> Optional[str]:
    """The active session id (LUC-795), so a prompt fetch resolves through the
    session's bound/pre-linked checkpoint. Never let context lookup break a fetch.
    """
    try:
        from ...sdk.context import current_session_id
        return current_session_id.get(None)
    except Exception:
        return None


@dataclass
class Prompt:
    """Represents a prompt retrieved from the Lucidic prompt database."""

    raw_content: str
    content: str
    metadata: Dict[str, Any]

    def __str__(self) -> str:
        return self.content

    def replace_variables(self, variables: Dict[str, Any]) -> "Prompt":
        """Replace template variables in the prompt content.

        Replaces {{key}} placeholders in raw_content with the provided
        variable values and updates content.

        Args:
            variables: Dictionary mapping variable names to their values.

        Returns:
            self, for method chaining.
        """
        content = self.raw_content
        for key, value in variables.items():
            content = content.replace(f"{{{{{key}}}}}", str(value))
        self.content = content
        return self


class PromptResource:
    """Handle prompt-related API operations."""

    def __init__(self, http: HttpClient, config: "SDKConfig", production: bool = False):
        """Initialize prompt resource.

        Args:
            http: HTTP client instance
            config: SDK configuration
            production: Whether to suppress errors in production mode
        """
        self.http = http
        self._config = config
        self._production = production
        # LUC-795: keyed by (prompt_name, label, checkpoint_id, session_id) so entries
        # for different sessions/checkpoints never collide. prompt_name stays first so
        # _invalidate_cache(prompt_name) still matches by k[0].
        self._cache: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

    def _invalidate_cache(self, prompt_name: str, label: Optional[str] = None) -> None:
        """Invalidate cached prompt entries.

        Args:
            prompt_name: Name of the prompt to invalidate.
            label: If provided, only invalidate the specific (prompt_name, label) entry.
                   If None, invalidate all entries matching prompt_name.
        """
        if label is not None:
            self._cache.pop((prompt_name, label), None)
        else:
            keys_to_remove = [k for k in self._cache if k[0] == prompt_name]
            for k in keys_to_remove:
                del self._cache[k]

    def _is_cache_valid(self, cache_key: Tuple[Any, ...], cache_ttl: int) -> bool:
        """Check if a cached prompt is still valid.

        Args:
            cache_key: The (prompt_name, label) tuple
            cache_ttl: Cache TTL in seconds (-1 = indefinite, 0 = no cache)

        Returns:
            True if cache is valid, False otherwise
        """
        if cache_ttl == 0:
            return False
        if cache_key not in self._cache:
            return False
        if cache_ttl == -1:
            return True
        cached = self._cache[cache_key]
        return (time.time() - cached["timestamp"]) < cache_ttl

    def get(
        self,
        prompt_name: str,
        variables: Optional[Dict[str, Any]] = None,
        label: str = "production",
        cache_ttl: int = 0,
        checkpoint_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Prompt:
        """Get a prompt from the prompt database.

        Args:
            prompt_name: Name of the prompt.
            variables: Variables to interpolate into the prompt.
            label: Prompt version label (default: "production").
            cache_ttl: Cache TTL in seconds. 0 = no cache, -1 = cache indefinitely,
                       positive value = seconds before refetching.
            checkpoint_id: Explicitly resolve against a loaded checkpoint (LUC-795).
                           Rarely needed inside a session — the active session already
                           carries its checkpoint server-side. Defaults to None.
            session_id: Override the session context. Defaults to the active session,
                        so a session running a checkpoint gets that checkpoint's
                        trained prompt version (else normal label resolution).

        Returns:
            A Prompt object with raw_content, content (with variables replaced),
            and metadata. Use str(prompt) for backward-compatible string access.
        """
        try:
            # LUC-795: resolve the session context so a session running a checkpoint
            # gets that checkpoint's trained prompt version. The backend overrides the
            # label only for prompts the checkpoint trained and falls back to the label
            # otherwise, so this is safe for non-checkpoint sessions.
            resolved_session_id = session_id if session_id is not None else _current_session_id()
            cache_key = (prompt_name, label, checkpoint_id, resolved_session_id)

            # Check cache
            if self._is_cache_valid(cache_key, cache_ttl):
                raw_content = self._cache[cache_key]["content"]
                metadata = self._cache[cache_key]["metadata"]
            else:
                params: Dict[str, Any] = {
                    "prompt_name": prompt_name,
                    "label": label,
                    "agent_id": self._config.agent_id,
                }
                if checkpoint_id is not None:
                    params["checkpoint_id"] = checkpoint_id
                if resolved_session_id is not None:
                    params["session_id"] = resolved_session_id
                response = self.http.get("sdk/prompts", params)
                raw_content = response.get("prompt_content", "")
                metadata = response.get("metadata", {})

                # Store in cache if caching is enabled. checkpoint/session-scoped
                # entries resolve to immutable versions, so a positive/indefinite ttl
                # needs no invalidation.
                if cache_ttl != 0:
                    self._cache[cache_key] = {
                        "content": raw_content,
                        "metadata": metadata,
                        "timestamp": time.time(),
                    }

            prompt = Prompt(raw_content=raw_content, content=raw_content, metadata=metadata)
            if variables:
                prompt.replace_variables(variables)
            return prompt
        except Exception as e:
            if self._production:
                logger.error(f"[PromptResource] Failed to get prompt: {e}")
                return Prompt(raw_content="", content="", metadata={})
            raise

    async def aget(
        self,
        prompt_name: str,
        variables: Optional[Dict[str, Any]] = None,
        label: str = "production",
        cache_ttl: int = 0,
        checkpoint_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Prompt:
        """Get a prompt from the prompt database (asynchronous).

        See get() for full documentation.
        """
        try:
            # LUC-795: resolve the session context (see get()).
            resolved_session_id = session_id if session_id is not None else _current_session_id()
            cache_key = (prompt_name, label, checkpoint_id, resolved_session_id)

            # Check cache
            if self._is_cache_valid(cache_key, cache_ttl):
                raw_content = self._cache[cache_key]["content"]
                metadata = self._cache[cache_key]["metadata"]
            else:
                params: Dict[str, Any] = {
                    "prompt_name": prompt_name,
                    "label": label,
                    "agent_id": self._config.agent_id,
                }
                if checkpoint_id is not None:
                    params["checkpoint_id"] = checkpoint_id
                if resolved_session_id is not None:
                    params["session_id"] = resolved_session_id
                response = await self.http.aget("sdk/prompts", params)
                raw_content = response.get("prompt_content", "")
                metadata = response.get("metadata", {})

                # Store in cache if caching is enabled (immutable when context-scoped).
                if cache_ttl != 0:
                    self._cache[cache_key] = {
                        "content": raw_content,
                        "metadata": metadata,
                        "timestamp": time.time(),
                    }

            prompt = Prompt(raw_content=raw_content, content=raw_content, metadata=metadata)
            if variables:
                prompt.replace_variables(variables)
            return prompt
        except Exception as e:
            if self._production:
                logger.error(f"[PromptResource] Failed to get prompt: {e}")
                return Prompt(raw_content="", content="", metadata={})
            raise

    def update(
        self,
        prompt_name: str,
        prompt_content: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        labels: Optional[List[str]] = None,
    ) -> Prompt:
        """Update a prompt, creating a new immutable version.

        Args:
            prompt_name: Name of the prompt to update.
            prompt_content: New content for the prompt.
            description: Optional description for the prompt version.
            metadata: Optional metadata dict to attach to the prompt version.
            labels: Optional list of labels to assign to the new version.

        Returns:
            A Prompt object with the new content and metadata from the response.
        """
        try:
            body: Dict[str, Any] = {
                "agent_id": self._config.agent_id,
                "prompt_name": prompt_name,
                "prompt_content": prompt_content,
            }
            if description is not None:
                body["description"] = description
            if metadata is not None:
                body["metadata"] = metadata
            if labels is not None:
                body["labels"] = labels

            response = self.http.put("sdk/prompts", data=body)
            response_metadata = response.get("metadata", {})

            self._invalidate_cache(prompt_name)

            return Prompt(raw_content=prompt_content, content=prompt_content, metadata=response_metadata)
        except Exception as e:
            if self._production:
                logger.error(f"[PromptResource] Failed to update prompt: {e}")
                return Prompt(raw_content="", content="", metadata={})
            raise

    async def aupdate(
        self,
        prompt_name: str,
        prompt_content: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        labels: Optional[List[str]] = None,
    ) -> Prompt:
        """Update a prompt, creating a new immutable version (asynchronous).

        See update() for full documentation.
        """
        try:
            body: Dict[str, Any] = {
                "agent_id": self._config.agent_id,
                "prompt_name": prompt_name,
                "prompt_content": prompt_content,
            }
            if description is not None:
                body["description"] = description
            if metadata is not None:
                body["metadata"] = metadata
            if labels is not None:
                body["labels"] = labels

            response = await self.http.aput("sdk/prompts", data=body)
            response_metadata = response.get("metadata", {})

            self._invalidate_cache(prompt_name)

            return Prompt(raw_content=prompt_content, content=prompt_content, metadata=response_metadata)
        except Exception as e:
            if self._production:
                logger.error(f"[PromptResource] Failed to update prompt: {e}")
                return Prompt(raw_content="", content="", metadata={})
            raise

    def update_metadata(
        self,
        prompt_name: str,
        label: str,
        metadata: Dict[str, Any],
    ) -> Prompt:
        """Update metadata on an existing prompt version.

        Sends a PATCH request to update only the metadata for the prompt version
        identified by (prompt_name, label). The prompt content is not returned
        by this endpoint, so the returned Prompt will have empty content fields.

        Args:
            prompt_name: Name of the prompt.
            label: Label identifying the prompt version to update.
            metadata: Metadata dict to set on the prompt version.

        Returns:
            A Prompt object with empty content and the updated metadata.
        """
        try:
            body: Dict[str, Any] = {
                "agent_id": self._config.agent_id,
                "prompt_name": prompt_name,
                "label": label,
                "metadata": metadata,
            }

            response = self.http.patch("sdk/prompts", data=body)
            response_metadata = response.get("metadata", {})

            self._invalidate_cache(prompt_name, label)

            return Prompt(raw_content="", content="", metadata=response_metadata)
        except Exception as e:
            if self._production:
                logger.error(f"[PromptResource] Failed to update prompt metadata: {e}")
                return Prompt(raw_content="", content="", metadata={})
            raise

    async def aupdate_metadata(
        self,
        prompt_name: str,
        label: str,
        metadata: Dict[str, Any],
    ) -> Prompt:
        """Update metadata on an existing prompt version (asynchronous).

        See update_metadata() for full documentation.
        """
        try:
            body: Dict[str, Any] = {
                "agent_id": self._config.agent_id,
                "prompt_name": prompt_name,
                "label": label,
                "metadata": metadata,
            }

            response = await self.http.apatch("sdk/prompts", data=body)
            response_metadata = response.get("metadata", {})

            self._invalidate_cache(prompt_name, label)

            return Prompt(raw_content="", content="", metadata=response_metadata)
        except Exception as e:
            if self._production:
                logger.error(f"[PromptResource] Failed to update prompt metadata: {e}")
                return Prompt(raw_content="", content="", metadata={})
            raise

    # ==================== v2 reads (LUC-909) ====================
    #
    # Prompt-ops discovery: list the agent's prompts, walk a prompt's full
    # version history, read the agent's label set. Data-bearing reads — they do
    # NOT swallow in production (they surface the typed transport error), unlike
    # the get/update methods above. ``agent_id`` defaults to the configured
    # agent. Each read has an async sibling.

    def list(
        self, agent_id: Optional[str] = None, *,
        ordering: Optional[str] = None, page_size: Optional[int] = None,
    ) -> Iterator[PromptInfo]:
        """Lazily iterate the agent's prompts (by name). ``ordering`` accepts
        ``name`` / ``id`` (± prefix)."""
        base = self._agent_params(agent_id, ordering, page_size)
        return paginate(lambda c: self._page_get("sdk/v2/prompts", base, c), model=PromptInfo)

    def alist(
        self, agent_id: Optional[str] = None, *,
        ordering: Optional[str] = None, page_size: Optional[int] = None,
    ) -> AsyncIterator[PromptInfo]:
        """Async sibling of ``list``."""
        base = self._agent_params(agent_id, ordering, page_size)
        return apaginate(lambda c: self._apage_get("sdk/v2/prompts", base, c), model=PromptInfo)

    def list_page(
        self, agent_id: Optional[str] = None, *, cursor: Optional[str] = None,
        ordering: Optional[str] = None, page_size: Optional[int] = None,
    ) -> CursorPage:
        """Fetch a single page of prompts (manual pagination control)."""
        body = self._page_get("sdk/v2/prompts", self._agent_params(agent_id, ordering, page_size), cursor)
        return CursorPage.from_body(body, model=PromptInfo)

    async def alist_page(
        self, agent_id: Optional[str] = None, *, cursor: Optional[str] = None,
        ordering: Optional[str] = None, page_size: Optional[int] = None,
    ) -> CursorPage:
        """Async sibling of ``list_page``."""
        body = await self._apage_get("sdk/v2/prompts", self._agent_params(agent_id, ordering, page_size), cursor)
        return CursorPage.from_body(body, model=PromptInfo)

    def versions(
        self, name: str, agent_id: Optional[str] = None, *,
        ordering: Optional[str] = None, page_size: Optional[int] = None,
    ) -> Iterator[PromptVersion]:
        """Lazily iterate one prompt's full version history, newest first.
        ``ordering`` accepts ``version_number`` / ``id`` (± prefix)."""
        base = self._agent_params(agent_id, ordering, page_size)
        base["prompt_name"] = name
        return paginate(lambda c: self._page_get("sdk/v2/prompts/versions", base, c), model=PromptVersion)

    def aversions(
        self, name: str, agent_id: Optional[str] = None, *,
        ordering: Optional[str] = None, page_size: Optional[int] = None,
    ) -> AsyncIterator[PromptVersion]:
        """Async sibling of ``versions``."""
        base = self._agent_params(agent_id, ordering, page_size)
        base["prompt_name"] = name
        return apaginate(lambda c: self._apage_get("sdk/v2/prompts/versions", base, c), model=PromptVersion)

    def versions_page(
        self, name: str, agent_id: Optional[str] = None, *, cursor: Optional[str] = None,
        ordering: Optional[str] = None, page_size: Optional[int] = None,
    ) -> CursorPage:
        """Fetch a single page of a prompt's version history."""
        base = self._agent_params(agent_id, ordering, page_size)
        base["prompt_name"] = name
        return CursorPage.from_body(self._page_get("sdk/v2/prompts/versions", base, cursor), model=PromptVersion)

    async def aversions_page(
        self, name: str, agent_id: Optional[str] = None, *, cursor: Optional[str] = None,
        ordering: Optional[str] = None, page_size: Optional[int] = None,
    ) -> CursorPage:
        """Async sibling of ``versions_page``."""
        base = self._agent_params(agent_id, ordering, page_size)
        base["prompt_name"] = name
        body = await self._apage_get("sdk/v2/prompts/versions", base, cursor)
        return CursorPage.from_body(body, model=PromptVersion)

    def labels(self, agent_id: Optional[str] = None) -> List[str]:
        """The agent's label names (shared across its prompts) — a small,
        non-paginated read."""
        agent = require_agent_id(agent_id or self._config.agent_id, "prompts.labels")
        resp = self.http.get("sdk/v2/prompts/labels", {"agent_id": agent})
        return resp.get("labels", [])

    async def alabels(self, agent_id: Optional[str] = None) -> List[str]:
        """Async sibling of ``labels``."""
        agent = require_agent_id(agent_id or self._config.agent_id, "prompts.labels")
        resp = await self.http.aget("sdk/v2/prompts/labels", {"agent_id": agent})
        return resp.get("labels", [])

    # ==================== v2 writes (LUC-914) ====================
    #
    # create makes a brand-new prompt (+ its first version, via POST /sdk/v2/prompts,
    # LUC-931); rename / set_labels / delete operate on EXISTING prompts. (``update()``
    # above, the gen-3 PUT /sdk/prompts, adds a *version* to an existing prompt.)
    # Data-bearing writes → no production swallow. Each has an async sibling.

    def create(
        self, name: str, prompt_content: str, *,
        icon: Optional[str] = None, description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None, labels: Optional[List[str]] = None,
        agent_id: Optional[str] = None,
    ) -> Tuple[PromptInfo, Optional[PromptVersion]]:
        """Create a brand-new prompt and its first version (POST /sdk/v2/prompts;
        needs ``prompt:write``). ``name`` is unique per agent — a duplicate →
        ``ConflictError`` (use ``update()`` to add a version to an existing prompt).
        The first version auto-gets the ``latest`` + ``production`` labels; any
        ``labels`` you pass are added on top. Returns ``(prompt, first_version)``
        as typed models — ``first_version`` is ``None`` only if a future backend
        omits the version subobject from the response."""
        response = self.http.post(
            "sdk/v2/prompts",
            self._create_body(name, prompt_content, icon, description, metadata, labels, agent_id),
        )
        self._invalidate_cache(name)  # a same-name entry (deleted elsewhere, recreated) is now stale
        return self._created(response)

    async def acreate(
        self, name: str, prompt_content: str, *,
        icon: Optional[str] = None, description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None, labels: Optional[List[str]] = None,
        agent_id: Optional[str] = None,
    ) -> Tuple[PromptInfo, Optional[PromptVersion]]:
        """Async sibling of ``create``."""
        response = await self.http.apost(
            "sdk/v2/prompts",
            self._create_body(name, prompt_content, icon, description, metadata, labels, agent_id),
        )
        self._invalidate_cache(name)
        return self._created(response)

    def rename(
        self, prompt_name: str, *, new_name: Optional[str] = None,
        icon: Optional[str] = None, agent_id: Optional[str] = None,
    ) -> PromptInfo:
        """Rename and/or re-icon a prompt (PATCH /sdk/v2/prompts/detail). Pass
        ``new_name`` and/or ``icon``. A name collision with another prompt →
        ``ConflictError``. When the renamed prompt has shipped versions, the
        result carries a ``warning`` (on ``.extra``): clients still fetching the
        old name will 404 until updated."""
        result = PromptInfo.from_dict(
            self.http.patch("sdk/v2/prompts/detail", self._rename_body(prompt_name, new_name, icon, agent_id))
        )
        self._invalidate_cache(prompt_name)  # old (agent, name) fetches are now stale
        return result

    async def arename(
        self, prompt_name: str, *, new_name: Optional[str] = None,
        icon: Optional[str] = None, agent_id: Optional[str] = None,
    ) -> PromptInfo:
        """Async sibling of ``rename``."""
        body = self._rename_body(prompt_name, new_name, icon, agent_id)
        result = PromptInfo.from_dict(await self.http.apatch("sdk/v2/prompts/detail", body))
        self._invalidate_cache(prompt_name)
        return result

    def set_labels(
        self, prompt_name: str, version_number: int, labels: List[str], *,
        agent_id: Optional[str] = None,
    ) -> PromptVersion:
        """Set which labels point at a specific version — promote / roll back
        (PUT /sdk/v2/prompts/labels). This is a **replace**: ``labels`` becomes
        that version's entire label set (any other label on it is removed), and
        each label is **moved off** whatever version currently holds it — e.g.
        ``set_labels("greeting", 5, ["production"])`` points ``production`` at
        v5 and removes it from wherever it was. Not additive. ``"latest"`` is
        auto-managed and can't be moved (→ ``ValidationError``); an unknown
        ``version_number`` → ``NotFoundError``. Returns the updated version (a
        promote onto a previously-unlabeled version carries a ``warning`` on
        ``.extra``)."""
        result = PromptVersion.from_dict(
            self.http.put("sdk/v2/prompts/labels", self._label_body(prompt_name, version_number, labels, agent_id))
        )
        self._invalidate_cache(prompt_name)  # label pointers moved -> label fetches stale
        return result

    async def aset_labels(
        self, prompt_name: str, version_number: int, labels: List[str], *,
        agent_id: Optional[str] = None,
    ) -> PromptVersion:
        """Async sibling of ``set_labels``."""
        body = self._label_body(prompt_name, version_number, labels, agent_id)
        result = PromptVersion.from_dict(await self.http.aput("sdk/v2/prompts/labels", body))
        self._invalidate_cache(prompt_name)
        return result

    def delete(self, prompt_name: str, *, agent_id: Optional[str] = None) -> None:
        """Hard-delete a prompt and all its versions (DELETE /sdk/v2/prompts/detail;
        needs ``prompt:delete``). A version bound to a checkpoint is protected →
        ``ConflictError``."""
        self.http.delete("sdk/v2/prompts/detail", self._name_params(prompt_name, agent_id))
        self._invalidate_cache(prompt_name)

    async def adelete(self, prompt_name: str, *, agent_id: Optional[str] = None) -> None:
        """Async sibling of ``delete``."""
        await self.http.adelete("sdk/v2/prompts/detail", self._name_params(prompt_name, agent_id))
        self._invalidate_cache(prompt_name)

    # ---- write internals ----

    def _create_body(
        self, name: str, prompt_content: str, icon: Optional[str],
        description: Optional[str], metadata: Optional[Dict[str, Any]],
        labels: Optional[List[str]], agent_id: Optional[str],
    ) -> Dict[str, Any]:
        agent = require_agent_id(agent_id or self._config.agent_id, "prompts.create")
        body: Dict[str, Any] = {
            "agent_id": agent, "name": name, "prompt_content": prompt_content,
        }
        # omit-None: let the backend apply its defaults (icon="compass",
        # description="", metadata={}, labels=[]) rather than sending explicit nulls.
        if icon is not None:
            body["icon"] = icon
        if description is not None:
            body["description"] = description
        if metadata is not None:
            body["metadata"] = metadata
        if labels is not None:
            body["labels"] = labels
        return body

    @staticmethod
    def _created(response: Dict[str, Any]) -> Tuple[PromptInfo, Optional[PromptVersion]]:
        """Split the create response into the created ``PromptInfo`` and its first
        ``PromptVersion``. The backend returns the prompt payload plus a ``version``
        subobject; ``version`` is ``None`` only if a future backend omits it."""
        info = PromptInfo.from_dict(response)
        raw_version = response.get("version")
        version = PromptVersion.from_dict(raw_version) if isinstance(raw_version, dict) else None
        return info, version

    def _rename_body(
        self, prompt_name: str, new_name: Optional[str], icon: Optional[str], agent_id: Optional[str]
    ) -> Dict[str, Any]:
        agent = require_agent_id(agent_id or self._config.agent_id, "prompts.rename")
        body: Dict[str, Any] = {"agent_id": agent, "prompt_name": prompt_name}
        if new_name is not None:
            body["name"] = new_name  # backend field is "name" (the new name)
        if icon is not None:
            body["icon"] = icon
        return body

    def _label_body(
        self, prompt_name: str, version_number: int, labels: List[str], agent_id: Optional[str]
    ) -> Dict[str, Any]:
        agent = require_agent_id(agent_id or self._config.agent_id, "prompts.set_labels")
        return {
            "agent_id": agent, "prompt_name": prompt_name,
            "version_number": version_number, "labels": labels,
        }

    def _name_params(self, prompt_name: str, agent_id: Optional[str]) -> Dict[str, Any]:
        agent = require_agent_id(agent_id or self._config.agent_id, "prompts.delete")
        return {"agent_id": agent, "prompt_name": prompt_name}

    # ---- read internals ----

    def _agent_params(
        self, agent_id: Optional[str], ordering: Optional[str], page_size: Optional[int]
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "agent_id": require_agent_id(agent_id or self._config.agent_id, "prompts.list"),
        }
        if ordering is not None:
            params["ordering"] = ordering
        if page_size is not None:
            params["page_size"] = page_size
        return params

    def _page_get(self, path: str, base: Dict[str, Any], cursor: Optional[str]) -> Dict[str, Any]:
        params = dict(base)
        if cursor:
            params["cursor"] = cursor
        return self.http.get(path, params)

    async def _apage_get(self, path: str, base: Dict[str, Any], cursor: Optional[str]) -> Dict[str, Any]:
        params = dict(base)
        if cursor:
            params["cursor"] = cursor
        return await self.http.aget(path, params)
