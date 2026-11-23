import asyncio
import codecs
import re
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from html.parser import HTMLParser
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urljoin, urlsplit

import asyncpg
import httpx
from rfnry_engine import InputError, Tool, Tools, current_call

from sandbox import Sandboxes
from setup import USER_AGENT

HOSTS = (
    "pds-geosciences.wustl.edu",
    "pds.nasa.gov",
    "astrogeology.usgs.gov",
    "ssd.jpl.nasa.gov",
    "ssd-api.jpl.nasa.gov",
    "ntrs.nasa.gov",
    "arxiv.org",
    "export.arxiv.org",
)
MAX_REDIRECTS = 5
CACHE_ITEM = 2_000_000
CHARSET = re.compile(r'charset=["\']?([\w.-]+)', re.IGNORECASE)
REDIRECTS = (301, 302, 303, 307, 308)
SKIPPED = ("script", "style")
BLOCKS = (
    "p",
    "div",
    "br",
    "li",
    "ul",
    "ol",
    "tr",
    "table",
    "section",
    "article",
    "pre",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "blockquote",
    "header",
    "footer",
)
TEXT_TYPES = ("application/json", "application/xml", "application/javascript", "application/csv")
WRITE = """
insert into fetches (scope, survey, run_id, url, status, bytes, saved, cached)
values ($1, $2, $3, $4, $5, $6, $7, $8)
"""
URLS = "select distinct url from fetches where scope = $1 and survey = $2 and run_id = $3"


class FetchError(Exception):
    pass


@dataclass(frozen=True)
class Page:
    url: str
    status: int
    content_type: str
    body: bytes
    cached: bool = False


def base_type(content_type: str) -> str:
    return content_type.split(";")[0].strip().lower()


def is_text(content_type: str) -> bool:
    kind = base_type(content_type)
    return kind.startswith("text/") or kind in TEXT_TYPES or kind.endswith(("+xml", "+json"))


class Reader(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skipped = 0

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in SKIPPED:
            self.skipped += 1
        elif tag in BLOCKS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in SKIPPED:
            self.skipped = max(0, self.skipped - 1)
        elif tag in BLOCKS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.skipped:
            self.parts.append(data)


def html_text(source: str) -> str:
    reader = Reader()
    reader.feed(source)
    reader.close()
    lines = (" ".join(line.split()) for line in "".join(reader.parts).splitlines())
    return "\n".join(line for line in lines if line)


def decode(body: bytes, content_type: str) -> str:
    found = CHARSET.search(content_type)
    encoding = "utf-8"
    if found:
        try:
            encoding = codecs.lookup(found.group(1)).name
        except LookupError:
            encoding = "utf-8"
    return body.decode(encoding, errors="replace")


def read_page(page: Page) -> str:
    text = decode(page.body, page.content_type)
    if base_type(page.content_type) in ("text/html", "application/xhtml+xml"):
        return html_text(text)
    return text


class Web:
    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        hosts: tuple[str, ...] = HOSTS,
        text_cap: int = 60_000,
        save_cap: int = 50_000_000,
        timeout: float = 30.0,
        pace: dict[str, float] | None = None,
        default_pace: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        cache_bytes: int = 200_000_000,
        downloads: int = 4,
    ) -> None:
        self.client = client
        self.hosts = hosts
        self.text_cap = text_cap
        self.save_cap = save_cap
        self.timeout = timeout
        self.pace = {"arxiv.org": 3.0, "export.arxiv.org": 3.0} if pace is None else pace
        self.default_pace = default_pace
        self.clock = clock
        self.sleep = sleep
        self.cache_bytes = cache_bytes
        self.cache: OrderedDict[str, Page] = OrderedDict()
        self.cached_total = 0
        self.slots = asyncio.Semaphore(downloads)
        self.flying: dict[str, asyncio.Task[Page]] = {}
        self.locks: dict[str, asyncio.Lock] = {}
        self.started: dict[str, float] = {}

    def check(self, url: str, *, redirect: bool = False) -> str:
        parts = urlsplit(url)
        host = (parts.hostname or "").lower()
        if redirect and host not in self.hosts:
            raise FetchError(f"fetch: a redirect to '{url}' leaves the list")
        if parts.scheme != "https":
            raise FetchError(f"fetch: only https is allowed; got '{url}'")
        if host not in self.hosts:
            raise FetchError(f"fetch: host '{host}' is not on the list")
        try:
            port = parts.port
        except ValueError:
            port = -1
        if port not in (None, 443):
            raise FetchError(f"fetch: only port 443 is allowed; got '{url}'")
        return host

    async def wait_turn(self, host: str) -> None:
        lock = self.locks.setdefault(host, asyncio.Lock())
        async with lock:
            gap = self.pace.get(host, self.default_pace)
            last = self.started.get(host)
            if last is not None:
                wait = last + gap - self.clock()
                if wait > 0:
                    await self.sleep(wait)
            self.started[host] = self.clock()

    async def request(self, url: str) -> tuple[Page, str | None]:
        try:
            async with (
                self.slots,
                asyncio.timeout(self.timeout),
                self.client.stream(
                    "GET",
                    url,
                    headers={"User-Agent": USER_AGENT},
                    timeout=self.timeout,
                    follow_redirects=False,
                ) as response,
            ):
                if (
                    response.status_code in REDIRECTS
                    and response.headers.get("location", "").strip()
                ):
                    return Page(url, response.status_code, "", b""), response.headers["location"]
                chunks: list[bytes] = []
                size = 0
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > self.save_cap:
                        raise FetchError(
                            f"fetch: '{url}' is larger than the cap of {self.save_cap} bytes"
                        )
                    chunks.append(chunk)
                page = Page(
                    url,
                    response.status_code,
                    response.headers.get("content-type", ""),
                    b"".join(chunks),
                )
                return page, None
        except TimeoutError:
            raise FetchError(
                f"fetch: '{url}' did not finish within {self.timeout} seconds"
            ) from None
        except httpx.HTTPError as error:
            raise FetchError(f"fetch: {type(error).__name__} for '{url}': {error}") from None

    def keep(self, url: str, page: Page) -> None:
        if not 200 <= page.status < 300 or len(page.body) > CACHE_ITEM:
            return
        old = self.cache.pop(url, None)
        if old is not None:
            self.cached_total -= len(old.body)
        self.cache[url] = page
        self.cached_total += len(page.body)
        while self.cached_total > self.cache_bytes and self.cache:
            _, dropped = self.cache.popitem(last=False)
            self.cached_total -= len(dropped.body)

    async def load(self, url: str) -> Page:
        current = url
        host = self.check(current)
        for _ in range(MAX_REDIRECTS + 1):
            await self.wait_turn(host)
            page, location = await self.request(current)
            if location is not None:
                current = urljoin(current, location)
                host = self.check(current, redirect=True)
                continue
            self.keep(url, page)
            return page
        raise FetchError(f"fetch: more than {MAX_REDIRECTS} redirects from '{url}'")

    async def get(self, url: str) -> Page:
        found = self.cache.get(url)
        if found is not None:
            self.cache.move_to_end(url)
            return replace(found, cached=True)
        flying = self.flying.get(url)
        if flying is not None:
            return replace(await asyncio.shield(flying), cached=True)
        task = asyncio.ensure_future(self.load(url))
        self.flying[url] = task
        task.add_done_callback(lambda _: self.flying.pop(url, None))
        return await asyncio.shield(task)


class FakeWeb:
    def __init__(
        self,
        pages: dict[str, tuple[str, bytes]],
        *,
        text_cap: int = 60_000,
        save_cap: int = 50_000_000,
    ) -> None:
        self.pages = pages
        self.text_cap = text_cap
        self.save_cap = save_cap

    async def get(self, url: str) -> Page:
        found = self.pages.get(url)
        if found is None:
            return Page(url, 404, "text/plain", b"not found")
        content_type, body = found
        if len(body) > self.save_cap:
            raise FetchError(f"fetch: '{url}' is larger than the cap of {self.save_cap} bytes")
        return Page(url, 200, content_type, body)


class FetchLog:
    def __init__(self, pool: asyncpg.Pool, scope: str, survey: str) -> None:
        self.pool = pool
        self.scope = scope
        self.survey = survey

    async def write(
        self, *, run_id: str, url: str, status: int, bytes: int, saved: str | None, cached: bool
    ) -> None:
        await self.pool.execute(
            WRITE, self.scope, self.survey, run_id, url, status, bytes, saved, cached
        )

    async def urls(self, run_id: str) -> set[str]:
        rows = await self.pool.fetch(URLS, self.scope, self.survey, run_id)
        return {row["url"] for row in rows}


def inside(save: str) -> str:
    if save.startswith("/") or ".." in PurePosixPath(save).parts:
        raise FetchError(f"fetch: save must be a path inside your sandbox; got '{save}'")
    return save


def fetch_tool(web: Any, sandboxes: Sandboxes, log: Any) -> Tool:
    async def fetch(args: dict[str, Any]) -> dict[str, Any]:
        call = current_call()
        if call is None:
            raise FetchError("fetch: no run is calling")
        run_id = call.run_id
        url = args["url"]
        save = args.get("save")
        target = inside(save) if save else None
        try:
            page = await web.get(url)
        except FetchError:
            await log.write(run_id=run_id, url=url, status=0, bytes=0, saved=None, cached=False)
            raise
        saved = None
        refused: Exception | None = None
        if target is not None and page.status == 200:
            try:
                await sandboxes.write(target, page.body)
                saved = save
            except InputError as error:
                refused = FetchError(f"fetch: could not write '{save}': {error}")
            except Exception as error:
                refused = error
        for logged in dict.fromkeys((url, page.url)):
            await log.write(
                run_id=run_id,
                url=logged,
                status=page.status,
                bytes=len(page.body),
                saved=saved,
                cached=page.cached,
            )
        if refused is not None:
            raise refused
        content_type = page.content_type
        if saved is not None:
            return {
                "url": page.url,
                "status": page.status,
                "content_type": content_type,
                "saved": saved,
                "bytes": len(page.body),
            }
        if page.body and not is_text(content_type):
            return {
                "url": page.url,
                "status": page.status,
                "content_type": content_type,
                "bytes": len(page.body),
                "note": "binary; fetch it again with save to write it into your sandbox",
            }
        text = await asyncio.to_thread(read_page, page)
        return {
            "url": page.url,
            "status": page.status,
            "content_type": content_type,
            "text": text[: web.text_cap],
            "cut": len(text) > web.text_cap,
        }

    return Tool(
        "fetch",
        "Read a web page, or save it into your sandbox. Only public science hosts answer; "
        "pages are text, and a binary file is written with save, a path inside your sandbox.",
        body=fetch,
        input_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "an https URL"},
                "save": {
                    "type": "string",
                    "description": "a path inside your sandbox to write the body to",
                },
            },
            "required": ["url"],
        },
    )


def boss_tools(surveyors: Any) -> Tools:
    async def progress(args: dict[str, Any]) -> dict[str, Any]:
        return await surveyors.progress()

    async def start(args: dict[str, Any]) -> dict[str, Any]:
        return await surveyors.start(args["region"])

    async def check(args: dict[str, Any]) -> dict[str, Any]:
        return await surveyors.check(args["job"])

    return Tools(
        [
            Tool(
                "progress",
                "Read the survey's board: its counts of landed, failed, claimed and free regions,"
                " the regions you may start now, and how many surveyors are running.",
                body=progress,
                input_schema={"type": "object", "properties": {}},
            ),
            Tool(
                "start",
                "Start a surveyor on a region that progress says you may start. Answers its job id"
                " at once, such as s1; the surveyor works while you go on.",
                body=start,
                input_schema={
                    "type": "object",
                    "properties": {"region": {"type": "string", "description": "A region id"}},
                    "required": ["region"],
                },
            ),
            Tool(
                "check",
                "Read a surveyor you started, by its job id: running, or how it ended, with its"
                " reply, its error and whether its region landed.",
                body=check,
                input_schema={
                    "type": "object",
                    "properties": {
                        "job": {"type": "string", "description": "A job id, such as s1"}
                    },
                    "required": ["job"],
                },
            ),
        ]
    )
