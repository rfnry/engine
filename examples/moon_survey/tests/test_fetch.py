import asyncio
import secrets
from pathlib import Path

import httpx
import pytest
from rfnry_engine import CallInfo

import sandbox as sandbox_module
import tools as tools_module
from sandbox import Sandboxes
from tools import FakeWeb, FetchError, FetchLog, Web, fetch_tool

EXAMPLE = Path(__file__).resolve().parent.parent

HTML = (
    b"<html><head><style>p{}</style><script>var x=1;</script></head>"
    b"<body><h1>Title</h1><p>one</p><p>two</p></body></html>"
)


def web_over(handler, **options):
    return Web(httpx.AsyncClient(transport=httpx.MockTransport(handler)), **options)


def ok(body=b"hello", content_type="text/plain", status=200):
    return httpx.Response(status, content=body, headers={"content-type": content_type})


class Recorder:
    def __init__(self):
        self.rows = []

    async def write(self, **row):
        self.rows.append(row)

    async def urls(self, run_id):
        return {r["url"] for r in self.rows if r["run_id"] == run_id}


@pytest.fixture
def caller(monkeypatch, tmp_path):
    async def run(tool, args, run_id="a"):
        call = CallInfo(run_id=run_id, agent="surveyor", scope="moon", tool="fetch", call_id="c")
        monkeypatch.setattr(tools_module, "current_call", lambda: call)
        monkeypatch.setattr(sandbox_module, "current_call", lambda: call)
        return await tool.body(args)

    return run


def tool_of(web, tmp_path, log=None):
    sandboxes = Sandboxes(
        tmp_path, agents=EXAMPLE / "agents", data=EXAMPLE / "tests" / "data" / "lp_cut"
    )
    return fetch_tool(web, sandboxes, log or Recorder())


async def test_a_host_off_the_list_is_refused():
    web = web_over(lambda request: ok())
    with pytest.raises(FetchError, match="fetch: host 'evil.example' is not on the list"):
        await web.get("https://evil.example/x")


async def test_only_https_is_allowed():
    web = web_over(lambda request: ok())
    with pytest.raises(FetchError, match="fetch: only https is allowed; got 'http://arxiv.org/x'"):
        await web.get("http://arxiv.org/x")


async def test_a_subdomain_off_the_list_is_refused():
    web = web_over(lambda request: ok())
    with pytest.raises(FetchError, match="is not on the list"):
        await web.get("https://www.arxiv.org/x")


async def test_a_redirect_inside_the_list_is_followed():
    def handler(request):
        if request.url.path == "/a":
            return httpx.Response(302, headers={"location": "https://pds.nasa.gov/b"})
        return ok(b"landed")

    page = await web_over(handler).get("https://arxiv.org/a")
    assert page.body == b"landed"
    assert page.url == "https://pds.nasa.gov/b"


async def test_a_redirect_leaving_the_list_is_refused():
    def handler(request):
        return httpx.Response(302, headers={"location": "https://evil.example/x"})

    with pytest.raises(
        FetchError, match="fetch: a redirect to 'https://evil.example/x' leaves the list"
    ):
        await web_over(handler).get("https://arxiv.org/a")


async def test_too_many_redirects_are_refused():
    def handler(request):
        return httpx.Response(302, headers={"location": "https://arxiv.org/again"})

    with pytest.raises(FetchError, match="fetch: more than 5 redirects"):
        await web_over(handler).get("https://arxiv.org/a")


async def test_a_network_error_is_a_fetch_error():
    def handler(request):
        raise httpx.ConnectError("down")

    with pytest.raises(FetchError, match="^fetch: "):
        await web_over(handler).get("https://arxiv.org/a")


async def test_html_reads_as_text_cut_at_the_cap(caller, tmp_path):
    web = web_over(lambda request: ok(HTML, "text/html; charset=utf-8"), text_cap=12)
    result = await caller(tool_of(web, tmp_path), {"url": "https://arxiv.org/a"})
    assert result["status"] == 200
    assert result["content_type"].startswith("text/html")
    assert result["cut"] is True
    assert result["text"] == "Title\none\ntwo"[:12]
    assert "var x" not in result["text"]

    whole = web_over(lambda request: ok(HTML, "text/html"))
    result = await caller(tool_of(whole, tmp_path), {"url": "https://arxiv.org/a"})
    assert result["text"] == "Title\none\ntwo"
    assert result["cut"] is False


async def test_a_non_200_is_a_result(caller, tmp_path):
    web = web_over(lambda request: ok(b"gone", status=404))
    result = await caller(tool_of(web, tmp_path), {"url": "https://arxiv.org/a"})
    assert result["status"] == 404
    assert result["text"] == "gone"


async def test_a_binary_body_is_not_read(caller, tmp_path):
    web = web_over(lambda request: ok(b"\x00\x01\x02", "application/octet-stream"))
    result = await caller(tool_of(web, tmp_path), {"url": "https://arxiv.org/a"})
    assert result == {
        "url": "https://arxiv.org/a",
        "status": 200,
        "content_type": "application/octet-stream",
        "bytes": 3,
        "note": "binary; fetch it again with save to write it into your sandbox",
    }


async def test_save_writes_into_the_callers_sandbox_only(caller, tmp_path):
    web = web_over(lambda request: ok(b"\x00\x01\x02", "application/octet-stream"))
    tool = tool_of(web, tmp_path)
    result = await caller(tool, {"url": "https://arxiv.org/a", "save": "data/x.bin"}, run_id="r1")
    assert result["saved"] == "data/x.bin"
    assert result["bytes"] == 3
    assert (tmp_path / "r1" / "data" / "x.bin").read_bytes() == b"\x00\x01\x02"
    assert not (tmp_path / "r2").exists()


@pytest.mark.parametrize("bad", ["../x", "/etc/x", "a/../../x"])
async def test_a_save_path_leaving_the_sandbox_is_refused(caller, tmp_path, bad):
    tool = tool_of(web_over(lambda request: ok()), tmp_path)
    with pytest.raises(
        FetchError, match=f"fetch: save must be a path inside your sandbox; got '{bad}'"
    ):
        await caller(tool, {"url": "https://arxiv.org/a", "save": bad})
    assert not (tmp_path / "x").exists()


async def test_a_symlink_leaving_the_sandbox_is_refused(caller, tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "outside").mkdir()
    (tmp_path / "a" / "link").symlink_to(tmp_path / "outside")
    tool = tool_of(web_over(lambda request: ok()), tmp_path)
    with pytest.raises(FetchError, match="^fetch: could not write 'link/x': .*outside"):
        await caller(tool, {"url": "https://arxiv.org/a", "save": "link/x"})
    assert not (tmp_path / "outside" / "x").exists()


@pytest.mark.parametrize("save", ["notes.txt", "data/x.bin"])
async def test_a_symlink_swapped_in_during_the_fetch_is_not_written_through(caller, tmp_path, save):
    place = tmp_path / "a"
    (place / "data").mkdir(parents=True)
    (place / "notes.txt").write_text("old")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "notes.txt").write_text("mine")

    def swap(request):
        first = Path(save).parts[0]
        target = place / first
        if target.is_dir():
            target.rmdir()
        else:
            target.unlink()
        target.symlink_to(outside / first if first == "notes.txt" else outside)
        return ok(b"page")

    tool = tool_of(web_over(swap), tmp_path)
    with pytest.raises(FetchError, match=f"^fetch: could not write '{save}': .*outside"):
        await caller(tool, {"url": "https://arxiv.org/a", "save": save})
    assert (outside / "notes.txt").read_text() == "mine"
    assert sorted(p.name for p in outside.iterdir()) == ["notes.txt"]


async def test_a_body_past_the_save_cap_is_refused(caller, tmp_path):
    web = web_over(lambda request: ok(b"x" * 20), save_cap=10)
    with pytest.raises(FetchError, match="^fetch: .*10"):
        await caller(tool_of(web, tmp_path), {"url": "https://arxiv.org/a", "save": "x"})
    assert not (tmp_path / "a" / "x").exists()


async def test_the_same_url_fetched_together_is_one_request():
    count = 0

    def handler(request):
        nonlocal count
        count += 1
        return ok()

    async def slow(seconds):
        await asyncio.sleep(0)

    web = web_over(handler, sleep=slow)
    pages = await asyncio.gather(*(web.get("https://pds.nasa.gov/a") for _ in range(20)))
    assert count == 1
    assert sum(not p.cached for p in pages) == 1
    again = await web.get("https://pds.nasa.gov/a")
    assert again.cached is True
    assert count == 1


async def test_a_failed_request_is_not_cached():
    count = 0

    def handler(request):
        nonlocal count
        count += 1
        if count == 1:
            raise httpx.ConnectError("down")
        return ok()

    web = web_over(handler)
    with pytest.raises(FetchError):
        await web.get("https://pds.nasa.gov/a")
    assert (await web.get("https://pds.nasa.gov/a")).status == 200


async def test_each_host_is_paced():
    now = [100.0]
    sleeps = []
    starts = []

    async def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    def handler(request):
        starts.append((request.url.host, now[0]))
        return ok()

    web = web_over(handler, clock=lambda: now[0], sleep=sleep)
    await web.get("https://arxiv.org/a")
    await web.get("https://arxiv.org/b")
    await web.get("https://pds.nasa.gov/a")
    await web.get("https://arxiv.org/a")
    assert starts == [("arxiv.org", 100.0), ("arxiv.org", 103.0), ("pds.nasa.gov", 103.0)]
    assert sleeps == [3.0]


async def test_every_fetch_is_logged(pool, caller, tmp_path):
    log = FetchLog(pool, "moon", secrets.token_hex(4))
    web = web_over(lambda request: ok(b"abc"))
    tool = tool_of(web, tmp_path, log)
    await caller(tool, {"url": "https://arxiv.org/a"}, run_id="r1")
    await caller(tool, {"url": "https://arxiv.org/a", "save": "a.txt"}, run_id="r1")
    await caller(tool, {"url": "https://arxiv.org/b"}, run_id="r2")
    rows = await pool.fetch(
        "select run_id, url, status, bytes, saved, cached from fetches"
        " where survey = $1 order by at, run_id",
        log.survey,
    )
    assert [tuple(r) for r in rows] == [
        ("r1", "https://arxiv.org/a", 200, 3, None, False),
        ("r1", "https://arxiv.org/a", 200, 3, "a.txt", True),
        ("r2", "https://arxiv.org/b", 200, 3, None, False),
    ]
    assert await log.urls("r1") == {"https://arxiv.org/a"}
    assert await log.urls("r3") == set()


async def test_a_refused_fetch_is_logged_and_still_an_error(caller, tmp_path):
    log = Recorder()
    tool = tool_of(web_over(lambda request: ok()), tmp_path, log)
    with pytest.raises(FetchError):
        await caller(tool, {"url": "https://evil.example/x"})
    assert [r["url"] for r in log.rows] == ["https://evil.example/x"]
    assert log.rows[0]["status"] == 0


async def test_a_redirect_logs_the_final_url_too(caller, tmp_path):
    def handler(request):
        if request.url.path == "/a":
            return httpx.Response(302, headers={"location": "https://pds.nasa.gov/b"})
        return ok()

    log = Recorder()
    await caller(tool_of(web_over(handler), tmp_path, log), {"url": "https://arxiv.org/a"})
    assert await log.urls("a") == {"https://arxiv.org/a", "https://pds.nasa.gov/b"}


async def test_the_fetch_tool_raises_outside_a_run(tmp_path):
    tool = tool_of(web_over(lambda request: ok()), tmp_path)
    with pytest.raises(FetchError, match="^fetch: "):
        await tool.body({"url": "https://arxiv.org/a"})


async def test_the_fake_web_serves_its_pages_and_404s_the_rest(caller, tmp_path):
    web = FakeWeb({"https://arxiv.org/a": ("text/html", HTML)})
    page = await web.get("https://arxiv.org/a")
    assert (page.status, page.content_type, page.body) == (200, "text/html", HTML)
    missing = await web.get("https://arxiv.org/none")
    assert missing.status == 404
    result = await caller(tool_of(web, tmp_path), {"url": "https://arxiv.org/a"})
    assert result["text"] == "Title\none\ntwo"


async def test_only_a_2xx_body_is_cached():
    count = 0

    def handler(request):
        nonlocal count
        count += 1
        return ok(status=503) if count == 1 else ok()

    web = web_over(handler)
    assert (await web.get("https://pds.nasa.gov/a")).status == 503
    again = await web.get("https://pds.nasa.gov/a")
    assert (again.status, again.cached) == (200, False)
    assert (await web.get("https://pds.nasa.gov/a")).cached is True


async def test_a_body_past_two_megabytes_is_fetched_again():
    count = 0

    def handler(request):
        nonlocal count
        count += 1
        return ok(b"x" * 2_000_001)

    web = web_over(handler)
    await web.get("https://pds.nasa.gov/a")
    again = await web.get("https://pds.nasa.gov/a")
    assert count == 2
    assert again.cached is False


async def test_the_cache_drops_the_least_recently_used_past_its_budget():
    count = {}

    def handler(request):
        count[request.url.path] = count.get(request.url.path, 0) + 1
        return ok(b"x" * 10)

    web = web_over(handler, cache_bytes=25, default_pace=0)
    await web.get("https://pds.nasa.gov/a")
    await web.get("https://pds.nasa.gov/b")
    await web.get("https://pds.nasa.gov/a")
    await web.get("https://pds.nasa.gov/c")
    assert web.cached_total == 20
    await web.get("https://pds.nasa.gov/a")
    await web.get("https://pds.nasa.gov/b")
    assert count == {"/a": 1, "/b": 2, "/c": 1}


async def test_real_downloads_are_bounded():
    running = 0
    peak = 0

    async def handler(request):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.01)
        running -= 1
        return ok()

    web = web_over(handler, downloads=2, default_pace=0)
    await asyncio.gather(*(web.get(f"https://pds.nasa.gov/{n}") for n in range(8)))
    assert peak == 2


async def test_a_client_that_follows_redirects_cannot_bypass_the_hop_check():
    def handler(request):
        return httpx.Response(302, headers={"location": "https://evil.example/x"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)
    with pytest.raises(FetchError, match="leaves the list"):
        await Web(client).get("https://arxiv.org/a")


@pytest.mark.parametrize("url", ["https://arxiv.org:8443/x", "https://arxiv.org:bad/x"])
async def test_a_port_other_than_443_is_refused(url):
    with pytest.raises(FetchError, match="only port 443"):
        await web_over(lambda request: ok()).get(url)


async def test_port_443_is_allowed():
    page = await web_over(lambda request: ok()).get("https://arxiv.org:443/x")
    assert page.status == 200


async def test_a_slow_drip_body_stops_at_the_deadline():
    async def drip():
        for _ in range(100):
            await asyncio.sleep(0.01)
            yield b"x"

    def handler(request):
        return httpx.Response(200, content=drip(), headers={"content-type": "text/plain"})

    with pytest.raises(FetchError, match="did not finish within"):
        await web_over(handler, timeout=0.05).get("https://arxiv.org/a")


async def test_a_save_path_that_is_a_directory_is_a_fetch_error(caller, tmp_path):
    (tmp_path / "a" / "d").mkdir(parents=True)
    tool = tool_of(web_over(lambda request: ok()), tmp_path)
    with pytest.raises(FetchError, match="^fetch: could not write 'd'"):
        await caller(tool, {"url": "https://arxiv.org/a", "save": "d"})


async def test_a_non_200_binary_body_is_not_read(caller, tmp_path):
    web = web_over(lambda request: ok(b"\x00\x01", "application/octet-stream", status=500))
    result = await caller(tool_of(web, tmp_path), {"url": "https://arxiv.org/a"})
    assert result["status"] == 500
    assert "text" not in result
    assert result["note"].startswith("binary;")


async def test_the_declared_charset_is_honoured(caller, tmp_path):
    web = web_over(lambda request: ok("café".encode("latin-1"), "text/plain; charset=iso-8859-1"))
    result = await caller(tool_of(web, tmp_path), {"url": "https://arxiv.org/a"})
    assert result["text"] == "café"
    odd = web_over(lambda request: ok(b"ok", "text/plain; charset=nonsense"))
    result = await caller(tool_of(odd, tmp_path), {"url": "https://arxiv.org/a"})
    assert result["text"] == "ok"


async def test_the_deadline_starts_when_the_download_slot_is_held():
    async def handler(request):
        await asyncio.sleep(0.06)
        return ok()

    web = web_over(handler, downloads=1, timeout=0.1, default_pace=0)
    pages = await asyncio.gather(*(web.get(f"https://pds.nasa.gov/{n}") for n in range(3)))
    assert [p.status for p in pages] == [200, 200, 200]


async def test_a_3xx_without_a_location_is_a_result(caller, tmp_path):
    web = web_over(lambda request: httpx.Response(304, content=b""))
    result = await caller(tool_of(web, tmp_path), {"url": "https://arxiv.org/a"})
    assert result["status"] == 304
    assert result["text"] == ""


async def test_a_3xx_with_a_body_and_no_content_type_is_not_a_hop():
    page = await web_over(lambda request: httpx.Response(300, content=b"choose")).get(
        "https://arxiv.org/a"
    )
    assert (page.status, page.body) == (300, b"choose")


async def test_an_unwritable_save_says_why(caller, tmp_path):
    place = tmp_path / "a"
    place.mkdir()
    place.chmod(0o500)
    tool = tool_of(web_over(lambda request: ok()), tmp_path)
    try:
        with pytest.raises(FetchError) as refused:
            await caller(tool, {"url": "https://arxiv.org/a", "save": "x"})
    finally:
        place.chmod(0o700)
    assert str(refused.value) == (
        "fetch: could not write 'x': tools/workspace: cannot write x: Permission denied"
    )


async def test_a_3xx_with_an_empty_location_is_a_result():
    def handler(request):
        return httpx.Response(302, content=b"moved", headers={"location": "  "})

    page = await web_over(handler).get("https://arxiv.org/a")
    assert (page.status, page.body) == (302, b"moved")
