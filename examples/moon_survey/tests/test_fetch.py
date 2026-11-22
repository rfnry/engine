import asyncio
import secrets

import httpx
import pytest
from rfnry_engine import CallInfo

import tools as tools_module
from tools import FakeWeb, FetchError, FetchLog, Web, fetch_tool

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
        monkeypatch.setattr(
            tools_module,
            "current_call",
            lambda: CallInfo(
                run_id=run_id, agent="surveyor", scope="moon", tool="fetch", call_id="c"
            ),
        )
        return await tool.body(args)

    return run


def tool_of(web, tmp_path, log=None):
    return fetch_tool(web, lambda run_id: tmp_path / run_id, log or Recorder())


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
    with pytest.raises(FetchError, match="save must be a path inside your sandbox"):
        await caller(tool, {"url": "https://arxiv.org/a", "save": "link/x"})
    assert not (tmp_path / "outside" / "x").exists()


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
