"""HTTP 接口层测试。需要先启动服务：

    python scripts/run_server.py

服务未启动时整组测试自动 skip，不会导致 CI 失败。
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import uuid

import pytest

BASE = "http://127.0.0.1:8000"

# 关键：urllib 会读取 Windows 注册表里的系统代理（WinINET），把 127.0.0.1 也路由到代理，
# 导致本机服务返回 404（本机实测系统代理为 127.0.0.1:26561）。用空代理 handler 强制直连。
_NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _request(method: str, path: str, payload: dict | None = None, timeout: int = 60):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    with _NO_PROXY_OPENER.open(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return resp.status, (json.loads(body) if body.strip().startswith(("{", "[")) else body)


@pytest.fixture(scope="module")
def server():
    """等待服务就绪后返回 base URL。

    为什么用轮询而非一次性探测：服务启动阶段会预热本地模型（本机 BGE-M3 在 CPU
    上约 10~20s），在 lifespan 完成前路由会返回 404 或连接拒绝。
    若只在开头探测一次，整组接口测试会被误判为"服务未启动"而跳过。
    """
    deadline = time.monotonic() + 120
    last_exc: Exception | None = None
    while time.monotonic() < deadline:
        try:
            _request("GET", "/api/v1/health", timeout=5)
            return BASE
        except Exception as exc:  # noqa: BLE001, PERF203  轮询等待属预期，逐次捕获
            last_exc = exc
            time.sleep(3)
    pytest.skip(f"HTTP 服务未在 120s 内就绪（{last_exc}），跳过接口测试")


DOC = """# HTTP 接口测试文档

## 缓存策略

父块内容写入 Redis，key 形如 psd:parent:123，读取使用 MGET 批量获取。

## 唯一标记

本段包含唯一的检索标记 ZQPX8842，用于验证「子块检索 → parent_id 回表」链路。
该标记只在本文档中出现，稀疏分支应当能精准命中它。

## 向量检索

子块在 Milvus 中同时保存稠密向量与稀疏向量，检索时做 RRF 融合。
"""


def test_health_and_stats(server) -> None:
    _, health = _request("GET", "/api/v1/health")
    assert health["mysql"] and health["redis"] and health["milvus"]
    assert health["detail"]["milvus"]["collection"]

    _, stats = _request("GET", "/api/v1/stats")
    assert stats["mysql_parent_count"] >= 0
    assert stats["milvus_child_count"] >= 0
    assert 0.0 <= stats["cache_hit_rate"] <= 1.0


def test_index_page_served(server) -> None:
    with _NO_PROXY_OPENER.open(BASE + "/", timeout=15) as resp:
        html = resp.read().decode("utf-8")
    assert resp.status == 200
    assert "父子分块 RAG 检索系统" in html


def test_ingest_search_chat_delete_flow(server) -> None:
    doc_id = f"http_{uuid.uuid4().hex[:12]}"

    # 入库
    status, created = _request("POST", "/api/v1/documents", {
        "doc_id": doc_id, "title": "HTTP 接口测试文档", "content": DOC, "source": "pytest",
    })
    assert status == 200
    assert created["parent_count"] >= 1
    parent_id = created["parent_ids"][0]

    # 文档列表
    _, docs = _request("GET", "/api/v1/documents")
    assert any(d["doc_id"] == doc_id for d in docs)

    # 检索：用文档内唯一标记查询，保证它一定被判为最相关
    _, search = _request("POST", "/api/v1/search", {
        "query": "唯一检索标记 ZQPX8842 在哪个父块里？", "top_k": 20, "final_top_k": 5,
    })
    assert search["parents"], "检索结果不应为空"
    assert any(p["parent_id"] == parent_id for p in search["parents"]), (
        f"应命中刚入库的父块 {parent_id}，实际命中 {[p['parent_id'] for p in search['parents']]}"
    )
    assert "milvus_ms" in search["timings"]

    # 按 parent_id 直取
    _, parent = _request("GET", f"/api/v1/parents/{parent_id}")
    assert parent["id"] == parent_id and parent["content"]

    # 问答（本地 7B 模型首次调用含加载时间，故放宽超时）
    _, chat = _request("POST", "/api/v1/chat", {
        "query": "唯一检索标记 ZQPX8842 是什么？", "top_k": 20, "final_top_k": 5,
    }, timeout=180)
    assert chat["answer"] and chat["citations"]
    assert chat["llm_mode"]

    # 三路对比
    _, cmp_result = _request("POST", "/api/v1/compare", {
        "query": "RRF 融合", "top_k": 10, "final_top_k": 3,
    })
    assert "dense_only" in cmp_result and "sparse_only" in cmp_result and "hybrid" in cmp_result

    # 删除
    _, deleted = _request("DELETE", f"/api/v1/documents/{doc_id}")
    assert deleted["deleted_parents"] >= 1
    _, after = _request("GET", "/api/v1/documents")
    assert not any(d["doc_id"] == doc_id for d in after)


def test_parent_not_found_returns_404(server) -> None:
    with pytest.raises(urllib.error.HTTPError) as exc:
        _request("GET", "/api/v1/parents/99999999")
    assert exc.value.code == 404


def test_empty_query_rejected(server) -> None:
    with pytest.raises(urllib.error.HTTPError) as exc:
        _request("POST", "/api/v1/search", {"query": "", "top_k": 5, "final_top_k": 2})
    assert exc.value.code == 422


# ----------------------------------------------------------------------
# 上传接口的防护
# ----------------------------------------------------------------------
def _multipart(field: str, filename: str, content: bytes) -> tuple[bytes, str]:
    """手工拼一个 multipart/form-data 请求体（避免为测试引入额外依赖）。"""
    boundary = "----parsubsplitboundary"
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode()
    tail = f"\r\n--{boundary}--\r\n".encode()
    return head + content + tail, f"multipart/form-data; boundary={boundary}"


def _upload(filename: str, content: bytes, server: str) -> tuple[int, dict]:
    body, ctype = _multipart("file", filename, content)
    req = urllib.request.Request(
        BASE + "/api/v1/documents/upload", data=body, method="POST",
        headers={"Content-Type": ctype},
    )
    with _NO_PROXY_OPENER.open(req, timeout=120) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def test_upload_rejects_disallowed_extension(server) -> None:
    """扩展名白名单：非白名单文件必须 400，且不应产生任何入库副作用。"""
    with pytest.raises(urllib.error.HTTPError) as exc:
        _upload("malicious.exe", b"MZ\x90\x00fake-binary", server)
    assert exc.value.code == 400
    assert "不支持的文件类型" in exc.value.read().decode("utf-8")


def test_upload_accepts_markdown(server) -> None:
    """白名单内的 .md 应正常入库，并可被检索到。"""
    marker = f"UPLDMK{uuid.uuid4().hex[:8]}"
    content = (
        f"# 上传测试文档\n\n## 唯一标记\n\n本文包含唯一上传标记 {marker}，用于验证上传链路。\n\n"
        "## 其余内容\n\n这一段用于凑出足够的长度以便切分。\n"
    ).encode()
    status, created = _upload(f"{marker}.md", content, server)
    assert status == 200
    assert created["parent_count"] >= 1

    try:
        _, search = _request("POST", "/api/v1/search", {
            "query": f"唯一上传标记 {marker}", "top_k": 20, "final_top_k": 5,
        })
        assert any(p["parent_id"] in created["parent_ids"] for p in search["parents"])
    finally:
        _request("DELETE", f"/api/v1/documents/{created['doc_id']}")


def test_parent_metadata_consistent_across_cache_states(server) -> None:
    """`/parents/{id}` 无论正文来自 Redis 还是 MySQL，元数据都必须完整。

    回归历史缺陷：缓存命中时只回正文，doc_id/title 变成空字符串。
    """
    doc_id = f"http_meta_{uuid.uuid4().hex[:8]}"
    _, created = _request("POST", "/api/v1/documents", {
        "doc_id": doc_id, "title": "元数据一致性测试", "content": DOC, "source": "pytest",
    })
    pid = created["parent_ids"][0]
    try:
        _, first = _request("GET", f"/api/v1/parents/{pid}")
        _, second = _request("GET", f"/api/v1/parents/{pid}")
        for view in (first, second):
            assert view["doc_id"] == doc_id
            assert view["title"] == "元数据一致性测试"
            assert view["child_count"] >= 1
            assert view["content"]
        # 两次读取至少有一次命中缓存（第一次可能回表）
        assert first["from_redis"] or second["from_redis"], "重复读取应命中 Redis 缓存"
    finally:
        _request("DELETE", f"/api/v1/documents/{doc_id}")


def test_stats_memoization_and_force(server) -> None:
    """`/stats` 有 5 秒记忆化，`?force=true` 应强制重新统计。"""
    _, a = _request("GET", "/api/v1/stats")
    _, b = _request("GET", "/api/v1/stats")
    assert a == b, "5 秒内两次调用应返回同一份记忆化结果"
    _, c = _request("GET", "/api/v1/stats?force=true")
    assert c["mysql_parent_count"] == a["mysql_parent_count"]


def test_admin_endpoint_without_token_when_unconfigured(server) -> None:
    """未配置 ADMIN_TOKEN 时管理接口不校验（本机开发模式）。

    令牌强制校验的分支由 tests/test_api_security.py 的单元测试覆盖，
    因为那里可以安全地 monkeypatch 配置而无需重启服务。
    """
    status, body = _request("POST", "/api/v1/admin/reset-cache-stats")
    assert status == 200 and body["ok"] is True


def test_internal_error_does_not_leak_details(server) -> None:
    """参数非法时返回 4xx 且不包含 Python 异常细节。"""
    with pytest.raises(urllib.error.HTTPError) as exc:
        _request("POST", "/api/v1/documents", {"title": "x", "content": ""})
    text = exc.value.read().decode("utf-8")
    assert exc.value.code == 422
    for leaked in ("Traceback", "pydantic", "site-packages", "File \""):
        assert leaked not in text, f"响应体不应包含内部实现细节：{leaked}"
