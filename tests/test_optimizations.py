"""针对本机适配与优化项的回归测试。

覆盖四类容易悄悄坏掉的地方：
    1. ModelScope 本地模型路径解析（Windows 反斜杠 + dotenv 转义坑）
    2. Markdown 表格 / 围栏代码块的"原子块"保护（不能被切断）
    3. LLM 接入判定（本地服务无需 API Key，不能被误判为降级）
    4. 重排器的空实现语义与请求默认值
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings, is_local_base
from app.schemas import SearchRequest
from app.services.chunker import split_document
from app.services.llm import AnswerGenerator
from app.services.reranker import NoopReranker


# ==============================================================================
# 一、本地模型路径解析
# ==============================================================================
def _fake_modelscope_cache(root: Path, repo_id: str = "BAAI/bge-m3") -> Path:
    """构造一个与 ModelScope 真实目录结构一致的假缓存。"""
    snapshot = root / repo_id.replace("/", "--") / "snapshots" / "master"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text('{"model_type": "xlm-roberta"}', encoding="utf-8")
    return snapshot


@pytest.mark.parametrize("name", ["models", "hub", "root"])
def test_resolve_local_model_tolerates_truncated_and_backslash_root(tmp_path: Path, name: str) -> None:
    """缓存根目录写成 ``<root>/hub``（被 dotenv 吞掉一层）或用反斜杠，都必须能解析到。

    真实踩坑：``.env`` 里写 ``D:\\modelscope\\hub\\models`` 会被解析成
    ``D:\\modelscope\\hub``，导致「模型明明在本地却报找不到」。
    """
    cache_root = tmp_path / "hub" / "models"
    snapshot = _fake_modelscope_cache(cache_root)

    raw_map = {
        "models": str(cache_root),                          # 正常写法
        "hub": str(tmp_path / "hub"),                       # 被吞掉一层
        "root": str(cache_root).replace("/", "\\"),         # 反斜杠写法
    }
    cfg = Settings(modelscope_cache=raw_map[name])
    assert cfg.resolve_local_model("BAAI/bge-m3") == str(snapshot)


def test_resolve_local_model_returns_empty_when_missing(tmp_path: Path) -> None:
    cfg = Settings(modelscope_cache=str(tmp_path / "not-exist"))
    assert cfg.resolve_local_model("BAAI/bge-m3") == ""
    assert cfg.model_cache_roots(), "根目录列表不应为空，否则无法给出有效报错信息"


def test_model_cache_roots_deduplicates() -> None:
    cfg = Settings(modelscope_cache="D:/modelscope/hub/models")
    roots = [str(p).lower().replace("\\", "/") for p in cfg.model_cache_roots()]
    assert len(roots) == len(set(roots))


# ==============================================================================
# 二、原子块保护
# ==============================================================================
TABLE_DOC = """# 故障手册

## 报警代码对照表

下表列出常见报警代码及其含义，处置前请先核对代码。

| 代码 | 含义 | 危险等级 |
| --- | --- | --- |
| E-1042 | 主轴轴承过热 | 高 |
| E-2201 | 冷却液液位低 | 中 |
| E-3310 | 伺服驱动器过载 | 高 |
"""


def test_markdown_table_kept_in_one_chunk() -> None:
    """表格必须整体落在同一个子块里，不能把表头和数据行切开。"""
    parents = split_document(TABLE_DOC, parent_size=1200, child_size=280)
    children = [c for p in parents for c in p.children]
    holder = [c for c in children if "| 代码 |" in c.content]
    assert holder, "未找到承载表格的子块"
    assert any("| E-3310 |" in c.content for c in holder), "表格被切断：表头与最后一行不在同一块"


def test_large_table_splits_only_at_row_boundaries() -> None:
    """超大表格允许拆分，但每个分片里的表格行必须是完整的行。"""
    rows = [f"| E-{i:04d} | 故障描述{i} | 高 |" for i in range(80)]
    doc = "# 手册\n\n## 大表\n\n" + "\n".join(
        ["| 代码 | 含义 | 等级 |", "| --- | --- | --- |", *rows]
    )
    parents = split_document(doc, parent_size=600, child_size=200)
    chunks = [c.content for p in parents for c in p.children]
    assert len(chunks) > 1, "超大表格应当被拆分，否则会产出超长块"

    for chunk in chunks:
        for line in chunk.split("\n"):
            stripped = line.strip()
            if stripped.startswith("|"):
                assert stripped.endswith("|"), f"表格行被截断：{line!r}"

    joined = "\n".join(chunks)
    missing = [r for r in rows if r not in joined]
    assert not missing, f"拆分过程丢失了表格行：{missing[:3]}"


FENCE_DOC = """# 配置说明

## 网关样例

下面是一段网关配置报文示例。

```json
{"device_id": "CNC-007", "alarm_code": "E-2201", "interval_ms": 5000}
```
"""


def test_fenced_code_block_kept_intact() -> None:
    """围栏代码块不能被拆散（否则 JSON/YAML 会变成不可解析的碎片）。"""
    parents = split_document(FENCE_DOC, parent_size=1200, child_size=120)
    joined = "\n".join(c.content for p in parents for c in p.children)
    assert "```json" in joined and '{"device_id": "CNC-007"' in joined
    holders = [c for p in parents for c in p.children if "```json" in c.content]
    assert any('"interval_ms": 5000}' in c.content for c in holders), "代码块被切断"


def test_no_markdown_heading_still_chunks() -> None:
    """无标题纯文本走兜底路径，也必须能正常分块。"""
    text = "这是一段没有任何标题的说明文字。" * 60
    parents = split_document(text, parent_size=300, child_size=100)
    assert parents and parents[0].children


# ==============================================================================
# 三、LLM 接入判定
# ==============================================================================
def test_is_local_base_detects_localhost() -> None:
    assert is_local_base("http://127.0.0.1:11434/v1")
    assert is_local_base("http://localhost:8000/v1")
    assert not is_local_base("https://api.openai.com/v1")
    assert not is_local_base("")


def test_ollama_provider_requires_no_api_key() -> None:
    """本地 Ollama 不校验 Key，不能被"Key 为空"误判为降级。"""
    gen = AnswerGenerator(provider="ollama", api_key="")
    assert gen.mode == "openai"
    assert gen.api_key, "OpenAI SDK 要求 key 非空，应为本地服务填充占位值"
    info = gen.describe()
    assert info["base_url"].startswith("http://127.0.0.1:11434")
    assert info["model"] == "qwen2.5:7b"


def test_auto_provider_with_local_base_needs_no_key() -> None:
    gen = AnswerGenerator(provider="auto", api_base="http://127.0.0.1:11434/v1", api_key="")
    assert gen.mode == "openai"


def test_auto_provider_with_remote_base_and_no_key_falls_back() -> None:
    gen = AnswerGenerator(provider="auto", api_base="https://api.openai.com/v1", api_key="")
    assert gen.mode == "extractive"


def test_mock_provider_forces_extractive() -> None:
    gen = AnswerGenerator(provider="mock", api_key="sk-should-be-ignored")
    assert gen.mode == "extractive"


def test_extractive_answer_cites_sources() -> None:
    """降级模式必须仍然标注出处，不能返回无法溯源的答案。"""
    gen = AnswerGenerator(provider="mock")
    answer, mode = gen.generate(
        "父子分块有什么好处？",
        [
            {
                "parent_id": 42,
                "title": "手册",
                "heading_path": "一",
                "content": "父子分块的核心理念是用子块检索，用父块生成。",
            }
        ],
    )
    assert mode == "extractive"
    assert "[父文档ID: 42]" in answer


def test_openai_prompt_forbids_easy_refusal() -> None:
    """Prompt 必须显式抑制"假拒绝"——小模型很容易照抄拒绝句式。"""
    from app.services.llm import USER_TEMPLATE

    assert "不要轻易回答" in USER_TEMPLATE
    assert "[父文档ID: 12]" in USER_TEMPLATE


# ==============================================================================
# 四、重排与请求默认值
# ==============================================================================
def test_noop_reranker_preserves_order_and_reports_unavailable() -> None:
    reranker = NoopReranker()
    assert reranker.available is False
    docs = ["a", "b", "c"]
    result = reranker.rerank("q", docs, top_n=2)
    assert result == [(0, 0.0), (1, 0.0)]
    assert reranker.describe()["available"] is False


def test_reranker_sigmoid_keeps_scores_positive() -> None:
    """重排分数必须归一化到正区间，否则下游「相对阈值过滤」会失效。"""
    from app.services.reranker import _sigmoid

    assert 0.0 < _sigmoid(-8.0) < 0.5
    assert 0.5 < _sigmoid(8.0) < 1.0
    assert _sigmoid(0.0) == pytest.approx(0.5)


def test_search_request_defaults_explain_off() -> None:
    """explain 会额外发起 2 次 Milvus 查询，默认必须关闭。"""
    req = SearchRequest(query="测试")
    assert req.explain is False
    assert req.top_k == 20
    assert req.final_top_k == 4
    assert req.use_sparse is True


def test_settings_sparse_and_milvus_defaults_consistent() -> None:
    """Embedding 维度必须与 Milvus 集合维度同源。"""
    cfg = Settings()
    assert cfg.milvus_dim_resolved == cfg.embedding_dim
    assert cfg.retrieve_top_k > cfg.final_top_k


def test_local_modelscope_cache_root_points_to_real_dir() -> None:
    """本机模型缓存根目录至少应存在一个候选（不存在则给出可读的失败信息）。"""
    cfg = Settings()
    existing = [p for p in cfg.model_cache_roots() if p.is_dir()]
    if not existing:
        pytest.skip(f"本机未配置 ModelScope 缓存：{cfg.modelscope_cache}")
    assert existing


def test_demo_doc_contains_unique_anchor() -> None:
    """演示文档的锚点必须唯一，否则演示的核心对照会失去意义。"""
    import importlib.util
    import sys as _sys

    spec = importlib.util.spec_from_file_location(
        "demo_anchor_mod", Path(__file__).with_name("test_parent_child_demo.py")
    )
    module = importlib.util.module_from_spec(spec)
    _sys.modules["demo_anchor_mod"] = module
    spec.loader.exec_module(module)

    manual = module.DOC_FAULT_MANUAL
    assert module.KEY_TERM in manual

    # 真正的不变量：所有处置步骤都必须出现在「处置步骤」标题之后，
    # 且不得提前出现在前文（否则"线索与答案分离"的对照就不成立）
    head, _, tail = manual.partition("处置步骤")
    assert tail, "演示文档缺少『处置步骤』小节"
    for step in module.ANSWER_STEPS:
        assert step in tail, f"处置步骤片段缺失：{step}"
        assert step not in head, f"处置步骤片段提前出现在前文，会使核心对照失效：{step}"

    # 其他演示文档不得包含同一锚点，避免检索竞争
    for _, content in module.DEMO_DOCS[1:]:
        assert module.KEY_TERM not in content
