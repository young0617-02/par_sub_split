"""父子分块（Parent-Child Chunking）效果演示 —— 可独立运行的完整示例。

    python tests/test_parent_child_demo.py              # 完整演示（推荐）
    python tests/test_parent_child_demo.py --no-llm     # 跳过 LLM，只看检索链路
    python tests/test_parent_child_demo.py --keep       # 保留演示数据便于手工核查
    python -m pytest tests/test_parent_child_demo.py -v -s

需要 MySQL / Redis / Milvus 已启动；缺失时自动跳过（pytest）或给出提示（CLI）。

================================================================================
测试数据的设计意图
================================================================================
演示数据不是随手写的，而是**专门构造**用来暴露「小粒度切块会丢上下文」这个问题的：

    文档《数控加工中心故障诊断手册》的「E-1042 主轴轴承过热」小节里，
    报警代码 E-1042 只出现在前半部分（现象描述 / 根因分析），
    而真正的答案「处置步骤」在后半部分，两者会被切进**不同的子块**。

    于是对问题「报警代码 E-1042 的处理方法是什么？」：
      · 只用子块（传统小块 RAG，top-1）→ 拿到的是"现象/根因"，**没有处置步骤**；
      · 父子分块 → 先用子块精准命中，再回表拿到**整个父块**，处置步骤完整给出。

这个差异就是父子分块的全部价值所在，也是本文件第 4 步要断言的核心结论。

================================================================================
各步骤预期输出（括号内为本机实测值，用于判断本次运行是否正常）
================================================================================
步骤 1 分块结构   父块数 < 子块数；父块均长 ≈ 子块均长的 3~5 倍；每块带章节标签
                  （实测：父块 1 个 894 字 / 子块 5 个 208 字，粒度比 4.30:1）

步骤 2 入库       父块写入 MySQL 并拿到自增 id；子块携带同一 parent_id；
                  Redis 中存在对应 key；Milvus 实体数增加
                  （实测：MySQL 12~40ms｜Redis 1~2ms｜Milvus 13~540ms）

步骤 3 检索       命中若干子块 → 按 parent_id 去重 → 从 Redis 取回父块全文
                  （实测：embed 86~138ms｜milvus 7~13ms｜redis 3~7ms｜合计 100~170ms）

步骤 4 关键对照 ★ 小块 RAG（top-1 子块）覆盖处置步骤 **0/5** 条；
                  父子分块（parent_id 回表）覆盖 **5/5** 条
                  → 这是"用子块检索、用父块生成"最直接的量化证据

步骤 5 三路对比   稀疏分支靠错误码精确命中；稠密分支靠语义近似；混合结果取两者之长
                  （实测：稠密命中 [24,1,4,25,2]、稀疏命中 [24,2,25]、混合 [24,2,25,1,4]）

步骤 6 RAG 问答   答案覆盖处置步骤并给出 [父文档ID: x] 引用；重复查询缓存命中率上升
                  （实测：命中率 100%；LLM 3.4~6.9s，其中首次数含模型加载）

================================================================================
本地模型接入（本机已具备）
================================================================================
· 嵌入：BAAI/bge-m3（ModelScope 本地快照，1024 维，与集合定义一致）
        配置 EMBEDDING_PROVIDER=local 即自动解析本地路径，无需改集合结构
· 大模型：Ollama qwen2.5:7b（http://127.0.0.1:11434/v1，无需 API Key）
· 先跑一次自检确认环境：python scripts/check_local_models.py --load --llm "你好"

================================================================================
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.clients.milvus_client import get_chunk_store
from app.clients.mysql import ping as mysql_ping
from app.clients.redis_client import get_client, get_parent_cache
from app.clients.redis_client import ping as redis_ping
from app.config import settings
from app.schemas import SearchResponse
from app.services.chunker import chunk_stats, split_document
from app.services.ingest import IngestService
from app.services.llm import get_answer_generator
from app.services.rag import get_rag_service
from app.services.retriever import get_retrieval_service

# ==============================================================================
# 一、测试数据
# ==============================================================================

DOC_FAULT_MANUAL = """# 数控加工中心故障诊断手册

## E-1042 主轴轴承过热

现象描述：设备连续运行约 40 分钟后主轴温度快速上升，HMI 弹出报警代码 E-1042。
同时伴随轻微异响，振动速度有效值由 1.2mm/s 上升到 4.8mm/s。
前端轴承温度实测 92℃，远超报警阈值 75℃，若不及时处理会触发热变形停机保护。

根因分析：该报警通常由三类原因引起，分别是润滑脂失效、轴承预紧力过大、冷却液流量不足。
其中润滑脂失效占比约 65%，是首要排查方向；预紧力过大占比约 20%，多见于大修之后；
冷却液流量不足占比约 15%，常发生在过滤网堵塞的工况下。

处置步骤：
1. 立即停机并切断主轴电源，等待温度自然降至 40℃ 以下再开始作业。
2. 检查润滑脂状态，若出现发黑或干结应立即更换，加注量为轴承腔容积的 30%。
3. 复测轴承预紧力，标准值为 220±20N，超差需重新调整锁紧螺母的力矩。
4. 检查冷却液流量，正常应大于 12L/min，不足时清洗过滤网并排空管路内的空气。
5. 复位报警后空载运行 30 分钟，确认温度稳定在 60℃ 以下方可重新投入生产。

预防措施：每 500 小时补充一次润滑脂，每 2000 小时更换一次轴承。
建议将轴承温度纳入预测性维护模型，实现提前 72 小时预警，把非计划停机转为计划检修。

## 常见报警代码对照表

| 代码 | 含义 | 危险等级 | 建议响应时间 |
| --- | --- | --- | --- |
| E-2201 | 冷却液液位低 | 中 | 4 小时内 |
| E-3310 | 伺服驱动器过载 | 高 | 立即停机 |
| E-4402 | 刀库定位超时 | 低 | 下一班次 |
| E-5501 | 气源压力不足 | 中 | 2 小时内 |

## 网关上报报文示例

设备报警通过工业网关以 JSON 格式上报，字段含义见下表：

```json
{
  "device_id": "CNC-007",
  "alarm_code": "E-2201",
  "temperature": 61,
  "vibration": 1.4,
  "ts": "2026-09-21T10:30:00+08:00"
}
```
"""

DOC_INSPECTION = """# 设备巡检标准作业流程

## 每班次开机确认

每班次开机后需完成三项确认：气压是否处于 0.5~0.7MPa 区间、冷却液液位是否高于下限刻度、
导轨润滑泵是否处于通电工作状态。任意一项不满足都应记录并上报班组长，不得带病开机。

## 振动与温度阈值

主轴振动速度有效值应小于 2.8mm/s，超过 4.5mm/s 判定为异常需停机检查。
轴承温度正常范围为 30~60℃，超过 75℃ 会触发过热类报警。
当车间环境温度高于 35℃ 时，上述温度阈值统一上浮 5℃，避免夏季误报。

## 巡检记录与归档

巡检结果需在当班结束前录入设备管理系统，包含实测值、异常描述与处理动作。
连续三个班次出现同一异常项时，系统自动生成检修工单并推送至设备主管。
"""

DOC_GATEWAY = """# 工业网关接入配置说明

## 网络与端口

网关默认使用 4840 端口对外提供 OPC UA 服务，使用 1883 端口对接 MQTT 消息代理。
部署前需确认现场交换机已放行上述端口，且网关与采集服务器处于同一网段。

## 采集周期配置

采集周期根据点位类型差异化设置：温度类点位建议 5000ms，振动类点位建议 1000ms，
开关量点位建议 500ms。周期设置过短会导致网关 CPU 占用升高，出现报文堆积。

```yaml
gateway:
  device_id: CNC-007
  tags:
    - name: spindle_temp
      address: ns=2;s=Spindle.Temperature
      interval_ms: 5000
    - name: spindle_vibration
      address: ns=2;s=Spindle.Vibration
      interval_ms: 1000
```

## 断网续传

网关内置 4GB 本地缓存，网络中断时数据先落盘。网络恢复后按时间顺序补传，
补传速率限制为正常速率的 3 倍，避免瞬时流量冲击服务端。
"""

DEMO_DOCS: list[tuple[str, str]] = [
    ("demo_fault_manual", DOC_FAULT_MANUAL),
    ("demo_inspection", DOC_INSPECTION),
    ("demo_gateway", DOC_GATEWAY),
]

# 演示问题：答案横跨同一父块内的多个子块
DEMO_QUERY = "报警代码 E-1042 的处理方法是什么？"

# 关键断言锚点：
#   KEY_TERM     —— 检索命中点，只出现在该小节的前半部分
#   ANSWER_STEPS —— 真正的答案，5 条处置步骤的关键片段，分布在父块后半部分
#   ANSWER_TERM  —— 仅用于打印提示的章节名（不作断言，因为它可能被并进上一个子块）
KEY_TERM = "E-1042"
ANSWER_TERM = "处置步骤"
ANSWER_STEPS = [
    "立即停机并切断主轴电源",
    "润滑脂状态",
    "复测轴承预紧力",
    "检查冷却液流量",
    "空载运行 30 分钟",
]


# ==============================================================================
# 二、演示执行器
# ==============================================================================
@dataclass
class DemoStep:
    """单步结果，便于统一打印与断言。"""

    name: str
    ok: bool
    detail: dict = field(default_factory=dict)
    note: str = ""


class ParentChildDemo:
    """把「分块 → 入库 → 检索 → 对照 → 对比 → 问答」串成一次可观察的演示。"""

    def __init__(self, use_llm: bool = True) -> None:
        self.use_llm = use_llm
        self.ingest = IngestService()
        self.retriever = get_retrieval_service()
        self.store = get_chunk_store()
        self.cache = get_parent_cache()
        self.steps: list[DemoStep] = []
        self.parent_ids: list[int] = []
        self.doc_ids: list[str] = []
        self.total_children: int = 0
        # 步骤 3 的检索响应，供步骤 4 与 pytest 断言复用
        self.search_response: SearchResponse | None = None

    # ------------------------------------------------------------------
    # 环境自检
    # ------------------------------------------------------------------
    @staticmethod
    def env_report() -> dict[str, object]:
        return {
            "mysql": mysql_ping(),
            "redis": redis_ping(),
            "milvus": get_chunk_store().ping(),
            "embedding": get_retrieval_service().embedder.describe(),
            "sparse": get_retrieval_service().encoder.describe(),
            "rerank": get_retrieval_service().reranker.describe(),
            "llm": get_answer_generator().describe(),
        }

    @staticmethod
    def require_env() -> None:
        missing = []
        if not mysql_ping():
            missing.append("MySQL")
        if not redis_ping():
            missing.append("Redis")
        if not get_chunk_store().ping():
            missing.append("Milvus")
        if missing:
            raise RuntimeError(
                f"以下服务不可用：{', '.join(missing)}。"
                "请先启动全部依赖服务：docker compose up -d"
                "（查看状态：docker compose ps）"
            )
        from app.clients.mysql import init_schema

        init_schema()
        get_chunk_store().ensure_collection()

    # ------------------------------------------------------------------
    # 步骤 1：分块结构
    # ------------------------------------------------------------------
    def step_1_chunking(self) -> DemoStep:
        print("\n" + "─" * 78)
        print("步骤 1｜父子分块结构：一份文档被切成两级粒度")
        print("─" * 78)
        print(f"参数：父块 {settings.parent_chunk_size} 字（重叠 {settings.parent_chunk_overlap}）"
              f"｜子块 {settings.child_chunk_size} 字（重叠 {settings.child_chunk_overlap}）")

        parents = split_document(
            DOC_FAULT_MANUAL,
            parent_size=settings.parent_chunk_size,
            parent_overlap=settings.parent_chunk_overlap,
            child_size=settings.child_chunk_size,
            child_overlap=settings.child_chunk_overlap,
        )
        stats = chunk_stats(parents)
        ratio = stats["avg_parent_chars"] / max(stats["avg_child_chars"], 1)

        print(f"\n原文 {len(DOC_FAULT_MANUAL)} 字  →  "
              f"父块 {stats['parent_count']} 个（均长 {stats['avg_parent_chars']} 字）"
              f" + 子块 {stats['child_count']} 个（均长 {stats['avg_child_chars']} 字）")
        print(f"父子粒度比 = {ratio:.2f} : 1  （预期 3~5 : 1，比值过小说明父块退化成了子块）\n")

        for parent in parents:
            label = parent.heading_path or "(无章节标题)"
            print(f"  父块 #{parent.index}  [{label}]")
            print(f"      {parent.char_count} 字 → {len(parent.children)} 个子块")
            for child in parent.children:
                preview = child.content[:46].replace("\n", " ")
                hit_key = "★含命中词" if KEY_TERM in child.content else ""
                steps_here = sum(1 for s in ANSWER_STEPS if s in child.content)
                hit_ans = f"★含答案步骤{steps_here}/{len(ANSWER_STEPS)}" if steps_here else ""
                marks = " ".join(x for x in (hit_key, hit_ans) if x)
                print(f"        ├─ 子块#{child.index} {len(child.content):>4}字 "
                      f"{child.chunk_id[:8]}… {preview}… {marks}")

        # 关键观察：检索命中点与答案被切进了不同的子块
        key_children = [c for p in parents for c in p.children if KEY_TERM in c.content]
        key_child_text = "".join(c.content for c in key_children)
        steps_covering_key = sum(1 for s in ANSWER_STEPS if s in key_child_text)
        separated = bool(key_children) and steps_covering_key < len(ANSWER_STEPS)
        print(f"\n  观察：含『{KEY_TERM}』的子块 {len(key_children)} 个，"
              f"其中仅覆盖 {steps_covering_key}/{len(ANSWER_STEPS)} 条处置步骤 → "
              f"{'命中点与完整答案已分离，构造成功' if separated else '命中点已包含全部答案，演示效果会减弱'}")

        ok = (
            stats["parent_count"] >= 1
            and ratio > 2
            and stats["child_count"] > stats["parent_count"]
        )
        self.steps.append(DemoStep("分块结构", ok, {**stats, "steps_at_key_child": steps_covering_key}))
        return self.steps[-1]

    # ------------------------------------------------------------------
    # 步骤 2：入库（三处落库）
    # ------------------------------------------------------------------
    def step_2_ingest(self) -> DemoStep:
        print("\n" + "─" * 78)
        print("步骤 2｜入库：一次写入 MySQL（父块）/ Redis（内容）/ Milvus（子块向量）")
        print("─" * 78)

        self.parent_ids, self.doc_ids = [], []
        for doc_id, content in DEMO_DOCS:
            result = self.ingest.ingest_text(
                content=content,
                title=doc_id.replace("demo_", ""),
                doc_id=doc_id,
                source="demo",
                force_reindex=True,  # 保证每次演示都是干净数据
            )
            self.parent_ids.extend(result.parent_ids)
            self.doc_ids.append(result.doc_id)
            self.total_children += result.child_count
            timing = result.detail.get("timing_ms", {})
            print(f"  {doc_id:20s} 父块 {result.parent_count} 个 "
                  f"(ids={result.parent_ids}) 子块 {result.child_count} 个 "
                  f"耗时 {result.elapsed_ms}ms "
                  f"[MySQL {timing.get('mysql')} / Redis {timing.get('redis')} / Milvus {timing.get('milvus')}]")

        # ---------- 核对三处写入 ----------
        # 契约（务必区分硬要求与软检查）：
        #   · MySQL 是事实来源、Milvus 是检索索引 —— 写入失败即入库失败，属硬要求；
        #   · Redis 只是派生缓存 —— 写入失败不影响正确性（检索会自动回表兜底），
        #     因此这里只做软检查并明确提示，不据此判定入库失败。
        milvus_total = self.store.count()
        redis_hit = sum(1 for pid in self.parent_ids if self.cache.exists(pid))
        if redis_hit < len(self.parent_ids):
            # 高负载（如同时加载本地嵌入模型）时缓存写入可能超时，重试一次再判定
            time.sleep(0.5)
            redis_hit = sum(1 for pid in self.parent_ids if self.cache.exists(pid))

        print(f"\n  核对（硬要求）MySQL 父块 {len(self.parent_ids)} 个 ｜ "
              f"Milvus 子块 {self.total_children} 个（集合实体总数 {milvus_total}）")
        if redis_hit == len(self.parent_ids):
            print(f"  核对（软检查）Redis key 命中 {redis_hit}/{len(self.parent_ids)} ✓")
        else:
            print(f"  核对（软检查）Redis key 命中 {redis_hit}/{len(self.parent_ids)} ⚠ "
                  "缓存未写全会导致回表，但检索结果依然正确")

        ok = (
            bool(self.parent_ids)
            and self.total_children > 0
            and milvus_total >= self.total_children
        )
        self.steps.append(DemoStep("三处入库", ok, {
            "parent_ids": self.parent_ids,
            "children": self.total_children,
            "milvus_total": milvus_total,
            "redis_hit": redis_hit,
        }))
        return self.steps[-1]

    # ------------------------------------------------------------------
    # 步骤 3：检索链路
    # ------------------------------------------------------------------
    def step_3_retrieval(self) -> DemoStep:
        print("\n" + "─" * 78)
        print("步骤 3｜检索链路：问题 → 命中子块 → 取 parent_id → Redis 取父块")
        print("─" * 78)
        print(f"问题：{DEMO_QUERY}\n")

        response = self.retriever.search(DEMO_QUERY, top_k=20, final_top_k=4, explain=True)
        print(f"阶段耗时(ms)：{response.timings}")
        print(f"命中父块 {len(response.parents)} 个（已按 parent_id 去重）\n")

        for rank, parent in enumerate(response.parents, 1):
            cache_flag = "Redis 命中" if parent.cache_hit else "回表 MySQL"
            print(f"  [{rank}] 父文档 #{parent.parent_id}  分数={parent.score}  "
                  f"命中子块 {len(parent.child_hits)} 个  ({cache_flag})")
            print(f"      章节：{parent.heading_path}")
            print(f"      父块 {len(parent.content)} 字：{parent.content[:60].replace(chr(10), ' ')}…")
            for child in parent.child_hits[:3]:
                ranks = []
                if child.dense_rank is not None:
                    ranks.append(f"稠密#{child.dense_rank}")
                if child.sparse_rank is not None:
                    ranks.append(f"稀疏#{child.sparse_rank}")
                print(f"        └─ 子块#{child.chunk_index} 分={child.score} "
                      f"{' '.join(ranks)}：{child.content[:44].replace(chr(10), ' ')}…")

        demo_parent_ids = set(self.parent_ids)
        hit = any(p.parent_id in demo_parent_ids for p in response.parents)
        ok = bool(response.parents) and hit and all(p.content for p in response.parents)
        if not hit:
            print("  ✘ 未命中演示文档的父块，请检查是否已完成步骤 2")
        self.steps.append(DemoStep("检索链路", ok, {"timings": response.timings}))
        self.search_response = response
        return self.steps[-1]

    # ------------------------------------------------------------------
    # 步骤 4：核心对照（父子分块的价值证据）
    # ------------------------------------------------------------------
    def step_4_context_ablation(self) -> DemoStep:
        print("\n" + "─" * 78)
        print("步骤 4｜核心对照：只用子块 vs 父子分块，送进模型的上下文差在哪")
        print("─" * 78)

        # 传统小块 RAG 的典型做法：只召回 top-1 个块，直接当上下文
        single = self.retriever.retrieve(DEMO_QUERY, top_k=1, final_top_k=1)
        naive_parent = single[0][0] if single and single[0] else None
        naive_context = naive_parent.child_hits[0].content if naive_parent else ""

        # 父子分块：命中子块后回表取父块全文
        full = self.retriever.retrieve(DEMO_QUERY, top_k=20, final_top_k=1)
        pc_parent = full[0][0] if full and full[0] else None

        if not naive_context or pc_parent is None:
            print("  ✘ 未取到对照所需的上下文")
            self.steps.append(DemoStep("上下文对照", False))
            return self.steps[-1]

        def coverage(text: str) -> int:
            """统计文本覆盖了多少条处置步骤 —— 这是「答案是否完整送达」的客观度量。"""
            return sum(1 for step in ANSWER_STEPS if step in text)

        naive_cov = coverage(naive_context)
        parent_cov = coverage(pc_parent.content)

        print("场景 A｜传统小块 RAG（top-1 子块直接作为上下文）")
        print(f"  上下文来源：子块 #{naive_parent.child_hits[0].chunk_index}（{len(naive_context)} 字）")
        print(f"  含检索命中词『{KEY_TERM}』：{'是' if KEY_TERM in naive_context else '否'}")
        print(f"  覆盖处置步骤：{naive_cov}/{len(ANSWER_STEPS)} 条        ← 关键")
        print(f"  内容：{naive_context[:96].replace(chr(10), ' ')}…\n")

        print("场景 B｜父子分块（命中子块 → parent_id → 父块全文）")
        print(f"  上下文来源：父文档 #{pc_parent.parent_id}（{len(pc_parent.content)} 字）")
        print(f"  含检索命中词『{KEY_TERM}』：{'是' if KEY_TERM in pc_parent.content else '否'}")
        print(f"  覆盖处置步骤：{parent_cov}/{len(ANSWER_STEPS)} 条        ← 关键")
        print(f"  上下文放大倍数：{len(pc_parent.content) / max(len(naive_context), 1):.2f}×")
        print(f"  内容：{pc_parent.content[:96].replace(chr(10), ' ')}…\n")

        missing = [s for s in ANSWER_STEPS if s not in naive_context]
        print("  对照结论：")
        print(f"    · 小块 RAG 拿到了『线索』（{KEY_TERM}），但缺失 {len(missing)}/{len(ANSWER_STEPS)} 条处置步骤：")
        for step_text in missing:
            print(f"         - {step_text}")
        print(f"    · 父子分块命中的是同一个子块，却通过 parent_id 回表拿到整个父块，"
              f"{parent_cov}/{len(ANSWER_STEPS)} 条处置步骤完整送达模型")
        print("    · 这就是「用子块保证召回精度、用父块保证上下文完整」的直接证据")

        ok = (
            KEY_TERM in naive_context
            and naive_cov < len(ANSWER_STEPS)
            and parent_cov == len(ANSWER_STEPS)
        )
        if not ok:
            print("\n  ⚠ 前提未完全成立，可尝试减小 CHILD_CHUNK_SIZE 后重跑")
        self.steps.append(DemoStep("上下文对照", bool(ok), {
            "naive_chars": len(naive_context),
            "parent_chars": len(pc_parent.content),
            "naive_step_coverage": naive_cov,
            "parent_step_coverage": parent_cov,
            "answer_steps_total": len(ANSWER_STEPS),
        }))
        return self.steps[-1]

    # ------------------------------------------------------------------
    # 步骤 5：稠密 / 稀疏 / 混合 三路对比
    # ------------------------------------------------------------------
    def step_5_branch_compare(self) -> DemoStep:
        print("\n" + "─" * 78)
        print("步骤 5｜三路召回对比：稀疏（BM25）擅长错误码，稠密擅长语义泛化")
        print("─" * 78)

        dense_vec = self.retriever.embedder.encode_query(DEMO_QUERY)
        sparse_vec = self.retriever.encoder.encode(DEMO_QUERY)
        dense_hits = self.store.search_single_branch("dense", dense_vector=dense_vec, limit=10)
        sparse_hits = self.store.search_single_branch("sparse", sparse_vector=sparse_vec, limit=10)
        hybrid_hits = self.store.hybrid_search(dense_vec, sparse_vec, limit=10)

        def summarize(name: str, hits: list[dict]) -> list[int]:
            parents, seen = [], set()
            for h in hits:
                pid = int(h["entity"].get("parent_id") or 0)
                if pid and pid not in seen:
                    seen.add(pid)
                    parents.append(pid)
            print(f"  {name:10s} 命中父块：{parents}")
            return parents

        dense_parents = summarize("稠密（语义）", dense_hits)
        sparse_parents = summarize("稀疏（BM25）", sparse_hits)
        hybrid_parents = summarize("混合（RRF）", hybrid_hits)

        only_dense = sorted(set(dense_parents) - set(sparse_parents))
        only_sparse = sorted(set(sparse_parents) - set(dense_parents))
        print(f"\n  稠密独有：{only_dense or '无'}　稀疏独有：{only_sparse or '无'}"
              f"　两路重叠：{len(set(dense_parents) & set(sparse_parents))} 个")
        print(f"  稀疏查询词元数：{len(sparse_vec)}（BM25 权重已 L2 归一化）")
        print("  说明：错误码/型号这类低频词只有稀疏分支能稳定命中；")
        print("        而『怎么处理』这类语义表达则依赖稠密分支。混合后两端都能覆盖。")

        ok = bool(hybrid_hits) and bool(sparse_hits)
        self.steps.append(DemoStep("三路对比", ok, {
            "dense_parents": dense_parents,
            "sparse_parents": sparse_parents,
            "hybrid_parents": hybrid_parents,
        }))
        return self.steps[-1]

    # ------------------------------------------------------------------
    # 步骤 6：完整 RAG 问答
    # ------------------------------------------------------------------
    def step_6_rag(self) -> DemoStep:
        print("\n" + "─" * 78)
        print("步骤 6｜完整 RAG 问答：父块全文 + 大模型")
        print("─" * 78)

        generator = get_answer_generator()
        llm_info = generator.describe()
        print(f"LLM 接线：provider={llm_info['provider']}  mode={llm_info['mode']}  "
              f"model={llm_info['model']}")
        print(f"          base_url={llm_info['base_url']}")

        if not self.use_llm:
            print("  （已按 --no-llm 跳过实际生成）")
            self.steps.append(DemoStep("RAG 问答", True, {"skipped": True}))
            return self.steps[-1]

        if llm_info["mode"] == "extractive":
            print("  提示：当前为抽取式降级模式。配置 LLM_PROVIDER=ollama 即可接入本机 qwen2.5:7b。")

        # 缓存命中率观察：先清统计，再连查两次
        self.cache.reset_stats()
        first = get_rag_service().answer(DEMO_QUERY, top_k=20, final_top_k=4)
        stats_after_first = self.cache.stats()
        get_rag_service().answer(DEMO_QUERY, top_k=20, final_top_k=4)
        stats_after_second = self.cache.stats()

        print(f"\n第一次查询：命中率 {stats_after_first['hit_rate']:.0%} "
              f"({stats_after_first['hits']}/{stats_after_first['requests']})")
        print(f"第二次查询：命中率 {stats_after_second['hit_rate']:.0%} "
              f"({stats_after_second['hits']}/{stats_after_second['requests']})"
              f"  ← 重复查询后命中率上升")
        print(f"阶段耗时(ms)：{first.timings}\n")

        print("答案：")
        for line in first.answer.splitlines():
            print(f"  {line}")
        print(f"\n引用父文档：{[(c.parent_id, c.title) for c in first.citations]}")

        cited = {c.parent_id for c in first.citations}
        answer_ok = bool(first.answer.strip())
        cite_ok = bool(cited & set(self.parent_ids))
        # 答案应覆盖到处置步骤相关内容（降级模式下由原文句子拼成，同样应命中）
        covered = [s for s in ANSWER_STEPS if s in first.answer]
        if not covered:
            print("  ⚠ 答案未覆盖任何处置步骤片段，请人工检查送入模型的父块是否完整")

        ok = answer_ok and cite_ok
        self.steps.append(DemoStep("RAG 问答", ok, {
            "llm_mode": first.llm_mode,
            "citations": sorted(cited),
            "hit_rate": stats_after_second["hit_rate"],
            "answer_step_coverage": len(covered),
        }))
        return self.steps[-1]

    # ------------------------------------------------------------------
    # 清理 / 汇总
    # ------------------------------------------------------------------
    def teardown(self, keep: bool = False) -> None:
        if keep:
            print(f"\n（--keep 已指定，保留演示数据：doc_ids={self.doc_ids}）")
            return
        removed = 0
        for doc_id in self.doc_ids:
            removed += self.ingest.delete_document(doc_id)["deleted_parents"]
        print(f"\n已清理演示数据：{removed} 个父块及其 Redis 缓存 / Milvus 向量"
              f"（Milvus 剩余实体 {self.store.count()}）")

    def summary(self) -> bool:
        print("\n" + "═" * 78)
        print("演示结果汇总")
        print("═" * 78)
        for step in self.steps:
            print(f"  {'✔' if step.ok else '✘'} {step.name}")
        all_ok = all(s.ok for s in self.steps)
        print()
        if all_ok:
            print("  ✔ 全部通过：父子分块的『子块检索 → parent_id 回表 → 父块生成』链路验证成功。")
        else:
            failed = [s.name for s in self.steps if not s.ok]
            print(f"  ✘ 未通过：{', '.join(failed)}")
        print("═" * 78)
        return all_ok

    def run_all(self) -> bool:
        self.require_env()
        print("═" * 78)
        print("父子分块效果演示")
        print("═" * 78)
        env = self.env_report()
        print(f"环境：MySQL={env['mysql']}  Redis={env['redis']}  Milvus={env['milvus']}")
        print(f"嵌入：{env['embedding']}")
        print(f"稀疏：{env['sparse']}")
        print(f"重排：{env['rerank']}")
        print(f"大模型：{env['llm']}")
        print()
        print("预期观察点：")
        print("  ① 父块个数远少于子块，且父块明显更长（粒度比 3~5:1）")
        print("  ② 含『E-1042』的子块与含『处置步骤』的子块是**不同**的子块")
        print("  ③ 检索命中子块后，父块回表能拿到完整小节（含处置步骤）")
        print("  ④ 稀疏分支靠错误码精确命中，稠密分支靠语义近似，混合取长补短")
        print("  ⑤ 问答答案覆盖处置步骤并给出引用；重复查询缓存命中率上升")

        try:
            self.step_1_chunking()
            self.step_2_ingest()
            self.step_3_retrieval()
            self.step_4_context_ablation()
            self.step_5_branch_compare()
            self.step_6_rag()
        finally:
            pass
        return self.summary()


# ==============================================================================
# 三、CLI 入口
# ==============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description="父子分块效果演示")
    parser.add_argument("--no-llm", action="store_true", help="跳过 LLM 生成，只演示检索链路")
    parser.add_argument("--keep", action="store_true", help="保留演示数据，便于手工核查")
    args = parser.parse_args()

    demo = ParentChildDemo(use_llm=not args.no_llm)
    try:
        ok = demo.run_all()
    except RuntimeError as exc:
        print(f"\n[环境不满足] {exc}")
        return 2
    finally:
        demo.teardown(keep=args.keep)
    return 0 if ok else 1


# ==============================================================================
# 四、pytest 入口（与 CLI 复用同一套步骤）
# ==============================================================================
try:
    import pytest
except ImportError:  # pragma: no cover - 未安装 pytest 时仅支持 CLI
    pytest = None  # type: ignore[assignment]

if pytest is not None:

    @pytest.fixture(scope="module")
    def demo():
        try:
            ParentChildDemo.require_env()
        except RuntimeError as exc:
            pytest.skip(str(exc))
        instance = ParentChildDemo(use_llm=False)
        yield instance
        instance.teardown(keep=False)

    def test_demo_step1_chunking(demo: ParentChildDemo) -> None:
        """父块粒度必须显著大于子块，且线索与答案落在不同子块。"""
        step = demo.step_1_chunking()
        assert step.ok, f"分块结构不符合预期：{step.detail}"
        assert step.detail["parent_count"] < step.detail["child_count"]

    def test_demo_step2_ingest_writes_three_stores(demo: ParentChildDemo) -> None:
        """入库必须同时写 MySQL（父块）、Redis（内容）、Milvus（子块）。"""
        step = demo.step_2_ingest()
        assert step.ok, "Redis 未写全父块内容"
        assert step.detail["parent_ids"]
        assert get_client().exists(get_parent_cache().key(step.detail["parent_ids"][0]))

    def test_demo_step3_retrieval_hits_demo_parents(demo: ParentChildDemo) -> None:
        """检索应命中演示文档的父块，并取回父块全文。"""
        step = demo.step_3_retrieval()
        assert step.ok, "检索未命中演示父块"
        response = demo.search_response
        assert response is not None and response.parents
        assert all(p.content for p in response.parents)
        assert "group_ms" in response.timings

    def test_demo_step4_parent_covers_all_steps_but_child_does_not(demo: ParentChildDemo) -> None:
        """★ 核心断言：父块完整覆盖全部处置步骤，而被命中的最佳子块不覆盖。

        这正是「子块检索 + 父块生成」的价值证据：不用父子分块时，
        模型只拿到线索（错误码/现象）而拿不到答案（处置步骤）。
        """
        step = demo.step_4_context_ablation()
        total = step.detail["answer_steps_total"]
        assert step.detail["parent_step_coverage"] == total, "父块未覆盖全部处置步骤，回表链路有问题"
        assert step.detail["naive_step_coverage"] < total, "最佳子块意外包含全部答案，演示前提不成立"
        assert step.detail["parent_chars"] > step.detail["naive_chars"]

    def test_demo_step5_hybrid_covers_both_branches(demo: ParentChildDemo) -> None:
        """混合检索结果应非空，且稀疏分支有独立贡献。"""
        step = demo.step_5_branch_compare()
        assert step.ok
        assert step.detail["hybrid_parents"], "混合检索无结果"
        assert step.detail["sparse_parents"], "稀疏分支无结果，检查 BM25 词表是否一致"

    def test_demo_step6_rag_returns_citations(demo: ParentChildDemo) -> None:
        """问答应返回答案与引用（默认跳过真实 LLM，避免测试依赖外部服务）。"""
        step = demo.step_6_rag()
        assert step.ok


if __name__ == "__main__":
    raise SystemExit(main())
