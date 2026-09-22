"""本地模型与推理服务的可用性自检（preflight）。

用途：在切换到本地 BGE-M3 / BGE-reranker / Ollama 之前，先确认
「模型文件在不在、依赖装了没、服务通不通」，避免上线后才发现问题。

用法：
    python scripts/check_local_models.py
    python scripts/check_local_models.py --load        # 真正加载模型做一次编码（慢）
    python scripts/check_local_models.py --llm "你好"   # 真调一次 LLM
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings

OK = "[OK]  "
WARN = "[WARN]"
FAIL = "[FAIL]"


def _try_import(name: str) -> tuple[bool, str]:
    try:
        mod = importlib.import_module(name)
        return True, getattr(mod, "__version__", "?")
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)[:80]


def check_models() -> dict[str, str]:
    print("=" * 74)
    print("一、本地模型文件")
    print("=" * 74)
    resolved: dict[str, str] = {}
    for repo in (settings.embedding_model, settings.rerank_model):
        path = settings.resolve_local_model(repo)
        if path:
            size_gb = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1e9
            print(f"{OK} {repo}")
            print(f"       路径: {path}")
            print(f"       大小: {size_gb:.2f} GB")
            resolved[repo] = path
        else:
            print(f"{FAIL} {repo} —— 在 {settings.modelscope_cache} 下未找到本地快照")
            print(f"       下载: modelscope download --model {repo}")
    return resolved


def check_deps() -> bool:
    print()
    print("=" * 74)
    print("二、Python 推理依赖")
    print("=" * 74)
    all_ok = True
    for name in ("torch", "transformers", "sentence_transformers"):
        ok, info = _try_import(name)
        extra = ""
        if ok and name == "torch":
            try:
                import torch

                extra = f" | cuda_available={torch.cuda.is_available()} | cuda={torch.version.cuda}"
            except Exception as exc:  # noqa: BLE001  取不到 CUDA 信息不影响依赖可用性判断
                extra = f" | CUDA 信息不可用：{type(exc).__name__}"
        print(f"{OK if ok else FAIL} {name:22s} {info}{extra}")
        all_ok = all_ok and ok
    if not all_ok:
        print()
        print("       安装命令（CPU 版，约 400MB；若要用 GPU 请装对应 CUDA 版）:")
        print("         python -m pip install torch --index-url https://download.pytorch.org/whl/cpu")
        print("         python -m pip install sentence-transformers transformers")
        print("       提示：可装到专用环境（本机预留了 D:\\py312-torch）以隔离主环境。")
    return all_ok


def check_gpu() -> None:
    print()
    print("=" * 74)
    print("三、显存预算（8GB 机器的关键约束）")
    print("=" * 74)
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=False,
        )
        if out.returncode != 0:
            print(f"{WARN} nvidia-smi 执行失败: {out.stderr.strip()[:120]}")
            return
        name, total, used = [x.strip() for x in out.stdout.strip().split(",")]
        total_mb, used_mb = int(total), int(used)
        free_mb = total_mb - used_mb
        print(f"  GPU={name}  总显存={total_mb}MiB  已用={used_mb}MiB  空闲={free_mb}MiB")
        print(f"  常驻服务占用 {used_mb}MiB（通常是 Ollama 的 7B 模型）")
        for label, mb in (("BGE-M3 fp16", 1200), ("BGE-M3 fp32", 2200),
                          ("reranker-v2-m3 fp16", 1100), ("reranker-v2-m3 fp32", 2100)):
            mark = OK if free_mb > mb + 500 else WARN
            print(f"{mark} {label:22s} 约需 {mb}MiB（含激活值需再留 300~500MiB）")
        print()
        print("  结论：三者（LLM + 嵌入 + 重排）不可同时驻留 GPU。")
        print("        推荐 LLM 常驻 GPU，嵌入与重排走 CPU，或按索引/查询分时调度。")
    except FileNotFoundError:
        print(f"{WARN} 未找到 nvidia-smi，跳过显存检查")


def check_llm(prompt: str | None) -> None:
    print()
    print("=" * 74)
    print("四、大模型服务")
    print("=" * 74)
    base = settings.llm_base_resolved
    print(f"  provider = {settings.llm_provider}")
    print(f"  base_url = {base}")
    print(f"  model    = {settings.llm_model_resolved}")
    print(f"  mode     = {'openai' if settings.llm_enabled else 'extractive（降级）'}")

    if "11434" in base:
        # 本机直连：禁用系统代理（WinINET 注册表代理会把 127.0.0.1 也路由走）
        no_proxy_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            # URL 来自本机配置（默认 http://127.0.0.1:11434），非外部输入
            with no_proxy_opener.open(base.replace("/v1", "") + "/api/tags", timeout=8) as resp:
                tags = json.loads(resp.read().decode())
            names = [m["name"] for m in tags.get("models", [])]
            print(f"{OK} Ollama 在线，已安装模型: {names}")
            if settings.llm_model_resolved not in names:
                print(f"{WARN} 模型 {settings.llm_model_resolved} 不在列表中，"
                      f"可执行: ollama pull {settings.llm_model_resolved}")
        except Exception as exc:  # noqa: BLE001
            print(f"{FAIL} 无法连接 Ollama: {exc}")

    if not prompt:
        return
    try:
        from openai import OpenAI

        client = OpenAI(base_url=base, api_key=settings.llm_api_key or "ollama",
                        timeout=settings.llm_timeout)
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=settings.llm_model_resolved,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=128,
        )
        elapsed = time.perf_counter() - t0
        content = (resp.choices[0].message.content or "").strip()
        print(f"{OK} 调用成功，耗时 {elapsed:.2f}s（首次调用含模型加载）")
        print(f"       回复: {content[:200]}")
    except Exception as exc:  # noqa: BLE001
        print(f"{FAIL} 调用失败: {type(exc).__name__}: {str(exc)[:200]}")


def check_embedding_load(path: str | None) -> None:
    print()
    print("=" * 74)
    print("五、本地嵌入模型实载测试")
    print("=" * 74)
    if not path:
        print(f"{WARN} 跳过（未找到本地模型目录）")
        return
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print(f"{WARN} 跳过（sentence-transformers 未安装）")
        return

    t0 = time.perf_counter()
    model = SentenceTransformer(path, device=settings.embedding_device)
    load_s = time.perf_counter() - t0
    get_dim = getattr(model, "get_embedding_dimension", None) or getattr(
        model, "get_sentence_embedding_dimension", None
    )
    dim = get_dim() if callable(get_dim) else None
    t1 = time.perf_counter()
    vecs = model.encode(["父子分块的检索链路", "用子块检索，用父块生成"],
                        normalize_embeddings=True, show_progress_bar=False)
    enc_s = time.perf_counter() - t1
    sim = float((vecs[0] * vecs[1]).sum())
    print(f"{OK} 加载成功: device={settings.embedding_device} 耗时={load_s:.2f}s")
    print(f"       稠密维度 = {dim}（集合定义 = {settings.milvus_dim_resolved}，"
          f"{'一致' if dim == settings.milvus_dim_resolved else '★ 不一致，需改 EMBEDDING_DIM 并重建集合'}）")
    print(f"       编码 2 条耗时 {enc_s * 1000:.0f}ms，语义相似度 = {sim:.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="本地模型与服务自检")
    parser.add_argument("--load", action="store_true", help="实际加载嵌入模型并编码一次")
    parser.add_argument("--llm", default=None, metavar="PROMPT", help="实际调用一次 LLM")
    args = parser.parse_args()

    resolved = check_models()
    deps_ok = check_deps()
    check_gpu()
    check_llm(args.llm)

    if args.load:
        if deps_ok:
            check_embedding_load(resolved.get(settings.embedding_model))
        else:
            print()
            print(f"{WARN} 依赖不全，跳过嵌入模型实载测试")

    print()
    print("=" * 74)
    print("自检结论")
    print("=" * 74)
    print(f"  当前 EMBEDDING_PROVIDER = {settings.embedding_provider}")
    if not deps_ok and settings.embedding_provider == "local":
        print(f"{FAIL} 配置为 local 但依赖缺失，启动时会直接报错；请先装依赖或改回 hash")
    elif not deps_ok:
        print(f"{WARN} 依赖缺失，暂不能切到 local；装好后把 EMBEDDING_PROVIDER 改为 local 并重建集合")
    else:
        print(f"{OK} 依赖齐备，可切换到 local 模式")
    print(f"  RERANK_ENABLED = {settings.rerank_enabled}")
    print(f"  LLM mode = {'openai' if settings.llm_enabled else 'extractive'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
