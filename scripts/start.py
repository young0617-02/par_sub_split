"""一键启动工具：把「依赖 → 配置 → 容器 → 建表 → 灌数据 → 起服务 → 健康检查」串成一条命令。

用法：
    python scripts/start.py                  # 运行按钮：自动重启（先停后起），无需额外命令
    python scripts/start.py restart          # 等价于无参（先停后起）
    python scripts/start.py start            # 只启动（运行中则复用）
    python scripts/start.py status           # 查看各组件状态
    python scripts/start.py selftest         # 只读功能自检（秒级，不写业务数据）
    python scripts/start.py logs [-f]        # 查看/跟踪服务日志
    python scripts/start.py stop [--force]   # 终端停止服务（不传 --force 时校验进程归属，避免误杀其他服务）
    python scripts/start.py --help           # 查看全部参数

设计取向：
1. **无参即重启**：双击 start.bat / 点击「运行按钮」执行自动重启，
   先停掉现有 Web 与容器，再完整拉起，保证回到已知健康状态；全程无需任何输入。
2. **终端可停止**：``stop`` 子命令随时停止 Web 与容器；退出码 0 表示已全部停止。
3. **复用优先、缺失才创建**：容器先按名称逐个 ``docker start``（优先本项目 compose 的
   ``par_sub_*`` 容器，兼容早期遗留容器名）；只有容器完全不存在时才回退
   ``docker compose up -d <service>`` 创建，避免重复建容器抢占 9999 / 19530 端口。
4. **轮询就绪**：所有等待都用「端口/接口探测 + 超时」，不用固定 sleep。
5. **不静默失败**：每类失败都打印「原因 + 修复命令」，不向外抛裸 Traceback。
6. **不跑 pytest**：start/restart 只做轻量健康探测；功能验证请用 selftest 子命令。
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

REQUIREMENTS = PROJECT_ROOT / "requirements.txt"
ENV_FILE = PROJECT_ROOT / ".env"
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"

PID_FILE = PROJECT_ROOT / "_server.pid"
LOG_FILE = PROJECT_ROOT / "_server.log"
ERR_FILE = PROJECT_ROOT / "_server.err"

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000

# 服务启动阶段要等的最长时间（Milvus 单机版首启较慢，给足余量）
DEFAULT_TIMEOUT = 180

# 关键：urllib 默认会读取 Windows 注册表里的系统代理（WinINET），把 127.0.0.1 也路由到代理，
# 导致本机服务返回 404（本机实测系统代理为 127.0.0.1:26561）。这里用空代理 handler 强制直连。
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

# 判定「端口上的进程属于本项目」的特征串（避免误杀同名端口的其他服务）
_OWN_SERVER_MARKERS = ("app.api.main", "run_server.py", "app\\api\\main", "app/api/main")


# ==========================================================================
# 终端输出
# ==========================================================================
def say(text: str = "") -> None:
    print(text, flush=True)


def ok(text: str) -> None:
    print(f"[OK]   {text}", flush=True)


def warn(text: str) -> None:
    print(f"[WARN] {text}", flush=True)


def bad(text: str) -> None:
    print(f"[FAIL] {text}", flush=True)


def head(text: str) -> None:
    print("=" * 68, flush=True)
    print(text, flush=True)
    print("=" * 68, flush=True)


def step(index: int, total: int, title: str) -> None:
    print(f"\n[{index}/{total}] {title}", flush=True)


# ==========================================================================
# 通用工具
# ==========================================================================
def _run(args: list[str], timeout: int = 60) -> subprocess.CompletedProcess[str]:
    """执行外部命令并回收输出；超时/命令缺失也返回结果对象，由调用方判断。"""
    try:
        return subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(args, 124, "", f"命令超时（{timeout}s）")
    except OSError as exc:
        return subprocess.CompletedProcess(args, 127, "", str(exc))


def _settings() -> Any:
    """延迟导入全局配置。

    必须延迟到「配置检查」步骤之后：``app.config`` 在导入时就会读取 .env，
    若此时 .env 尚未从 .env.example 复制生成，读到的是代码默认值（端口/Provider 都可能不一致）。
    """
    from app.config import settings

    return settings


def port_open(host: str, port: int, timeout: float = 0.6) -> bool:
    """TCP 连接探测：端口是否已在监听。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def split_host_port(uri: str, default_port: int) -> tuple[str, int]:
    """从 ``http://host:port`` 之类的 URI 解析出宿主机与端口（用于端口探测）。"""
    body = (uri or "").strip().split("://")[-1]
    body = body.split("/")[0]
    host, sep, port = body.rpartition(":")
    if not sep:
        return (body or SERVER_HOST), default_port
    host = host.split("@")[-1] or SERVER_HOST  # 去掉 user:pass@ 前缀
    try:
        return host, int(port)
    except ValueError:
        return host, default_port


def http_json(method: str, path: str, payload: dict | None = None, timeout: float = 10.0) -> Any:
    """访问本机 Web 接口（强制直连，绕过系统代理）。"""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        f"http://{SERVER_HOST}:{SERVER_PORT}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    with _OPENER.open(request, timeout=timeout) as resp:  # 仅访问本机固定地址，非用户输入
        raw = resp.read().decode("utf-8")
    return json.loads(raw) if raw.strip() else {}


def probe_health(timeout: float = 5.0) -> dict | None:
    """探测 /health；不可达时返回 None（不抛异常，便于轮询）。"""
    try:
        return http_json("GET", "/api/v1/health", timeout=timeout)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
        return None


# ==========================================================================
# 容器
# ==========================================================================
@dataclass(frozen=True)
class Service:
    """一个依赖服务（含候选容器名）。

    ``candidates`` 按优先级排列：**本项目 compose 定义的 ``par_sub_*`` 容器名优先**，
    其后是早期独立部署时使用的遗留容器名（mysql84 / milvus-standalone / redis8 …），
    用于兼容尚未迁移到统一编排的机器。
    """

    label: str
    candidates: tuple[str, ...]
    compose_name: str
    port: int | None = None
    host: str = SERVER_HOST


def build_services() -> list[Service]:
    """按「etcd → minio → MySQL → Redis → Milvus」的顺序给出服务清单。

    全部服务由本仓库的 ``docker-compose.yml``（项目名 ``par_sub_split``）统一编排，
    因此容器名一律以 ``par_sub_`` 前缀为准。
    """
    settings = _settings()
    milvus_host, milvus_port = split_host_port(settings.milvus_uri, 19530)
    return [
        Service("etcd", ("par_sub_etcd", "milvus-etcd"), "etcd"),
        Service("MinIO", ("par_sub_minio", "milvus-minio"), "minio"),
        Service(
            "MySQL",
            ("par_sub_mysql", "mysql84"),
            "mysql",
            port=settings.mysql_port,
            host=settings.mysql_host or SERVER_HOST,
        ),
        Service(
            "Redis",
            ("par_sub_redis", "redis8"),
            "redis",
            port=settings.redis_port,
            host=settings.redis_host,
        ),
        Service(
            "Milvus",
            ("par_sub_milvus", "milvus-standalone"),
            "milvus",
            port=milvus_port,
            host=milvus_host,
        ),
    ]


def container_table() -> dict[str, str]:
    """``容器名 -> 状态文本``（含已停止的容器）。"""
    result = _run(["docker", "ps", "-a", "--format", "{{.Names}}|{{.Status}}"], timeout=30)
    table: dict[str, str] = {}
    if result.returncode != 0:
        return table
    for line in (result.stdout or "").splitlines():
        name, _, status = line.partition("|")
        if name.strip():
            table[name.strip()] = status.strip()
    return table


def resolve_containers(services: list[Service], table: dict[str, str]) -> dict[str, str | None]:
    """为每个服务挑选「机器上真实存在」的容器名。"""
    resolved: dict[str, str | None] = {}
    for svc in services:
        resolved[svc.label] = next((c for c in svc.candidates if c in table), None)
    return resolved


def docker_running() -> tuple[bool, str]:
    """Docker 守护进程是否可用（返回是否可用 + 版本或错误摘要）。"""
    if not shutil.which("docker"):
        return False, "未找到 docker 命令"
    result = _run(["docker", "info", "--format", "{{.ServerVersion}}"], timeout=30)
    if result.returncode != 0:
        return False, (result.stderr or result.stdout or "").strip()[:200]
    return True, (result.stdout or "").strip()


def start_containers(services: list[Service]) -> list[Service]:
    """逐个 ``docker start``；返回「机器上完全没有该容器」的服务（需回退 compose）。"""
    table = container_table()
    resolved = resolve_containers(services, table)
    missing: list[Service] = []
    for svc in services:
        name = resolved[svc.label]
        if name is None:
            missing.append(svc)
            continue
        if table.get(name, "").startswith("Up"):
            ok(f"{svc.label} 容器 {name} 已在运行，复用")
            continue
        result = _run(["docker", "start", name], timeout=120)
        if result.returncode == 0:
            ok(f"{svc.label} 容器 {name} 已启动")
        else:
            warn(f"{svc.label} 容器 {name} 启动失败：{(result.stderr or '').strip()[:160]}")
    return missing


def compose_up(missing: list[Service]) -> bool:
    """仅当容器完全不存在时才回退到 compose 创建（首次部署场景）。

    统一编排后所有服务都在默认 profile 中，因此一次 ``compose up -d`` 即可；
    超时给到 600s，因为 Milvus 首次启动需要初始化元数据与对象存储。
    """
    if not missing:
        return True
    names = [s.compose_name for s in missing]
    cmd = ["docker", "compose", "up", "-d", *names]
    say(f"       未找到容器，回退创建： {' '.join(cmd)}")
    result = _run(cmd, timeout=600)
    if result.returncode != 0:
        bad(f"compose 启动失败：{(result.stderr or result.stdout or '').strip()[:200]}")
        return False
    return True


def service_probes() -> dict[str, Any]:
    """各服务的真实连通探测函数（复用项目自身的客户端）。

    为什么不能只看端口：Docker 在容器启动瞬间就建立了宿主端口转发，
    端口立刻「可连接」，而容器内的服务可能还没开始监听
    （实测 Milvus 会出现「端口已通、客户端连接被拒」）。
    """
    from app.clients.milvus_client import get_chunk_store
    from app.clients.mysql import ping as mysql_ping
    from app.clients.redis_client import ping as redis_ping

    return {
        "MySQL": mysql_ping,
        "Redis": redis_ping,
        "Milvus": lambda: get_chunk_store().ping(),
    }


def call_probe(probe: Any) -> bool:
    """执行探测并归一化为布尔；轮询期间任何异常都只代表「尚未就绪」。"""
    try:
        return bool(probe())
    except Exception:  # noqa: BLE001
        return False


def wait_for_ready(services: list[Service], timeout: int) -> list[Service]:
    """轮询等待服务真正可用；返回超时仍未就绪的服务。"""
    probes = service_probes()
    pending = [s for s in services if s.label in probes]
    started = time.monotonic()
    deadline = started + timeout
    while pending and time.monotonic() < deadline:
        waiting: list[Service] = []
        for svc in pending:
            if port_open(svc.host, svc.port or 0) and call_probe(probes[svc.label]):
                ok(f"{svc.label} 已就绪 {svc.host}:{svc.port}（等待 {time.monotonic() - started:.0f}s）")
            else:
                waiting.append(svc)
        pending = waiting
        if pending:
            time.sleep(2)
    return pending


def stop_containers(services: list[Service]) -> None:
    """停止本项目相关的运行中容器（按真实容器名）。"""
    table = container_table()
    if not table:
        warn("无法读取容器列表（Docker 未运行？），跳过容器停止")
        return
    resolved = resolve_containers(services, table)
    any_running = False
    for svc in services:
        name = resolved[svc.label]
        if name is None or not table.get(name, "").startswith("Up"):
            continue
        any_running = True
        result = _run(["docker", "stop", name], timeout=120)
        if result.returncode == 0:
            ok(f"{svc.label} 容器 {name} 已停止")
        else:
            warn(f"{svc.label} 容器 {name} 停止失败：{(result.stderr or '').strip()[:160]}")
    if not any_running:
        say("       容器均未在运行，无需停止")


# ==========================================================================
# Web 服务进程
# ==========================================================================
def read_pid() -> int | None:
    try:
        return int(PID_FILE.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def process_cmdline(pid: int) -> str:
    """读取进程命令行（用于确认目标进程确实是本项目服务）。"""
    if os.name == "nt":
        result = _run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f'(Get-CimInstance Win32_Process -Filter "ProcessId={pid}").CommandLine',
            ],
            timeout=30,
        )
        return (result.stdout or "").strip()
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\x00", b" ").decode("utf-8", "replace").strip()


def is_own_server(cmdline: str) -> bool:
    return bool(cmdline) and any(marker in cmdline for marker in _OWN_SERVER_MARKERS)


def pid_on_port(port: int) -> int | None:
    """兜底：反查监听指定端口的进程 PID。"""
    if os.name == "nt":
        result = _run(["netstat", "-ano", "-p", "tcp"], timeout=30)
        for line in (result.stdout or "").splitlines():
            fields = line.split()
            if len(fields) >= 5 and fields[-2] == "LISTENING" and fields[1].endswith(f":{port}"):
                try:
                    return int(fields[-1])
                except ValueError:
                    return None
        return None
    lsof = shutil.which("lsof")
    if not lsof:
        return None
    result = _run([lsof, "-ti", f"tcp:{port}"], timeout=30)
    for line in (result.stdout or "").splitlines():
        if line.strip().isdigit():
            return int(line.strip())
    return None


def kill_process(pid: int) -> bool:
    if os.name == "nt":
        result = _run(["taskkill", "/PID", str(pid), "/T", "/F"], timeout=60)
        return result.returncode == 0
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return False
    for _ in range(10):
        if not process_cmdline(pid):
            return True
        time.sleep(0.5)
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        return False
    return True


def spawn_server_foreground() -> subprocess.Popen:
    """前台启动 uvicorn 子进程（共享当前终端）。

    为什么必须前台而不是后台 (DETACHED)：
    - 用户按 Ctrl+C 时，终端会把中断信号同时发给本进程与它；
      uvicorn 收到后执行**优雅关闭**（停收新请求 → 等待在途请求完成 →
      触发 lifespan shutdown 释放数据库连接池 / Redis / Milvus 句柄），
      再安全退出 —— 满足「优雅终止、保存状态、完成正在处理的任务」的要求。
    - 本进程随后等待子进程退出，再停止依赖容器，确保资源先被 Web 层释放。
    """
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.api.main:app",
            "--host",
            SERVER_HOST,
            "--port",
            str(SERVER_PORT),
        ],
        cwd=PROJECT_ROOT,
        stdin=subprocess.DEVNULL,
    )
    PID_FILE.write_text(str(proc.pid), encoding="utf-8")
    return proc


def wait_for_graceful_server(proc: subprocess.Popen) -> None:
    """等待前台 uvicorn 优雅退出；Ctrl+C 由 uvicorn 处理，本进程随后收割子进程。

    首次 Ctrl+C：本进程的 ``wait()`` 会抛 KeyboardInterrupt，
    此时 uvicorn 正在做优雅关闭，需再次 ``wait()`` 等它真正退出。"""
    say()
    say("  ---------------------------------------------------------------")
    say("  Web 服务前台运行中，按 Ctrl+C 即可优雅停止整个服务（无需其它命令）。")
    say(f"  调试台 http://{SERVER_HOST}:{SERVER_PORT}/ | 接口文档 /docs")
    say("  ---------------------------------------------------------------")
    try:
        proc.wait()
    except KeyboardInterrupt:
        say("\n收到 Ctrl+C，正在优雅关闭：等待 Web 处理完在途请求、释放连接...")
        with suppress(KeyboardInterrupt):  # 用户连按 Ctrl+C：子进程已被信号终止，直接继续
            proc.wait()  # 收割子进程：uvicorn 完成优雅关闭后自身退出


def stop_web(force: bool = False) -> bool:
    """停止 Web 服务：先按 PID 文件，再兜底按端口反查。返回是否真的停掉了进程。"""
    pid = read_pid()
    if pid is not None:
        cmdline = process_cmdline(pid)
        if is_own_server(cmdline) or force:
            if kill_process(pid):
                ok(f"Web 服务已停止（PID {pid}）")
                PID_FILE.unlink(missing_ok=True)
                return True
            warn(f"PID {pid} 停止失败，请手动检查")
        elif not cmdline:
            warn(f"PID 文件中的进程 {pid} 已不存在，改为按端口 {SERVER_PORT} 反查")
        else:
            warn(f"PID {pid} 命令行与本项目服务不匹配，已跳过：{cmdline[:120]}")
        PID_FILE.unlink(missing_ok=True)

    pid = pid_on_port(SERVER_PORT)
    if pid is None:
        say("       Web 服务未在运行")
        return False
    cmdline = process_cmdline(pid)
    if not (is_own_server(cmdline) or force):
        warn(f"端口 {SERVER_PORT} 被 PID {pid} 占用，但不属于本项目，已跳过：{cmdline[:120]}")
        return False
    if kill_process(pid):
        ok(f"Web 服务已停止（按端口反查 PID {pid}）")
        return True
    warn(f"PID {pid} 停止失败，请手动检查")
    return False


def wait_for_retrieval(timeout: int) -> tuple[bool, str]:
    """轮询 /health 直到三存储连通且检索组件预热完成。"""
    deadline = time.monotonic() + timeout
    last = "服务不可达"
    while time.monotonic() < deadline:
        body = probe_health(timeout=5)
        if body is not None:
            stores = {k: bool(body.get(k)) for k in ("mysql", "redis", "milvus")}
            ready = bool((body.get("detail") or {}).get("retrieval_ready"))
            if all(stores.values()) and ready:
                return True, "ready"
            down = [k for k, v in stores.items() if not v]
            last = f"三存储未全部连通（异常：{', '.join(down)}）" if down else "组件预热尚未完成"
        time.sleep(2)
    return False, last


# ==========================================================================
# 依赖 / 配置
# ==========================================================================
def normalize_pkg(name: str) -> str:
    """PEP 503 归一化：大小写、下划线、点号统一为连字符。"""
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def required_packages() -> list[str]:
    """解析 requirements.txt 里的直接依赖名（忽略注释与可选依赖块）。"""
    packages: list[str] = []
    for raw in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        name = re.split(r"[<>=!~;\[\s]", line, maxsplit=1)[0].strip()
        if name:
            packages.append(name)
    return packages


def missing_packages() -> list[str]:
    installed = set()
    for dist in importlib.metadata.distributions():
        meta = dist.metadata
        meta_name = meta["Name"] if meta else None
        if meta_name:
            installed.add(normalize_pkg(meta_name))
    return [p for p in required_packages() if normalize_pkg(p) not in installed]


def ensure_env_file() -> bool:
    """确保 .env 存在；返回是否新建。"""
    if ENV_FILE.exists():
        return False
    if not ENV_EXAMPLE.exists():
        bad(f"缺少 {ENV_EXAMPLE.name}，无法生成 .env")
        return False
    shutil.copyfile(ENV_EXAMPLE, ENV_FILE)
    ok(f"未发现 .env，已从 {ENV_EXAMPLE.name} 复制生成")
    return True


# ==========================================================================
# 子命令：start
# ==========================================================================
def cmd_start(args: argparse.Namespace) -> int:
    total = 10
    head("父子分块 RAG 检索系统 —— 一键启动")

    # ---------- 1 环境自检 ----------
    step(1, total, "环境自检")
    if sys.version_info[:2] < (3, 10):  # noqa: UP036  运行时兜底：用户可能用旧解释器执行本脚本
        bad(f"Python 版本过低：{sys.version.split()[0]}，本项目要求 >= 3.10")
        say("       修复：安装 Python 3.10+ 后重新执行")
        return 1
    ok(f"Python {sys.version.split()[0]}（{sys.executable}）")
    markers = ("app/api/main.py", "requirements.txt", "scripts/init_db.py")
    absent = [m for m in markers if not (PROJECT_ROOT / m).exists()]
    if absent:
        bad(f"项目根目录不完整，缺少：{', '.join(absent)}（当前根目录 {PROJECT_ROOT}）")
        say("       修复：在项目根目录执行本脚本")
        return 1
    if not shutil.which("docker"):
        bad("未找到 docker 命令")
        say("       修复：安装 Docker Desktop 并确保 docker 已加入 PATH")
        return 1
    ok("项目根目录与 docker 命令检查通过")

    # ---------- 2 依赖检查 ----------
    step(2, total, "依赖检查")
    missing = missing_packages()
    if not missing:
        ok("依赖已齐全（requirements.txt 全部满足）")
    elif args.no_install:
        warn(f"缺少依赖：{', '.join(missing)}（已指定 --no-install，跳过安装）")
    else:
        warn(f"缺少依赖：{', '.join(missing)}，开始安装")
        say(f"       执行： {sys.executable} -m pip install -r {REQUIREMENTS}")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS)],
            cwd=PROJECT_ROOT,
            check=False,
        )
        if result.returncode != 0:
            bad("依赖安装失败")
            say("       修复：手动执行上述 pip 命令查看详细报错")
            return 1
        ok("依赖安装完成")

    # ---------- 3 配置检查 ----------
    step(3, total, "配置检查")
    ensure_env_file()
    if not ENV_FILE.exists():
        bad(f"缺少配置文件 {ENV_FILE}")
        say("       修复：参考 .env.example 手动创建 .env")
        return 1
    try:
        settings = _settings()
    except Exception as exc:  # noqa: BLE001  配置非法时给出可操作的提示
        bad(f".env 配置解析失败：{exc}")
        say("       修复：对照 .env.example 检查配置项名称与取值")
        return 1
    ok(f".env 已加载 | MySQL {settings.mysql_host}:{settings.mysql_port}/{settings.mysql_database}"
       f" | Redis {settings.redis_host}:{settings.redis_port}"
       f" | Milvus {settings.milvus_uri}")
    if not settings.milvus_token.strip():
        warn("MILVUS_TOKEN 为空：若 Milvus 开启了鉴权会连接失败（本机默认 root:123456）")
    if not settings.admin_enabled:
        warn("ADMIN_TOKEN 为空：/api/v1/admin/* 无需鉴权，仅供本机开发")
    if not settings.llm_enabled:
        warn("大模型未配置可用凭据：/chat 会走抽取式降级模式（检索链路不受影响）")

    # ---------- 4 Docker 守护进程 ----------
    step(4, total, "Docker 守护进程")
    available, info = docker_running()
    if not available:
        bad(f"Docker 不可用：{info}")
        say("       修复：启动 Docker Desktop，等待其状态变为 Running 后重试")
        return 1
    ok(f"Docker 守护进程正常（Server {info}）")

    # ---------- 5 启动容器 ----------
    step(5, total, "启动依赖容器（复用优先）")
    services = build_services()
    missing_services = start_containers(services)
    if missing_services:
        if not compose_up(missing_services):
            bad("部分容器创建失败")
            say("       修复：手动执行 docker compose up -d 后重试")
            return 1
        start_containers(services)

    # ---------- 6 等待就绪 ----------
    step(6, total, f"等待服务就绪（超时 {args.timeout}s）")
    not_ready = wait_for_ready(services, args.timeout)
    if not_ready:
        for svc in not_ready:
            bad(f"{svc.label} 未在 {args.timeout}s 内可用（{svc.host}:{svc.port}）")
        say("       修复：查看容器日志定位，如 docker logs --tail 50 par_sub_milvus")
        return 1

    # ---------- 7 初始化三存储 ----------
    step(7, total, "初始化 MySQL 表 / Milvus 集合")
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "init_db.py")],
        cwd=PROJECT_ROOT,
        check=False,
    )
    if result.returncode != 0:
        bad("三存储初始化未通过")
        say("       修复：按上面 init_db.py 的输出逐项处理（常见原因是容器未起或 MILVUS_TOKEN 不正确）")
        return 1

    # ---------- 8 灌入样例数据（仅当库为空） ----------
    step(8, total, "样例数据")
    if args.no_ingest:
        warn("已指定 --no-ingest，跳过样例数据检查")
    else:
        parent_count = count_parents()
        if parent_count is None:
            warn("无法统计 MySQL 父块数量，跳过样例灌入")
        elif parent_count > 0:
            ok(f"MySQL 已有 {parent_count} 个父块，跳过灌入（幂等）")
        else:
            say("       MySQL 无父块，正在灌入 data/samples 样例数据")
            result = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "scripts" / "ingest_samples.py")],
                cwd=PROJECT_ROOT,
                check=False,
            )
            if result.returncode != 0:
                warn("样例数据灌入未成功，检索结果可能为空（可用 selftest 复核）")

    # ---------- 9 启动 Web 服务（前台，Ctrl+C 优雅停止） ----------
    step(9, total, "启动 Web 服务（前台，Ctrl+C 可优雅停止）")
    if port_open(SERVER_HOST, SERVER_PORT) and probe_health(timeout=5) is not None:
        ok(f"端口 {SERVER_PORT} 已有本机服务在响应，复用现有进程，不再重复拉起")
        server_proc = None
    elif port_open(SERVER_HOST, SERVER_PORT):
        bad(f"端口 {SERVER_PORT} 已被占用，但不是本项目的服务（/health 无响应）")
        say("       修复：先执行 python scripts/start.py stop --force，或释放该端口")
        return 1
    else:
        server_proc = spawn_server_foreground()
        ok(f"Web 服务已启动（PID {server_proc.pid}，前台运行）")

    # ---------- 10 健康检查 ----------
    step(10, total, f"等待检索组件预热（最多 {args.timeout}s）")
    ready, reason = wait_for_retrieval(args.timeout)
    if not ready:
        bad(f"服务未就绪：{reason}")
        say(f"       修复：查看日志 python scripts/start.py logs --err；错误日志 {ERR_FILE}")
        if server_proc is not None:
            server_proc.kill()
        return 1
    ok("健康检查通过：三存储连通，检索组件已预热")

    print_panel(services)

    # ---------- 前台驻留：直到用户按 Ctrl+C 优雅停止 ----------
    if server_proc is not None:
        wait_for_graceful_server(server_proc)
        say("\nWeb 服务已优雅退出。正在停止依赖容器...")
        _stop_all(force=True)
        say("=" * 68)
        say("所有服务均已安全停止。")
    return 0


def count_parents() -> int | None:
    """统计 MySQL 父块数量；不可用时返回 None。"""
    try:
        from sqlalchemy import func, select

        from app.clients.mysql import session_scope
        from app.models import ParentDocument

        with session_scope() as session:
            return int(session.execute(select(func.count(ParentDocument.id))).scalar() or 0)
    except Exception as exc:  # noqa: BLE001  统计失败不影响启动主流程
        say(f"       统计父块数量失败：{exc}")
        return None


def print_panel(services: list[Service]) -> None:
    """打印启动结果面板。"""
    table = container_table()
    resolved = resolve_containers(services, table)
    pid = read_pid()
    say()
    head("启动完成")
    for svc in services:
        name = resolved[svc.label]
        state = table.get(name or "", "容器不存在")
        say(f"  {svc.label:<7} {name or '-':<20} {state}")
    say(f"  {'Web':<7} {'-' if pid is None else f'PID {pid}':<20} http://{SERVER_HOST}:{SERVER_PORT}/ 已就绪")
    say()
    say(f"  调试台   http://{SERVER_HOST}:{SERVER_PORT}/")
    say(f"  接口文档 http://{SERVER_HOST}:{SERVER_PORT}/docs")
    say(f"  日志     {LOG_FILE}  |  {ERR_FILE}")
    say("  常用命令 python scripts/start.py status | selftest | logs | stop | restart")
    say("=" * 68)


# ==========================================================================
# 子命令：status / selftest / stop / restart / logs
# ==========================================================================
def cmd_status(_args: argparse.Namespace) -> int:
    head("父子分块 RAG 检索系统 —— 运行状态")
    settings = _settings()

    table = container_table()
    if not table:
        warn("无法读取容器列表（Docker 未运行？）")
    for svc in build_services():
        resolved = resolve_containers([svc], table)[svc.label]
        state = table.get(resolved or "", "容器不存在")
        say(f"  容器 {svc.label:<7} {resolved or '-':<20} {state}")

    say()
    milvus_host, milvus_port = split_host_port(settings.milvus_uri, 19530)
    for label, host, port in (
        ("MySQL", settings.mysql_host, settings.mysql_port),
        ("Redis", settings.redis_host, settings.redis_port),
        ("Milvus", milvus_host, milvus_port),
        ("Web", SERVER_HOST, SERVER_PORT),
    ):
        state = "占用" if port_open(host, port) else "空闲"
        say(f"  端口 {label:<7} {host}:{port:<8} {state}")

    say()
    pid = read_pid()
    if pid is not None:
        cmdline = process_cmdline(pid)
        say(f"  Web  服务 PID {pid} {'存活' if cmdline else '已退出'}")
    else:
        say("  Web  服务 无 PID 记录（可能由其他方式启动）")

    body = probe_health(timeout=5)
    if body is None:
        warn(f"/api/v1/health 不可达（http://{SERVER_HOST}:{SERVER_PORT}）")
        say("       修复：python scripts/start.py start")
        return 1
    detail = body.get("detail") or {}
    say(f"  健康检查 mysql={body.get('mysql')} redis={body.get('redis')} milvus={body.get('milvus')} "
        f"retrieval_ready={detail.get('retrieval_ready')}")
    say("=" * 68)
    return 0


SELFTEST_QUERY = "父子分块检索的父块和子块分别承担什么职责？"


def cmd_selftest(_args: argparse.Namespace) -> int:
    """只读功能自检：健康检查 + 一次检索召回。不写任何业务数据。"""
    head("父子分块 RAG 检索系统 —— 功能自检（只读）")
    failures: list[str] = []

    body = probe_health(timeout=10)
    if body is None:
        bad(f"/api/v1/health 不可达（http://{SERVER_HOST}:{SERVER_PORT}），请先执行 start")
        return 1
    ok("服务可达：/api/v1/health")

    for key, label in (("mysql", "MySQL"), ("redis", "Redis"), ("milvus", "Milvus")):
        if body.get(key):
            ok(f"{label} 连通正常")
        else:
            bad(f"{label} 连通异常")
            failures.append(f"{label} 未连通")

    detail = body.get("detail") or {}
    if detail.get("retrieval_ready"):
        ok("检索组件已预热（BM25 / 嵌入 / 重排）")
    else:
        bad("检索组件未就绪：预热可能失败，详见服务端日志")
        failures.append("retrieval_ready=false")

    try:
        resp = http_json(
            "POST",
            "/api/v1/search",
            {"query": SELFTEST_QUERY, "top_k": 10, "final_top_k": 3},
            timeout=60,
        )
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ValueError) as exc:
        bad(f"检索请求失败：{exc}")
        return 1

    parents = resp.get("parents") or []
    if parents:
        ok(f"检索链路正常：召回 {len(parents)} 个父块，首个 parent_id={parents[0].get('parent_id')}"
           f"，耗时 {resp.get('timings', {}).get('total_ms', '-')}ms")
    else:
        bad("检索未召回任何父块：库中可能没有数据")
        say("       修复：python scripts/ingest_samples.py")
        failures.append("检索结果为空")

    say()
    if failures:
        bad(f"自检未通过（{len(failures)} 项）：{'；'.join(failures)}")
        return 1
    say("自检全部通过。")
    return 0


def _stop_all(force: bool) -> None:
    """停止 Web 服务与依赖容器（stop / restart / 无参运行共用），不重复打印大标题。"""
    stop_web(force=force)
    stop_containers(build_services())


def cmd_stop(args: argparse.Namespace) -> int:
    head("父子分块 RAG 检索系统 —— 停止服务")
    _stop_all(force=args.force)
    say("=" * 68)
    return 0


def cmd_restart(args: argparse.Namespace) -> int:
    """重启（无参「运行按钮」与 restart 子命令共用）：先停后起，保证回到已知健康状态。"""
    head("父子分块 RAG 检索系统 —— 自动重启")
    say("第一步：停止现有服务")
    _stop_all(force=True)
    say()
    return cmd_start(args)


def cmd_logs(args: argparse.Namespace) -> int:
    path = ERR_FILE if args.err else LOG_FILE
    if not path.exists():
        bad(f"日志文件不存在：{path}")
        say("       修复：先执行 python scripts/start.py start")
        return 1

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in lines[-max(1, args.lines):]:
        say(line)
    if not args.follow:
        return 0

    say(f"--- 跟踪 {path.name} 中，按 Ctrl+C 退出 ---")
    position = path.stat().st_size
    try:
        while True:
            time.sleep(0.5)
            size = path.stat().st_size
            if size < position:  # 日志被清空/轮转
                position = 0
            if size > position:
                with path.open("r", encoding="utf-8", errors="replace") as handle:
                    handle.seek(position)
                    print(handle.read(), end="")
                position = size
    except KeyboardInterrupt:
        say("\n已停止跟踪日志。")
    return 0


# ==========================================================================
# 入口
# ==========================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="start.py",
        description="父子分块 RAG 检索系统 —— 运行按钮(自动重启) / 启动 / 停止 / 自检工具",
    )
    sub = parser.add_subparsers(dest="command", metavar="{start,stop,status,restart,logs,selftest}")

    p_start = sub.add_parser("start", help="只启动（运行中则复用，不停止）")
    p_start.add_argument("--no-install", action="store_true", help="跳过缺失依赖的自动安装")
    p_start.add_argument("--no-ingest", action="store_true", help="库为空时不自动灌入样例数据")
    p_start.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="等待就绪的超时秒数")
    p_start.set_defaults(func=cmd_start)

    p_stop = sub.add_parser("stop", help="停止 Web 服务与依赖容器")
    p_stop.add_argument("--force", action="store_true", help="跳过进程归属校验，直接结束占用端口的进程")
    p_stop.set_defaults(func=cmd_stop)

    sub.add_parser("status", help="查看容器 / 端口 / 服务状态").set_defaults(func=cmd_status)
    sub.add_parser("selftest", help="只读功能自检（健康检查 + 一次检索）").set_defaults(func=cmd_selftest)

    p_restart = sub.add_parser("restart", help="重启（先停后起），与无参「运行按钮」等价")
    p_restart.add_argument("--no-install", action="store_true", help="跳过缺失依赖的自动安装")
    p_restart.add_argument("--no-ingest", action="store_true", help="库为空时不自动灌入样例数据")
    p_restart.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="等待就绪的超时秒数")
    p_restart.set_defaults(func=cmd_restart)

    p_logs = sub.add_parser("logs", help="查看/跟踪服务日志")
    p_logs.add_argument("-n", "--lines", type=int, default=50, help="先打印末尾多少行（默认 50）")
    p_logs.add_argument("--err", action="store_true", help="改看错误日志 _server.err")
    p_logs.add_argument("--no-follow", dest="follow", action="store_false", help="只看当前日志，不持续跟踪")
    p_logs.set_defaults(func=cmd_logs, follow=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        with suppress(AttributeError, ValueError):
            stream.reconfigure(encoding="utf-8", errors="replace")  # 非 TTY（被重定向）时忽略

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        # 不带子命令（例如双击 start.bat 的「运行按钮」）：自动执行重启（先停后起），无需任何额外输入
        args = argparse.Namespace(
            func=cmd_restart,
            command="restart",
            no_install=False,
            no_ingest=False,
            timeout=DEFAULT_TIMEOUT,
        )
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        say("\n已中断。")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
