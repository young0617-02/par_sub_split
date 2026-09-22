"""Redis 父文档缓存 —— 整条链路的"提速器"。

为什么用 Redis 而不是直接回查 MySQL？
    Milvus 只能给出 ``parent_id``，最终要拿的是几百~几千字的父块全文。
    MySQL 单次主键回表在 1~5ms 量级，而 Redis 在 0.1~0.5ms 量级，
    且父块内容天然"读多写少、可容忍短暂不一致"，是典型的缓存友好场景。
    批量取多个父块时应使用 ``mget``，把 N 次往返压成 1 次（避免 N+1 查询）。

key 设计：
    ``psd:parent:{parent_id}`` -> 父块全文（string）
    另用 ``psd:cache:hits`` / ``psd:cache:misses`` 做跨进程命中率统计。
"""

from __future__ import annotations

import logging

import redis

from app.config import settings

logger = logging.getLogger(__name__)

_HIT_KEY = "psd:cache:hits"
_MISS_KEY = "psd:cache:misses"
_REQ_KEY = "psd:cache:requests"

_client: redis.Redis | None = None


def get_client() -> redis.Redis:
    """进程内单例 Redis 连接池。"""
    global _client
    if _client is None:
        _client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=settings.redis_password or None,
            decode_responses=True,
            socket_timeout=settings.redis_socket_timeout,
            socket_connect_timeout=settings.redis_socket_timeout,
            health_check_interval=30,
            max_connections=32,
        )
    return _client


class ParentCache:
    """父文档内容缓存操作封装。"""

    def __init__(self, ttl: int | None = None, prefix: str | None = None) -> None:
        self.ttl = settings.redis_parent_ttl if ttl is None else ttl
        self.prefix = settings.redis_parent_prefix if prefix is None else prefix

    # ------------------------------------------------------------------
    def key(self, parent_id: int) -> str:
        return f"{self.prefix}{parent_id}"

    # ------------------------------------------------------------------
    def set(self, parent_id: int, content: str, ttl: int | None = None) -> None:
        """写入单个父块内容。"""
        ttl = self.ttl if ttl is None else ttl
        if ttl and ttl > 0:
            get_client().set(self.key(parent_id), content, ex=ttl)
        else:
            get_client().set(self.key(parent_id), content)

    def set_many(self, items: dict[int, str], ttl: int | None = None) -> int:
        """批量写入（pipeline，一次 RTT）。"""
        if not items:
            return 0
        ttl = self.ttl if ttl is None else ttl
        pipe = get_client().pipeline(transaction=False)
        for pid, content in items.items():
            if ttl and ttl > 0:
                pipe.set(self.key(pid), content, ex=ttl)
            else:
                pipe.set(self.key(pid), content)
        pipe.execute()
        return len(items)

    def get(self, parent_id: int) -> str | None:
        """读取单个父块内容，并累计命中率。"""
        content = get_client().get(self.key(parent_id))
        self._record(1 if content is not None else 0)
        return content

    def mget(self, parent_ids: list[int]) -> dict[int, str]:
        """批量读取：**一次 RTT 拿回 N 个父块**，这是缓存价值的核心体现。"""
        if not parent_ids:
            return {}
        keys = [self.key(pid) for pid in parent_ids]
        values = get_client().mget(keys)
        result: dict[int, str] = {}
        hits = 0
        for pid, val in zip(parent_ids, values, strict=True):
            if val is not None:
                result[pid] = val
                hits += 1
        self._record(hits, total=len(parent_ids))
        return result

    def delete(self, parent_ids: list[int]) -> int:
        if not parent_ids:
            return 0
        return int(get_client().delete(*[self.key(pid) for pid in parent_ids]))

    def exists(self, parent_id: int) -> bool:
        return bool(get_client().exists(self.key(parent_id)))

    # ------------------------------------------------------------------
    def count_keys(self) -> int:
        """统计缓存中的父块 key 数量（scan_iter 避免 KEYS 阻塞）。"""
        return sum(1 for _ in get_client().scan_iter(match=f"{self.prefix}*", count=500))

    def _record(self, hits: int, total: int | None = None) -> None:
        """命中率统计放进 Redis，天然支持多进程/多实例聚合。"""
        total = hits if total is None else total
        try:
            pipe = get_client().pipeline(transaction=False)
            pipe.incrby(_HIT_KEY, hits)
            pipe.incrby(_MISS_KEY, total - hits)
            pipe.incrby(_REQ_KEY, total)
            pipe.execute()
        except redis.RedisError as exc:  # 统计失败不影响主流程
            logger.debug("缓存统计写入失败：%s", exc)

    def stats(self) -> dict[str, float]:
        client = get_client()
        hits = int(client.get(_HIT_KEY) or 0)
        misses = int(client.get(_MISS_KEY) or 0)
        requests = int(client.get(_REQ_KEY) or 0)
        return {
            "requests": requests,
            "hits": hits,
            "misses": misses,
            "hit_rate": round(hits / requests, 4) if requests else 0.0,
        }

    def reset_stats(self) -> None:
        get_client().delete(_HIT_KEY, _MISS_KEY, _REQ_KEY)

    def clear_all(self) -> int:
        """清空所有父块缓存（慎用）。"""
        keys = list(get_client().scan_iter(match=f"{self.prefix}*", count=500))
        if not keys:
            return 0
        return int(get_client().delete(*keys))


def ping() -> bool:
    try:
        return bool(get_client().ping())
    except Exception as exc:  # noqa: BLE001
        logger.warning("Redis 连通性检查失败：%s", exc)
        return False


_parent_cache: ParentCache | None = None


def get_parent_cache() -> ParentCache:
    global _parent_cache
    if _parent_cache is None:
        _parent_cache = ParentCache()
    return _parent_cache
