"""父子分块（Parent-Child Chunking）RAG 检索系统。

核心链路：
    入库: 原始文档 -> 父块(MySQL) + 子块(Milvus 稠密/稀疏双向量, 携带 parent_id)
          -> 父块内容镜像进 Redis (key=parent_id)
    查询: 问题 -> Milvus 混合检索命中子块 -> 取 parent_id
          -> Redis 取父块全文(毫秒级) -> 组装 Prompt -> 大模型生成
"""

__version__ = "1.2.0"
