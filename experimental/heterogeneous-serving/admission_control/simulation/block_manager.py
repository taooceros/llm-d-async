"""
Physical Block Allocator for Paged KV-Cache Memory Management.
==============================================================
Mirrors vLLM PagedAttention physical block allocator semantics.
"""

from typing import Dict, Any, Optional


class BlockManager:
    """
    Physical Block Allocator for Paged KV-Cache Memory Management.
    Mirrors vLLM PagedAttention physical block allocator semantics with deterministic
    unique block tracking and reference-counted radix prefix caching.
    """

    def __init__(
        self,
        capacity_tokens: int,
        block_size: int = 16,
        enable_prefix_caching: bool = False,
    ) -> None:
        self.capacity_tokens = capacity_tokens
        self.block_size = block_size
        self.total_blocks = capacity_tokens // block_size
        self.free_blocks = self.total_blocks
        self.enable_prefix_caching = enable_prefix_caching

        # Allocation tracking: req_id -> unique physical blocks
        self.allocated_blocks: Dict[Any, int] = {}
        # Prefix cache tracking: prefix_hash -> block count
        self.prefix_blocks: Dict[str, int] = {}
        # Prefix ref count: prefix_hash -> active reference count
        self.prefix_refs: Dict[str, int] = {}
        # Mapping: req_id -> prefix_hash
        self.req_prefix: Dict[Any, str] = {}

        self.prefix_hits: int = 0
        self.saved_tokens: int = 0

    def tokens_to_blocks(self, tokens: int) -> int:
        """Converts token count to required physical block count."""
        if tokens <= 0:
            return 0
        return (tokens + self.block_size - 1) // self.block_size

    def can_allocate(self, num_blocks: int) -> bool:
        """Predicate checking whether num_blocks can be physically allocated."""
        return self.free_blocks >= num_blocks

    def allocate(
        self,
        req_id: Any,
        num_blocks: int,
        prefix_hash: Optional[str] = None,
        prefix_tokens: int = 0,
    ) -> None:
        """
        Allocates physical KV cache blocks for req_id.
        Integrates deterministic reference-counted radix prefix caching when enabled.
        """
        if num_blocks <= 0:
            return

        p_blocks = self.tokens_to_blocks(prefix_tokens) if prefix_tokens > 0 else 0
        p_blocks = min(num_blocks, p_blocks)

        if self.enable_prefix_caching and prefix_hash is not None and p_blocks > 0:
            shared_b = self.prefix_blocks.get(prefix_hash)
            if shared_b is not None:
                self.prefix_refs[prefix_hash] = self.prefix_refs.get(prefix_hash, 0) + 1
                self.prefix_hits += 1
                self.saved_tokens += shared_b * self.block_size
                unique_b = max(0, num_blocks - shared_b)
            else:
                assert self.free_blocks >= p_blocks, (
                    f"KV OOM on prefix: Free={self.free_blocks}, Needed={p_blocks}"
                )
                self.free_blocks -= p_blocks
                self.prefix_blocks[prefix_hash] = p_blocks
                self.prefix_refs[prefix_hash] = 1
                unique_b = num_blocks - p_blocks

            self.req_prefix[req_id] = prefix_hash
        else:
            unique_b = num_blocks

        assert self.free_blocks >= unique_b, (
            f"KV OOM: Free={self.free_blocks}, Requested={unique_b}"
        )
        self.free_blocks -= unique_b
        self.allocated_blocks[req_id] = self.allocated_blocks.get(req_id, 0) + unique_b

    def free(self, req_id: Any) -> int:
        """
        Deallocates all physical blocks assigned to req_id.
        Decrements prefix reference counts and releases prefix blocks if references drop to zero.
        Returns total number of physical blocks reclaimed.
        """
        freed = self.allocated_blocks.pop(req_id, 0)
        self.free_blocks += freed

        if self.enable_prefix_caching:
            p_hash = self.req_prefix.pop(req_id, None)
            if p_hash is not None:
                current_refs = self.prefix_refs.get(p_hash, 0) - 1
                if current_refs <= 0:
                    self.prefix_refs.pop(p_hash, None)
                    p_b = self.prefix_blocks.pop(p_hash, 0)
                    self.free_blocks += p_b
                    freed += p_b
                else:
                    self.prefix_refs[p_hash] = current_refs
        return freed

    def free_all(self) -> None:
        """Releases all unique and prefix blocks, restoring total capacity."""
        self.free_blocks = self.total_blocks
        self.allocated_blocks.clear()
        self.prefix_blocks.clear()
        self.prefix_refs.clear()
        self.req_prefix.clear()

    def get_allocated(self, req_id: Any) -> int:
        """Returns the total number of blocks (unique + prefix) allocated to req_id."""
        alloc = self.allocated_blocks.get(req_id, 0)
        if self.enable_prefix_caching:
            p_hash = self.req_prefix.get(req_id)
            if p_hash is not None:
                alloc += self.prefix_blocks.get(p_hash, 0)
        return alloc

    @property
    def used_blocks(self) -> int:
        return self.total_blocks - self.free_blocks

    @property
    def used_tokens(self) -> int:
        return self.used_blocks * self.block_size

    @property
    def free_tokens(self) -> int:
        return self.free_blocks * self.block_size
