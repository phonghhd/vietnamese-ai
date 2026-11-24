"""Extension Web3 / DePIN - Mạng P2P, Tokenomics và TEE/ZKP mô phỏng (extra: `web3`)."""

from vietnamese_ai.utils.lazy_loader import tao_lazy_getattr

_LAZY = {
    "P2PTracker": ("vietnamese_ai.edge.p2p_network", "P2PTracker"),
    "TokenLedger": ("vietnamese_ai.edge.p2p_network", "TokenLedger"),
    "SecureEdgeNode": ("vietnamese_ai.edge.tee_zkp_node", "SecureEdgeNode"),
    "WebBrowserNode": ("vietnamese_ai.edge.wasm_node", "WebBrowserNode"),
}

__all__ = list(_LAZY.keys())
__getattr__ = tao_lazy_getattr(__name__, _LAZY, globals())
