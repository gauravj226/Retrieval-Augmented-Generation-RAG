"""
MCP (Model Context Protocol) integration.
fastapi-mcp auto-exposes selected FastAPI endpoints as MCP tools.
Connect via: http://localhost:3002/api/mcp  (through nginx proxy)
"""
from fastapi import FastAPI


def setup_mcp(app: FastAPI):
    """
    Call this after all routers are registered.
    Exposes chat and document query endpoints as MCP tools.
    """
    try:
        from fastapi_mcp import FastApiMCP

        mcp = FastApiMCP(
            app,
            name="RAG Platform MCP Server",
            description=(
                "Query department knowledge bases using RAG. "
                "Tools: list available knowledge bases, send messages, "
                "list and upload documents."
            ),
            # Selectively expose only these operation IDs as MCP tools
            include_operations=[
                "get_available_kbs_chat_knowledge_bases_get",
                "send_message_chat_message_post",
                "list_documents_documents__kb_id__get",
            ],
        )
        mcp.mount()
        return mcp
    except ImportError:
        import logging
        logging.getLogger(__name__).warning(
            "fastapi-mcp not installed; MCP server disabled. "
            "Install with: pip install fastapi-mcp"
        )
        return None
