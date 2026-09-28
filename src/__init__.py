"""
AI Security Gateway package.

A local LLM-based system for preventing semantic data leaks to external AI services.
"""

from .gateway import AISecurityGateway

__version__ = "1.0.0"
__all__ = ["AISecurityGateway"]
