# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc

"""Plagiarism detection package - tools for detecting and managing plagiarism.

Note: Imports are lazy to avoid circular import issues.
Use explicit imports from submodules:
    from vali_objects.plagiarism.plagiarism_manager import PlagiarismManager
    from vali_objects.plagiarism.plagiarism_server import PlagiarismServer, PlagiarismClient
"""

def __getattr__(name):
    """Lazy import to avoid circular dependencies."""
    if name == 'PlagiarismManager':
        from vali_objects.plagiarism.plagiarism_manager import PlagiarismManager
        return PlagiarismManager
    elif name == 'PlagiarismServer':
        from vali_objects.plagiarism.plagiarism_server import PlagiarismServer
        return PlagiarismServer
    elif name == 'PlagiarismClient':
        from vali_objects.plagiarism.plagiarism_client import PlagiarismClient
        return PlagiarismClient
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'PlagiarismManager',
    'PlagiarismServer',
    'PlagiarismClient',
]
