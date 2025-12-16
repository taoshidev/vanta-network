# developer: jbonilla
# Copyright © 2024 Taoshi Inc

"""Plagiarism detection package - tools for detecting and managing plagiarism.

Note: Imports are lazy to avoid circular import issues.
Use explicit imports from submodules:
    from vali_objects.plagiarism.plagiarism_events import PlagiarismEvents
    from vali_objects.plagiarism.plagiarism_detector import PlagiarismDetector
    from vali_objects.plagiarism.plagiarism_detector_server import PlagiarismDetectorServer, PlagiarismDetectorClient
    from vali_objects.plagiarism.plagiarism_pipeline import PlagiarismPipeline
    from vali_objects.plagiarism.plagiarism_definitions import FollowPercentage, LagDetection, CopySimilarity, TwoCopySimilarity, ThreeCopySimilarity
    from vali_objects.plagiarism.plagiarism_manager import PlagiarismManager
    from vali_objects.plagiarism.plagiarism_server import PlagiarismServer, PlagiarismClient
"""

def __getattr__(name):
    """Lazy import to avoid circular dependencies."""
    if name == 'PlagiarismEvents':
        from vali_objects.plagiarism.plagiarism_events import PlagiarismEvents
        return PlagiarismEvents
    elif name == 'PlagiarismDetector':
        from vali_objects.plagiarism.plagiarism_detector import PlagiarismDetector
        return PlagiarismDetector
    elif name == 'PlagiarismDetectorServer':
        from vali_objects.plagiarism.plagiarism_detector_server import PlagiarismDetectorServer
        return PlagiarismDetectorServer
    elif name == 'PlagiarismDetectorClient':
        from vali_objects.plagiarism.plagiarism_detector_server import PlagiarismDetectorClient
        return PlagiarismDetectorClient
    elif name == 'PlagiarismPipeline':
        from vali_objects.plagiarism.plagiarism_pipeline import PlagiarismPipeline
        return PlagiarismPipeline
    elif name == 'FollowPercentage':
        from vali_objects.plagiarism.plagiarism_definitions import FollowPercentage
        return FollowPercentage
    elif name == 'LagDetection':
        from vali_objects.plagiarism.plagiarism_definitions import LagDetection
        return LagDetection
    elif name == 'CopySimilarity':
        from vali_objects.plagiarism.plagiarism_definitions import CopySimilarity
        return CopySimilarity
    elif name == 'TwoCopySimilarity':
        from vali_objects.plagiarism.plagiarism_definitions import TwoCopySimilarity
        return TwoCopySimilarity
    elif name == 'ThreeCopySimilarity':
        from vali_objects.plagiarism.plagiarism_definitions import ThreeCopySimilarity
        return ThreeCopySimilarity
    elif name == 'PlagiarismManager':
        from vali_objects.plagiarism.plagiarism_manager import PlagiarismManager
        return PlagiarismManager
    elif name == 'PlagiarismServer':
        from vali_objects.plagiarism.plagiarism_server import PlagiarismServer
        return PlagiarismServer
    elif name == 'PlagiarismClient':
        from vali_objects.plagiarism.plagiarism_server import PlagiarismClient
        return PlagiarismClient
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'PlagiarismEvents',
    'PlagiarismDetector',
    'PlagiarismDetectorServer',
    'PlagiarismDetectorClient',
    'PlagiarismPipeline',
    'FollowPercentage',
    'LagDetection',
    'CopySimilarity',
    'TwoCopySimilarity',
    'ThreeCopySimilarity',
    'PlagiarismManager',
    'PlagiarismServer',
    'PlagiarismClient',
]
