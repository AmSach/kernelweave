"""Kernel library - 20+ kernels across diverse task families.

Each kernel is handcrafted from real execution patterns with:
- Specific preconditions (what must be true)
- Postconditions (what the output must satisfy)
- Evidence requirements (what must be captured)
- Realistic steps (actual execution path)
"""
from .analysis import *
from .comparison import *
from .generation import *
from .search import *
from .transformation import *
from .debugging import *
from .testing import *
from .documentation import *
from .library import ALL_KERNELS, install_kernel_library, kernel_for_task_family, kernel_summary

__all__ = [
    # Analysis family
    "CODE_ANALYSIS_KERNEL",
    "SECURITY_AUDIT_KERNEL",
    "PERFORMANCE_PROFILING_KERNEL",
    
    # Comparison family
    "ARTIFACT_COMPARISON_KERNEL",
    "DIFF_ANALYSIS_KERNEL",
    "VERSION_COMPARISON_KERNEL",
    
    # Generation family
    "CODE_GENERATION_KERNEL",
    "TEST_GENERATION_KERNEL",
    "DOCUMENTATION_GENERATION_KERNEL",
    "CONFIG_GENERATION_KERNEL",
    
    # Search family
    "CODE_SEARCH_KERNEL",
    "PATTERN_SEARCH_KERNEL",
    "DEPENDENCY_SEARCH_KERNEL",
    
    # Transformation family
    "FORMAT_CONVERSION_KERNEL",
    "REFACTORING_KERNEL",
    "MIGRATION_KERNEL",
    
    # Debugging family
    "ERROR_DIAGNOSIS_KERNEL",
    "LOG_ANALYSIS_KERNEL",
    
    # Testing family
    "TEST_DEBUGGING_KERNEL",
    "COVERAGE_ANALYSIS_KERNEL",
    
    # Documentation family
    "API_DOCS_KERNEL",
    "README_GENERATION_KERNEL",
    
    # All kernels
    "ALL_KERNELS",
    "install_kernel_library",
]
