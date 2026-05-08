"""Kernel library aggregator."""
from .analysis import CODE_ANALYSIS_KERNEL, SECURITY_AUDIT_KERNEL, PERFORMANCE_PROFILING_KERNEL
from .comparison import ARTIFACT_COMPARISON_KERNEL, DIFF_ANALYSIS_KERNEL, VERSION_COMPARISON_KERNEL
from .generation import CODE_GENERATION_KERNEL, TEST_GENERATION_KERNEL, DOCUMENTATION_GENERATION_KERNEL, CONFIG_GENERATION_KERNEL
from .search import CODE_SEARCH_KERNEL, PATTERN_SEARCH_KERNEL, DEPENDENCY_SEARCH_KERNEL
from .transformation import FORMAT_CONVERSION_KERNEL, REFACTORING_KERNEL, MIGRATION_KERNEL
from .debugging import ERROR_DIAGNOSIS_KERNEL, LOG_ANALYSIS_KERNEL
from .testing import TEST_DEBUGGING_KERNEL, COVERAGE_ANALYSIS_KERNEL
from .documentation import API_DOCS_KERNEL, README_GENERATION_KERNEL

ALL_KERNELS = [
    # Analysis (3)
    CODE_ANALYSIS_KERNEL,
    SECURITY_AUDIT_KERNEL,
    PERFORMANCE_PROFILING_KERNEL,
    
    # Comparison (3)
    ARTIFACT_COMPARISON_KERNEL,
    DIFF_ANALYSIS_KERNEL,
    VERSION_COMPARISON_KERNEL,
    
    # Generation (4)
    CODE_GENERATION_KERNEL,
    TEST_GENERATION_KERNEL,
    DOCUMENTATION_GENERATION_KERNEL,
    CONFIG_GENERATION_KERNEL,
    
    # Search (3)
    CODE_SEARCH_KERNEL,
    PATTERN_SEARCH_KERNEL,
    DEPENDENCY_SEARCH_KERNEL,
    
    # Transformation (3)
    FORMAT_CONVERSION_KERNEL,
    REFACTORING_KERNEL,
    MIGRATION_KERNEL,
    
    # Debugging (2)
    ERROR_DIAGNOSIS_KERNEL,
    LOG_ANALYSIS_KERNEL,
    
    # Testing (2)
    TEST_DEBUGGING_KERNEL,
    COVERAGE_ANALYSIS_KERNEL,
    
    # Documentation (2)
    API_DOCS_KERNEL,
    README_GENERATION_KERNEL,
]

def install_kernel_library(store) -> int:
    """Install all kernels to a store.
    
    Returns:
        Number of kernels installed
    """
    count = 0
    for kernel in ALL_KERNELS:
        store.add_kernel(kernel)
        count += 1
    return count

def kernel_for_task_family(task_family: str) -> list:
    """Find kernels matching a task family."""
    return [k for k in ALL_KERNELS if k.task_family == task_family or task_family in k.task_family]

def kernel_summary() -> dict:
    """Summary of the kernel library."""
    families = {}
    for k in ALL_KERNELS:
        families[k.task_family] = families.get(k.task_family, 0) + 1
    
    return {
        "total_kernels": len(ALL_KERNELS),
        "families": families,
        "avg_confidence": sum(k.status.confidence for k in ALL_KERNELS) / len(ALL_KERNELS),
        "verified_count": sum(1 for k in ALL_KERNELS if k.status.state == "verified"),
    }
