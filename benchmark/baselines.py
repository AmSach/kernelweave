"""Baseline routing implementations for comparison.

DSPy: Signature-based routing with examples
LangGraph: State machine routing with nodes
KernelWeave: Postcondition-based routing
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Protocol
import random


class RoutingDecision(Protocol):
    """Protocol for routing decisions."""
    mode: str
    confidence: float
    reason: str


@dataclass
class DSPyRouting:
    """Simulates DSPy-style signature routing.
    
    DSPy uses:
    - Signature definitions (input/output specs)
    - Few-shot examples
    - LM-based routing decisions
    
    This is a simulation that captures the routing behavior without
    requiring actual DSPy installation.
    """
    
    def route(self, instruction: str, signatures: list[dict]) -> dict[str, Any]:
        """Route based on signature matching.
        
        Args:
            instruction: The user instruction
            signatures: List of available signatures (like kernels)
        
        Returns:
            Routing decision with signature match or fallback
        """
        # DSPy routes by finding signature that best matches instruction
        # This is typically done by embedding similarity or LM prompting
        
        best_match = None
        best_score = 0.0
        
        for sig in signatures:
            # Simulate embedding-based matching
            score = self._signature_similarity(instruction, sig)
            if score > best_score:
                best_score = score
                best_match = sig
        
        # DSPy threshold is typically around 0.5-0.6 for routing
        if best_score > 0.55:
            return {
                "mode": "signature",
                "signature_id": best_match.get("id"),
                "confidence": best_score,
                "reason": f"matched signature: {best_match.get('name', 'unknown')}",
            }
        
        return {
            "mode": "generate",
            "signature_id": None,
            "confidence": best_score,
            "reason": "no signature match above threshold",
        }
    
    def _signature_similarity(self, instruction: str, signature: dict) -> float:
        """Compute similarity between instruction and signature.
        
        DSPy uses embedding similarity or LM-based matching.
        We simulate this with keyword matching and task family detection.
        """
        inst_lower = instruction.lower()
        sig_name = signature.get("name", "").lower()
        sig_desc = signature.get("description", "").lower()
        
        # Keyword overlap
        inst_words = set(inst_lower.split())
        sig_words = set((sig_name + " " + sig_desc).split())
        overlap = len(inst_words & sig_words) / max(1, len(inst_words))
        
        # Task family boost
        family_boost = 0.0
        if "compare" in inst_lower and "compare" in sig_name:
            family_boost = 0.3
        elif "analyze" in inst_lower and "analyze" in sig_name:
            family_boost = 0.3
        elif "generate" in inst_lower and "generate" in sig_name:
            family_boost = 0.3
        elif "search" in inst_lower and "search" in sig_name:
            family_boost = 0.3
        
        # Add noise to simulate LM variability
        noise = random.uniform(-0.1, 0.1)
        
        return min(1.0, overlap + family_boost + noise)


@dataclass
class LangGraphRouting:
    """Simulates LangGraph-style state machine routing.
    
    LangGraph uses:
    - State graph with nodes
    - Conditional edges
    - Tools and actions as nodes
    
    This is a simulation that captures the routing behavior.
    """
    
    def route(self, instruction: str, nodes: list[dict]) -> dict[str, Any]:
        """Route based on graph traversal.
        
        Args:
            instruction: The user instruction
            nodes: List of available nodes (like kernels)
        
        Returns:
            Routing decision with node selection
        """
        # LangGraph typically starts with an intent classifier node
        # Then routes to specialized nodes based on intent
        
        intent = self._classify_intent(instruction)
        
        # Find matching node
        best_node = None
        best_score = 0.0
        
        for node in nodes:
            score = self._node_relevance(intent, node)
            if score > best_score:
                best_score = score
                best_node = node
        
        # LangGraph threshold varies by graph design
        if best_score > 0.50:
            return {
                "mode": "node",
                "node_id": best_node.get("id"),
                "confidence": best_score,
                "reason": f"routed to node: {best_node.get('name', 'unknown')}",
            }
        
        return {
            "mode": "generate",
            "node_id": None,
            "confidence": best_score,
            "reason": "no matching node in graph",
        }
    
    def _classify_intent(self, instruction: str) -> str:
        """Classify the intent of the instruction.
        
        LangGraph typically uses a classifier node for this.
        """
        inst_lower = instruction.lower()
        
        if "compare" in inst_lower or "diff" in inst_lower:
            return "comparison"
        elif "analyze" in inst_lower or "review" in inst_lower:
            return "analysis"
        elif "generate" in inst_lower or "create" in inst_lower or "write" in inst_lower:
            return "generation"
        elif "search" in inst_lower or "find" in inst_lower:
            return "search"
        elif "convert" in inst_lower or "transform" in inst_lower:
            return "transformation"
        elif "debug" in inst_lower or "fix" in inst_lower or "error" in inst_lower:
            return "debugging"
        elif "test" in inst_lower:
            return "testing"
        elif "document" in inst_lower or "readme" in inst_lower:
            return "documentation"
        
        return "general"
    
    def _node_relevance(self, intent: str, node: dict) -> float:
        """Compute relevance between intent and node.
        
        LangGraph uses predefined edges, but we simulate relevance.
        """
        node_name = node.get("name", "").lower()
        node_intent = node.get("intent", "general")
        
        if intent == node_intent:
            return 0.7 + random.uniform(0, 0.2)
        
        # Partial match
        if intent in node_name or node_intent in intent:
            return 0.5 + random.uniform(0, 0.15)
        
        return random.uniform(0.2, 0.4)


@dataclass
class KernelWeaveRouting:
    """Actual KernelWeave routing implementation.
    
    Uses postcondition verification as routing signal.
    """
    
    def route(self, instruction: str, kernels: list[Any]) -> dict[str, Any]:
        """Route based on postcondition matching.
        
        Args:
            instruction: The user instruction
            kernels: List of available kernels
        
        Returns:
            Routing decision with kernel match or fallback
        """
        from kernelweave.runtime import KernelRuntime
        from kernelweave.kernel import KernelStore
        
        # Create temporary store with kernels
        # (In real use, this would use the existing store)
        # For simulation, we just use semantic matching
        
        best_kernel = None
        best_score = 0.0
        
        for kernel in kernels:
            score = self._kernel_score(instruction, kernel)
            if score > best_score:
                best_score = score
                best_kernel = kernel
        
        # KernelWeave threshold is 0.50
        if best_score > 0.50:
            # Verify preconditions
            if self._check_preconditions(instruction, best_kernel):
                return {
                    "mode": "kernel",
                    "kernel_id": best_kernel.kernel_id,
                    "confidence": best_score,
                    "reason": f"matched kernel: {best_kernel.name}",
                    "verification": "passed",
                }
            else:
                return {
                    "mode": "generate",
                    "kernel_id": None,
                    "confidence": best_score * 0.5,  # Reduced due to precondition fail
                    "reason": "preconditions not met",
                    "verification": "failed",
                }
        
        return {
            "mode": "generate",
            "kernel_id": None,
            "confidence": best_score,
            "reason": "no kernel match above threshold",
        }
    
    def _kernel_score(self, instruction: str, kernel) -> float:
        """Compute kernel score using KernelWeave's scoring."""
        from kernelweave.metrics import cosine_similarity, jaccard_similarity, coverage
        
        kernel_text = f"{kernel.task_family} {kernel.description}"
        
        # Semantic similarity
        semantic = cosine_similarity(instruction, kernel_text)
        jaccard = jaccard_similarity(instruction, kernel.description)
        
        # Evidence bonus
        evidence_bonus = coverage(kernel.evidence_requirements, instruction)
        
        # Confidence boost for verified kernels
        confidence_boost = kernel.status.confidence * 0.3
        
        # Precondition check (new addition)
        precondition_penalty = 0.0
        if self._check_preconditions(instruction, kernel):
            precondition_penalty = 0.0
        else:
            precondition_penalty = 0.3
        
        score = (0.5 * semantic + 0.3 * jaccard + 0.2 * evidence_bonus + 
                 confidence_boost - precondition_penalty)
        
        return min(1.0, max(0.0, score))
    
    def _check_preconditions(self, instruction: str, kernel) -> bool:
        """Check if instruction satisfies kernel preconditions."""
        inst_lower = instruction.lower()
        
        # Check for artifact/file indicators for comparison kernels
        if "artifact comparison" in kernel.task_family:
            # Must have file references, names, or paths
            has_files = any(kw in inst_lower for kw in 
                          ["file", ".py", ".js", ".json", ".yaml", ".xml", 
                           "artifact", "schema", "document", "v1", "v2", "version"])
            if not has_files:
                return False
        
        # Check for code indicators for code analysis kernels
        if "code analysis" in kernel.task_family:
            has_code = any(kw in inst_lower for kw in 
                          ["code", "function", "class", "module", "complexity", "quality"])
            if not has_code:
                return False
        
        # Other precondition checks would go here
        return True


def create_baselines():
    """Create baseline routing implementations."""
    return {
        "dspy": DSPyRouting(),
        "langgraph": LangGraphRouting(),
        "kernelweave": KernelWeaveRouting(),
    }
