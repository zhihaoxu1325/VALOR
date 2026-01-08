"""
数据结构定义模块
定义所有模块间共享的数据结构、枚举类型和工具函数
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Union
from enum import Enum
import numpy as np
import json
import time


# =============================================================================
# 枚举类型定义
# =============================================================================

class StrategyType(Enum):
    """优化策略类型枚举"""
    ORIGINAL = "original"
    WEIGHT_QUANTIZATION = "weight_quantization"
    ACTIVATION_QUANTIZATION = "activation_quantization"
    LOW_RANK = "low_rank"
    SPLIT_CONSTRUCTION = "split_construction"
    MIXED = "mixed"


class QuantizationType(Enum):
    """量化类型枚举"""
    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"


class QuantizationGranularity(Enum):
    """量化粒度枚举"""
    PER_TENSOR = "per_tensor"
    PER_CHANNEL = "per_channel"


class DecompositionMethod(Enum):
    """分解方法枚举"""
    SVD = "svd"
    NMF = "nmf"
    TUCKER = "tucker"


class OptimizationStatus(Enum):
    """优化状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# =============================================================================
# 核心数据结构
# =============================================================================

@dataclass
class LayerInfo:
    """层信息数据结构"""
    name: str
    onnx_node_name: str
    op_type: str
    weight_shape: Optional[Tuple[int, ...]]
    has_weights: bool
    mac_count: int
    original_latency_ms: float
    input_shape: Optional[Tuple[int, ...]] = None
    output_shape: Optional[Tuple[int, ...]] = None
    layer_index: int = 0
    
    def __post_init__(self):
        """数据验证"""
        if self.mac_count < 0:
            raise ValueError(f"MAC count cannot be negative: {self.mac_count}")
        if self.original_latency_ms < 0:
            raise ValueError(f"Latency cannot be negative: {self.original_latency_ms}")
    
    @property
    def weight_size(self) -> int:
        """权重参数数量"""
        if self.weight_shape:
            return np.prod(self.weight_shape)
        return 0
    
    @property
    def complexity_score(self) -> float:
        """层复杂度评分（用于排序和预算分配）"""
        return self.mac_count * self.original_latency_ms
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "onnx_node_name": self.onnx_node_name,
            "op_type": self.op_type,
            "weight_shape": self.weight_shape,
            "has_weights": self.has_weights,
            "mac_count": self.mac_count,
            "original_latency_ms": self.original_latency_ms,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "weight_size": self.weight_size,
            "complexity_score": self.complexity_score
        }


@dataclass
class OptimizationStrategy:
    """优化策略数据结构"""
    layer_name: str
    strategy_type: StrategyType
    parameters: Dict[str, Any]
    target: str  # "weight", "activation", "both", "none"
    expected_speedup: float = 1.0
    estimated_accuracy_loss: float = 0.0
    memory_reduction: float = 0.0
    strategy_id: str = field(default="")
    
    def __post_init__(self):
        """生成策略ID和验证"""
        if not self.strategy_id:
            self.strategy_id = self._generate_strategy_id()
        
        # 参数验证
        if self.expected_speedup <= 0:
            raise ValueError(f"Expected speedup must be positive: {self.expected_speedup}")
        if self.estimated_accuracy_loss < 0:
            raise ValueError(f"Accuracy loss cannot be negative: {self.estimated_accuracy_loss}")
    
    def _generate_strategy_id(self) -> str:
        """生成唯一的策略ID"""
        params_str = "_".join(f"{k}={v}" for k, v in sorted(self.parameters.items()))
        return f"{self.layer_name}_{self.strategy_type.value}_{self.target}_{hash(params_str) % 10000:04d}"
    
    def get_quantization_info(self) -> Optional[Dict[str, Any]]:
        """获取量化相关信息"""
        if self.strategy_type in [StrategyType.WEIGHT_QUANTIZATION, 
                                StrategyType.ACTIVATION_QUANTIZATION, 
                                StrategyType.MIXED]:
            return {
                "bits": self.parameters.get("bits", 32),
                "quantization_type": self.parameters.get("quantization_type", "symmetric"),
                "granularity": self.parameters.get("per_channel", False)
            }
        return None
    
    def get_low_rank_info(self) -> Optional[Dict[str, Any]]:
        """获取低秋分解相关信息"""
        if self.strategy_type in [StrategyType.LOW_RANK, StrategyType.SPLIT_CONSTRUCTION, StrategyType.MIXED]:
            return {
                "rank": self.parameters.get("rank", self.parameters.get("d_mid", None)),
                "decomposition_method": self.parameters.get("decomposition_method", "svd")
            }
        return None
    
    def calculate_compression_ratio(self, original_weight_shape: Tuple[int, ...]) -> float:
        """计算压缩比"""
        if not original_weight_shape:
            return 1.0
        
        original_size = np.prod(original_weight_shape)
        
        if self.strategy_type == StrategyType.WEIGHT_QUANTIZATION:
            bits = self.parameters.get("bits", 32)
            return 32.0 / bits
        
        elif self.strategy_type == StrategyType.LOW_RANK:
            rank = self.parameters.get("rank", min(original_weight_shape))
            if len(original_weight_shape) == 4:  # Conv layer
                out_c, in_c, h, w = original_weight_shape
                compressed_size = (in_c * h * w * rank) + (rank * out_c)
            elif len(original_weight_shape) == 2:  # FC layer
                m, k = original_weight_shape
                compressed_size = (k * rank) + (rank * m)
            else:
                return 1.0
            
            return original_size / compressed_size

        elif self.strategy_type == StrategyType.SPLIT_CONSTRUCTION:
            rank = self.parameters.get("d_mid", min(original_weight_shape))
            if len(original_weight_shape) == 2:  # FC layer
                m, k = original_weight_shape
                compressed_size = (k * rank) + (rank * m)
                return original_size / compressed_size
            return 1.0
        
        elif self.strategy_type == StrategyType.MIXED:
            # 综合考虑低秋和量化的压缩比
            low_rank_ratio = self.calculate_compression_ratio(original_weight_shape)
            quant_bits = self.parameters.get("quantization_bits", 32)
            quant_ratio = 32.0 / quant_bits
            return low_rank_ratio * quant_ratio
        
        return 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "strategy_id": self.strategy_id,
            "layer_name": self.layer_name,
            "strategy_type": self.strategy_type.value,
            "parameters": self.parameters,
            "target": self.target,
            "expected_speedup": self.expected_speedup,
            "estimated_accuracy_loss": self.estimated_accuracy_loss,
            "memory_reduction": self.memory_reduction,
            "quantization_info": self.get_quantization_info(),
            "low_rank_info": self.get_low_rank_info()
        }
    
    def __str__(self) -> str:
        return f"{self.strategy_type.value}({self.target}): {self.parameters}"


@dataclass
class SearchResult:
    """搜索结果数据结构"""
    strategies: List[OptimizationStrategy]
    accuracy_loss: float
    estimated_latency_improvement: float
    actual_latency_improvement: Optional[float]
    search_time_seconds: float
    total_evaluations: int
    convergence_info: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """验证搜索结果"""
        if self.accuracy_loss < 0:
            raise ValueError(f"Accuracy loss cannot be negative: {self.accuracy_loss}")
        if self.estimated_latency_improvement <= 0:
            raise ValueError(f"Latency improvement must be positive: {self.estimated_latency_improvement}")
        if self.total_evaluations < 0:
            raise ValueError(f"Total evaluations cannot be negative: {self.total_evaluations}")
    
    @property
    def total_expected_speedup(self) -> float:
        """计算总预期加速比"""
        total_speedup = 1.0
        for strategy in self.strategies:
            if strategy.strategy_type != StrategyType.ORIGINAL:
                total_speedup *= strategy.expected_speedup
        return total_speedup
    
    @property
    def optimization_efficiency(self) -> float:
        """优化效率：加速比/精度损失比值"""
        if self.accuracy_loss == 0:
            return float('inf') if self.estimated_latency_improvement > 1.0 else 1.0
        return self.estimated_latency_improvement / self.accuracy_loss
    
    def get_strategy_distribution(self) -> Dict[str, int]:
        """获取策略类型分布"""
        distribution = {}
        for strategy in self.strategies:
            strategy_type = strategy.strategy_type.value
            distribution[strategy_type] = distribution.get(strategy_type, 0) + 1
        return distribution
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "strategies": [strategy.to_dict() for strategy in self.strategies],
            "accuracy_loss": self.accuracy_loss,
            "estimated_latency_improvement": self.estimated_latency_improvement,
            "actual_latency_improvement": self.actual_latency_improvement,
            "search_time_seconds": self.search_time_seconds,
            "total_evaluations": self.total_evaluations,
            "total_expected_speedup": self.total_expected_speedup,
            "optimization_efficiency": self.optimization_efficiency,
            "strategy_distribution": self.get_strategy_distribution(),
            "convergence_info": self.convergence_info
        }


@dataclass
class OptimizationResult:
    """最终优化结果数据结构"""
    original_model_path: str
    optimized_model_path: str
    optimization_config: Dict[str, Any]
    search_result: SearchResult
    final_performance: Dict[str, Any]
    timing_breakdown: Dict[str, float]
    warnings: List[str] = field(default_factory=list)
    optimization_id: str = field(default="")
    
    def __post_init__(self):
        """生成优化ID"""
        if not self.optimization_id:
            timestamp = int(time.time())
            self.optimization_id = f"opt_{timestamp}_{hash(self.original_model_path) % 10000:04d}"
    
    @property
    def success(self) -> bool:
        """优化是否成功"""
        return len(self.warnings) == 0 and self.search_result.accuracy_loss >= 0
    
    @property
    def total_time(self) -> float:
        """总优化时间"""
        return sum(self.timing_breakdown.values())
    
    def get_summary(self) -> Dict[str, Any]:
        """获取优化摘要"""
        return {
            "optimization_id": self.optimization_id,
            "success": self.success,
            "accuracy_loss": self.search_result.accuracy_loss,
            "latency_improvement": self.search_result.estimated_latency_improvement,
            "actual_latency_improvement": self.search_result.actual_latency_improvement,
            "model_size_reduction": self.final_performance.get("model_size_reduction_ratio", 0),
            "total_time_seconds": self.total_time,
            "total_evaluations": self.search_result.total_evaluations,
            "optimization_efficiency": self.search_result.optimization_efficiency,
            "warnings_count": len(self.warnings)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为完整字典"""
        return {
            "optimization_id": self.optimization_id,
            "original_model_path": self.original_model_path,
            "optimized_model_path": self.optimized_model_path,
            "optimization_config": self.optimization_config,
            "search_result": self.search_result.to_dict(),
            "final_performance": self.final_performance,
            "timing_breakdown": self.timing_breakdown,
            "warnings": self.warnings,
            "summary": self.get_summary(),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def save_to_json(self, file_path: str):
        """保存到JSON文件"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


# =============================================================================
# 工具数据结构
# =============================================================================

@dataclass
class BudgetAllocation:
    """预算分配数据结构"""
    layer_name: str
    allocated_budget: float
    used_budget: float
    remaining_budget: float
    is_locked: bool = False
    
    @property
    def budget_utilization(self) -> float:
        """预算利用率"""
        if self.allocated_budget == 0:
            return 0.0
        return self.used_budget / self.allocated_budget
    
    def can_afford(self, cost: float) -> bool:
        """是否可以承担指定成本"""
        return self.remaining_budget >= cost
    
    def spend_budget(self, amount: float) -> bool:
        """花费预算"""
        if self.can_afford(amount):
            self.used_budget += amount
            self.remaining_budget -= amount
            return True
        return False


@dataclass
class PerformanceMetrics:
    """性能指标数据结构"""
    accuracy_loss: float
    latency_improvement: float
    memory_reduction: float
    model_size_reduction: float
    throughput_improvement: float = 1.0
    energy_reduction: float = 0.0
    
    def __post_init__(self):
        """验证指标"""
        if self.accuracy_loss < 0:
            raise ValueError("Accuracy loss cannot be negative")
        if self.latency_improvement <= 0:
            raise ValueError("Latency improvement must be positive")
    
    @property
    def overall_score(self) -> float:
        """综合评分"""
        # 简单的加权评分公式
        return (self.latency_improvement * 0.4 + 
                (1 + self.memory_reduction) * 0.3 + 
                (1 + self.model_size_reduction) * 0.2 + 
                (1 - self.accuracy_loss) * 0.1)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "accuracy_loss": self.accuracy_loss,
            "latency_improvement": self.latency_improvement,
            "memory_reduction": self.memory_reduction,
            "model_size_reduction": self.model_size_reduction,
            "throughput_improvement": self.throughput_improvement,
            "energy_reduction": self.energy_reduction,
            "overall_score": self.overall_score
        }


@dataclass
class ValidationResult:
    """验证结果数据结构"""
    is_valid: bool
    error_message: str = ""
    warnings: List[str] = field(default_factory=list)
    validation_details: Dict[str, Any] = field(default_factory=dict)
    
    def add_warning(self, warning: str):
        """添加警告"""
        self.warnings.append(warning)
    
    def set_error(self, error: str):
        """设置错误"""
        self.is_valid = False
        self.error_message = error


# =============================================================================
# 工具函数
# =============================================================================

def create_original_strategy(layer_name: str) -> OptimizationStrategy:
    """创建原始策略（不优化）"""
    return OptimizationStrategy(
        layer_name=layer_name,
        strategy_type=StrategyType.ORIGINAL,
        parameters={},
        target="none",
        expected_speedup=1.0,
        estimated_accuracy_loss=0.0
    )


def create_weight_quantization_strategy(layer_name: str, bits: int, 
                                      quantization_type: str = "symmetric",
                                      per_channel: bool = False) -> OptimizationStrategy:
    """创建权重量化策略"""
    return OptimizationStrategy(
        layer_name=layer_name,
        strategy_type=StrategyType.WEIGHT_QUANTIZATION,
        parameters={
            "bits": bits,
            "quantization_type": quantization_type,
            "per_channel": per_channel
        },
        target="weight",
        expected_speedup=32.0 / bits,  # 简单估算
        estimated_accuracy_loss=0.01 * (8 - bits) if bits < 8 else 0.005
    )


def create_low_rank_strategy(layer_name: str, rank: int, 
                           decomposition_method: str = "svd") -> OptimizationStrategy:
    """创建低秩分解策略"""
    return OptimizationStrategy(
        layer_name=layer_name,
        strategy_type=StrategyType.LOW_RANK,
        parameters={
            "rank": rank,
            "decomposition_method": decomposition_method
        },
        target="weight",
        expected_speedup=2.0,  # 需要根据实际情况计算
        estimated_accuracy_loss=0.005 + (1.0 / rank) * 0.01
    )


def create_mixed_strategy(layer_name: str, rank: int, quantization_bits: int) -> OptimizationStrategy:
    """创建混合策略（低秋+量化）"""
    return OptimizationStrategy(
        layer_name=layer_name,
        strategy_type=StrategyType.MIXED,
        parameters={
            "rank": rank,
            "quantization_bits": quantization_bits,
            "quantization_type": "symmetric",
            "per_channel": False
        },
        target="weight",
        expected_speedup=2.0 * (32.0 / quantization_bits),
        estimated_accuracy_loss=0.01 + (1.0 / rank) * 0.005
    )


def validate_strategy_compatibility(strategy: OptimizationStrategy, 
                                  layer_info: LayerInfo) -> ValidationResult:
    """验证策略与层的兼容性"""
    result = ValidationResult(is_valid=True)
    
    # 检查是否有权重
    if (strategy.strategy_type != StrategyType.ORIGINAL and 
        strategy.target in ["weight", "both"] and 
        not layer_info.has_weights):
        result.set_error(f"Strategy {strategy.strategy_type.value} requires weights but layer {layer_info.name} has none")
        return result
    
    # 检查量化参数
    if strategy.strategy_type in [StrategyType.WEIGHT_QUANTIZATION, StrategyType.ACTIVATION_QUANTIZATION]:
        bits = strategy.parameters.get("bits", 32)
        if bits not in [4, 8, 16, 32]:
            result.add_warning(f"Unusual quantization bits: {bits}")
    
    # 检查低秋参数
    if strategy.strategy_type in [StrategyType.LOW_RANK, StrategyType.MIXED]:
        rank = strategy.parameters.get("rank")
        if rank and layer_info.weight_shape:
            min_dim = min(layer_info.weight_shape)
            if rank >= min_dim:
                result.set_error(f"Rank {rank} too large for weight shape {layer_info.weight_shape}")
                return result
    
    return result


def calculate_total_performance_impact(strategies: List[OptimizationStrategy], 
                                     layer_infos: List[LayerInfo]) -> PerformanceMetrics:
    """计算策略组合的总体性能影响"""
    total_latency_improvement = 1.0
    total_accuracy_loss = 0.0
    total_memory_reduction = 0.0
    total_size_reduction = 0.0
    
    # 创建层信息映射
    layer_map = {layer.name: layer for layer in layer_infos}
    
    for strategy in strategies:
        if strategy.strategy_type == StrategyType.ORIGINAL:
            continue
        
        # 累积性能影响
        total_latency_improvement *= strategy.expected_speedup
        total_accuracy_loss += strategy.estimated_accuracy_loss
        
        # 计算内存和模型大小影响
        if strategy.layer_name in layer_map:
            layer = layer_map[strategy.layer_name]
            if layer.weight_shape:
                compression_ratio = strategy.calculate_compression_ratio(layer.weight_shape)
                weight_contribution = layer.weight_size / sum(l.weight_size for l in layer_infos)
                
                total_memory_reduction += (1 - 1/compression_ratio) * weight_contribution
                total_size_reduction += (1 - 1/compression_ratio) * weight_contribution
    
    return PerformanceMetrics(
        accuracy_loss=total_accuracy_loss,
        latency_improvement=total_latency_improvement,
        memory_reduction=total_memory_reduction,
        model_size_reduction=total_size_reduction
    )


# =============================================================================
# 常量定义
# =============================================================================

class OptimizationConstants:
    """优化相关常量"""
    
    # 默认配置
    DEFAULT_RVV_LENGTH = 128
    DEFAULT_ACCURACY_THRESHOLD = 0.01
    DEFAULT_CALIBRATION_SAMPLES = 32
    
    # 量化配置
    SUPPORTED_QUANTIZATION_BITS = [4, 8]
    DEFAULT_QUANTIZATION_BITS = 8
    
    # 低秋配置
    DEFAULT_RANK_CANDIDATES = [32, 64, 128]
    MIN_RANK = 16
    MAX_RANK = 256
    
    # 搜索配置
    DEFAULT_BOHB_TRIALS = 300
    DEFAULT_EARLY_STOP_MULTIPLIER = 1.2
    SAMPLE_PROGRESSION = [2, 4, 8, 16, 32]
    
    # 性能配置
    LATENCY_MEASUREMENT_WARMUP = 3
    LATENCY_MEASUREMENT_RUNS = 20
    
    # 文件扩展名
    ONNX_EXTENSION = ".onnx"
    REPORT_EXTENSION = ".json"
    
    # 支持的层类型
    SUPPORTED_CONV_TYPES = ["Conv", "ConvTranspose"]
    SUPPORTED_LINEAR_TYPES = ["MatMul", "Gemm"]
    SUPPORTED_ACTIVATION_TYPES = ["Relu", "Sigmoid", "Tanh"]


if __name__ == "__main__":
    # 测试数据结构
    print("Testing data structures...")
    
    # 测试LayerInfo
    layer = LayerInfo(
        name="test_conv",
        onnx_node_name="Conv_1",
        op_type="Conv",
        weight_shape=(64, 32, 3, 3),
        has_weights=True,
        mac_count=1000000,
        original_latency_ms=50.0
    )
    print(f"Layer info: {layer.to_dict()}")
    
    # 测试OptimizationStrategy
    strategy = create_weight_quantization_strategy("test_conv", 8, per_channel=True)
    print(f"Strategy: {strategy.to_dict()}")
    
    # 测试兼容性验证
    validation = validate_strategy_compatibility(strategy, layer)
    print(f"Validation: valid={validation.is_valid}, warnings={validation.warnings}")
    
    # 测试性能计算
    strategies = [strategy]
    layers = [layer]
    metrics = calculate_total_performance_impact(strategies, layers)
    print(f"Performance metrics: {metrics.to_dict()}")
    
    print("All data structure tests passed!")
