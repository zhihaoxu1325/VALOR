"""
策略生成模块
基于RVV长度和层信息生成优化策略候选，包括量化策略和低秩分解策略
"""

import math
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
from dataclasses import dataclass

from model_config import ModelConfig
from onnx_info_extractor import LayerInfo


class StrategyType(Enum):
    """策略类型枚举"""
    ORIGINAL = "original"
    WEIGHT_QUANTIZATION = "weight_quantization"
    ACTIVATION_QUANTIZATION = "activation_quantization"
    LOW_RANK = "low_rank"
    SPLIT_CONSTRUCTION = "split_construction"
    MIXED = "mixed"


@dataclass
class OptimizationStrategy:
    """优化策略数据结构"""
    layer_name: str
    strategy_type: StrategyType
    parameters: Dict[str, Any]
    target: str  # "weight", "activation", "both"
    expected_speedup: float = 1.0
    
    def __str__(self):
        return f"{self.strategy_type.value}({self.target}): {self.parameters}"


class RVVAwareStrategyGenerator:
    """基于RVV的策略生成器"""
    
    def __init__(self, rvv_length: int = 128):
        self.rvv_length = rvv_length
        self.w = rvv_length // 32  # FP32情况下的向量元素数
        
        # 量化策略配置
        self.weight_quantization_bits = [8, 4]
        self.activation_quantization_bits = [8]
        
        # 低秩策略配置
        self.rank_divisors = [4, 8]  # K/4, K/8
        self.min_rank = 32
        self.max_rank = 128
        self.rank_candidates = [32, 64, 128]
    
    def generate_strategies(self, layer_info: LayerInfo) -> List[OptimizationStrategy]:
        """为指定层生成所有可用的优化策略"""
        strategies = []
        
        # 1. 原始策略（baseline）
        strategies.append(OptimizationStrategy(
            layer_name=layer_info.name,
            strategy_type=StrategyType.ORIGINAL,
            parameters={},
            target="none"
        ))
        
        # 2. 如果没有权重，直接返回原始策略
        if not layer_info.has_weights or layer_info.weight_shape is None:
            return strategies
        
        # 3. 生成量化策略
        strategies.extend(self._generate_quantization_strategies(layer_info))

        # 4. 生成split construction策略（clogging节点）
        strategies.extend(self._generate_split_strategies(layer_info))

        # 5. 生成低秩分解策略（仅适用于某些层类型）
        if self._supports_low_rank(layer_info):
            strategies.extend(self._generate_low_rank_strategies(layer_info))

        # 6. 生成混合策略（低秩+量化）
        if self._supports_low_rank(layer_info):
            strategies.extend(self._generate_mixed_strategies(layer_info))
        
        return strategies
    
    def _generate_quantization_strategies(self, layer_info: LayerInfo) -> List[OptimizationStrategy]:
        """生成量化策略"""
        strategies = []
        
        # 权重量化策略
        for bits in self.weight_quantization_bits:
            strategies.append(OptimizationStrategy(
                layer_name=layer_info.name,
                strategy_type=StrategyType.WEIGHT_QUANTIZATION,
                parameters={
                    "bits": bits,
                    "quantization_type": "symmetric" if bits == 8 else "symmetric",
                    "per_channel": True if bits == 8 else False
                },
                target="weight",
                expected_speedup=self._estimate_quantization_speedup(bits)
            ))
        
        # 激活量化策略（仅对支持的层类型）
        if layer_info.op_type in ["Conv", "MatMul", "Gemm"]:
            for bits in self.activation_quantization_bits:
                strategies.append(OptimizationStrategy(
                    layer_name=layer_info.name,
                    strategy_type=StrategyType.ACTIVATION_QUANTIZATION,
                    parameters={
                        "bits": bits,
                        "quantization_type": "asymmetric",
                        "per_tensor": True
                    },
                    target="activation",
                    expected_speedup=self._estimate_quantization_speedup(bits, is_activation=True)
                ))
        
        return strategies

    def _generate_split_strategies(self, layer_info: LayerInfo) -> List[OptimizationStrategy]:
        """生成split construction策略"""
        if layer_info.op_type not in ["MatMul", "Gemm"]:
            return []
        if not layer_info.weight_shape or len(layer_info.weight_shape) < 2:
            return []

        K, M = self._get_matrix_dimensions(layer_info.weight_shape, layer_info.op_type)
        eta = self.w

        if eta <= 0 or K % eta == 0:
            return []

        max_d_mid = self._calculate_d_mid_upper_bound(K, M, eta)
        max_d_mid = min(max_d_mid, K, M)
        if max_d_mid < eta:
            return []

        candidates = []
        for d_mid in range(eta, max_d_mid + 1, eta):
            if self._calculate_clogging_level(K, M, d_mid, eta) < 0:
                candidates.append(d_mid)

        strategies = []
        for d_mid in candidates:
            strategies.append(OptimizationStrategy(
                layer_name=layer_info.name,
                strategy_type=StrategyType.SPLIT_CONSTRUCTION,
                parameters={
                    "d_mid": d_mid,
                    "eta": eta
                },
                target="weight",
                expected_speedup=self._estimate_split_speedup(K, M, d_mid, eta)
            ))

        return strategies
    
    def _generate_low_rank_strategies(self, layer_info: LayerInfo) -> List[OptimizationStrategy]:
        """生成低秋分解策略"""
        strategies = []
        weight_shape = layer_info.weight_shape
        
        # 计算合适的rank值
        valid_ranks = self._calculate_valid_ranks(weight_shape, layer_info.op_type)
        
        for rank in valid_ranks:
            strategies.append(OptimizationStrategy(
                layer_name=layer_info.name,
                strategy_type=StrategyType.LOW_RANK,
                parameters={
                    "rank": rank,
                    "decomposition_method": "svd"
                },
                target="weight",
                expected_speedup=self._estimate_low_rank_speedup(weight_shape, rank)
            ))
        
        return strategies
    
    def _generate_mixed_strategies(self, layer_info: LayerInfo) -> List[OptimizationStrategy]:
        """生成混合策略（低秋+量化）"""
        strategies = []
        weight_shape = layer_info.weight_shape
        
        # 获取有效的rank值
        valid_ranks = self._calculate_valid_ranks(weight_shape, layer_info.op_type)
        
        # 混合策略：低秋分解 + 权重量化
        for rank in valid_ranks:
            for bits in self.weight_quantization_bits:
                strategies.append(OptimizationStrategy(
                    layer_name=layer_info.name,
                    strategy_type=StrategyType.MIXED,
                    parameters={
                        "rank": rank,
                        "quantization_bits": bits,
                        "quantization_type": "symmetric",
                        "per_channel": False  # 混合策略中使用per-tensor量化
                    },
                    target="weight",
                    expected_speedup=self._estimate_mixed_speedup(weight_shape, rank, bits)
                ))
        
        return strategies
    
    def _supports_low_rank(self, layer_info: LayerInfo) -> bool:
        """判断层是否支持低秋分解"""
        # 支持低秋分解的层类型
        supported_ops = ["Conv", "MatMul", "Gemm"]
        if layer_info.op_type not in supported_ops:
            return False
        
        # 检查权重shape是否适合低秋分解
        weight_shape = layer_info.weight_shape
        if not weight_shape or len(weight_shape) < 2:
            return False
        
        # 获取矩阵维度K（用于判断是否适合RVV优化）
        K, M = self._get_matrix_dimensions(weight_shape, layer_info.op_type)
        
        # 按照你的规则：如果max(K,M) < 4*w 或 K % w == 0，优先量化；否则考虑低秋
        if max(K, M) < 4 * self.w or K % self.w == 0:
            return False  # 这种情况更适合量化
        
        # 检查矩阵是否足够大，值得做低秋分解
        min_size_for_decomposition = 64
        return min(K, M) >= min_size_for_decomposition
    
    def _get_matrix_dimensions(self, weight_shape: Tuple[int, ...], op_type: str) -> Tuple[int, int]:
        """获取权重矩阵的K和M维度"""
        if op_type == "Conv":
            # Conv权重: (out_c, in_c, h, w)
            out_c, in_c, h, w = weight_shape
            K = in_c * h * w  # 卷积核"长度"
            M = out_c
        elif op_type in ["MatMul", "Gemm"]:
            # 全连接权重: (out_dim, in_dim) 或 (in_dim, out_dim)
            # 这里假设是(out_dim, in_dim)格式
            M, K = weight_shape
        else:
            # 其他情况的fallback
            K, M = weight_shape[-1], weight_shape[0]
        
        return K, M

    def _calculate_d_mid_upper_bound(self, K: int, M: int, eta: int) -> int:
        """计算满足CL<0的d_mid上界"""
        if eta <= 0:
            return 0

        k_tiles = math.ceil(K / eta)
        denom = k_tiles + (M / eta)
        if denom <= 0:
            return 0

        upper = (M * k_tiles) / denom
        upper_int = max(0, math.floor(upper - 1e-9))
        return (upper_int // eta) * eta

    def _calculate_clogging_level(self, K: int, M: int, d_mid: int, eta: int) -> float:
        """计算clogging level (I(G') - I(G))"""
        original = self._instruction_count(K, M, eta)
        split = self._split_instruction_count(K, M, d_mid, eta)
        return split - original

    def _instruction_count(self, K: int, M: int, eta: int) -> int:
        """估算指令数（基于tail近似模型）"""
        return M * math.ceil(K / eta)

    def _split_instruction_count(self, K: int, M: int, d_mid: int, eta: int) -> int:
        """估算split后的指令数（两段矩阵乘）"""
        return (d_mid * math.ceil(K / eta)) + (M * math.ceil(d_mid / eta))

    def _estimate_split_speedup(self, K: int, M: int, d_mid: int, eta: int) -> float:
        """估算split策略的加速比"""
        original = self._instruction_count(K, M, eta)
        split = self._split_instruction_count(K, M, d_mid, eta)
        if split <= 0:
            return 1.0
        return min(original / split, 5.0)
    
    def _calculate_valid_ranks(self, weight_shape: Tuple[int, ...], op_type: str) -> List[int]:
        """计算有效的rank值"""
        K, M = self._get_matrix_dimensions(weight_shape, op_type)
        
        valid_ranks = []
        
        # 基于K的分数计算候选rank
        for divisor in self.rank_divisors:
            candidate_rank = K // divisor
            
            # 向下取整到预定义的rank候选
            for rank in sorted(self.rank_candidates):
                if rank <= candidate_rank:
                    if rank not in valid_ranks:
                        valid_ranks.append(rank)
                    break
        
        # 过滤有效范围
        valid_ranks = [r for r in valid_ranks 
                      if self.min_rank <= r <= min(self.max_rank, min(K, M) // 2)]
        
        return sorted(valid_ranks, reverse=True)  # 从大到小排序
    
    def _estimate_quantization_speedup(self, bits: int, is_activation: bool = False) -> float:
        """估算量化策略的加速比"""
        if is_activation:
            # 激活量化的加速比相对保守
            return {8: 1.2, 4: 1.8}.get(bits, 1.0)
        else:
            # 权重量化的加速比
            return {8: 1.5, 4: 2.5}.get(bits, 1.0)
    
    def _estimate_low_rank_speedup(self, weight_shape: Tuple[int, ...], rank: int) -> float:
        """估算低秋分解的加速比"""
        if len(weight_shape) < 2:
            return 1.0
        
        # 计算压缩比
        original_ops = np.prod(weight_shape)
        if len(weight_shape) == 4:  # Conv层
            out_c, in_c, h, w = weight_shape
            compressed_ops = (in_c * h * w * rank) + (rank * out_c)
        else:  # FC层
            M, K = weight_shape
            compressed_ops = (K * rank) + (rank * M)
        
        compression_ratio = original_ops / compressed_ops
        
        # 考虑RVV的额外加速
        rvv_bonus = 1.2 if rank % self.w == 0 else 1.0
        
        return min(compression_ratio * rvv_bonus, 5.0)  # 限制最大加速比
    
    def _estimate_mixed_speedup(self, weight_shape: Tuple[int, ...], rank: int, bits: int) -> float:
        """估算混合策略的加速比"""
        low_rank_speedup = self._estimate_low_rank_speedup(weight_shape, rank)
        quant_speedup = self._estimate_quantization_speedup(bits)
        
        # 混合策略的加速比不是简单相乘，而是有一定的效率损失
        efficiency_factor = 0.8
        return low_rank_speedup * quant_speedup * efficiency_factor
    
    def get_strategy_compatibility_matrix(self, layer_infos: List[LayerInfo]) -> Dict[str, List[str]]:
        """获取策略兼容性矩阵，用于搜索时的预过滤"""
        compatibility = {}
        
        for layer_info in layer_infos:
            strategies = self.generate_strategies(layer_info)
            strategy_names = [f"{s.strategy_type.value}_{s.target}" for s in strategies]
            compatibility[layer_info.name] = strategy_names
        
        return compatibility
    
    def filter_strategies_by_budget(self, strategies: List[OptimizationStrategy], 
                                   layer_budget: float) -> List[OptimizationStrategy]:
        """根据层预算过滤策略（这里是预估，实际精度损失需要MSE评估）"""
        # 简单的启发式过滤：更激进的策略预估精度损失更大
        strategy_risk_scores = {
            StrategyType.ORIGINAL: 0.0,
            StrategyType.WEIGHT_QUANTIZATION: 0.3,
            StrategyType.ACTIVATION_QUANTIZATION: 0.4,
            StrategyType.SPLIT_CONSTRUCTION: 0.5,
            StrategyType.LOW_RANK: 0.6,
            StrategyType.MIXED: 0.8
        }
        
        filtered = []
        for strategy in strategies:
            base_risk = strategy_risk_scores[strategy.strategy_type]
            
            # 根据参数调整风险分数
            if strategy.strategy_type == StrategyType.WEIGHT_QUANTIZATION:
                bits = strategy.parameters.get("bits", 8)
                risk = base_risk * (8 / bits)
            elif strategy.strategy_type in [StrategyType.LOW_RANK, StrategyType.SPLIT_CONSTRUCTION]:
                # rank越小风险越大（这里需要更复杂的计算，暂时简化）
                risk = base_risk
            else:
                risk = base_risk
            
            # 如果预估风险在预算范围内，保留该策略
            if risk <= layer_budget * 10:  # 10是放大系数，避免过度保守
                filtered.append(strategy)
        
        return filtered


def create_strategy_summary(strategies: List[OptimizationStrategy]) -> Dict[str, Any]:
    """创建策略摘要信息"""
    summary = {
        "total_strategies": len(strategies),
        "by_type": {},
        "by_target": {},
        "expected_speedup_range": [1.0, 1.0]
    }
    
    speedups = []
    for strategy in strategies:
        # 按类型统计
        type_name = strategy.strategy_type.value
        summary["by_type"][type_name] = summary["by_type"].get(type_name, 0) + 1
        
        # 按目标统计
        target = strategy.target
        summary["by_target"][target] = summary["by_target"].get(target, 0) + 1
        
        # 收集加速比
        speedups.append(strategy.expected_speedup)
    
    if speedups:
        summary["expected_speedup_range"] = [min(speedups), max(speedups)]
    
    return summary


if __name__ == "__main__":
    # 测试代码
    from onnx_info_extractor import LayerInfo
    
    # 创建测试层信息
    test_layer = LayerInfo(
        name="test_conv",
        onnx_node_name="Conv_1",
        op_type="Conv",
        weight_shape=(512, 256, 3, 3),
        has_weights=True,
        mac_count=1000000,
        original_latency_ms=50.0
    )
    
    # 创建策略生成器
    generator = RVVAwareStrategyGenerator(rvv_length=128)
    
    # 生成策略
    strategies = generator.generate_strategies(test_layer)
    
    print(f"Generated {len(strategies)} strategies for {test_layer.name}")
    for i, strategy in enumerate(strategies):
        print(f"{i+1}. {strategy}")
    
    # 创建策略摘要
    summary = create_strategy_summary(strategies)
    print(f"\nStrategy summary: {summary}")
