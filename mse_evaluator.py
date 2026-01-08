"""
MSE精度评估模块
负责将策略应用到ONNX模型并评估精度损失
"""

import os
import copy
import numpy as np
import onnx
import onnxruntime as ort
import torch
from typing import List, Dict, Tuple, Optional, Any, Union
import tempfile
import time

from model_config import ModelConfig
from onnx_info_extractor import LayerInfo
from strategy_generator import OptimizationStrategy, StrategyType


class MSEAccuracyEstimator:
    """MSE精度评估器"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.original_model = onnx.load(config.onnx_path)
        self.original_session = None
        self._warmup_count = 3
        self._measurement_count = 20
    
    def apply_strategies_to_onnx(self, strategies: List[OptimizationStrategy]) -> onnx.ModelProto:
        """将策略列表应用到ONNX模型"""
        # 深拷贝原始模型
        modified_model = copy.deepcopy(self.original_model)
        
        # 按策略逐层应用
        for strategy in strategies:
            if strategy.strategy_type == StrategyType.ORIGINAL:
                continue  # 跳过原始策略
            
            try:
                modified_model = self._apply_single_strategy(modified_model, strategy)
            except Exception as e:
                print(f"Warning: Failed to apply strategy {strategy} to layer {strategy.layer_name}: {e}")
                # 策略应用失败时，保持原始状态
                continue
        
        return modified_model
    
    def _apply_single_strategy(self, model: onnx.ModelProto, strategy: OptimizationStrategy) -> onnx.ModelProto:
        """应用单个策略到模型"""
        if strategy.strategy_type == StrategyType.WEIGHT_QUANTIZATION:
            return self._apply_weight_quantization(model, strategy)
        elif strategy.strategy_type == StrategyType.ACTIVATION_QUANTIZATION:
            return self._apply_activation_quantization(model, strategy)
        elif strategy.strategy_type == StrategyType.LOW_RANK:
            return self._apply_low_rank_decomposition(model, strategy)
        elif strategy.strategy_type == StrategyType.SPLIT_CONSTRUCTION:
            return self._apply_split_construction(model, strategy)
        elif strategy.strategy_type == StrategyType.MIXED:
            return self._apply_mixed_strategy(model, strategy)
        else:
            return model
    
    def _apply_weight_quantization(self, model: onnx.ModelProto, strategy: OptimizationStrategy) -> onnx.ModelProto:
        """应用权重量化策略"""
        # 找到目标节点对应的权重
        target_node = self._find_node_by_strategy(model, strategy)
        if not target_node:
            return model
        
        weight_name = self._get_weight_name(target_node)
        if not weight_name:
            return model
        
        # 获取权重数据
        weight_data = self._get_weight_data(model, weight_name)
        if weight_data is None:
            return model
        
        # 执行量化
        bits = strategy.parameters["bits"]
        quantization_type = strategy.parameters.get("quantization_type", "symmetric")
        per_channel = strategy.parameters.get("per_channel", False)
        
        if quantization_type == "symmetric":
            quantized_weight = self._symmetric_quantize(weight_data, bits, per_channel)
        else:
            quantized_weight = self._asymmetric_quantize(weight_data, bits)
        
        # 更新模型中的权重
        self._update_weight_in_model(model, weight_name, quantized_weight)
        
        return model
    
    def _apply_activation_quantization(self, model: onnx.ModelProto, strategy: OptimizationStrategy) -> onnx.ModelProto:
        """应用激活量化策略"""
        # 激活量化需要插入QuantizeLinear/DequantizeLinear节点
        # 这里实现简化版本：在目标节点后插入量化节点
        
        target_node = self._find_node_by_strategy(model, strategy)
        if not target_node:
            return model
        
        bits = strategy.parameters["bits"]
        
        # 生成量化参数（简化：使用固定的scale和zero_point）
        scale = 0.1  # 实际应用中需要通过校准数据计算
        zero_point = 128 if bits == 8 else 8
        
        # 插入量化/反量化节点
        self._insert_quantization_nodes(model, target_node, scale, zero_point, bits)
        
        return model
    
    def _apply_low_rank_decomposition(self, model: onnx.ModelProto, strategy: OptimizationStrategy) -> onnx.ModelProto:
        """应用低秋分解策略"""
        target_node = self._find_node_by_strategy(model, strategy)
        if not target_node:
            return model
        
        weight_name = self._get_weight_name(target_node)
        if not weight_name:
            return model
        
        weight_data = self._get_weight_data(model, weight_name)
        if weight_data is None:
            return model
        
        rank = strategy.parameters["rank"]
        
        # 执行SVD分解
        U, V = self._svd_decompose_weight(weight_data, rank, target_node.op_type)
        
        # 替换原始节点为两个矩阵乘法节点
        self._replace_with_low_rank_nodes(model, target_node, U, V)
        
        return model

    def _apply_split_construction(self, model: onnx.ModelProto, strategy: OptimizationStrategy) -> onnx.ModelProto:
        """应用split construction策略"""
        target_node = self._find_node_by_strategy(model, strategy)
        if not target_node:
            return model

        if target_node.op_type not in ["MatMul", "Gemm"]:
            return model

        weight_name = self._get_weight_name(target_node)
        if not weight_name:
            return model

        weight_data = self._get_weight_data(model, weight_name)
        if weight_data is None:
            return model

        d_mid = strategy.parameters.get("d_mid")
        if not d_mid or d_mid <= 0:
            return model

        U, V = self._svd_decompose_weight(weight_data, d_mid, target_node.op_type)
        self._replace_with_low_rank_nodes(model, target_node, U, V)

        return model
    
    def _apply_mixed_strategy(self, model: onnx.ModelProto, strategy: OptimizationStrategy) -> onnx.ModelProto:
        """应用混合策略（低秋+量化）"""
        # 先应用低秋分解
        low_rank_strategy = OptimizationStrategy(
            layer_name=strategy.layer_name,
            strategy_type=StrategyType.LOW_RANK,
            parameters={"rank": strategy.parameters["rank"]},
            target="weight"
        )
        model = self._apply_low_rank_decomposition(model, low_rank_strategy)
        
        # 然后对分解后的权重应用量化
        # 这里需要找到新创建的U和V权重进行量化
        quant_bits = strategy.parameters["quantization_bits"]
        self._quantize_decomposed_weights(model, strategy.layer_name, quant_bits)
        
        return model
    
    def _symmetric_quantize(self, weight: np.ndarray, bits: int, per_channel: bool = False) -> np.ndarray:
        """对称量化实现"""
        if per_channel and weight.ndim >= 2:
            # per-channel量化（沿第0维）
            scales = np.max(np.abs(weight), axis=tuple(range(1, weight.ndim)), keepdims=True)
        else:
            # per-tensor量化
            scales = np.max(np.abs(weight))
        
        # 避免除零
        scales = np.maximum(scales, 1e-8)
        
        # 量化范围
        qmax = 2**(bits-1) - 1
        qmin = -2**(bits-1)
        
        # 计算scale
        scale = scales / qmax
        
        # 量化
        quantized = np.round(weight / scale)
        quantized = np.clip(quantized, qmin, qmax)
        
        # 反量化回float32
        dequantized = quantized * scale
        
        return dequantized.astype(np.float32)
    
    def _asymmetric_quantize(self, weight: np.ndarray, bits: int) -> np.ndarray:
        """非对称量化实现"""
        # 计算min/max
        w_min = np.min(weight)
        w_max = np.max(weight)
        
        # 量化范围
        qmin = 0
        qmax = 2**bits - 1
        
        # 计算scale和zero_point
        scale = (w_max - w_min) / (qmax - qmin)
        scale = max(scale, 1e-8)  # 避免除零
        
        zero_point = qmin - w_min / scale
        zero_point = np.round(np.clip(zero_point, qmin, qmax))
        
        # 量化
        quantized = np.round(weight / scale + zero_point)
        quantized = np.clip(quantized, qmin, qmax)
        
        # 反量化
        dequantized = (quantized - zero_point) * scale
        
        return dequantized.astype(np.float32)
    
    def _svd_decompose_weight(self, weight: np.ndarray, rank: int, op_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """SVD分解权重矩阵"""
        original_shape = weight.shape
        
        if op_type == "Conv":
            # Conv权重: (out_c, in_c, h, w) -> (out_c, in_c*h*w)
            out_c = original_shape[0]
            weight_matrix = weight.reshape(out_c, -1)
        else:
            # FC权重: 直接使用
            weight_matrix = weight
        
        # 执行SVD
        U, S, Vt = np.linalg.svd(weight_matrix, full_matrices=False)
        
        # 截断到指定rank
        rank = min(rank, min(U.shape[1], Vt.shape[0]))
        U_truncated = U[:, :rank]
        S_truncated = S[:rank]
        Vt_truncated = Vt[:rank, :]
        
        # 构造分解后的矩阵
        U_new = U_truncated @ np.diag(S_truncated)  # (out_c, rank)
        V_new = Vt_truncated  # (rank, in_c*h*w)
        
        return U_new.astype(np.float32), V_new.astype(np.float32)
    
    def _replace_with_low_rank_nodes(self, model: onnx.ModelProto, original_node: onnx.NodeProto, 
                                   U: np.ndarray, V: np.ndarray):
        """用两个MatMul节点替换原始节点"""
        graph = model.graph
        
        # 生成唯一名称
        U_name = self._generate_unique_name(graph, f"U_{original_node.name}")
        V_name = self._generate_unique_name(graph, f"V_{original_node.name}")
        Y1_name = self._generate_unique_name(graph, f"Y1_{original_node.name}")
        
        # 添加新的权重initializer
        U_init = onnx.helper.make_tensor(U_name, onnx.TensorProto.FLOAT, U.shape, U.flatten())
        V_init = onnx.helper.make_tensor(V_name, onnx.TensorProto.FLOAT, V.shape, V.flatten())
        graph.initializer.extend([U_init, V_init])
        
        # 创建两个新的MatMul节点
        matmul_v = onnx.helper.make_node(
            'MatMul',
            inputs=[original_node.input[0], V_name],
            outputs=[Y1_name],
            name=f"MatMul_V_{original_node.name}"
        )
        
        matmul_u = onnx.helper.make_node(
            'MatMul',
            inputs=[U_name, Y1_name],
            outputs=[original_node.output[0]],  # 复用原输出名
            name=f"MatMul_U_{original_node.name}"
        )
        
        # 从图中移除原始节点
        graph.node.remove(original_node)
        
        # 添加新节点
        graph.node.extend([matmul_v, matmul_u])
        
        # 移除原始权重
        original_weight_name = self._get_weight_name(original_node)
        if original_weight_name:
            for init in graph.initializer:
                if init.name == original_weight_name:
                    graph.initializer.remove(init)
                    break
    
    def evaluate_mse(self, original_model: onnx.ModelProto, modified_model: onnx.ModelProto, 
                    test_data: List[np.ndarray]) -> float:
        """评估修改后模型与原模型的MSE差异"""
        try:
            # 创建临时文件保存修改后的模型
            with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as temp_file:
                onnx.save(modified_model, temp_file.name)
                temp_model_path = temp_file.name
            
            # 创建推理会话
            original_session = ort.InferenceSession(self.config.onnx_path)
            modified_session = ort.InferenceSession(temp_model_path)
            
            input_name = original_session.get_inputs()[0].name
            
            mse_values = []
            
            for data in test_data:
                # 原模型推理
                original_outputs = original_session.run(None, {input_name: data})
                
                # 修改后模型推理
                try:
                    modified_outputs = modified_session.run(None, {input_name: data})
                except Exception as e:
                    print(f"Warning: Modified model inference failed: {e}")
                    return float('inf')  # 返回无穷大表示策略不可行
                
                # 计算logits的MSE
                original_logits = original_outputs[0]  # 假设第一个输出是logits
                modified_logits = modified_outputs[0]
                
                mse = np.mean((original_logits - modified_logits) ** 2)
                mse_values.append(mse)
            
            # 清理临时文件
            os.unlink(temp_model_path)
            
            return np.mean(mse_values)
        
        except Exception as e:
            print(f"Error in MSE evaluation: {e}")
            return float('inf')
    
    def measure_latency(self, model_path: str, test_data: List[np.ndarray]) -> float:
        """测量模型推理延迟"""
        try:
            session = ort.InferenceSession(model_path)
            input_name = session.get_inputs()[0].name
            
            # 预热
            dummy_data = test_data[0] if test_data else np.random.randn(*self.config.input_shape).astype(np.float32)
            for _ in range(self._warmup_count):
                session.run(None, {input_name: dummy_data})
            
            # 测量
            latencies = []
            for i in range(min(self._measurement_count, len(test_data))):
                data = test_data[i % len(test_data)]
                
                start_time = time.perf_counter()
                session.run(None, {input_name: data})
                end_time = time.perf_counter()
                
                latencies.append((end_time - start_time) * 1000)  # 转换为毫秒
            
            return np.mean(latencies)
        
        except Exception as e:
            print(f"Error in latency measurement: {e}")
            return float('inf')
    
    # 辅助方法
    def _find_node_by_strategy(self, model: onnx.ModelProto, strategy: OptimizationStrategy) -> Optional[onnx.NodeProto]:
        """根据策略找到对应的ONNX节点"""
        # 这里需要一个mapping从layer_name到onnx_node_name
        # 简化实现：假设strategy.layer_name就是node名称或包含足够信息
        for node in model.graph.node:
            if strategy.layer_name in node.name or node.name in strategy.layer_name:
                return node
        return None
    
    def _get_weight_name(self, node: onnx.NodeProto) -> Optional[str]:
        """获取节点的权重名称"""
        # 查找initializer中的权重
        for input_name in node.input:
            # 简单启发式：第二个输入通常是权重（第一个是输入数据）
            if len(node.input) > 1 and input_name == node.input[1]:
                return input_name
        return None
    
    def _get_weight_data(self, model: onnx.ModelProto, weight_name: str) -> Optional[np.ndarray]:
        """从模型中获取权重数据"""
        for init in model.graph.initializer:
            if init.name == weight_name:
                return onnx.numpy_helper.to_array(init)
        return None
    
    def _update_weight_in_model(self, model: onnx.ModelProto, weight_name: str, new_weight: np.ndarray):
        """更新模型中的权重数据"""
        for init in model.graph.initializer:
            if init.name == weight_name:
                # 清除原始数据
                init.ClearField('raw_data')
                init.ClearField('float_data')
                init.ClearField('double_data')
                init.ClearField('int32_data')
                init.ClearField('int64_data')
                
                # 设置新数据
                init.raw_data = new_weight.tobytes()
                # 更新shape（如果改变了）
                init.dims[:] = new_weight.shape
                break
    
    def _generate_unique_name(self, graph: onnx.GraphProto, base_name: str) -> str:
        """生成唯一的名称"""
        existing_names = {init.name for init in graph.initializer}
        existing_names.update({node.name for node in graph.node})
        
        if base_name not in existing_names:
            return base_name
        
        counter = 1
        while f"{base_name}_{counter}" in existing_names:
            counter += 1
        return f"{base_name}_{counter}"
    
    def _insert_quantization_nodes(self, model: onnx.ModelProto, target_node: onnx.NodeProto, 
                                 scale: float, zero_point: int, bits: int):
        """插入量化/反量化节点"""
        # 简化实现：这里只是占位，实际需要更复杂的图修改逻辑
        pass
    
    def _quantize_decomposed_weights(self, model: onnx.ModelProto, layer_name: str, bits: int):
        """对分解后的U、V权重进行量化"""
        # 简化实现：找到对应的U、V权重并量化
        for init in model.graph.initializer:
            if f"U_{layer_name}" in init.name or f"V_{layer_name}" in init.name:
                weight_data = onnx.numpy_helper.to_array(init)
                quantized_weight = self._symmetric_quantize(weight_data, bits, per_channel=False)
                self._update_weight_in_model(model, init.name, quantized_weight)


if __name__ == "__main__":
    # 测试代码
    from model_config import create_default_config
    from strategy_generator import OptimizationStrategy, StrategyType
    
    # 创建测试配置
    config = create_default_config(
        onnx_path="test_model.onnx",
        layers_json_path="test_layers.json",
        input_shape=(1, 3, 224, 224)
    )
    
    # 创建评估器
    evaluator = MSEAccuracyEstimator(config)
    
    # 创建测试策略
    test_strategy = OptimizationStrategy(
        layer_name="test_conv",
        strategy_type=StrategyType.WEIGHT_QUANTIZATION,
        parameters={"bits": 8, "quantization_type": "symmetric", "per_channel": True},
        target="weight"
    )
    
    print("MSE Evaluator created successfully")
    print(f"Test strategy: {test_strategy}")
