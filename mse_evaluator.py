"""
MSE accuracy evaluation module.
Applies strategies to ONNX models and evaluates accuracy loss.
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
    """MSE accuracy estimator."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.original_model = onnx.load(config.onnx_path)
        self.original_session = None
        self._warmup_count = 3
        self._measurement_count = 20
    
    def apply_strategies_to_onnx(self, strategies: List[OptimizationStrategy]) -> onnx.ModelProto:
        """Apply a list of strategies to an ONNX model."""
        # Deep copy the original model
        modified_model = copy.deepcopy(self.original_model)
        
        # Apply per-strategy
        for strategy in strategies:
            if strategy.strategy_type == StrategyType.ORIGINAL:
                continue  # Skip original strategy
            
            try:
                modified_model = self._apply_single_strategy(modified_model, strategy)
            except Exception as e:
                print(f"Warning: Failed to apply strategy {strategy} to layer {strategy.layer_name}: {e}")
                # Keep original state if strategy application fails
                continue
        
        return modified_model
    
    def _apply_single_strategy(self, model: onnx.ModelProto, strategy: OptimizationStrategy) -> onnx.ModelProto:
        """Apply a single strategy to the model."""
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
        """Apply weight quantization."""
        # Find target node weights
        target_node = self._find_node_by_strategy(model, strategy)
        if not target_node:
            return model
        
        weight_name = self._get_weight_name(target_node)
        if not weight_name:
            return model
        
        # Fetch weight data
        weight_data = self._get_weight_data(model, weight_name)
        if weight_data is None:
            return model
        
        # Quantize
        bits = strategy.parameters["bits"]
        quantization_type = strategy.parameters.get("quantization_type", "symmetric")
        per_channel = strategy.parameters.get("per_channel", False)
        
        if quantization_type == "symmetric":
            quantized_weight = self._symmetric_quantize(weight_data, bits, per_channel)
        else:
            quantized_weight = self._asymmetric_quantize(weight_data, bits)
        
        # Update model weights
        self._update_weight_in_model(model, weight_name, quantized_weight)
        
        return model
    
    def _apply_activation_quantization(self, model: onnx.ModelProto, strategy: OptimizationStrategy) -> onnx.ModelProto:
        """Apply activation quantization."""
        # Activation quantization requires inserting QuantizeLinear/DequantizeLinear.
        # This is a simplified placeholder that inserts after the target node.
        
        target_node = self._find_node_by_strategy(model, strategy)
        if not target_node:
            return model
        
        bits = strategy.parameters["bits"]
        
        # Build quantization params (simplified fixed scale/zero_point)
        scale = 0.1  # Should be computed from calibration data in practice
        zero_point = 128 if bits == 8 else 8
        
        # Insert quant/dequant nodes
        self._insert_quantization_nodes(model, target_node, scale, zero_point, bits)
        
        return model
    
    def _apply_low_rank_decomposition(self, model: onnx.ModelProto, strategy: OptimizationStrategy) -> onnx.ModelProto:
        """Apply low-rank decomposition."""
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
        
        # Run SVD decomposition
        U, V = self._svd_decompose_weight(weight_data, rank, target_node.op_type)
        
        # Replace node with two MatMul nodes
        self._replace_with_low_rank_nodes(model, target_node, U, V)
        
        return model

    def _apply_split_construction(self, model: onnx.ModelProto, strategy: OptimizationStrategy) -> onnx.ModelProto:
        """Apply split-construction strategy."""

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

        U, V = self._build_split_weights(weight_data, d_mid)
        if U is None or V is None:
            return model

        self._replace_with_split_nodes(model, target_node, U, V)

        U, V = self._svd_decompose_weight(weight_data, d_mid, target_node.op_type)
        self._replace_with_low_rank_nodes(model, target_node, U, V)
        return model
    
    def _apply_mixed_strategy(self, model: onnx.ModelProto, strategy: OptimizationStrategy) -> onnx.ModelProto:
        """Apply mixed strategy (low-rank + quantization)."""
        # Apply low-rank decomposition first
        low_rank_strategy = OptimizationStrategy(
            layer_name=strategy.layer_name,
            strategy_type=StrategyType.LOW_RANK,
            parameters={"rank": strategy.parameters["rank"]},
            target="weight"
        )
        model = self._apply_low_rank_decomposition(model, low_rank_strategy)
        
        # Then quantize decomposed weights
        # This finds the new U/V weights and quantizes them
        quant_bits = strategy.parameters["quantization_bits"]
        self._quantize_decomposed_weights(model, strategy.layer_name, quant_bits)
        
        return model
    
    def _symmetric_quantize(self, weight: np.ndarray, bits: int, per_channel: bool = False) -> np.ndarray:
        """Symmetric quantization."""
        if per_channel and weight.ndim >= 2:
            # Per-channel quantization (axis 0)
            scales = np.max(np.abs(weight), axis=tuple(range(1, weight.ndim)), keepdims=True)
        else:
            # Per-tensor quantization
            scales = np.max(np.abs(weight))
        
        # Avoid division by zero
        scales = np.maximum(scales, 1e-8)
        
        # Quantization range
        qmax = 2**(bits-1) - 1
        qmin = -2**(bits-1)
        
        # Compute scale
        scale = scales / qmax
        
        # Quantize
        quantized = np.round(weight / scale)
        quantized = np.clip(quantized, qmin, qmax)
        
        # Dequantize back to float32
        dequantized = quantized * scale
        
        return dequantized.astype(np.float32)
    
    def _asymmetric_quantize(self, weight: np.ndarray, bits: int) -> np.ndarray:
        """Asymmetric quantization."""
        # Min/max
        w_min = np.min(weight)
        w_max = np.max(weight)
        
        # Quantization range
        qmin = 0
        qmax = 2**bits - 1
        
        # Compute scale and zero point
        scale = (w_max - w_min) / (qmax - qmin)
        scale = max(scale, 1e-8)  # Avoid division by zero
        
        zero_point = qmin - w_min / scale
        zero_point = np.round(np.clip(zero_point, qmin, qmax))
        
        # Quantize
        quantized = np.round(weight / scale + zero_point)
        quantized = np.clip(quantized, qmin, qmax)
        
        # Dequantize
        dequantized = (quantized - zero_point) * scale
        
        return dequantized.astype(np.float32)
    
    def _svd_decompose_weight(self, weight: np.ndarray, rank: int, op_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """SVD decomposition for a weight matrix."""
        original_shape = weight.shape
        
        if op_type == "Conv":
            # Conv weights: (out_c, in_c, h, w) -> (out_c, in_c*h*w)
            out_c = original_shape[0]
            weight_matrix = weight.reshape(out_c, -1)
        else:
            # FC weights: use directly
            weight_matrix = weight
        
        # Run SVD
        U, S, Vt = np.linalg.svd(weight_matrix, full_matrices=False)
        
        # Truncate to rank
        rank = min(rank, min(U.shape[1], Vt.shape[0]))
        U_truncated = U[:, :rank]
        S_truncated = S[:rank]
        Vt_truncated = Vt[:rank, :]
        
        # Build decomposed matrices
        U_new = U_truncated @ np.diag(S_truncated)  # (out_c, rank)
        V_new = Vt_truncated  # (rank, in_c*h*w)
        
        return U_new.astype(np.float32), V_new.astype(np.float32)
    
    def _replace_with_low_rank_nodes(self, model: onnx.ModelProto, original_node: onnx.NodeProto, 
                                   U: np.ndarray, V: np.ndarray):
        """Replace original node with two MatMul nodes."""
        graph = model.graph
        
        # Unique names
        U_name = self._generate_unique_name(graph, f"U_{original_node.name}")
        V_name = self._generate_unique_name(graph, f"V_{original_node.name}")
        Y1_name = self._generate_unique_name(graph, f"Y1_{original_node.name}")
        
        # Add new weight initializers
        U_init = onnx.helper.make_tensor(U_name, onnx.TensorProto.FLOAT, U.shape, U.flatten())
        V_init = onnx.helper.make_tensor(V_name, onnx.TensorProto.FLOAT, V.shape, V.flatten())
        graph.initializer.extend([U_init, V_init])
        
        # Create MatMul nodes
        matmul_v = onnx.helper.make_node(
            'MatMul',
            inputs=[original_node.input[0], V_name],
            outputs=[Y1_name],
            name=f"MatMul_V_{original_node.name}"
        )
        
        matmul_u = onnx.helper.make_node(
            'MatMul',
            inputs=[U_name, Y1_name],
            outputs=[original_node.output[0]],  # Reuse original output name
            name=f"MatMul_U_{original_node.name}"
        )
        
        # Remove original node
        graph.node.remove(original_node)
        
        # Add new nodes
        graph.node.extend([matmul_v, matmul_u])
        
        # Remove original weight
        original_weight_name = self._get_weight_name(original_node)
        if original_weight_name:
            for init in graph.initializer:
                if init.name == original_weight_name:
                    graph.initializer.remove(init)
                    break

    def _replace_with_split_nodes(self, model: onnx.ModelProto, original_node: onnx.NodeProto,
                                  U: np.ndarray, V: np.ndarray):
        """Replace original node with two MatMul nodes (split construction)."""
        graph = model.graph

        U_name = self._generate_unique_name(graph, f"U_{original_node.name}")
        V_name = self._generate_unique_name(graph, f"V_{original_node.name}")
        Y1_name = self._generate_unique_name(graph, f"Y1_{original_node.name}")

        U_init = onnx.helper.make_tensor(U_name, onnx.TensorProto.FLOAT, U.shape, U.flatten())
        V_init = onnx.helper.make_tensor(V_name, onnx.TensorProto.FLOAT, V.shape, V.flatten())
        graph.initializer.extend([U_init, V_init])

        matmul_u = onnx.helper.make_node(
            'MatMul',
            inputs=[original_node.input[0], U_name],
            outputs=[Y1_name],
            name=f"MatMul_U_{original_node.name}"
        )

        matmul_v = onnx.helper.make_node(
            'MatMul',
            inputs=[Y1_name, V_name],
            outputs=[original_node.output[0]],
            name=f"MatMul_V_{original_node.name}"
        )

        graph.node.remove(original_node)
        graph.node.extend([matmul_u, matmul_v])

        original_weight_name = self._get_weight_name(original_node)
        if original_weight_name:
            for init in graph.initializer:
                if init.name == original_weight_name:
                    graph.initializer.remove(init)
                    break

    def _replace_with_split_nodes(self, model: onnx.ModelProto, original_node: onnx.NodeProto,
                                  U: np.ndarray, V: np.ndarray):
        """用两个MatMul节点替换原始节点（split construction）"""
        graph = model.graph

        U_name = self._generate_unique_name(graph, f"U_{original_node.name}")
        V_name = self._generate_unique_name(graph, f"V_{original_node.name}")
        Y1_name = self._generate_unique_name(graph, f"Y1_{original_node.name}")

        U_init = onnx.helper.make_tensor(U_name, onnx.TensorProto.FLOAT, U.shape, U.flatten())
        V_init = onnx.helper.make_tensor(V_name, onnx.TensorProto.FLOAT, V.shape, V.flatten())
        graph.initializer.extend([U_init, V_init])

        matmul_u = onnx.helper.make_node(
            'MatMul',
            inputs=[original_node.input[0], U_name],
            outputs=[Y1_name],
            name=f"MatMul_U_{original_node.name}"
        )

        matmul_v = onnx.helper.make_node(
            'MatMul',
            inputs=[Y1_name, V_name],
            outputs=[original_node.output[0]],
            name=f"MatMul_V_{original_node.name}"
        )

        graph.node.remove(original_node)
        graph.node.extend([matmul_u, matmul_v])

        original_weight_name = self._get_weight_name(original_node)
        if original_weight_name:
            for init in graph.initializer:
                if init.name == original_weight_name:
                    graph.initializer.remove(init)
                    break
    
    def evaluate_mse(self, original_model: onnx.ModelProto, modified_model: onnx.ModelProto, 
                    test_data: List[np.ndarray]) -> float:
        """Evaluate MSE between original and modified models."""
        try:
            # Save modified model to a temp file
            with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as temp_file:
                onnx.save(modified_model, temp_file.name)
                temp_model_path = temp_file.name
            
            # Create inference sessions
            original_session = ort.InferenceSession(self.config.onnx_path)
            modified_session = ort.InferenceSession(temp_model_path)
            
            input_name = original_session.get_inputs()[0].name
            
            mse_values = []
            
            for data in test_data:
            # Original model inference
                original_outputs = original_session.run(None, {input_name: data})
                
                # Modified model inference
                try:
                    modified_outputs = modified_session.run(None, {input_name: data})
                except Exception as e:
                    print(f"Warning: Modified model inference failed: {e}")
                    return float('inf')  # Strategy is infeasible
                
                # Compute output MSE
                original_logits = original_outputs[0]  # Assume first output is logits
                modified_logits = modified_outputs[0]
                
                mse = np.mean((original_logits - modified_logits) ** 2)
                mse_values.append(mse)
            
            # Clean up temp file
            os.unlink(temp_model_path)
            
            return np.mean(mse_values)
        
        except Exception as e:
            print(f"Error in MSE evaluation: {e}")
            return float('inf')

    def predict_accuracy_loss(self, strategies: List[OptimizationStrategy],
                              layer_infos: List[LayerInfo]) -> float:
        """FAP: Predict accuracy loss for strategy combinations."""

        if not strategies:
            return 0.0

        layer_order = {layer.name: idx for idx, layer in enumerate(layer_infos)}
        ordered = sorted(
            [s for s in strategies if s.strategy_type != StrategyType.ORIGINAL],
            key=lambda s: layer_order.get(s.layer_name, float('inf'))
        )

        layer_errors = {}
        layer_vectors = {}

        for strategy in ordered:
            node = self._find_node_by_strategy(self.original_model, strategy)
            if not node:
                continue
            weight_name = self._get_weight_name(node)
            if not weight_name:
                continue
            weight_data = self._get_weight_data(self.original_model, weight_name)
            if weight_data is None:
                continue

            layer_vectors[strategy.layer_name] = weight_data.flatten()
            layer_errors[strategy.layer_name] = self._estimate_strategy_error(
                strategy, weight_data
            )

        total_loss = 0.0
        for strategy in ordered:
            layer_name = strategy.layer_name
            if layer_name not in layer_errors:
                continue

            alpha = 1.0
            for prev in ordered:
                if layer_order.get(prev.layer_name, float('inf')) >= layer_order.get(layer_name, float('inf')):
                    break
                prev_error = layer_errors.get(prev.layer_name, 0.0)
                kappa = self._estimate_coupling(prev.layer_name, layer_name, layer_vectors)
                alpha += prev_error * kappa

            total_loss += alpha * layer_errors[layer_name]

        return total_loss
    
    def measure_latency(self, model_path: str, test_data: List[np.ndarray]) -> float:
        """Measure model inference latency."""
        try:
            session = ort.InferenceSession(model_path)
            input_name = session.get_inputs()[0].name
            
            # Warmup
            dummy_data = test_data[0] if test_data else np.random.randn(*self.config.input_shape).astype(np.float32)
            for _ in range(self._warmup_count):
                session.run(None, {input_name: dummy_data})
            
            # Measure
            latencies = []
            for i in range(min(self._measurement_count, len(test_data))):
                data = test_data[i % len(test_data)]
                
                start_time = time.perf_counter()
                session.run(None, {input_name: data})
                end_time = time.perf_counter()
                
                latencies.append((end_time - start_time) * 1000)  # Convert to ms
            
            return np.mean(latencies)
        
        except Exception as e:
            print(f"Error in latency measurement: {e}")
            return float('inf')
    
    # Helpers
    def _find_node_by_strategy(self, model: onnx.ModelProto, strategy: OptimizationStrategy) -> Optional[onnx.NodeProto]:
        """Find the ONNX node for a strategy."""
        # A proper mapping from layer_name to onnx_node_name is needed.
        # This is a simplified heuristic.
        for node in model.graph.node:
            if strategy.layer_name in node.name or node.name in strategy.layer_name:
                return node
        return None
    
    def _get_weight_name(self, node: onnx.NodeProto) -> Optional[str]:
        """Get the node weight name."""
        # Find weight in initializers
        for input_name in node.input:
            # Heuristic: second input is usually the weight
            if len(node.input) > 1 and input_name == node.input[1]:
                return input_name
        return None
    
    def _get_weight_data(self, model: onnx.ModelProto, weight_name: str) -> Optional[np.ndarray]:
        """Get weight data from the model."""
        for init in model.graph.initializer:
            if init.name == weight_name:
                return onnx.numpy_helper.to_array(init)
        return None
    
    def _update_weight_in_model(self, model: onnx.ModelProto, weight_name: str, new_weight: np.ndarray):
        """Update weight data in the model."""
        for init in model.graph.initializer:
            if init.name == weight_name:
                # Clear existing data
                init.ClearField('raw_data')
                init.ClearField('float_data')
                init.ClearField('double_data')
                init.ClearField('int32_data')
                init.ClearField('int64_data')
                
                # Set new data
                init.raw_data = new_weight.tobytes()
                # Update shape if needed
                init.dims[:] = new_weight.shape
                break
    
    def _generate_unique_name(self, graph: onnx.GraphProto, base_name: str) -> str:
        """Generate a unique name."""
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
        """Insert quantization/dequantization nodes."""
        # Placeholder implementation; real graph edits are more complex.
        pass
    
    def _quantize_decomposed_weights(self, model: onnx.ModelProto, layer_name: str, bits: int):
        """Quantize decomposed U/V weights."""
        # Simplified: find U/V weights and quantize them
        for init in model.graph.initializer:
            if f"U_{layer_name}" in init.name or f"V_{layer_name}" in init.name:
                weight_data = onnx.numpy_helper.to_array(init)
                quantized_weight = self._symmetric_quantize(weight_data, bits, per_channel=False)
                self._update_weight_in_model(model, init.name, quantized_weight)

    def _build_split_weights(self, weight: np.ndarray, d_mid: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Build split-construction weights."""
        if weight.ndim != 2:
            return None, None

        k_dim, m_dim = weight.shape
        if d_mid >= k_dim:
            u = np.zeros((k_dim, d_mid), dtype=np.float32)
            u[:, :k_dim] = np.eye(k_dim, dtype=np.float32)
            v = np.zeros((d_mid, m_dim), dtype=np.float32)
            v[:k_dim, :] = weight.astype(np.float32)
            return u, v

        u_svd, v_svd = self._svd_decompose_weight(weight, d_mid, "MatMul")
        return u_svd, v_svd

    def _estimate_strategy_error(self, strategy: OptimizationStrategy, weight: np.ndarray) -> float:
        """Estimate reconstruction error for a strategy."""
        if weight.ndim < 2:
            return 0.0

        weight_norm = np.linalg.norm(weight) + 1e-8

        if strategy.strategy_type == StrategyType.WEIGHT_QUANTIZATION:
            bits = strategy.parameters.get("bits", 8)
            quantized = self._symmetric_quantize(weight, bits, per_channel=False)
            return float(np.linalg.norm(weight - quantized) ** 2 / (weight_norm ** 2))

        if strategy.strategy_type == StrategyType.ACTIVATION_QUANTIZATION:
            return 0.01

        if strategy.strategy_type == StrategyType.LOW_RANK:
            rank = strategy.parameters.get("rank", min(weight.shape))
            u, v = self._svd_decompose_weight(weight, rank, "MatMul")
            recon = u @ v
            return float(np.linalg.norm(weight - recon) ** 2 / (weight_norm ** 2))

        if strategy.strategy_type == StrategyType.SPLIT_CONSTRUCTION:
            d_mid = strategy.parameters.get("d_mid", min(weight.shape))
            u, v = self._build_split_weights(weight, d_mid)
            if u is None or v is None:
                return 0.0
            recon = u @ v
            return float(np.linalg.norm(weight - recon) ** 2 / (weight_norm ** 2))

        if strategy.strategy_type == StrategyType.MIXED:
            rank = strategy.parameters.get("rank", min(weight.shape))
            bits = strategy.parameters.get("quantization_bits", 8)
            u, v = self._svd_decompose_weight(weight, rank, "MatMul")
            recon = u @ v
            quantized = self._symmetric_quantize(recon, bits, per_channel=False)
            return float(np.linalg.norm(weight - quantized) ** 2 / (weight_norm ** 2))

        return 0.0

    def _estimate_coupling(self, layer_a: str, layer_b: str,
                           layer_vectors: Dict[str, np.ndarray]) -> float:
        """Estimate inter-layer coupling."""
        vec_a = layer_vectors.get(layer_a)
        vec_b = layer_vectors.get(layer_b)
        if vec_a is None or vec_b is None:
            return 0.0
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(abs(np.dot(vec_a, vec_b)) / (norm_a * norm_b))


if __name__ == "__main__":
    # Test code
    from model_config import create_default_config
    from strategy_generator import OptimizationStrategy, StrategyType
    
    # Create test config
    config = create_default_config(
        onnx_path="test_model.onnx",
        layers_json_path="test_layers.json",
        input_shape=(1, 3, 224, 224)
    )
    
    # Create evaluator
    evaluator = MSEAccuracyEstimator(config)
    
    # Create a test strategy
    test_strategy = OptimizationStrategy(
        layer_name="test_conv",
        strategy_type=StrategyType.WEIGHT_QUANTIZATION,
        parameters={"bits": 8, "quantization_type": "symmetric", "per_channel": True},
        target="weight"
    )
    
    print("MSE Evaluator created successfully")
    print(f"Test strategy: {test_strategy}")
