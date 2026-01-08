"""
策略搜索模块
实现两阶段搜索：BOHB全局粗搜 + 贪心局部精调
"""

import numpy as np
import optuna
from typing import List, Dict, Tuple, Optional, Any, Union
import json
import time
from dataclasses import dataclass

from model_config import ModelConfig
from onnx_info_extractor import LayerInfo
from strategy_generator import OptimizationStrategy, StrategyType, RVVAwareStrategyGenerator
from mse_evaluator import MSEAccuracyEstimator


@dataclass
class SearchResult:
    """搜索结果数据结构"""
    strategies: List[OptimizationStrategy]
    accuracy_loss: float
    estimated_latency_improvement: float
    search_time_seconds: float
    total_evaluations: int


class BudgetAllocator:
    """预算分配器"""
    
    def __init__(self, global_threshold: float):
        self.global_threshold = global_threshold
        self.layer_budgets = {}
        self.remaining_pool = 0.0
        self.locked_layers = set()
    
    def allocate_initial_budgets(self, layer_infos: List[LayerInfo]) -> Dict[str, float]:
        """初始预算分配"""
        # 1. 计算总MAC
        total_mac = sum(layer.mac_count for layer in layer_infos if layer.has_weights)
        if total_mac == 0:
            return {layer.name: 0.0 for layer in layer_infos}
        
        # 2. 按MAC比例初分配
        initial_budgets = {}
        for layer in layer_infos:
            if layer.has_weights:
                initial_budgets[layer.name] = self.global_threshold * (layer.mac_count / total_mac)
            else:
                initial_budgets[layer.name] = 0.0
        
        # 3. 识别需要锁死的层（首层/末层的启发式判断）
        self._identify_locked_layers(layer_infos)
        
        # 4. 重新分配锁死层的预算
        locked_budget = sum(initial_budgets[layer.name] for layer in layer_infos 
                           if layer.name in self.locked_layers)
        
        remaining_budget = self.global_threshold - locked_budget
        remaining_mac = sum(layer.mac_count for layer in layer_infos 
                           if layer.has_weights and layer.name not in self.locked_layers)
        
        final_budgets = {}
        for layer in layer_infos:
            if layer.name in self.locked_layers:
                final_budgets[layer.name] = 0.0
            elif layer.has_weights and remaining_mac > 0:
                final_budgets[layer.name] = remaining_budget * (layer.mac_count / remaining_mac)
            else:
                final_budgets[layer.name] = 0.0
        
        self.layer_budgets = final_budgets
        self.remaining_pool = 0.0
        
        return final_budgets
    
    def _identify_locked_layers(self, layer_infos: List[LayerInfo]):
        """识别需要锁死的敏感层"""
        # 简单启发式：锁死第一层和最后一层
        if layer_infos:
            # 第一个有权重的层
            for layer in layer_infos:
                if layer.has_weights:
                    self.locked_layers.add(layer.name)
                    break
            
            # 最后一个有权重的层
            for layer in reversed(layer_infos):
                if layer.has_weights:
                    self.locked_layers.add(layer.name)
                    break
    
    def update_remaining_pool(self, layer_name: str, actual_loss: float):
        """更新剩余预算池"""
        allocated_budget = self.layer_budgets.get(layer_name, 0.0)
        if actual_loss < allocated_budget:
            saved_budget = allocated_budget - actual_loss
            self.remaining_pool += saved_budget
            self.layer_budgets[layer_name] = actual_loss
    
    def can_borrow_budget(self, layer_name: str, requested_budget: float) -> bool:
        """检查是否可以从剩余池借用预算"""
        current_budget = self.layer_budgets.get(layer_name, 0.0)
        return (current_budget + self.remaining_pool) >= requested_budget
    
    def borrow_budget(self, layer_name: str, requested_budget: float) -> bool:
        """从剩余池借用预算"""
        current_budget = self.layer_budgets.get(layer_name, 0.0)
        shortfall = requested_budget - current_budget
        
        if shortfall <= self.remaining_pool:
            self.remaining_pool -= shortfall
            self.layer_budgets[layer_name] = requested_budget
            return True
        return False


class GreedyStrategySearcher:
    """策略搜索器"""
    
    def __init__(self, config: ModelConfig, evaluator: MSEAccuracyEstimator):
        self.config = config
        self.evaluator = evaluator
        self.budget_allocator = BudgetAllocator(config.accuracy_threshold)
        
        # BOHB搜索参数
        self.bohb_trials = 300
        self.early_stop_multiplier = 1.2
        self.min_samples = 2
        self.max_samples = 32
        self.sample_progression = [2, 4, 8, 16, 32]
        
        # 搜索统计
        self.total_evaluations = 0
        self.search_start_time = 0
    
    def search_optimal_strategies(self, layer_infos: List[LayerInfo], 
                                strategies_per_layer: Dict[str, List[OptimizationStrategy]]) -> SearchResult:
        """主搜索流程"""
        self.search_start_time = time.time()
        
        # 分配预算
        layer_budgets = self.budget_allocator.allocate_initial_budgets(layer_infos)
        
        print(f"Starting optimization with {len(layer_infos)} layers, global threshold: {self.config.accuracy_threshold}")
        print(f"Layer budgets: {layer_budgets}")
        
        # 阶段A: BOHB全局粗搜
        print("Phase A: Global search with BOHB...")
        pareto_candidates = self.global_search_bohb(layer_infos, strategies_per_layer, layer_budgets)
        
        if not pareto_candidates:
            print("Warning: No feasible candidates found in global search")
            # 返回原始策略
            original_strategies = [OptimizationStrategy(
                layer_name=layer.name,
                strategy_type=StrategyType.ORIGINAL,
                parameters={},
                target="none"
            ) for layer in layer_infos]
            
            return SearchResult(
                strategies=original_strategies,
                accuracy_loss=0.0,
                estimated_latency_improvement=1.0,
                search_time_seconds=time.time() - self.search_start_time,
                total_evaluations=self.total_evaluations
            )
        
        # 阶段B: 贪心局部精调
        print(f"Phase B: Local refinement with {len(pareto_candidates)} candidates...")
        best_result = self.local_greedy_refine(pareto_candidates, layer_infos, strategies_per_layer)
        
        search_time = time.time() - self.search_start_time
        print(f"Search completed in {search_time:.2f} seconds with {self.total_evaluations} evaluations")
        
        return SearchResult(
            strategies=best_result["strategies"],
            accuracy_loss=best_result["accuracy_loss"],
            estimated_latency_improvement=best_result["latency_improvement"],
            search_time_seconds=search_time,
            total_evaluations=self.total_evaluations
        )
    
    def global_search_bohb(self, layer_infos: List[LayerInfo], 
                          strategies_per_layer: Dict[str, List[OptimizationStrategy]],
                          layer_budgets: Dict[str, float]) -> List[Dict[str, Any]]:
        """BOHB全局粗搜"""
        
        # 准备测试数据
        from onnx_info_extractor import ONNXNodeInfoExtractor
        extractor = ONNXNodeInfoExtractor(self.config)
        test_data = extractor.generate_test_data(num_samples=self.max_samples)
        
        # 创建Optuna study
        study = optuna.create_study(
            direction="minimize",  # 最小化延迟
            pruner=optuna.pruners.HyperbandPruner(
                min_resource=self.min_samples,
                max_resource=self.max_samples,
                reduction_factor=2
            ),
            sampler=optuna.samplers.TPESampler(n_startup_trials=20)
        )
        
        def objective(trial):
            return self._bohb_objective(trial, layer_infos, strategies_per_layer, 
                                      layer_budgets, test_data)
        
        # 运行优化
        study.optimize(objective, n_trials=self.bohb_trials, 
                      callbacks=[self._trial_callback])
        
        # 提取Pareto最优解
        pareto_candidates = self._extract_pareto_candidates(study)
        
        print(f"BOHB completed: {len(study.trials)} trials, {len(pareto_candidates)} Pareto candidates")
        
        return pareto_candidates
    
    def _bohb_objective(self, trial, layer_infos: List[LayerInfo], 
                       strategies_per_layer: Dict[str, List[OptimizationStrategy]],
                       layer_budgets: Dict[str, float], 
                       test_data: List[np.ndarray]) -> float:
        """BOHB目标函数"""
        try:
            # 为每层选择策略
            selected_strategies = []
            for layer in layer_infos:
                if layer.name not in strategies_per_layer:
                    continue
                
                available_strategies = strategies_per_layer[layer.name]
                if not available_strategies:
                    continue
                
                # 创建策略选择参数
                strategy_names = [f"{s.strategy_type.value}_{hash(str(s.parameters))}" 
                                for s in available_strategies]
                
                chosen_idx = trial.suggest_categorical(f"strategy_{layer.name}", 
                                                     list(range(len(available_strategies))))
                selected_strategies.append(available_strategies[chosen_idx])
            
            # 获取本次trial的样本数预算
            n_samples = trial.suggest_categorical("n_samples", self.sample_progression)
            current_test_data = test_data[:n_samples]
            
            # Early stopping: 快速精度检查
            if n_samples == self.min_samples:
                quick_mse = self._evaluate_strategies_mse(selected_strategies, current_test_data)
                if quick_mse > self.config.accuracy_threshold * self.early_stop_multiplier:
                    raise optuna.TrialPruned()
            
            # 完整评估
            mse_loss = self._evaluate_strategies_mse(selected_strategies, current_test_data)
            estimated_latency = self._estimate_latency_improvement(selected_strategies)
            
            # 检查精度约束
            if mse_loss > self.config.accuracy_threshold:
                # 返回一个大的惩罚值，但不prune（因为后续更大的budget可能有救）
                return 1000.0 + mse_loss
            
            # 记录trial信息用于后续Pareto提取
            trial.set_user_attr("strategies", selected_strategies)
            trial.set_user_attr("mse_loss", mse_loss)
            trial.set_user_attr("latency_improvement", estimated_latency)
            
            # 返回延迟目标（越小越好）
            return 1.0 / estimated_latency  # 最小化倒数 = 最大化加速比
        
        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"Trial failed: {e}")
            return float('inf')
    
    def _evaluate_strategies_mse(self, strategies: List[OptimizationStrategy], 
                               test_data: List[np.ndarray]) -> float:
        """评估策略组合的MSE损失"""
        try:
            # 应用策略到模型
            modified_model = self.evaluator.apply_strategies_to_onnx(strategies)
            
            # 计算MSE
            mse = self.evaluator.evaluate_mse(self.evaluator.original_model, 
                                            modified_model, test_data)
            
            self.total_evaluations += 1
            return mse
        
        except Exception as e:
            print(f"MSE evaluation failed: {e}")
            return float('inf')
    
    def _estimate_latency_improvement(self, strategies: List[OptimizationStrategy]) -> float:
        """估算延迟改善（基于策略的预期加速比）"""
        total_speedup = 1.0
        for strategy in strategies:
            if hasattr(strategy, 'expected_speedup'):
                total_speedup *= strategy.expected_speedup
        
        return total_speedup
    
    def _trial_callback(self, study, trial):
        """Trial回调函数，用于监控进度"""
        if trial.number % 50 == 0:
            print(f"Trial {trial.number}: Best value so far: {study.best_value}")
    
    def _extract_pareto_candidates(self, study) -> List[Dict[str, Any]]:
        """从study中提取Pareto最优候选"""
        candidates = []
        
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                user_attrs = trial.user_attrs
                if "strategies" in user_attrs and "mse_loss" in user_attrs:
                    mse_loss = user_attrs["mse_loss"]
                    latency_improvement = user_attrs["latency_improvement"]
                    
                    # 只保留满足精度约束的candidate
                    if mse_loss <= self.config.accuracy_threshold:
                        candidates.append({
                            "strategies": user_attrs["strategies"],
                            "mse_loss": mse_loss,
                            "latency_improvement": latency_improvement,
                            "trial_number": trial.number
                        })
        
        # Pareto排序：优先选择延迟改善大且精度损失小的
        candidates.sort(key=lambda x: (-x["latency_improvement"], x["mse_loss"]))
        
        # 返回top-K个候选（避免后续阶段计算量过大）
        return candidates[:10]
    
    def local_greedy_refine(self, pareto_candidates: List[Dict[str, Any]], 
                           layer_infos: List[LayerInfo],
                           strategies_per_layer: Dict[str, List[OptimizationStrategy]]) -> Dict[str, Any]:
        """贪心局部精调"""
        
        # 选择最优候选作为起点
        best_candidate = pareto_candidates[0]  # 已经按延迟改善排序
        current_strategies = best_candidate["strategies"].copy()
        current_mse = best_candidate["mse_loss"]
        current_latency = best_candidate["latency_improvement"]
        
        print(f"Starting local refinement from candidate with latency improvement: {current_latency:.2f}x")
        
        # 按MAC降序排列层（优先优化高价值层）
        sorted_layers = sorted([layer for layer in layer_infos if layer.has_weights], 
                             key=lambda x: x.mac_count, reverse=True)
        
        # 准备小批量测试数据用于快速评估
        from onnx_info_extractor import ONNXNodeInfoExtractor
        extractor = ONNXNodeInfoExtractor(self.config)
        quick_test_data = extractor.generate_test_data(num_samples=8)
        
        improvements_made = 0
        
        for layer in sorted_layers:
            if layer.name not in strategies_per_layer:
                continue
            
            available_strategies = strategies_per_layer[layer.name]
            current_strategy_for_layer = None
            
            # 找到当前层使用的策略
            for strategy in current_strategies:
                if strategy.layer_name == layer.name:
                    current_strategy_for_layer = strategy
                    break
            
            if not current_strategy_for_layer:
                continue
            
            # 尝试更激进的策略
            better_strategies = self._get_more_aggressive_strategies(
                current_strategy_for_layer, available_strategies)
            
            for candidate_strategy in better_strategies:
                # 创建临时策略组合
                temp_strategies = current_strategies.copy()
                for i, s in enumerate(temp_strategies):
                    if s.layer_name == layer.name:
                        temp_strategies[i] = candidate_strategy
                        break
                
                # 快速评估
                temp_mse = self._evaluate_strategies_mse(temp_strategies, quick_test_data)
                temp_latency = self._estimate_latency_improvement(temp_strategies)
                
                # 检查是否改善（延迟更好且精度仍满足要求）
                if (temp_latency > current_latency and 
                    temp_mse <= self.config.accuracy_threshold):
                    
                    print(f"Layer {layer.name}: {current_strategy_for_layer.strategy_type.value} -> "
                          f"{candidate_strategy.strategy_type.value}, "
                          f"latency: {current_latency:.2f}x -> {temp_latency:.2f}x")
                    
                    current_strategies = temp_strategies
                    current_mse = temp_mse
                    current_latency = temp_latency
                    improvements_made += 1
                    break  # 找到改善就立即锁定，不再尝试该层的其他策略
        
        print(f"Local refinement completed: {improvements_made} layers improved")
        
        return {
            "strategies": current_strategies,
            "accuracy_loss": current_mse,
            "latency_improvement": current_latency
        }
    
    def _get_more_aggressive_strategies(self, current_strategy: OptimizationStrategy, 
                                      available_strategies: List[OptimizationStrategy]) -> List[OptimizationStrategy]:
        """获取比当前策略更激进的策略选项"""
        more_aggressive = []
        
        # 定义激进程度排序
        aggressiveness_order = {
            StrategyType.ORIGINAL: 0,
            StrategyType.WEIGHT_QUANTIZATION: 1,
            StrategyType.ACTIVATION_QUANTIZATION: 2,
            StrategyType.SPLIT_CONSTRUCTION: 3,
            StrategyType.LOW_RANK: 4,
            StrategyType.MIXED: 5
        }
        
        current_aggressiveness = aggressiveness_order.get(current_strategy.strategy_type, 0)
        
        for strategy in available_strategies:
            strategy_aggressiveness = aggressiveness_order.get(strategy.strategy_type, 0)
            
            # 更激进的策略类型
            if strategy_aggressiveness > current_aggressiveness:
                more_aggressive.append(strategy)
            
            # 同类型但参数更激进
            elif (strategy_aggressiveness == current_aggressiveness and 
                  self._is_more_aggressive_params(current_strategy, strategy)):
                more_aggressive.append(strategy)
        
        # 按预期加速比排序
        more_aggressive.sort(key=lambda x: getattr(x, 'expected_speedup', 1.0), reverse=True)
        
        return more_aggressive
    
    def _is_more_aggressive_params(self, current: OptimizationStrategy, 
                                 candidate: OptimizationStrategy) -> bool:
        """判断候选策略的参数是否比当前策略更激进"""
        if current.strategy_type != candidate.strategy_type:
            return False
        
        if current.strategy_type == StrategyType.WEIGHT_QUANTIZATION:
            current_bits = current.parameters.get("bits", 32)
            candidate_bits = candidate.parameters.get("bits", 32)
            return candidate_bits < current_bits
        
        elif current.strategy_type == StrategyType.LOW_RANK:
            current_rank = current.parameters.get("rank", float('inf'))
            candidate_rank = candidate.parameters.get("rank", float('inf'))
            return candidate_rank < current_rank

        elif current.strategy_type == StrategyType.SPLIT_CONSTRUCTION:
            current_rank = current.parameters.get("d_mid", float('inf'))
            candidate_rank = candidate.parameters.get("d_mid", float('inf'))
            return candidate_rank < current_rank
        
        elif current.strategy_type == StrategyType.MIXED:
            # 混合策略的激进程度综合考虑rank和quantization
            current_rank = current.parameters.get("rank", float('inf'))
            current_bits = current.parameters.get("quantization_bits", 32)
            candidate_rank = candidate.parameters.get("rank", float('inf'))
            candidate_bits = candidate.parameters.get("quantization_bits", 32)
            
            return (candidate_rank < current_rank or 
                   (candidate_rank == current_rank and candidate_bits < current_bits))
        
        return False


if __name__ == "__main__":
    # 测试代码
    from model_config import create_default_config
    from onnx_info_extractor import ONNXNodeInfoExtractor
    from strategy_generator import RVVAwareStrategyGenerator
    from mse_evaluator import MSEAccuracyEstimator
    
    try:
        # 创建配置
        config = create_default_config(
            onnx_path="test_model.onnx",
            layers_json_path="test_layers.json",
            input_shape=(1, 3, 224, 224)
        )
        
        # 创建各个组件
        extractor = ONNXNodeInfoExtractor(config)
        generator = RVVAwareStrategyGenerator(config.rvv_length)
        evaluator = MSEAccuracyEstimator(config)
        searcher = GreedyStrategySearcher(config, evaluator)
        
        # 提取层信息
        layer_infos = extractor.extract_layer_info()
        
        # 生成策略
        strategies_per_layer = {}
        for layer in layer_infos:
            strategies_per_layer[layer.name] = generator.generate_strategies(layer)
        
        print(f"Generated strategies for {len(layer_infos)} layers")
        
        # 运行搜索（模拟）
        print("Starting strategy search...")
        # result = searcher.search_optimal_strategies(layer_infos, strategies_per_layer)
        # print(f"Search result: {result}")
        
    except Exception as e:
        print(f"Test failed: {e}")
