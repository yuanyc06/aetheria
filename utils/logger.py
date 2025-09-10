# utils/logger.py
import logging
import functools
import time
from typing import Callable, Any

def get_logger(name: str) -> logging.Logger:
    """获取带有指定名称的日志记录器"""
    return logging.getLogger(name)

def log_execution(logger: logging.Logger = None):
    """记录函数执行的装饰器"""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 自动获取记录器
            _logger = logger or get_logger(func.__module__)
            
            func_name = func.__name__
            start_time = time.perf_counter()
            
            _logger.debug("开始执行: %s", func_name)
            
            try:
                result = func(*args, **kwargs)
                duration = time.perf_counter() - start_time
                _logger.debug("完成执行: %s (耗时: %.4f秒)", func_name, duration)
                return result
            except Exception as e:
                _logger.exception("执行失败: %s | 错误: %s", func_name, str(e))
                raise
        return wrapper
    return decorator

def log_state_transition(logger: logging.Logger, from_node: str, to_node: str, state: dict):
    """记录状态转换"""
    logger.info("状态转换: %s → %s", from_node, to_node)
    logger.debug("当前状态摘要: %s", {
        "modalities": state.get("modalities", []),
        "status": state.get("status", "unknown"),
        "background_length": len(state.get("background", "")),
        "debate_history_count": len(state.get("debate_history", [])),
        "verdict": state.get("verdict", {}).get("decision", "pending")
    })