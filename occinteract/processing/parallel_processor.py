import concurrent.futures
import json
import logging
import os
import threading
import time
from typing import List, Dict, Optional, Any


class ParallelProcessor:
    """一个通用的并行处理器，用于批量处理任务。

    特性:
    - 支持自定义结果字段名，兼容不同的 TextRewriter
    - 并行处理以提高效率
    - Checkpoint 机制，支持断点续传
    - 错误重试机制，自动重试失败的任务
    - 将最终结果保存为 JSON 文件
    - 支持任务超时，超时后自动终止并标记为错误
    - 启动时自动清理 checkpoint 中的错误记录，使出错任务可以重新开始
    """

    def __init__(
            self,
            text_rewriter: Any,
            texts_to_process: List[str],
            output_path: str = "results.json",
            max_retries: int = 3,
            max_workers: int = 8,
            result_schema: Optional[Dict[str, str]] = None,
    ) -> None:
        """初始化并行处理器。

        Args:
            text_rewriter: 文本处理器实例，参考 dataset.base_handler.BaseHandler.process 接口：
                - process 方法接收任务配置（如 JSON 字符串）并返回处理结果字典
                - 结果字典需包含 input 字段对应的值以及可选的 output 或 error 字段
                - 支持超时控制（通过 @timeout 装饰器）
            texts_to_process: 待处理的文本列表。
            output_path: 最终结果 JSON 文件的路径。
            max_retries: 单个任务的最大重试次数。
            max_workers: 并发工作线程数。
            result_schema: 结果字典的字段配置，格式为:
                {
                    'input': '输入字段名',
                    'output': '输出字段名'
                }

        Raises:
            ValueError: 当输入文本列表为空或 result_schema 格式不正确时。
        """
        if not texts_to_process:
            raise ValueError("输入文本列表不能为空。")

        self.all_texts_set = set(texts_to_process)
        self.output_path = output_path
        self.checkpoint_path = output_path + '.ckpt.jsonl'
        self.max_retries = max_retries
        self.max_workers = max_workers

        if 'input' not in result_schema or 'output' not in result_schema:
            raise ValueError("result_schema 必须包含 'input' 和 'output' 两个键")

        self.input_key = result_schema['input']
        self.output_key = result_schema['output']

        logging.warning(f"使用结果字段配置: 输入='{self.input_key}', 输出='{self.output_key}'")
        logging.warning(f"并发工作线程数: {self.max_workers}")
        logging.warning(f"最大重试次数: {self.max_retries}")

        self.rewriter = text_rewriter
        self.checkpoint_lock = threading.Lock()

        self.processed_texts = {}
        self.failure_counts = {}

    def _load_checkpoint(self) -> None:
        """从 checkpoint 文件加载已完成的任务。

        只保留成功的任务，移除所有出错的记录，使出错任务可以重新开始。
        """
        if not os.path.exists(self.checkpoint_path):
            logging.info("未找到 checkpoint 文件，将从头开始处理。")
            return

        successful_results = []
        error_count = 0

        # 读取 checkpoint 文件，只保留成功的记录
        with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    result = json.loads(line)
                    text = result.get(self.input_key)
                    if not text:
                        continue

                    # 只保留成功的记录（没有 error 键且包含输出字段）
                    if "error" not in result and self.output_key in result:
                        self.processed_texts[text] = result
                        successful_results.append(result)
                    else:
                        # 统计移除的错误记录数
                        error_count += 1

                except json.JSONDecodeError:
                    logging.warning(f"无法解析 checkpoint 文件中的行: {line.strip()}")

        num_successful = len(self.processed_texts)

        if error_count > 0:
            logging.warning(f"从 checkpoint 中移除了 {error_count} 个出错的记录，这些任务将重新开始。")

        logging.warning(f"从 checkpoint 加载了 {num_successful} 个成功任务。")

        # 重写 checkpoint 文件，只保留成功的记录
        if error_count > 0:
            with self.checkpoint_lock:
                with open(self.checkpoint_path, 'w', encoding='utf-8') as f:
                    for result in successful_results:
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
                logging.info(f"已清理 checkpoint 文件，移除了 {error_count} 个错误记录。")

    def _append_to_checkpoint(self, result: Dict[str, Any]) -> None:
        """将单个结果以线程安全的方式追加到 checkpoint 文件。

        Args:
            result: 要追加的结果字典。
        """
        with self.checkpoint_lock:
            with open(self.checkpoint_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

    def _process_task(self, text: str) -> Dict[str, Any]:
        """处理单个文本任务，包括调用重写器和保存到 checkpoint。

        包含超时处理逻辑。只有成功时才写入 checkpoint，失败时返回
        错误结果但不写入。

        Args:
            text: 待处理的文本任务。

        Returns:
            Dict[str, Any]: 处理结果字典，包含 input/output 字段或 error 字段。
        """
        try:
            # 由 rewriter.process 实现超时限制 -> from timeout_decorator import timeout
            # @timeout
            result = self.rewriter.process(text)

            if self.input_key not in result:
                logging.error(f"处理 '{text[:30]}...' 时返回了无效格式: {result}")
                return {
                    self.input_key: text,
                    self.output_key: None,
                    "error": f"Invalid format: missing '{self.input_key}' key"
                }

            if "error" in result:
                logging.warning(f"处理 '{text[:30]}...' 时返回错误: {result['error']}")
                return result  # 返回错误但不写入 checkpoint，等待重试

            # 只有成功时才写入 checkpoint
            self._append_to_checkpoint(result)
            return result

        except Exception as e:
            logging.error(f"处理 '{text[:30]}...' 时发生异常: {e}")
            return {
                self.input_key: text,
                self.output_key: None,
                "error": str(e)
            }

    def run(self) -> None:
        """执行整个并行处理流程，支持自动重试失败的任务。"""
        logging.warning('=' * 70)
        logging.warning('开始执行并行处理任务')
        logging.warning('=' * 70)

        # 1. 加载 checkpoint（会自动清理错误记录）
        self._load_checkpoint()

        # 2. 确定待办任务
        tasks_to_do = [
            text for text in self.all_texts_set
            if text not in self.processed_texts
        ]

        if not tasks_to_do:
            logging.info("所有任务均已成功处理。")
            self._save_final_json()
            return

        logging.warning(f"任务计划：总计 {len(self.all_texts_set)} 个任务，"
                        f"已完成 {len(self.processed_texts)} 个，"
                        f"待处理 {len(tasks_to_do)} 个。")

        # 3. 多轮重试处理
        current_retry = 0
        remaining_tasks = tasks_to_do.copy()

        while remaining_tasks and current_retry <= self.max_retries:
            if current_retry == 0:
                retry_info = "初次处理"
            else:
                retry_info = f"第 {current_retry}/{self.max_retries} 次重试"

            logging.warning(f"\n{retry_info}：开始处理 {len(remaining_tasks)} 个任务...")

            # 处理当前轮次的任务
            round_start_time = time.time()

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 动态导入 tqdm（如果没有则不显示进度条）
                try:
                    from tqdm import tqdm
                    use_tqdm = True
                except ImportError:
                    use_tqdm = False
                    logging.warning("未安装 tqdm，将不显示进度条")

                futures = {
                    executor.submit(self._process_task, text): text
                    for text in remaining_tasks
                }

                if use_tqdm:
                    desc = f"处理中 (轮次 {current_retry + 1}/{self.max_retries + 1})"
                    with tqdm(total=len(remaining_tasks), desc=desc) as pbar:
                        for future in concurrent.futures.as_completed(futures):
                            pbar.update(1)
                            self._handle_task_result(future)
                else:
                    for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                        self._handle_task_result(future)
                        if i % 10 == 0 or i == len(remaining_tasks):
                            logging.info(f"进度: {i}/{len(remaining_tasks)}")

            round_duration = time.time() - round_start_time

            # 统计当前轮次的结果
            round_success = len([t for t in remaining_tasks if t in self.processed_texts])
            round_failed = len(remaining_tasks) - round_success

            logging.warning(f"本轮完成：成功 {round_success} 个，失败 {round_failed} 个，"
                            f"耗时 {round_duration:.2f} 秒")

            # 确定下一轮需要重试的任务（本轮失败的任务）
            next_round_tasks = [
                text for text in remaining_tasks
                if text not in self.processed_texts  # 没有成功处理的任务
            ]

            if not next_round_tasks:
                logging.warning(f"✓ 所有任务在第 {current_retry + 1} 轮全部成功处理！")
                break

            if current_retry < self.max_retries:
                logging.warning(
                    f"→ 还有 {len(next_round_tasks)} 个任务失败，"
                    f"将在 2 秒后进入下一轮重试..."
                )
                # 短暂延迟，避免立即重试
                time.sleep(2)
            else:
                logging.warning(f"✗ 已达到最大重试次数 ({self.max_retries})，停止重试。")

            remaining_tasks = next_round_tasks
            current_retry += 1

        # 4. 将最终失败的任务写入 checkpoint
        final_failed_tasks = [
            text for text in self.all_texts_set
            if text not in self.processed_texts
        ]

        if final_failed_tasks:
            logging.warning(f"\n将 {len(final_failed_tasks)} 个最终失败的任务写入 checkpoint...")
            for text in final_failed_tasks:
                error_result = {
                    self.input_key: text,
                    self.output_key: None,
                    "error": f"Failed after {self.failure_counts.get(text, 0)} attempts"
                }
                self._append_to_checkpoint(error_result)

        # 5. 统计最终结果
        final_success = len(self.processed_texts)
        final_failed = len(self.all_texts_set) - final_success

        logging.warning('\n' + '=' * 70)
        logging.warning(f"所有任务处理完毕！")
        logging.warning(f"总计: {len(self.all_texts_set)} 个任务")
        logging.warning(f"成功: {final_success} 个 ({final_success / len(self.all_texts_set) * 100:.1f}%)")
        logging.warning(f"失败: {final_failed} 个 ({final_failed / len(self.all_texts_set) * 100:.1f}%)")
        logging.warning('=' * 70)

        if final_failed > 0:
            logging.warning(f"\n以下 {final_failed} 个任务最终失败：")
            for text in self.all_texts_set:
                if text not in self.processed_texts:
                    retry_count = self.failure_counts.get(text, 0)
                    logging.warning(f"  ✗ {text[:60]}... (失败 {retry_count} 次)")

        # 6. 保存最终结果
        self._save_final_json()

    def _handle_task_result(self, future: concurrent.futures.Future) -> None:
        """处理单个任务的结果。

        Args:
            future: 任务返回的 Future 对象。
        """
        result = future.result()
        text = result[self.input_key]

        # 更新处理状态
        if "error" in result:
            self.failure_counts[text] = self.failure_counts.get(text, 0) + 1
        else:
            # 成功处理
            self.processed_texts[text] = result
            if text in self.failure_counts:
                del self.failure_counts[text]

    def _save_final_json(self) -> None:
        """从 checkpoint 文件整理并保存最终的 JSON 结果。

        采用"后来者居上"的策略，确保每个任务只保留其最终状态
        （无论是成功还是最后一次失败）。
        """
        logging.info(f"\n正在将所有结果整合到最终文件: {self.output_path}")
        if not os.path.exists(self.checkpoint_path):
            logging.warning("Checkpoint 文件不存在，无法生成最终输出。")
            return

        final_status = {}
        with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    result = json.loads(line)
                    text = result.get(self.input_key)
                    if text:
                        # 无论成功或失败，都用最新的记录覆盖旧的记录
                        final_status[text] = result
                except json.JSONDecodeError:
                    continue  # 跳过损坏的行

        # 将字典的值转换为列表
        final_results_list = list(final_status.values())

        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(final_results_list, f, indent=2, ensure_ascii=False)

        num_final_success = sum(1 for res in final_results_list if "error" not in res)
        num_final_failed = len(final_results_list) - num_final_success
        logging.warning(
            f"✓ 成功保存 {len(final_results_list)} 条最终记录到 {self.output_path}\n"
            f"  ({num_final_success} 成功, {num_final_failed} 失败)"
        )
