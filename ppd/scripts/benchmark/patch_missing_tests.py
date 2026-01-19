#!/usr/bin/env python3
"""
Patch Missing & Failed Tests - 补测缺失和失败的测试点

运行缺失的测试点 + 重跑0%成功率的测试点，结果直接保存到现有目录。

缺失统计：
- 1P_3D: 30个缺失 (large_mid_paste, large_big_paste, large_huge_paste)
       + 9个失败 (large_mid_bal高QPS, large_short_gen, large_tiny多点)
- 3P_1D: 57个缺失 (large_very_long_gen部分 + 5个large workload)
       + 9个失败 (large_long_gen多点, large_short_gen, large_very_long_gen前3点)

总计: ~105个测试点
预计时间: ~2小时
"""

import os
import sys
import asyncio
import json

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_DIR)

from scripts.benchmark.comprehensive_benchmark import (
    run_benchmark_point,
    T1_CONFIGS,
    T2_CONFIGS,
    QPS_POINTS,
    start_config,
    run_cleanup,
    check_server_health,
    warmup_servers,
    save_result,
)
from pathlib import Path

# ============================================================================
# 缺失测试点定义 (文件不存在)
# ============================================================================

MISSING_TESTS = {
    "1P_3D": {
        "large_mid_paste": QPS_POINTS,      # 全部10个QPS点
        "large_big_paste": QPS_POINTS,      # 全部10个QPS点
        "large_huge_paste": QPS_POINTS,     # 全部10个QPS点
    },
    "3P_1D": {
        "large_very_long_gen": [4, 6, 8, 10, 12, 16, 20],  # 缺7个点
        "large_small_bal": QPS_POINTS,      # 全部10个QPS点
        "large_mid_bal": QPS_POINTS,        # 全部10个QPS点
        "large_mid_paste": QPS_POINTS,      # 全部10个QPS点
        "large_big_paste": QPS_POINTS,      # 全部10个QPS点
        "large_huge_paste": QPS_POINTS,     # 全部10个QPS点
    },
}

# ============================================================================
# 失败测试点定义 (文件存在但 success_rate = 0%)
# ============================================================================

FAILED_TESTS = {
    "1P_3D": {
        "large_mid_bal": [12, 16, 20],      # 高QPS失败
        "large_short_gen": [0.5],           # 低QPS失败
        "large_tiny": [0.5, 1, 2, 16, 20],  # 多点失败
    },
    "3P_1D": {
        "large_long_gen": [0.5, 1, 10, 12, 16],  # 多点失败
        "large_short_gen": [20],             # 高QPS失败
        "large_very_long_gen": [0.5, 1, 2],  # 前3点失败(会覆盖写入)
    },
}

OUTPUT_DIR = Path(PROJECT_DIR) / "results" / "comprehensive"


def count_tests():
    """统计测试点总数"""
    total_missing = 0
    total_failed = 0
    for config, workloads in MISSING_TESTS.items():
        for workload, qps_list in workloads.items():
            total_missing += len(qps_list)
    for config, workloads in FAILED_TESTS.items():
        for workload, qps_list in workloads.items():
            total_failed += len(qps_list)
    return total_missing, total_failed


def check_needs_rerun(config: str, workload: str, qps: float) -> tuple[bool, str]:
    """
    检查测试点是否需要运行
    返回: (需要运行, 原因)
    """
    result_file = OUTPUT_DIR / config / f"{config}_{workload}_{qps}.json"

    if not result_file.exists():
        return True, "missing"

    # 检查是否是失败测试点
    try:
        with open(result_file) as f:
            data = json.load(f)
            if data.get("success_rate", 100) == 0:
                return True, "failed (0% success)"
    except:
        return True, "corrupt file"

    return False, "exists"


def merge_test_lists(config: str) -> dict:
    """合并缺失和失败的测试点"""
    merged = {}

    # 添加缺失测试点
    if config in MISSING_TESTS:
        for workload, qps_list in MISSING_TESTS[config].items():
            if workload not in merged:
                merged[workload] = set()
            merged[workload].update(qps_list)

    # 添加失败测试点
    if config in FAILED_TESTS:
        for workload, qps_list in FAILED_TESTS[config].items():
            if workload not in merged:
                merged[workload] = set()
            merged[workload].update(qps_list)

    # 转换回list
    return {k: sorted(list(v)) for k, v in merged.items()}


async def run_tests_for_config(config: str):
    """运行单个配置的所有需要补测的测试点"""

    print(f"\n{'='*80}")
    print(f"补测配置: {config}")
    print(f"{'='*80}")

    # 合并缺失和失败的测试点
    workloads = merge_test_lists(config)

    # 统计需要补测的点
    tests_to_run = []
    for workload, qps_list in workloads.items():
        for qps in qps_list:
            needs_run, reason = check_needs_rerun(config, workload, qps)
            if needs_run:
                tests_to_run.append((workload, qps, reason))
            else:
                print(f"  跳过: {workload} QPS={qps} ({reason})")

    if not tests_to_run:
        print(f"  {config} 没有需要补测的点，跳过")
        return True

    # 显示待运行的测试点
    missing_count = sum(1 for _, _, r in tests_to_run if r == "missing")
    failed_count = sum(1 for _, _, r in tests_to_run if "failed" in r)
    print(f"  需要补测: {len(tests_to_run)} 个测试点 (缺失: {missing_count}, 失败: {failed_count})")

    # 启动服务器
    print(f"\n启动 {config} 服务器...")
    run_cleanup()
    await asyncio.sleep(8)

    if not start_config(config):
        print(f"ERROR: 无法启动 {config}")
        return False

    await asyncio.sleep(15)

    # 健康检查
    healthy, error = await check_server_health()
    if not healthy:
        print(f"ERROR: 服务器不健康: {error}")
        run_cleanup()
        return False

    # 预热
    print("预热服务器...")
    if not await warmup_servers():
        print("WARNING: 预热有问题，但继续执行")

    await asyncio.sleep(5)

    # 运行测试点
    success_count = 0
    fail_count = 0

    for workload, qps, reason in tests_to_run:
        # 解析workload
        parts = workload.split("_", 1)
        t1_name = parts[0]
        t2_name = parts[1]

        t1_config = T1_CONFIGS[t1_name]
        t2_config = T2_CONFIGS[t2_name]

        reason_short = "重跑" if "failed" in reason else "补测"
        print(f"\n  [{workload}] QPS={qps} ({reason_short})...", end=" ", flush=True)

        try:
            result, server_healthy = await run_benchmark_point(
                config, workload, qps, t1_config, t2_config
            )

            # 保存结果（覆盖失败的旧文件或创建新文件）
            save_result(result, OUTPUT_DIR / config)

            print(f"T1={result.turn1.avg_ttft_ms:.0f}ms T2={result.turn2.avg_ttft_ms:.0f}ms ({result.success_rate:.0f}% success)")

            if result.success_rate >= 90:
                success_count += 1
            else:
                fail_count += 1

            if not server_healthy:
                print("  WARNING: 服务器可能不健康")
                # 尝试重启
                print("  尝试重启服务器...")
                run_cleanup()
                await asyncio.sleep(10)
                if not start_config(config):
                    print(f"ERROR: 重启失败，跳过剩余测试")
                    break
                await asyncio.sleep(15)
                await warmup_servers()
                await asyncio.sleep(5)

        except Exception as e:
            print(f"FAILED: {e}")
            fail_count += 1

    # 清理
    print(f"\n清理 {config}...")
    run_cleanup()
    await asyncio.sleep(10)

    print(f"\n{config} 补测完成: {success_count} 成功, {fail_count} 失败")
    return True


async def main():
    """主入口"""

    total_missing, total_failed = count_tests()

    # 获取所有需要处理的配置
    all_configs = set(MISSING_TESTS.keys()) | set(FAILED_TESTS.keys())

    print("="*80)
    print("补测缺失和失败的测试点")
    print("="*80)
    print(f"处理配置: {sorted(all_configs)}")
    print(f"缺失测试点: {total_missing}")
    print(f"失败测试点(0% success): {total_failed}")
    print(f"总计: {total_missing + total_failed} 个测试点")
    print(f"预计时间: ~{(total_missing + total_failed) * 1.0 / 60:.1f} 小时")
    print("="*80)

    # 逐个配置运行
    for config in sorted(all_configs):
        await run_tests_for_config(config)

    print("\n" + "="*80)
    print("所有补测完成！")
    print("="*80)

    # 验证结果
    print("\n验证补测结果:")
    for config in sorted(all_configs):
        config_dir = OUTPUT_DIR / config
        count = len(list(config_dir.glob("*.json")))

        # 统计成功率>0的测试点
        valid_count = 0
        for f in config_dir.glob("*.json"):
            try:
                with open(f) as fp:
                    data = json.load(fp)
                    if data.get("success_rate", 0) > 0:
                        valid_count += 1
            except:
                pass

        print(f"  {config}: {count}/180 测试点, {valid_count} 有效(success>0%)")


if __name__ == "__main__":
    asyncio.run(main())
