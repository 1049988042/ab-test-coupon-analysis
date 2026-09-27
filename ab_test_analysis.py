from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, chisquare, ttest_ind
from statsmodels.stats.proportion import proportions_ztest, proportion_effectsize
from statsmodels.stats.power import NormalIndPower
import matplotlib
matplotlib.use("Agg")  # 无需图形界面，运行后直接保存 PNG。
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 优先使用可用中文字体；无中文字体时，渠道图使用英文名称。
font_candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "Arial Unicode MS"]
available_fonts = {font.name for font in font_manager.fontManager.ttflist}
chinese_font = next((font for font in font_candidates if font in available_fonts), None)
plt.rcParams["font.sans-serif"] = [chinese_font] if chinese_font else ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# 0. 实验前样本量设计：给定设计参数，与实验后显著性推断分开。
# 本节为事前设计情景的补充演示，不声称这些参数已在原实验中预注册。
historical_control_rate = 0.085
mde_pp = 1.0
design_alpha = 0.05
design_power = 0.80
allocation_ratio = 1.0
design_treatment_rate = historical_control_rate + mde_pp / 100
cohens_h = proportion_effectsize(design_treatment_rate, historical_control_rate)
required_sample_size_raw = NormalIndPower().solve_power(
    effect_size=cohens_h, alpha=design_alpha, power=design_power,
    ratio=allocation_ratio, alternative="two-sided",
)
required_sample_size_per_group = int(np.ceil(required_sample_size_raw))
required_sample_size_total = required_sample_size_per_group * 2
print("\n0. 实验前样本量设计（双侧、1:1 分流）")
print(f"历史转化率={historical_control_rate:.1%}, MDE={mde_pp:g} pp, alpha={design_alpha}, power={design_power:.0%}")
print(f"Cohen's h={cohens_h:.8f}")
print(f"每组未取整样本量={required_sample_size_raw:.6f}，向上取整={required_sample_size_per_group}，总样本量={required_sample_size_total}")

# 1. 数据读取与基础质量检查：不修改、填补或去重原始数据。
data_path = Path(__file__).resolve().parent / "ab_test_coupon_experiment.csv"
data = pd.read_csv(data_path)
group_order = ["control", "treatment"]
print("\n1. 数据读取与基础质量检查")
print(f"数据规模：{data.shape[0]} 行，{data.shape[1]} 列")
print(pd.DataFrame({"dtype": data.dtypes, "missing_count": data.isna().sum()}).to_string())
print(f"重复记录数：{data.duplicated().sum()}")
print(f"重复 user_id 数（不含首次）：{data['user_id'].duplicated().sum()}")
print(f"user_id 是否唯一：{data['user_id'].is_unique}")
group_counts = data["group"].value_counts().reindex(group_order, fill_value=0)
print("各组样本量：\n" + group_counts.to_string())

# 遇到影响分母或分析单位的异常时停止，不静默排除数据。
required_columns = ["user_id", "group", "channel", "user_type", "device",
                    "converted", "order_amount", "coupon_cost", "profit"]
if data[required_columns].isna().any().any():
    raise ValueError("分析字段存在缺失值，请先确认统计口径。")
if not data["user_id"].is_unique:
    raise ValueError("user_id 不唯一，请先确认实验分析单位。")
if not data["group"].isin(group_order).all() or (group_counts == 0).any():
    raise ValueError("实验分组必须包含且仅包含 control 和 treatment。")
if not data["converted"].isin([0, 1]).all():
    raise ValueError("converted 必须为 0/1。")

# 2. 实验有效性：SRM 期望人数由实际总样本量和 50:50 比例计算。
print("\n2. 实验有效性检查")
srm_chi_square, srm_p_value = chisquare(group_counts, f_exp=np.repeat(group_counts.sum() / 2, 2))
validity_rows = [{"check": "SRM (50:50)", "chi_square": srm_chi_square,
                  "p_value": srm_p_value, "degrees_of_freedom": len(group_order) - 1}]
for feature in ["channel", "user_type", "device"]:
    balance_counts = pd.crosstab(data["group"], data[feature]).reindex(group_order)
    balance_percent = balance_counts.div(balance_counts.sum(axis=1), axis=0) * 100
    print(f"\n{feature} 样本量：\n{balance_counts.to_string()}")
    print(f"{feature} 组内占比（%）：\n{balance_percent.round(4).to_string()}")
    # 保留 SciPy 默认设置：自由度为 1 时采用 Yates 连续性校正。
    balance_chi_square, balance_p_value, balance_dof, _ = chi2_contingency(balance_counts)
    validity_rows.append({"check": feature, "chi_square": balance_chi_square,
                          "p_value": balance_p_value, "degrees_of_freedom": balance_dof})
validity_results = pd.DataFrame(validity_rows).set_index("check")
print("\n有效性检验：\n" + validity_results.to_string(float_format=lambda x: f"{x:.8g}"))

# 3. 核心指标：统一聚合，计算保留完整精度，仅在输出时四舍五入。
group_metrics = data.groupby("group").agg(
    sample_size=("converted", "count"), conversions=("converted", "sum"),
    conversion_rate=("converted", "mean"), revenue=("order_amount", "sum"),
    arpu=("order_amount", "mean"), coupon_cost_per_user=("coupon_cost", "mean"),
    profit_per_user=("profit", "mean"),
).reindex(group_order)
control_rate = group_metrics.loc["control", "conversion_rate"]
treatment_rate = group_metrics.loc["treatment", "conversion_rate"]
conversion_difference = treatment_rate - control_rate
absolute_lift_pp = conversion_difference * 100
relative_lift_pct = conversion_difference / control_rate * 100 if control_rate != 0 else np.nan
# 保留原双侧 Z 检验顺序：Control - Treatment。
conversion_z_stat, conversion_p_value = proportions_ztest(
    count=group_metrics["conversions"], nobs=group_metrics["sample_size"], alternative="two-sided",
)
# 保留原非合并标准误的 Wald 区间及 1.96 系数，方向为 Treatment - Control。
conversion_standard_error = np.sqrt(
    control_rate * (1 - control_rate) / group_metrics.loc["control", "sample_size"]
    + treatment_rate * (1 - treatment_rate) / group_metrics.loc["treatment", "sample_size"]
)
ci_lower_pp = (conversion_difference - 1.96 * conversion_standard_error) * 100
ci_upper_pp = (conversion_difference + 1.96 * conversion_standard_error) * 100
conversion_results = group_metrics[["sample_size", "conversions"]].copy()
conversion_results["conversion_rate_pct"] = group_metrics["conversion_rate"] * 100
print("\n3. 核心指标分析")
print(conversion_results.to_string(float_format=lambda x: f"{x:.6f}"))
print(f"Absolute lift (Treatment - Control)：{absolute_lift_pp:.6f} pp")
print(f"Relative lift：{relative_lift_pct:.6f}%")
print(f"两比例 Z 检验（Control - Treatment）：z={conversion_z_stat:.8g}, p-value={conversion_p_value:.8g}")
print(f"转化率差异 95% CI：[{ci_lower_pp:.6f}, {ci_upper_pp:.6f}] pp")

# 4. 护栏指标：保留 AOV = 总收入 / 转化用户数，并非由订单数计算。
group_metrics["aov"] = group_metrics["revenue"] / group_metrics["conversions"].replace(0, np.nan)
guardrail_results = group_metrics[["arpu", "aov", "coupon_cost_per_user", "profit_per_user"]]
control_profit = data.loc[data["group"].eq("control"), "profit"]
treatment_profit = data.loc[data["group"].eq("treatment"), "profit"]
profit_t_stat, profit_p_value = ttest_ind(
    treatment_profit,
    control_profit,
    equal_var=False
)
print("\n4. 护栏指标分析（金额单位与原始 CSV 一致）")
print(guardrail_results.to_string(float_format=lambda x: f"{x:.6f}"))
print("AOV 口径：总收入 / 转化用户数；每个转化用户一单时等同于订单客单价。")
print(
    f"人均利润 Welch t-test（Treatment - Control）："
    f"t={profit_t_stat:.8g}, p-value={profit_p_value:.8g}"
)

# 5. 渠道异质性：复用渠道聚合，变化率以 Control 为基准。
channel_metrics = data.groupby(["channel", "group"]).agg(
    sample_size=("converted", "count"), conversions=("converted", "sum"),
    conversion_rate=("converted", "mean"), profit_per_user=("profit", "mean"),
)
channel_conversion = channel_metrics["conversion_rate"].unstack().reindex(columns=group_order)
channel_profit = channel_metrics["profit_per_user"].unstack().reindex(columns=group_order)
channel_results = channel_conversion.mul(100).rename(
    columns={"control": "control_conversion_pct", "treatment": "treatment_conversion_pct"}
)
channel_results["conversion_lift_pp"] = (channel_conversion["treatment"] - channel_conversion["control"]) * 100
channel_results["control_profit_per_user"] = channel_profit["control"]
channel_results["treatment_profit_per_user"] = channel_profit["treatment"]
channel_results["profit_change_pct"] = (
    (channel_profit["treatment"] - channel_profit["control"])
    / channel_profit["control"].replace(0, np.nan) * 100
)
print("\n5. 渠道异质性分析")
print(channel_results.to_string(float_format=lambda x: f"{x:.6f}"))
douyin_metrics = channel_metrics.loc["抖音广告"].reindex(group_order)
douyin_z_stat, douyin_conversion_p_value = proportions_ztest(
    count=douyin_metrics["conversions"], nobs=douyin_metrics["sample_size"], alternative="two-sided",
)
douyin_data = data.loc[data["channel"].eq("抖音广告")]
# 保留当前总体及渠道利润检验方向：Treatment - Control。
douyin_t_stat, douyin_profit_p_value = ttest_ind(
    douyin_data.loc[douyin_data["group"].eq("treatment"), "profit"],
    douyin_data.loc[douyin_data["group"].eq("control"), "profit"], equal_var=False,
)
print(f"抖音广告两比例 Z 检验（Control - Treatment）：z={douyin_z_stat:.8g}, p-value={douyin_conversion_p_value:.8g}")
print(f"抖音广告利润 Welch t-test（Treatment - Control）：t={douyin_t_stat:.8g}, p-value={douyin_profit_p_value:.8g}")

# 保留原有新老用户转化率描述分析，不增加额外统计检验。
user_type_conversion = data.groupby(["user_type", "group"])["converted"].mean().unstack().reindex(columns=group_order)
user_type_results = user_type_conversion.mul(100).rename(
    columns={"control": "control_conversion_pct", "treatment": "treatment_conversion_pct"}
)
user_type_results["conversion_lift_pp"] = (user_type_conversion["treatment"] - user_type_conversion["control"]) * 100
print("\n补充：新老用户转化率\n" + user_type_results.round(6).to_string())

# 6. 仅汇总计算值，不硬编码业务判断。所有 p 值均未做多重比较校正。
summary_results = pd.Series({
    "control_conversion_pct": control_rate * 100,
    "treatment_conversion_pct": treatment_rate * 100,
    "absolute_lift_pp": absolute_lift_pp, "relative_lift_pct": relative_lift_pct,
    "conversion_z_test_p_value": conversion_p_value,
    "conversion_ci_lower_pp": ci_lower_pp, "conversion_ci_upper_pp": ci_upper_pp,
    "profit_welch_p_value": profit_p_value, "srm_p_value": srm_p_value,
    "douyin_conversion_z_test_p_value": douyin_conversion_p_value,
    "douyin_profit_welch_p_value": douyin_profit_p_value,
}, name="value")
print("\n6. 关键结果汇总")
print(summary_results.to_string(float_format=lambda x: f"{x:.8g}"))

# 7. 保存三张业务分析图，所有数值和显著性标签均来自以上计算。
images_dir = Path(__file__).resolve().parent / "images"
images_dir.mkdir(exist_ok=True)
group_labels = ["Control", "Treatment"]
group_colors = ["#64748b", "#167d9a"]
profit_change_pct = (group_metrics.loc["treatment", "profit_per_user"]
                     / group_metrics.loc["control", "profit_per_user"] - 1) * 100
print(f"总体人均利润变化率：{profit_change_pct:.6f}%")
conversion_p_label = "p < 0.001" if conversion_p_value < 0.001 else f"p = {conversion_p_value:.3f}"
profit_p_label = "p < 0.001" if profit_p_value < 0.001 else f"p = {profit_p_value:.3f}"

fig, ax = plt.subplots(figsize=(8, 5.2))
rates_pct = group_metrics["conversion_rate"] * 100
bars = ax.bar(group_labels, rates_pct, color=group_colors, width=0.55)
ax.bar_label(bars, labels=[f"{value:.2f}%" for value in rates_pct], padding=5, fontsize=12)
ax.set(ylabel="Purchase conversion rate (%)", ylim=(0, rates_pct.max() * 1.35))
ax.set_title("Purchase conversion: Control vs Treatment", pad=50)
ax.text(0.5, 1.025, f"Absolute lift: {absolute_lift_pp:+.2f} pp\n"
        f"95% CI: [{ci_lower_pp:.2f} pp, {ci_upper_pp:.2f} pp]  |  {conversion_p_label}",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=11)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
plt.savefig(images_dir / "conversion_comparison.png", dpi=180, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5.2))
profits = group_metrics["profit_per_user"]
bars = ax.bar(group_labels, profits, color=group_colors, width=0.55)
ax.bar_label(bars, labels=[f"{value:.2f}" for value in profits], padding=5, fontsize=12)
ax.set(ylabel="Profit per user (source currency unit)", ylim=(0, profits.max() * 1.30))
ax.set_title("Profit per user: Control vs Treatment", pad=35)
ax.text(0.5, 1.025, f"Profit change: {profit_change_pct:+.1f}%  |  Welch t-test {profit_p_label}",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=11)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
plt.savefig(images_dir / "profit_comparison.png", dpi=180, bbox_inches="tight")
plt.close(fig)

channel_plot = channel_results.sort_values("conversion_lift_pp", ascending=False)
channel_english = {"抖音广告": "Douyin ads", "信息流广告": "Feed ads",
                   "自然搜索": "Organic search", "直接访问": "Direct visits"}
channel_labels = channel_plot.index.tolist() if chinese_font else [channel_english.get(name, name) for name in channel_plot.index]
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, metric, title, unit in zip(
    axes, ["conversion_lift_pp", "profit_change_pct"],
    ["Conversion lift by channel", "Profit per user change by channel"], ["pp", "%"],
):
    values = channel_plot[metric]
    bars = ax.barh(np.arange(len(channel_plot)), values, height=0.55, color="#167d9a")
    ax.bar_label(bars, labels=[f"{value:+.2f} {unit}" for value in values], padding=6, fontsize=11)
    ax.axvline(0, color="#475569", linewidth=1)
    span = max(values.max(), 0) - min(values.min(), 0)
    ax.set_xlim(min(values.min(), 0) - span * 0.30, max(values.max(), 0) + span * 0.30)
    ax.set_title(title, pad=14)
    ax.set_xlabel(f"Treatment vs Control ({unit})")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_yticks(np.arange(len(channel_plot)), channel_labels)
axes[0].tick_params(axis="y", length=0, pad=12)
axes[1].tick_params(axis="y", left=False, labelleft=False)
axes[0].invert_yaxis()
fig.suptitle("Exploratory channel analysis", fontsize=14)
fig.tight_layout(w_pad=3)
plt.savefig(images_dir / "channel_effect.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"三张分析图已保存至：{images_dir}")
