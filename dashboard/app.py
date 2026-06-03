import json
import sys
from pathlib import Path
from typing import Dict

import pandas as pd
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ru_liquidity_sentinel.config import LSI_THRESHOLDS, PipelineConfig
from ru_liquidity_sentinel.incremental import incremental_update, offline_initialize


st.set_page_config(
    page_title="RU Liquidity Sentinel",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    [data-testid="stMetricValue"] { font-size: 28px; color: #00D4FF; }
    .stButton>button {
        width: 100%; border-radius: 10px; border: 1px solid #00D4FF;
        background-color: #161B22; color: #00D4FF; font-weight: bold; transition: 0.3s;
    }
    .stButton>button:hover { background-color: #00D4FF; color: #0E1117; border: 1px solid #00D4FF; }
    [data-testid="stDataFrame"], [data-testid="stTable"],
    [data-testid="stArrowDataTable"], [data-testid="stVegaLiteChart"],
    div[data-testid="stVizContainer"], .stPlotlyChart {
        background-color: transparent !important;
    }
    svg g[class^="marks"] path.domain,
    svg g[class^="marks"] g.role-axis-grid line {
        stroke: #444B5A !important;
        stroke-opacity: 0.4 !important;
    }
    div[data-testid="stDataFrame"] > div, div[data-testid="stTable"] > div,
    div[data-testid="stArrowDataTable"] > div, [data-testid="stVegaLiteChart"] > div {
        background-color: transparent !important;
    }
    [data-testid="stVegaLiteChart"] canvas, [data-testid="stVegaLiteChart"] svg,
    [data-testid="stVegaLiteChart"] {
        background: transparent !important;
    }
    iframe { background-color: transparent !important; }
    .status-red { color: #FF4B4B; font-weight: bold; border: 1px solid #FF4B4B; padding: 2px 8px; border-radius: 5px; }
    .status-yellow { color: #FFD700; font-weight: bold; border: 1px solid #FFD700; padding: 2px 8px; border-radius: 5px; }
    .status-green { color: #00FF7F; font-weight: bold; border: 1px solid #00FF7F; padding: 2px 8px; border-radius: 5px; }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    [data-testid="stVerticalBlock"] > div {
        animation: fadeIn 0.5s ease-out forwards;
    }
    .block-container { padding-top: 2rem; }
</style>
""",
    unsafe_allow_html=True,
)


def is_fresh(path: Path, ref: Path) -> bool:
    """Return true when an optional dashboard artifact matches current LSI."""

    return path.exists() and ref.exists() and path.stat().st_mtime >= ref.stat().st_mtime


@st.cache_data(show_spinner=False)
def load_artifacts(artifacts_dir: Path) -> dict:
    lsi_path = artifacts_dir / "historical_lsi.parquet"
    if not lsi_path.exists():
        lsi_path = artifacts_dir / "lsi_daily.parquet"

    feat = pd.read_parquet(artifacts_dir / "features_daily.parquet")
    lsi = pd.read_parquet(lsi_path)
    feat.index = pd.to_datetime(feat.index)
    lsi.index = pd.to_datetime(lsi.index)

    p_target = artifacts_dir / "proxy_target.parquet"
    target = pd.read_parquet(p_target)["proxy_stress"] if p_target.exists() else pd.Series(dtype=float)

    p_sens = artifacts_dir / "sensitivity.parquet"
    sens = pd.read_parquet(p_sens) if is_fresh(p_sens, lsi_path) else pd.DataFrame(index=lsi.index)
    if not sens.empty:
        sens.index = pd.to_datetime(sens.index)

    p_sens_summary = artifacts_dir / "sensitivity_summary.csv"
    sens_summary = pd.read_csv(p_sens_summary) if is_fresh(p_sens_summary, lsi_path) else pd.DataFrame()

    backtest = pd.read_csv(artifacts_dir / "backtest_episodes.csv")
    metadata = json.loads((artifacts_dir / "metadata.json").read_text(encoding="utf-8"))

    p_auto = artifacts_dir / "auto_episodes.csv"
    auto_episodes = pd.DataFrame()
    if p_auto.exists() and p_auto.stat().st_size > 0:
        auto_episodes = pd.read_csv(p_auto, parse_dates=["start", "end"])

    p_pred = artifacts_dir / "model_predictions.parquet"
    model_predictions = pd.DataFrame()
    if is_fresh(p_pred, lsi_path) and p_pred.stat().st_size > 0:
        model_predictions = pd.read_parquet(p_pred)
        model_predictions.index = pd.to_datetime(model_predictions.index)
    if model_predictions.empty and "lsi" in lsi.columns:
        model_predictions = lsi[["lsi"]].rename(columns={"lsi": "nsvm"})

    p_metrics = artifacts_dir / "model_metrics.csv"
    model_metrics = pd.read_csv(p_metrics) if p_metrics.exists() else pd.DataFrame()

    p_cv = artifacts_dir / "model_cv_results.csv"
    cv_results = pd.read_csv(p_cv) if is_fresh(p_cv, lsi_path) else pd.DataFrame()

    p_best = artifacts_dir / "model_best_params.csv"
    best_params = pd.read_csv(p_best) if is_fresh(p_best, lsi_path) else pd.DataFrame()

    p_noise = artifacts_dir / "noise_breakdown.csv"
    noise = pd.read_csv(p_noise) if p_noise.exists() else pd.DataFrame()

    model_attributions: Dict[str, pd.DataFrame] = {}
    p_attr = artifacts_dir / "model_nsvm_module_attribution.parquet"
    if is_fresh(p_attr, lsi_path):
        model_attributions["nsvm"] = pd.read_parquet(p_attr)
        model_attributions["nsvm"].index = pd.to_datetime(model_attributions["nsvm"].index)

    return {
        "features": feat,
        "lsi": lsi,
        "target": target,
        "sensitivity": sens,
        "sensitivity_summary": sens_summary,
        "backtest": backtest,
        "auto_episodes": auto_episodes,
        "metadata": metadata,
        "model_predictions": model_predictions,
        "model_metrics": model_metrics,
        "model_cv_results": cv_results,
        "model_best_params": best_params,
        "noise_breakdown": noise,
        "model_attributions": model_attributions,
    }


def render_status(value: float) -> None:
    """Render the traffic-light LSI status."""

    if value is None or pd.isna(value):
        st.markdown("-")
    elif value <= LSI_THRESHOLDS["green_max"]:
        st.markdown('<span class="status-green">GREEN</span>', unsafe_allow_html=True)
    elif value <= LSI_THRESHOLDS["yellow_max"]:
        st.markdown('<span class="status-yellow">YELLOW</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-red">RED</span>', unsafe_allow_html=True)


def source_mtime() -> float:
    paths = [
        SRC / "ru_liquidity_sentinel" / "aggregate.py",
        SRC / "ru_liquidity_sentinel" / "features.py",
        SRC / "ru_liquidity_sentinel" / "nsvm_lsi.py",
        SRC / "ru_liquidity_sentinel" / "incremental.py",
        SRC / "ru_liquidity_sentinel" / "nsvm_model.py",
        *list((SRC / "ru_liquidity_sentinel" / "modules").glob("*.py")),
    ]
    mtimes = [p.stat().st_mtime for p in paths if p.exists()]
    return max(mtimes) if mtimes else 0.0


def main() -> None:
    cfg = PipelineConfig()
    cfg.ensure_dirs()
    artifacts_dir = cfg.artifacts_dir

    st.sidebar.title("Sentinel Control")
    st.sidebar.markdown("**ПСБ Казначейство**")

    if st.sidebar.button("Обновить данные"):
        with st.spinner("Инкрементальное обновление NSVM..."):
            result = incremental_update(cfg, fetch=True)
            st.cache_data.clear()
        if result.new_rows:
            msg = f"Добавлено дней: {result.new_rows}."
            if result.partial_fit_ran:
                msg += " Обнаружена смена режима, NSVM дообучен."
            st.sidebar.success(msg)
        else:
            st.sidebar.info("Новых дней нет.")

    if not (artifacts_dir / "historical_lsi.parquet").exists():
        if st.sidebar.button("Первичная инициализация"):
            with st.spinner(f"Обучение NSVM до {cfg.nsvm_init_cutoff_date} и сборка истории LSI..."):
                offline_initialize(cfg)
                st.cache_data.clear()
            st.sidebar.success("Инициализация готова.")
        else:
            st.warning("Артефакты incremental-контура не найдены. Запустите первичную инициализацию.")
            return

    if (artifacts_dir / "historical_lsi.parquet").stat().st_mtime < source_mtime():
        st.sidebar.warning("Артефакты старше кода модели. Пересоберите их через первичную инициализацию.")

    art = load_artifacts(artifacts_dir)
    lsi = art["lsi"]
    feat = art["features"]
    last_date = lsi.index.max()
    last_row = lsi.loc[last_date]

    st.title("Система раннего предупреждения стресса ликвидности")

    m_cols = st.columns(3)
    with m_cols[0]:
        st.metric("Дата", last_date.date().isoformat())
    with m_cols[1]:
        st.metric("LSI Score", f"{last_row['lsi']:.1f}")
    with m_cols[2]:
        st.write("Статус")
        render_status(last_row["lsi"])

    st.markdown("---")

    min_date, max_date = lsi.index.min().date(), lsi.index.max().date()
    rng = st.sidebar.date_input(
        "Интервал анализа",
        value=(max(min_date, max_date.replace(year=max_date.year - 2)), max_date),
    )
    if isinstance(rng, tuple) and len(rng) == 2:
        start, end = pd.Timestamp(rng[0]), pd.Timestamp(rng[1])
    else:
        start, end = pd.Timestamp(min_date), pd.Timestamp(max_date)

    lsi_w = lsi.loc[start:end]
    feat_w = feat.loc[start:end]

    tabs = st.tabs(["LSI", "Модули", "Модель", "Backtest", "Авто-эпизоды", "Чувствительность", "Алерты"])

    with tabs[0]:
        st.subheader("Линейка индекса (NSVM)")
        st.line_chart(lsi_w[["lsi"]].rename(columns={"lsi": "LSI Index"}), color="#00D4FF", height=350)

        attr_df = art["model_attributions"].get("nsvm", lsi).loc[start:end]
        contrib_cols = [c for c in attr_df.columns if c.startswith("contrib_")]
        if contrib_cols:
            st.subheader("Структура вклада модулей (SHAP GradientExplainer)")
            st.area_chart(attr_df[contrib_cols], height=300)

    with tabs[1]:
        for module in ("M1", "M2", "M3", "M5"):
            mod_lower = module.lower()
            mad_cols = [c for c in feat_w.columns if c.startswith(f"{mod_lower}_mad_")]
            if mad_cols:
                st.markdown(f"**Анализ модуля {module}**")
                st.line_chart(feat_w[mad_cols], height=200)

    with tabs[2]:
        st.subheader("Метрики агрегатора")
        if not art["model_metrics"].empty:
            st.dataframe(art["model_metrics"].style.format(precision=3), use_container_width=True)
        if not art["model_predictions"].empty:
            st.line_chart(art["model_predictions"].loc[start:end][["nsvm"]], color="#FF4B4B")

    with tabs[3]:
        st.subheader("Историческая верификация")
        st.dataframe(art["backtest"], use_container_width=True)

    with tabs[4]:
        st.subheader("Выявленные аномальные периоды")
        st.dataframe(art["auto_episodes"], use_container_width=True)

    with tabs[5]:
        st.subheader("Устойчивость к гиперпараметрам +/-20%")
        sens_w = art["sensitivity"].loc[start:end]
        if not sens_w.empty:
            st.line_chart(sens_w, height=400)
        if not art["sensitivity_summary"].empty:
            st.dataframe(art["sensitivity_summary"], use_container_width=True)
        if sens_w.empty and art["sensitivity_summary"].empty:
            st.info("Актуальная чувствительность ещё не рассчитана. Запустите первичную инициализацию или обновление.")

    with tabs[6]:
        st.subheader("Критическое давление (Top-30)")
        st.dataframe(lsi.sort_values("lsi", ascending=False).head(30), use_container_width=True)


if __name__ == "__main__":
    main()
