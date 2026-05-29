import streamlit as st
import pandas as pd
import json
import sys
from pathlib import Path
from typing import Dict

# --- ИМПОРТЫ ПРОЕКТА ---
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from src.ru_liquidity_sentinel.config import LSI_THRESHOLDS, PipelineConfig
from src.ru_liquidity_sentinel.pipeline import run_pipeline
from src.ru_liquidity_sentinel.reporting import write_all_reports

# --- КОНФИГУРАЦИЯ СТРАНИЦЫ ---
st.set_page_config(
    page_title="RU Liquidity Sentinel",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- КАСТОМНЫЙ CSS ДЛЯ ТЕМНОЙ ТЕМЫ И АНИМАЦИИ ---
st.markdown("""
<style> 
    .stApp { background-color: #0E1117; color: #FAFAFA; } 
    [data-testid="stMetricValue"] { font-size: 28px; color: #00D4FF; } 

    /* Стилизация кнопок */ 
    .stButton>button { 
        width: 100%; border-radius: 10px; border: 1px solid #00D4FF; 
        background-color: #161B22; color: #00D4FF; font-weight: bold; transition: 0.3s; 
    } 
    .stButton>button:hover { background-color: #00D4FF; color: #0E1117; border: 1px solid #00D4FF; }

    /* Прозрачность графиков */ 
    [data-testid="stDataFrame"], [data-testid="stTable"],
    [data-testid="stArrowDataTable"], [data-testid="stVegaLiteChart"],
    div[data-testid="stVizContainer"], .stPlotlyChart {
        background-color: transparent !important;
    }

    /* Смягчаем линии сетки */
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

    /* Цветовые стили для статусов (Markdown) */
    .status-red { color: #FF4B4B; font-weight: bold; border: 1px solid #FF4B4B; padding: 2px 8px; border-radius: 5px; }
    .status-yellow { color: #FFD700; font-weight: bold; border: 1px solid #FFD700; padding: 2px 8px; border-radius: 5px; }
    .status-green { color: #00FF7F; font-weight: bold; border: 1px solid #00FF7F; padding: 2px 8px; border-radius: 5px; }

    /* Анимация появления */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    [data-testid="stVerticalBlock"] > div {
        animation: fadeIn 0.5s ease-out forwards;
    }

    .block-container { padding-top: 2rem; } 
</style> """, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_artifacts(artifacts_dir: Path) -> dict:
    feat = pd.read_parquet(artifacts_dir / "features_daily.parquet")
    lsi = pd.read_parquet(artifacts_dir / "lsi_daily.parquet")
    target = pd.read_parquet(artifacts_dir / "proxy_target.parquet")["proxy_stress"]
    sens = pd.read_parquet(artifacts_dir / "sensitivity.parquet")
    _p_sens_sum = artifacts_dir / "sensitivity_summary.csv"
    if _p_sens_sum.exists() and _p_sens_sum.stat().st_size > 0:
        try:
            sens_summary = pd.read_csv(_p_sens_sum)
        except Exception:
            sens_summary = pd.DataFrame()
    else:
        sens_summary = pd.DataFrame()
    backtest = pd.read_csv(artifacts_dir / "backtest_episodes.csv")
    metadata = json.loads((artifacts_dir / "metadata.json").read_text())

    auto_episodes = pd.DataFrame()
    p_auto = artifacts_dir / "auto_episodes.csv"
    if p_auto.exists() and p_auto.stat().st_size > 0:
        try:
            auto_episodes = pd.read_csv(p_auto, parse_dates=["start", "end"])
        except Exception:
            auto_episodes = pd.DataFrame()

    model_predictions = pd.DataFrame()
    p_pred = artifacts_dir / "model_predictions.parquet"
    if p_pred.exists() and p_pred.stat().st_size > 0:
        try:
            model_predictions = pd.read_parquet(p_pred)
        except Exception:
            model_predictions = pd.DataFrame()

    model_metrics = pd.read_csv(artifacts_dir / "model_metrics.csv") if (
            artifacts_dir / "model_metrics.csv").exists() else pd.DataFrame()
    cv_results = pd.read_csv(artifacts_dir / "model_cv_results.csv") if (
            artifacts_dir / "model_cv_results.csv").exists() else pd.DataFrame()
    best_params = pd.read_csv(artifacts_dir / "model_best_params.csv") if (
            artifacts_dir / "model_best_params.csv").exists() else pd.DataFrame()
    noise = pd.read_csv(artifacts_dir / "noise_breakdown.csv") if (
            artifacts_dir / "noise_breakdown.csv").exists() else pd.DataFrame()

    model_attributions: Dict[str, pd.DataFrame] = {}
    p_attr = artifacts_dir / "model_nsvm_module_attribution.parquet"
    if p_attr.exists():
        model_attributions["nsvm"] = pd.read_parquet(p_attr)

    return {
        "features": feat, "lsi": lsi, "target": target, "sensitivity": sens,
        "sensitivity_summary": sens_summary, "backtest": backtest,
        "auto_episodes": auto_episodes, "metadata": metadata,
        "model_predictions": model_predictions, "model_metrics": model_metrics,
        "model_cv_results": cv_results, "model_best_params": best_params,
        "noise_breakdown": noise, "model_attributions": model_attributions,
    }


def render_status(value: float):
    """Отрисовка стилизованного статуса"""
    if value is None or pd.isna(value):
        st.markdown("—")
    elif value <= LSI_THRESHOLDS["green_max"]:
        st.markdown(f'<span class="status-green">🟢 GREEN</span>', unsafe_allow_html=True)
    elif value <= LSI_THRESHOLDS["yellow_max"]:
        st.markdown(f'<span class="status-yellow">🟡 YELLOW</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="status-red">🔴 RED</span>', unsafe_allow_html=True)


def main() -> None:
    cfg = PipelineConfig()
    cfg.ensure_dirs()
    artifacts_dir = cfg.artifacts_dir

    st.sidebar.title("💎 Sentinel Control")
    st.sidebar.markdown("**ПСБ Казначейство**")

    if st.sidebar.button("⏵ ОБНОВИТЬ ДАННЫЕ"):
        with st.spinner("Пересчет пайплайна..."):
            run = run_pipeline(cfg)
            write_all_reports(run, cfg)
            st.cache_data.clear()
        st.sidebar.success("Готово.")

    if not (artifacts_dir / "lsi_daily.parquet").exists():
        st.warning("Артефакты не найдены. Запустите пайплайн.")
        return

    art = load_artifacts(artifacts_dir)
    lsi = art["lsi"]
    feat = art["features"]
    last_date = lsi.index.max()
    last_row = lsi.loc[last_date]

    st.title("Система раннего предупреждения стресса ликвидности")

    # Сетка метрик
    m_cols = st.columns(3)
    with m_cols[0]:
        st.metric("Дата", last_date.date().isoformat())
    with m_cols[1]:
        st.metric("LSI Score", f"{last_row['lsi']:.1f}")
    with m_cols[2]:
        st.write("Статус")
        render_status(last_row['lsi'])

    st.markdown("---")

    min_date, max_date = lsi.index.min().date(), lsi.index.max().date()
    rng = st.sidebar.date_input(
        "Интервал анализа",
        value=(max(min_date, max_date.replace(year=max_date.year - 2)), max_date)
    )

    if isinstance(rng, tuple) and len(rng) == 2:
        start, end = pd.Timestamp(rng[0]), pd.Timestamp(rng[1])
    else:
        start, end = pd.Timestamp(min_date), pd.Timestamp(max_date)

    lsi_w = lsi.loc[start:end]
    feat_w = feat.loc[start:end]

    # Список вкладок
    tabs = st.tabs(["LSI", "Модули", "Модели", "Backtest", "Авто-эпизоды", "Чувствительность", "Алерты"])

    with tabs[0]:
        st.subheader("Линейка индекса (NSVM)")  # Исправили заголовок
        st.line_chart(lsi_w[["lsi"]].rename(columns={"lsi": "LSI Index"}), color="#00D4FF", height=350)

        # Берем данные о вкладе (атрибуции) напрямую из загруженных артефактов NSVM
        if "nsvm" in art["model_attributions"] and not art["model_attributions"]["nsvm"].empty:
            attr_df = art["model_attributions"]["nsvm"].loc[start:end]
            contrib_cols = [c for c in attr_df.columns if c.startswith("contrib_")]

            if contrib_cols:
                st.subheader("Структура вклада модулей (SHAP GradientExplainer)")
                st.area_chart(attr_df[contrib_cols], height=300)
        else:
            # Fallback на случай, если артефакт не загрузился, ищем внутри самого lsi
            contrib_cols = [c for c in lsi_w.columns if c.startswith("contrib_")]
            if contrib_cols:
                st.subheader("Структура вклада модулей (SHAP GradientExplainer)")
                st.area_chart(lsi_w[contrib_cols], height=300)

    with tabs[1]:
        # Отображаем графики для всех модулей, кроме M4
        for module in ("M1", "M2", "M3", "M4", "M5"):
            if module == "M4":
                continue
            mod_lower = module.lower()
            mad_cols = [c for c in feat_w.columns if c.startswith(f"{mod_lower}_mad_")]
            if mad_cols:
                st.markdown(f"**Анализ модуля {module}**")
                st.line_chart(feat_w[mad_cols], height=200)

    with tabs[2]:
        st.subheader("Метрики агрегатора")
        if not art["model_metrics"].empty:
            st.dataframe(art["model_metrics"].style.format(precision=3))
        if not art["model_predictions"].empty:
            st.line_chart(art["model_predictions"].loc[start:end][["nsvm"]], color="#FF4B4B")

    with tabs[3]:
        st.subheader("Историческая верификация")
        st.dataframe(art["backtest"], use_container_width=True)

    with tabs[4]:
        st.subheader("Выявленные аномальные периоды")
        st.dataframe(art["auto_episodes"], use_container_width=True)

    with tabs[5]:
        st.subheader("Устойчивость к гиперпараметрам ±20%")
        sens_w = art["sensitivity"].loc[start:end]
        if not sens_w.empty:
            st.line_chart(sens_w, height=400)

    with tabs[6]:
        st.subheader("Критическое давление (Top-30)")
        st.dataframe(lsi.sort_values("lsi", ascending=False).head(30), use_container_width=True)


if __name__ == "__main__":
    main()