import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import io
import os
from pathlib import Path

# =========================
# Configuración de página
# =========================
st.set_page_config(
    page_title="Productividad Profesionales - Multimensual",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Productividad de Profesionales (Multi-mes)")
st.caption(
    "Carga varios meses, consolida, compara por periodo (profesional y especialidad), "
    "y guarda una base persistente para mantener los análisis."
)

# =========================
# Paths de persistencia
# =========================
BASE_DIR = Path("./data")
BASE_DIR.mkdir(parents=True, exist_ok=True)
PARQUET_PATH = BASE_DIR / "historias_consolidadas.parquet"

# =========================
# Utilidades
# =========================
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres de columnas a MAYÚSCULAS, sin saltos de línea ni espacios duplicados."""
    import re
    df = df.copy()
    new_cols = []
    for c in df.columns:
        c2 = str(c).strip().upper()
        c2 = c2.replace("\n", " ")
        c2 = re.sub(r"\s+", " ", c2)
        new_cols.append(c2)
    df.columns = new_cols
    return df


def infer_period_from_dates(df: pd.DataFrame) -> pd.Series:
    """Devuelve el PERIODO (YYYY-MM) priorizando FECHA ATENCIÓN, luego FECHA REGISTRO."""
    fecha_cols = []
    if "FECHA ATENCIÓN" in df.columns:
        fecha_cols.append("FECHA ATENCIÓN")
    if "FECHA REGISTRO" in df.columns:
        fecha_cols.append("FECHA REGISTRO")

    if not fecha_cols:
        # Sin fechas, inferimos como 'SIN_FECHA'
        return pd.Series(["SIN_FECHA"] * len(df), index=df.index)

    # Elegimos la primera fecha disponible por fila (ATENCIÓN > REGISTRO)
    fecha = None
    for col in fecha_cols:
        col_dt = pd.to_datetime(df[col], errors="coerce")
        if fecha is None:
            fecha = col_dt
        else:
            fecha = fecha.fillna(col_dt)

    # Periodo YYYY-MM
    per = fecha.dt.to_period("M").astype(str)
    per = per.fillna("SIN_FECHA")
    return per


def ensure_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte columnas de fecha si existen."""
    for col in ["FECHA ATENCIÓN", "FECHA REGISTRO", "FECHA NACIMIENTO", "FECHA PROGRAMADA"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def classify_entorno(df: pd.DataFrame) -> pd.DataFrame:
    """Clasifica ENTORNO (PPL/Cárceles vs Comunidad) usando PROGRAMA y DIRECCION/DIRECION."""
    df = df.copy()
    dir_col = "DIRECCION" if "DIRECCION" in df.columns else ("DIRECION" if "DIRECION" in df.columns else None)
    programa_col = "PROGRAMA" if "PROGRAMA" in df.columns else None

    direccion_up = df[dir_col].astype(str).str.upper() if dir_col else pd.Series("", index=df.index)
    programa_up = df[programa_col].astype(str).str.upper() if programa_col else pd.Series("", index=df.index)

    df["ENTORNO"] = np.where(
        programa_up.str.contains("PPL", na=False)
        | direccion_up.str.contains("CPMS", na=False)
        | direccion_up.str.contains("EPMSC", na=False),
        "PPL / Cárceles",
        "Comunidad / Otros",
    )
    return df


def split_profesional(df: pd.DataFrame) -> pd.DataFrame:
    """Separa ID_PROFESIONAL y NOMBRE_PROFESIONAL desde 'PROFESIONAL ATIENDE'."""
    df = df.copy()
    if "PROFESIONAL ATIENDE" in df.columns:
        prof_raw = df["PROFESIONAL ATIENDE"].astype(str).str.split("-", n=1, expand=True)
        df["ID_PROFESIONAL"] = prof_raw[0].str.strip().replace({"nan": np.nan})
        if prof_raw.shape[1] > 1:
            df["NOMBRE_PROFESIONAL"] = prof_raw[1].str.strip()
        else:
            df["NOMBRE_PROFESIONAL"] = df["PROFESIONAL ATIENDE"]
    else:
        df["ID_PROFESIONAL"] = np.nan
        df["NOMBRE_PROFESIONAL"] = np.nan
    return df


def standardize_patient_id(df: pd.DataFrame) -> pd.DataFrame:
    """Define CEDULA_PACIENTE usando NUMERO PACIENTE (fallback IDENTIFICACION)."""
    df = df.copy()
    if "NUMERO PACIENTE" in df.columns:
        df["CEDULA_PACIENTE"] = df["NUMERO PACIENTE"].astype(str).str.strip()
    elif "IDENTIFICACION" in df.columns:
        df["CEDULA_PACIENTE"] = df["IDENTIFICACION"].astype(str).str.strip()
    else:
        df["CEDULA_PACIENTE"] = np.nan
    return df


@st.cache_data(show_spinner=False)
def load_one_excel(file_bytes: bytes) -> pd.DataFrame:
    """Carga un Excel en bytes, limpia columnas y tipa fechas."""
    df = pd.read_excel(file_bytes)
    df = clean_columns(df)
    df = ensure_dates(df)
    df = standardize_patient_id(df)
    df = split_profesional(df)
    df = classify_entorno(df)
    # Periodo
    df["PERIODO"] = infer_period_from_dates(df)
    return df


def load_multiple(files) -> pd.DataFrame:
    """Carga múltiples archivos (uploader) y retorna un solo DF unido."""
    frames = []
    for f in files:
        try:
            df = load_one_excel(f)
            frames.append(df)
        except Exception as e:
            st.warning(f"⚠️ No se pudo leer '{getattr(f, 'name', 'archivo')}' → {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_saved_base() -> pd.DataFrame:
    """Lee la base consolidada guardada (Parquet), si existe."""
    if PARQUET_PATH.exists():
        try:
            df = pd.read_parquet(PARQUET_PATH)
            # Seguridad por si un repo viejo no tenía estas columnas limpias
            df = clean_columns(df)
            df = ensure_dates(df)
            return df
        except Exception as e:
            st.warning(f"No se pudo leer la base guardada: {e}")
    return pd.DataFrame()


def save_base(df: pd.DataFrame):
    """Guarda la base consolidada a Parquet, forzando columnas object (como EPS) a texto."""
    try:
        df2 = df.copy()

        # Forzar todas las columnas tipo object a string (incluye EPS)
        obj_cols = df2.select_dtypes(include=["object"]).columns
        for col in obj_cols:
            df2[col] = df2[col].astype(str)

        # Guardar ya con tipos homogéneos
        df2.to_parquet(PARQUET_PATH, index=False)

        st.success(f"✅ Base consolidada guardada en '{PARQUET_PATH}'.")
    except Exception as e:
        st.error(f"Error guardando base consolidada: {e}")



def merge_and_dedup(df_base: pd.DataFrame, df_new: pd.DataFrame) -> pd.DataFrame:
    """Anexa df_new a df_base y elimina duplicados por ID ATENCION si existe, si no por combinación robusta."""
    if df_base is None or df_base.empty:
        merged = df_new.copy()
    else:
        merged = pd.concat([df_base, df_new], ignore_index=True)

    # Estrategia de duplicados:
    if "ID ATENCION" in merged.columns:
        merged = merged.sort_values(by=["ID ATENCION", "FECHA REGISTRO"], ascending=[True, True], na_position="last")
        merged = merged.drop_duplicates(subset=["ID ATENCION"], keep="last")
    else:
        # Fallback robusto si no hay ID ATENCION
        key_cols = [c for c in ["CEDULA_PACIENTE", "FECHA ATENCIÓN", "ESPECIALIDAD", "NOMBRE_PROFESIONAL"] if c in merged.columns]
        if key_cols:
            merged = merged.drop_duplicates(subset=key_cols, keep="last")
        else:
            merged = merged.drop_duplicates(keep="last")

    return merged.reset_index(drop=True)


def df_to_excel_bytes(dfs: dict) -> bytes:
    """Recibe dict nombre_hoja -> DataFrame, retorna bytes de un .xlsx con múltiples hojas."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, df in dfs.items():
            # Limitar nombre de hoja a 31 caracteres
            sheet = sheet_name[:31]
            df.to_excel(writer, index=False, sheet_name=sheet)
    output.seek(0)
    return output.getvalue()


def apply_common_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica filtros comunes desde la sidebar y retorna el DF filtrado."""
    if df.empty:
        return df

    st.sidebar.header("🔎 Filtros")

    # Periodos
    if "PERIODO" in df.columns:
        periodos = sorted(df["PERIODO"].dropna().unique().tolist())
        sel_per = st.sidebar.multiselect("Periodo (YYYY-MM)", periodos, default=periodos)
        if sel_per:
            df = df[df["PERIODO"].isin(sel_per)]

    # Rango de fecha (opcional)
    if "FECHA ATENCIÓN" in df.columns and df["FECHA ATENCIÓN"].notna().any():
        min_date = df["FECHA ATENCIÓN"].min()
        max_date = df["FECHA ATENCIÓN"].max()
        if pd.notna(min_date) and pd.notna(max_date):
            fecha_ini, fecha_fin = st.sidebar.date_input(
                "Rango de FECHA ATENCIÓN",
                value=(min_date.date(), max_date.date()),
                min_value=min_date.date(),
                max_value=max_date.date(),
            )
            mask_fecha = (df["FECHA ATENCIÓN"].dt.date >= fecha_ini) & (df["FECHA ATENCIÓN"].dt.date <= fecha_fin)
            df = df[mask_fecha]

    # Otros
    for col, label in [
        ("SEDE", "Sede"),
        ("PROGRAMA", "Programa"),
        ("CIUDAD", "Ciudad"),
        ("ENTORNO", "Entorno"),
        ("ESPECIALIDAD", "Especialidad"),
        ("NOMBRE_PROFESIONAL", "Profesional"),
    ]:
        if col in df.columns:
            vals = sorted(df[col].dropna().unique().tolist())
            sel = st.sidebar.multiselect(label, vals)
            if sel:
                df = df[df[col].isin(sel)]

    return df


def show_kpis(df: pd.DataFrame):
    if df.empty:
        st.info("No hay datos con los filtros aplicados.")
        return

    total_registros = len(df)
    historias_unicas = df["ID ATENCION"].nunique() if "ID ATENCION" in df.columns else total_registros
    pacientes_unicos = df["CEDULA_PACIENTE"].nunique() if "CEDULA_PACIENTE" in df.columns else np.nan
    profesionales_activos = df["ID_PROFESIONAL"].nunique() if "ID_PROFESIONAL" in df.columns else np.nan

    historias_por_paciente = None
    if pacientes_unicos and pacientes_unicos > 0:
        historias_por_paciente = historias_unicas / pacientes_unicos

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Historias únicas (ID ATENCION)", f"{historias_unicas:,}")
    c2.metric("Pacientes únicos (NUMERO PACIENTE)", f"{pacientes_unicos:,}" if not np.isnan(pacientes_unicos) else "N/D")
    c3.metric("Profesionales activos", f"{profesionales_activos:,}" if not np.isnan(profesionales_activos) else "N/D")
    c4.metric("Historias por paciente", f"{historias_por_paciente:.2f}" if historias_por_paciente is not None else "N/D")

    if "ID ATENCION" in df.columns:
        duplicados = total_registros - historias_unicas
        st.info(
            f"📌 Registros totales: **{total_registros:,}** · "
            f"Historias únicas: **{historias_unicas:,}** · "
            f"Posibles duplicados por ID ATENCION: **{duplicados:,}**"
        )


def chart_bar_with_labels(data: pd.DataFrame, x: str, y: str, label_col: str, color: str = None, title: str = "", height:int=480):
    if data.empty:
        st.info("No hay datos para este gráfico.")
        return

    base = alt.Chart(data).mark_bar().encode(
        x=alt.X(f"{x}:Q"),
        y=alt.Y(f"{y}:N", sort="-x"),
        tooltip=list(data.columns),
    )
    if color and color in data.columns:
        base = base.encode(color=alt.Color(f"{color}:N"))

    text = alt.Chart(data).mark_text(
        align="left",
        baseline="middle",
        dx=3,
    ).encode(
        x=alt.X(f"{x}:Q"),
        y=alt.Y(f"{y}:N", sort="-x"),
        text=alt.Text(f"{label_col}:N"),
    )

    st.altair_chart((base + text).properties(title=title, height=height), use_container_width=True)


def view_by_profesional(df: pd.DataFrame):
    if df.empty:
        st.warning("No hay datos para esta vista.")
        return

    group_cols = [c for c in ["ID_PROFESIONAL", "NOMBRE_PROFESIONAL", "ESPECIALIDAD", "ENTORNO"] if c in df.columns]
    if not group_cols:
        st.warning("No se encontraron columnas de profesional para agrupar.")
        return

    resumen = (
        df.groupby(group_cols, dropna=False)
        .agg(
            historias=("ID ATENCION", "nunique") if "ID ATENCION" in df.columns else ("CEDULA_PACIENTE", "size"),
            atenciones=("ID ATENCION", "size"),
            pacientes_unicos=("CEDULA_PACIENTE", "nunique"),
        )
        .reset_index()
    )

    top_n = st.slider("Top profesionales por historias", 5, 50, 20)
    resumen_top = resumen.sort_values("historias", ascending=False).head(top_n)
    resumen_top["lbl_hist_pac"] = resumen_top.apply(lambda r: f"H: {int(r['historias'])} | P: {int(r['pacientes_unicos'])}", axis=1)

    st.subheader("🏆 Ranking por profesional")
    chart_bar_with_labels(
        resumen_top,
        x="historias",
        y="NOMBRE_PROFESIONAL" if "NOMBRE_PROFESIONAL" in resumen_top.columns else group_cols[0],
        label_col="lbl_hist_pac",
        color="ENTORNO" if "ENTORNO" in resumen_top.columns else None,
        title="Historias únicas por profesional",
    )

    with st.expander("Tabla y descarga"):
        st.dataframe(resumen.sort_values("historias", ascending=False), use_container_width=True)
        xlsx = df_to_excel_bytes({"Profesionales": resumen.sort_values("historias", ascending=False)})
        st.download_button("⬇️ Descargar resumen por profesional (XLSX)", xlsx, file_name="resumen_profesionales.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # Pacientes repetidos por profesional
    st.markdown("### 👥 Pacientes repetidos por profesional (más de una atención)")
    if {"CEDULA_PACIENTE", "ID_PROFESIONAL"}.issubset(df.columns):
        rep_cols = ["ID_PROFESIONAL"]
        if "NOMBRE_PROFESIONAL" in df.columns:
            rep_cols.append("NOMBRE_PROFESIONAL")
        rep_cols.append("CEDULA_PACIENTE")

        agg_rep = {
            "atenciones": ("ID ATENCION", "size"),
            "historias": ("ID ATENCION", "nunique"),
        }
        if "NOMBRE PACIENTE" in df.columns:
            agg_rep["NOMBRE PACIENTE"] = ("NOMBRE PACIENTE", "first")

        rep = df.groupby(rep_cols, dropna=False).agg(**agg_rep).reset_index()
        rep = rep[rep["atenciones"] > 1]
        if rep.empty:
            st.info("No se encontraron cédulas repetidas por profesional.")
        else:
            st.dataframe(rep.sort_values("atenciones", ascending=False), use_container_width=True)
            xlsx_rep = df_to_excel_bytes({"Pacientes_repetidos": rep.sort_values("atenciones", ascending=False)})
            st.download_button("⬇️ Descargar pacientes repetidos (XLSX)", xlsx_rep,
                               file_name="pacientes_repetidos_por_profesional.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


def view_by_especialidad(df: pd.DataFrame):
    if df.empty:
        st.warning("No hay datos para esta vista.")
        return
    if "ESPECIALIDAD" not in df.columns:
        st.warning("No existe la columna ESPECIALIDAD.")
        return

    group_cols = ["ESPECIALIDAD"] + (["ENTORNO"] if "ENTORNO" in df.columns else [])
    resumen = (
        df.groupby(group_cols, dropna=False)
        .agg(
            historias=("ID ATENCION", "nunique") if "ID ATENCION" in df.columns else ("CEDULA_PACIENTE", "size"),
            atenciones=("ID ATENCION", "size"),
            pacientes_unicos=("CEDULA_PACIENTE", "nunique"),
            profesionales=("ID_PROFESIONAL", "nunique") if "ID_PROFESIONAL" in df.columns else ("PROFESIONAL ATIENDE", "nunique"),
        )
        .reset_index()
    )

    st.subheader("🩺 Productividad por especialidad")
    top_n = st.slider("Top especialidades por historias", 5, 30, 15, key="top_especialidades")
    resumen_top = resumen.sort_values("historias", ascending=False).head(top_n)
    resumen_top["lbl_hist_pac"] = resumen_top.apply(lambda r: f"H: {int(r['historias'])} | P: {int(r['pacientes_unicos'])}", axis=1)

    chart_bar_with_labels(
        resumen_top,
        x="historias",
        y="ESPECIALIDAD",
        label_col="lbl_hist_pac",
        color="ENTORNO" if "ENTORNO" in resumen_top.columns else None,
        title="Historias únicas por especialidad",
    )

    with st.expander("Tabla y descarga"):
        st.dataframe(resumen.sort_values("historias", ascending=False), use_container_width=True)
        xlsx = df_to_excel_bytes({"Especialidades": resumen.sort_values("historias", ascending=False)})
        st.download_button("⬇️ Descargar resumen por especialidad (XLSX)", xlsx, file_name="resumen_especialidades.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


def view_by_ciudad(df: pd.DataFrame):
    if df.empty:
        st.warning("No hay datos para esta vista.")
        return
    if "CIUDAD" not in df.columns:
        st.warning("No existe la columna CIUDAD.")
        return

    group_cols = ["CIUDAD"] + (["ENTORNO"] if "ENTORNO" in df.columns else [])
    resumen = (
        df.groupby(group_cols, dropna=False)
        .agg(
            historias=("ID ATENCION", "nunique") if "ID ATENCION" in df.columns else ("CEDULA_PACIENTE", "size"),
            atenciones=("ID ATENCION", "size"),
            pacientes_unicos=("CEDULA_PACIENTE", "nunique"),
            profesionales=("ID_PROFESIONAL", "nunique") if "ID_PROFESIONAL" in df.columns else ("PROFESIONAL ATIENDE", "nunique"),
        )
        .reset_index()
    )

    st.subheader("🌎 Ciudades y entorno")
    top_n = st.slider("Top ciudades por historias", 5, 40, 20, key="top_ciudades")
    resumen_top = resumen.sort_values("historias", ascending=False).head(top_n)
    resumen_top["lbl_hist_pac"] = resumen_top.apply(lambda r: f"H: {int(r['historias'])} | P: {int(r['pacientes_unicos'])}", axis=1)

    chart_bar_with_labels(
        resumen_top,
        x="historias",
        y="CIUDAD",
        label_col="lbl_hist_pac",
        color="ENTORNO" if "ENTORNO" in resumen_top.columns else None,
        title="Historias únicas por ciudad",
    )

    with st.expander("Tabla y descarga"):
        st.dataframe(resumen.sort_values("historias", ascending=False), use_container_width=True)
        xlsx = df_to_excel_bytes({"Ciudades": resumen.sort_values("historias", ascending=False)})
        st.download_button("⬇️ Descargar resumen por ciudad (XLSX)", xlsx, file_name="resumen_ciudades.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# =========================
# Comparativos mensuales
# =========================
def monthly_by_profesional(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "PERIODO" not in df.columns:
        return pd.DataFrame()
    group = ["PERIODO"]
    for c in ["ID_PROFESIONAL", "NOMBRE_PROFESIONAL"]:
        if c in df.columns:
            group.append(c)
    agg = (
        df.groupby(group, dropna=False)
        .agg(
            historias=("ID ATENCION", "nunique"),
            atenciones=("ID ATENCION", "size"),
            pacientes_unicos=("CEDULA_PACIENTE", "nunique"),
        ).reset_index()
    )
    return agg


def monthly_by_especialidad(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "PERIODO" not in df.columns or "ESPECIALIDAD" not in df.columns:
        return pd.DataFrame()
    group = ["PERIODO", "ESPECIALIDAD"]
    agg = (
        df.groupby(group, dropna=False)
        .agg(
            historias=("ID ATENCION", "nunique"),
            atenciones=("ID ATENCION", "size"),
            pacientes_unicos=("CEDULA_PACIENTE", "nunique"),
        ).reset_index()
    )
    return agg


def view_monthly_comparison(df: pd.DataFrame):
    st.subheader("📆 Comparativo mensual")

    by_prof = monthly_by_profesional(df)
    by_esp = monthly_by_especialidad(df)

    tab_prof, tab_esp = st.tabs(["👨‍⚕️ Por profesional (mensual)", "🩺 Por especialidad (mensual)"])

    with tab_prof:
        if by_prof.empty:
            st.info("No hay datos mensuales por profesional.")
        else:
            # Seleccionar top profesionales por último periodo (por historias)
            ult = by_prof["PERIODO"].max()
            top_candidates = (
                by_prof[by_prof["PERIODO"] == ult]
                .sort_values("historias", ascending=False)
            )
            nombres = sorted(by_prof["NOMBRE_PROFESIONAL"].dropna().unique().tolist()) if "NOMBRE_PROFESIONAL" in by_prof.columns else []
            default_sel = top_candidates["NOMBRE_PROFESIONAL"].head(10).tolist() if "NOMBRE_PROFESIONAL" in top_candidates.columns else []

            sel_pro = st.multiselect("Profesionales a comparar", nombres, default=default_sel)

            metrica = st.selectbox("Métrica", ["historias", "pacientes_unicos", "atenciones"], index=0)
            if sel_pro:
                plot_df = by_prof[by_prof["NOMBRE_PROFESIONAL"].isin(sel_pro)].copy()
            else:
                # Si no selecciona, mostramos top 10 por defecto
                plot_df = by_prof[by_prof["NOMBRE_PROFESIONAL"].isin(default_sel)].copy()

            chart = alt.Chart(plot_df).mark_line(point=True).encode(
                x=alt.X("PERIODO:N", sort=None),
                y=alt.Y(f"{metrica}:Q"),
                color="NOMBRE_PROFESIONAL:N",
                tooltip=list(plot_df.columns),
            ).properties(height=420)
            st.altair_chart(chart, use_container_width=True)

            with st.expander("Tabla y descarga"):
                st.dataframe(by_prof.sort_values(["NOMBRE_PROFESIONAL","PERIODO"]), use_container_width=True)
                xlsx = df_to_excel_bytes({"Profesional_mensual": by_prof})
                st.download_button("⬇️ Descargar comparativo mensual por profesional (XLSX)", xlsx,
                                   file_name="comparativo_mensual_profesional.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with tab_esp:
        if by_esp.empty:
            st.info("No hay datos mensuales por especialidad.")
        else:
            # Seleccionar top especialidades por último periodo
            ult = by_esp["PERIODO"].max()
            top_candidates = (
                by_esp[by_esp["PERIODO"] == ult]
                .sort_values("historias", ascending=False)
            )
            especialidades = sorted(by_esp["ESPECIALIDAD"].dropna().unique().tolist())
            default_sel = top_candidates["ESPECIALIDAD"].head(10).tolist()

            sel_esp = st.multiselect("Especialidades a comparar", especialidades, default=default_sel)
            metrica = st.selectbox("Métrica (esp.)", ["historias", "pacientes_unicos", "atenciones"], index=0, key="met_esp")

            if sel_esp:
                plot_df = by_esp[by_esp["ESPECIALIDAD"].isin(sel_esp)].copy()
            else:
                plot_df = by_esp[by_esp["ESPECIALIDAD"].isin(default_sel)].copy()

            chart = alt.Chart(plot_df).mark_line(point=True).encode(
                x=alt.X("PERIODO:N", sort=None),
                y=alt.Y(f"{metrica}:Q"),
                color="ESPECIALIDAD:N",
                tooltip=list(plot_df.columns),
            ).properties(height=420)
            st.altair_chart(chart, use_container_width=True)

            with st.expander("Tabla y descarga"):
                st.dataframe(by_esp.sort_values(["ESPECIALIDAD","PERIODO"]), use_container_width=True)
                xlsx = df_to_excel_bytes({"Especialidad_mensual": by_esp})
                st.download_button("⬇️ Descargar comparativo mensual por especialidad (XLSX)", xlsx,
                                   file_name="comparativo_mensual_especialidad.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# =========================
# Sidebar: Carga y gestión
# =========================
st.sidebar.header("📂 Cargar varios meses")
files = st.sidebar.file_uploader(
    "Sube uno o más archivos Excel de historias",
    type=["xlsx", "xls"],
    accept_multiple_files=True,
)

col_btn1, col_btn2 = st.sidebar.columns(2)
append_clicked = col_btn1.button("➕ Anexar a base")
replace_clicked = col_btn2.button("🧹 Reemplazar base")

# Base en memoria
if "df_base" not in st.session_state:
    st.session_state.df_base = load_saved_base()

# Previsualización de lo cargado (sin tocar la base)
if files:
    df_preview = load_multiple(files)
    with st.expander("👀 Previsualizar archivos cargados (sin guardar aún)"):
        st.dataframe(df_preview.head(300), use_container_width=True)
        st.caption(f"Filas cargadas: {len(df_preview):,}")
else:
    df_preview = pd.DataFrame()

# Anexar o reemplazar base
if append_clicked and not df_preview.empty:
    st.session_state.df_base = merge_and_dedup(st.session_state.df_base, df_preview)
    save_base(st.session_state.df_base)

if replace_clicked and not df_preview.empty:
    st.session_state.df_base = merge_and_dedup(pd.DataFrame(), df_preview)
    save_base(st.session_state.df_base)

# Gestión de periodos en la base
with st.sidebar.expander("🗂 Gestión de base consolidada"):
    if st.session_state.df_base is not None and not st.session_state.df_base.empty:
        per_list = sorted(st.session_state.df_base["PERIODO"].dropna().unique().tolist()) if "PERIODO" in st.session_state.df_base.columns else []
        st.write(f"Periodos en base: {', '.join(per_list) if per_list else 'N/A'}")
        drop_sel = st.multiselect("Eliminar periodos de la base", per_list)
        if st.button("❌ Eliminar periodos seleccionados"):
            before = len(st.session_state.df_base)
            st.session_state.df_base = st.session_state.df_base[~st.session_state.df_base["PERIODO"].isin(drop_sel)]
            after = len(st.session_state.df_base)
            save_base(st.session_state.df_base)
            st.success(f"Eliminados {before - after} registros de periodos {drop_sel}.")
    else:
        st.info("No hay base consolidada aún. Anexa o reemplaza con archivos.")

    if st.button("🧼 Vaciar base completa"):
        st.session_state.df_base = pd.DataFrame()
        if PARQUET_PATH.exists():
            try:
                os.remove(PARQUET_PATH)
            except Exception:
                pass
        st.success("Base vaciada.")

    if st.session_state.df_base is not None and not st.session_state.df_base.empty:
        xlsx_all = df_to_excel_bytes({"Base_consolidada": st.session_state.df_base})
        st.download_button("⬇️ Descargar base consolidada (XLSX)", xlsx_all, file_name="base_consolidada.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# =========================
# Filtros + vistas
# =========================
df_current = st.session_state.df_base.copy()
df_current = apply_common_filters(df_current)

st.markdown("### 🔍 Resumen general")
show_kpis(df_current)

t1, t2, t3, t4, t5 = st.tabs([
    "👨‍⚕️ Por profesional",
    "🩺 Por especialidad",
    "🌎 Ciudades y cárceles",
    "📆 Comparativo mensual",
    "📋 Detalle"
])

with t1:
    view_by_profesional(df_current)

with t2:
    view_by_especialidad(df_current)

with t3:
    view_by_ciudad(df_current)

with t4:
    view_monthly_comparison(df_current)

with t5:
    st.subheader("📋 Detalle filtrado")
    st.dataframe(df_current, use_container_width=True)
    xlsx_det = df_to_excel_bytes({"Detalle_filtrado": df_current})
    st.download_button("⬇️ Descargar detalle filtrado (XLSX)", xlsx_det, file_name="detalle_filtrado.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

