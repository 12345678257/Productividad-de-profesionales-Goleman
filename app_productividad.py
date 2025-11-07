import streamlit as st
import pandas as pd
import numpy as np
import io

# Intentamos usar Altair para gráficos elegantes
try:
    import altair as alt
    ALT_AVAILABLE = True
except ImportError:
    ALT_AVAILABLE = False

# =========================
# Configuración de página
# =========================
st.set_page_config(
    page_title="Productividad Profesionales - Multi-mes",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Productividad de Profesionales")
st.caption(
    "Dashboard para analizar historias clínicas por profesional, "
    "especialidad, ciudad y periodo (multi-mes)."
)


# =========================
# Utilidades básicas
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


def ensure_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte columnas de fecha a datetime si existen."""
    df = df.copy()
    for col in ["FECHA ATENCIÓN", "FECHA REGISTRO", "FECHA NACIMIENTO", "FECHA PROGRAMADA"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def infer_period(df: pd.DataFrame) -> pd.Series:
    """Devuelve una serie PERIODO (YYYY-MM) usando FECHA ATENCIÓN o FECHA REGISTRO."""
    fecha = None
    if "FECHA ATENCIÓN" in df.columns:
        fecha = pd.to_datetime(df["FECHA ATENCIÓN"], errors="coerce")
    elif "FECHA REGISTRO" in df.columns:
        fecha = pd.to_datetime(df["FECHA REGISTRO"], errors="coerce")

    if fecha is None:
        return pd.Series(["SIN_FECHA"] * len(df), index=df.index)

    periodo = fecha.dt.to_period("M").astype(str)
    periodo = periodo.fillna("SIN_FECHA")
    return periodo


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


def classify_entorno(df: pd.DataFrame) -> pd.DataFrame:
    """Crea ENTORNO (PPL/Cárceles vs Comunidad) usando PROGRAMA + DIRECCION/DIRECION."""
    df = df.copy()
    dir_col = None
    if "DIRECCION" in df.columns:
        dir_col = "DIRECCION"
    elif "DIRECION" in df.columns:
        dir_col = "DIRECION"

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


@st.cache_data
def load_data(file) -> pd.DataFrame:
    """Carga un Excel, lo limpia y añade PERIODO."""
    df = pd.read_excel(file)
    df = clean_columns(df)
    df = ensure_dates(df)
    df = standardize_patient_id(df)
    df = split_profesional(df)
    df = classify_entorno(df)
    df["PERIODO"] = infer_period(df)
    return df


def df_to_excel_bytes(dfs: dict) -> bytes:
    """Dict nombre_hoja -> DataFrame  ->  bytes XLSX."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, df in dfs.items():
            sheet = sheet_name[:31]  # límite Excel
            df.to_excel(writer, index=False, sheet_name=sheet)
    output.seek(0)
    return output.getvalue()


# =========================
# Filtros y KPIs
# =========================
def aplicar_filtros(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica filtros de sidebar (multi-mes, fechas, sedes, etc.)."""
    df = df.copy()
    st.sidebar.header("🔎 Filtros")

    # Periodo (YYYY-MM)
    if "PERIODO" in df.columns:
        periodos = sorted(df["PERIODO"].dropna().unique().tolist())
        per_sel = st.sidebar.multiselect("Periodo (YYYY-MM)", periodos, default=periodos)
        if per_sel:
            df = df[df["PERIODO"].isin(per_sel)]

    # Rango fecha de atención
    if "FECHA ATENCIÓN" in df.columns and df["FECHA ATENCIÓN"].notna().any():
        min_date = df["FECHA ATENCIÓN"].min()
        max_date = df["FECHA ATENCIÓN"].max()
        fecha_ini, fecha_fin = st.sidebar.date_input(
            "Rango de FECHA ATENCIÓN",
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date(),
        )
        mask_fecha = (df["FECHA ATENCIÓN"].dt.date >= fecha_ini) & (df["FECHA ATENCIÓN"].dt.date <= fecha_fin)
        df = df[mask_fecha]

    # Otros filtros categóricos
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


def mostrar_kpis(df: pd.DataFrame):
    if df.empty:
        st.warning("No hay datos para los filtros seleccionados.")
        return

    total_registros = len(df)
    if "ID ATENCION" in df.columns:
        historias_unicas = df["ID ATENCION"].nunique()
    else:
        historias_unicas = total_registros

    pacientes_unicos = df["CEDULA_PACIENTE"].nunique() if "CEDULA_PACIENTE" in df.columns else np.nan
    profesionales_activos = df["ID_PROFESIONAL"].nunique() if "ID_PROFESIONAL" in df.columns else np.nan

    historias_por_paciente = None
    if pacientes_unicos and pacientes_unicos > 0:
        historias_por_paciente = historias_unicas / pacientes_unicos

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Historias únicas (ID ATENCION)", f"{historias_unicas:,}")
    c2.metric(
        "Pacientes únicos (NUMERO PACIENTE)",
        f"{pacientes_unicos:,}" if not np.isnan(pacientes_unicos) else "N/D",
    )
    c3.metric(
        "Profesionales activos",
        f"{profesionales_activos:,}" if not np.isnan(profesionales_activos) else "N/D",
    )
    c4.metric(
        "Historias por paciente",
        f"{historias_por_paciente:.2f}" if historias_por_paciente is not None else "N/D",
    )

    if "ID ATENCION" in df.columns:
        duplicados = total_registros - historias_unicas
        st.info(
            f"📌 Registros totales: **{total_registros:,}** · "
            f"Historias únicas: **{historias_unicas:,}** · "
            f"Posibles duplicados por ID ATENCION: **{duplicados:,}**"
        )


# =========================
# Gráfico de barras
# =========================
def chart_bar_with_labels(
    data: pd.DataFrame,
    x: str,
    y: str,
    label_col: str,
    color: str = None,
    title: str = "",
    height: int = 480,
):
    """Barras horizontales con etiqueta de datos."""
    if data.empty:
        st.info("No hay datos para este gráfico.")
        return

    if ALT_AVAILABLE:
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

        st.altair_chart(
            (base + text).properties(title=title, height=height),
            use_container_width=True,
        )
    else:
        st.write(f"*(Altair no está instalado; se muestra gráfico básico para: {title})*")
        st.bar_chart(data.set_index(y)[x])


# =========================
# Vistas principales
# =========================
def vista_por_profesional(df: pd.DataFrame):
    if df.empty:
        st.warning("No hay datos para esta vista.")
        return

    group_cols = []
    if "ID_PROFESIONAL" in df.columns:
        group_cols.append("ID_PROFESIONAL")
    if "NOMBRE_PROFESIONAL" in df.columns:
        group_cols.append("NOMBRE_PROFESIONAL")
    if "ESPECIALIDAD" in df.columns:
        group_cols.append("ESPECIALIDAD")
    if "ENTORNO" in df.columns:
        group_cols.append("ENTORNO")

    if not group_cols:
        st.warning("No se encontraron columnas de profesional para agrupar.")
        return

    agg_dict = {}
    if "ID ATENCION" in df.columns:
        agg_dict["historias"] = ("ID ATENCION", "nunique")
        agg_dict["atenciones"] = ("ID ATENCION", "size")
    else:
        agg_dict["historias"] = ("CEDULA_PACIENTE", "size")
        agg_dict["atenciones"] = ("CEDULA_PACIENTE", "size")

    if "CEDULA_PACIENTE" in df.columns:
        agg_dict["pacientes_unicos"] = ("CEDULA_PACIENTE", "nunique")

    resumen = df.groupby(group_cols, dropna=False).agg(**agg_dict).reset_index()

    top_n = st.slider("Top profesionales por historias", 5, 50, 20)
    resumen_top = resumen.sort_values("historias", ascending=False).head(top_n)

    if "pacientes_unicos" in resumen_top.columns:
        resumen_top["lbl_hist_pac"] = resumen_top.apply(
            lambda r: f"H: {int(r['historias'])} | P: {int(r['pacientes_unicos'])}",
            axis=1,
        )
    else:
        resumen_top["lbl_hist_pac"] = resumen_top["historias"].astype(int).astype(str)

    st.subheader("🏆 Ranking por profesional")

    y_col = "NOMBRE_PROFESIONAL" if "NOMBRE_PROFESIONAL" in resumen_top.columns else group_cols[0]

    chart_bar_with_labels(
        resumen_top,
        x="historias",
        y=y_col,
        label_col="lbl_hist_pac",
        color="ENTORNO" if "ENTORNO" in resumen_top.columns else None,
        title="Historias únicas por profesional",
    )

    with st.expander("Tabla y descarga"):
        st.dataframe(
            resumen.sort_values("historias", ascending=False),
            use_container_width=True,
        )
        xlsx = df_to_excel_bytes({"Profesionales": resumen.sort_values("historias", ascending=False)})
        st.download_button(
            "⬇️ Descargar resumen por profesional (XLSX)",
            xlsx,
            file_name="resumen_productividad_profesional.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    # Pacientes repetidos por profesional
    st.markdown("### 👥 Pacientes repetidos por profesional (más de una atención)")
    if {"CEDULA_PACIENTE", "ID_PROFESIONAL"}.issubset(df.columns) and "ID ATENCION" in df.columns:
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
            st.dataframe(
                rep.sort_values("atenciones", ascending=False),
                use_container_width=True,
            )
            xlsx_rep = df_to_excel_bytes({"Pacientes_repetidos": rep.sort_values("atenciones", ascending=False)})
            st.download_button(
                "⬇️ Descargar pacientes repetidos (XLSX)",
                xlsx_rep,
                file_name="pacientes_repetidos_por_profesional.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
    else:
        st.info("No es posible calcular pacientes repetidos (falta CEDULA_PACIENTE, ID_PROFESIONAL o ID ATENCION).")


def vista_por_especialidad(df: pd.DataFrame):
    if df.empty:
        st.warning("No hay datos para esta vista.")
        return
    if "ESPECIALIDAD" not in df.columns:
        st.warning("No existe la columna ESPECIALIDAD en el archivo.")
        return

    group_cols = ["ESPECIALIDAD"]
    if "ENTORNO" in df.columns:
        group_cols.append("ENTORNO")

    agg_dict = {}
    if "ID ATENCION" in df.columns:
        agg_dict["historias"] = ("ID ATENCION", "nunique")
        agg_dict["atenciones"] = ("ID ATENCION", "size")
    else:
        agg_dict["historias"] = ("CEDULA_PACIENTE", "size")
        agg_dict["atenciones"] = ("CEDULA_PACIENTE", "size")

    if "CEDULA_PACIENTE" in df.columns:
        agg_dict["pacientes_unicos"] = ("CEDULA_PACIENTE", "nunique")
    if "ID_PROFESIONAL" in df.columns:
        agg_dict["profesionales"] = ("ID_PROFESIONAL", "nunique")

    resumen = df.groupby(group_cols, dropna=False).agg(**agg_dict).reset_index()

    resumen["hist_x_paciente"] = np.where(
        (resumen.get("pacientes_unicos", 0) > 0),
        resumen["historias"] / resumen.get("pacientes_unicos", 1),
        np.nan,
    )

    st.subheader("🩺 Productividad por especialidad")
    top_n = st.slider("Top especialidades por historias", 5, 30, 15, key="slider_especialidades")
    resumen_top = resumen.sort_values("historias", ascending=False).head(top_n)

    if "pacientes_unicos" in resumen_top.columns:
        resumen_top["lbl_hist_pac"] = resumen_top.apply(
            lambda r: f"H: {int(r['historias'])} | P: {int(r['pacientes_unicos'])}",
            axis=1,
        )
    else:
        resumen_top["lbl_hist_pac"] = resumen_top["historias"].astype(int).astype(str)

    chart_bar_with_labels(
        resumen_top,
        x="historias",
        y="ESPECIALIDAD",
        label_col="lbl_hist_pac",
        color="ENTORNO" if "ENTORNO" in resumen_top.columns else None,
        title="Historias únicas por especialidad",
    )

    with st.expander("Tabla y descarga"):
        st.dataframe(
            resumen.sort_values("historias", ascending=False),
            use_container_width=True,
        )
        xlsx = df_to_excel_bytes({"Especialidades": resumen.sort_values("historias", ascending=False)})
        st.download_button(
            "⬇️ Descargar resumen por especialidad (XLSX)",
            xlsx,
            file_name="resumen_productividad_especialidad.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


def vista_ciudades_carceles(df: pd.DataFrame):
    if df.empty:
        st.warning("No hay datos para esta vista.")
        return
    if "CIUDAD" not in df.columns:
        st.warning("No existe la columna CIUDAD en el archivo.")
        return

    group_cols = ["CIUDAD"]
    if "ENTORNO" in df.columns:
        group_cols.append("ENTORNO")

    agg_dict = {}
    if "ID ATENCION" in df.columns:
        agg_dict["historias"] = ("ID ATENCION", "nunique")
        agg_dict["atenciones"] = ("ID ATENCION", "size")
    else:
        agg_dict["historias"] = ("CEDULA_PACIENTE", "size")
        agg_dict["atenciones"] = ("CEDULA_PACIENTE", "size")

    if "CEDULA_PACIENTE" in df.columns:
        agg_dict["pacientes_unicos"] = ("CEDULA_PACIENTE", "nunique")
    if "ID_PROFESIONAL" in df.columns:
        agg_dict["profesionales"] = ("ID_PROFESIONAL", "nunique")

    resumen = df.groupby(group_cols, dropna=False).agg(**agg_dict).reset_index()

    st.subheader("🌎 Mapeo por ciudad y entorno")
    top_n = st.slider("Top ciudades por historias", 5, 40, 20, key="slider_ciudades")
    resumen_top = resumen.sort_values("historias", ascending=False).head(top_n)

    if "pacientes_unicos" in resumen_top.columns:
        resumen_top["lbl_hist_pac"] = resumen_top.apply(
            lambda r: f"H: {int(r['historias'])} | P: {int(r['pacientes_unicos'])}",
            axis=1,
        )
    else:
        resumen_top["lbl_hist_pac"] = resumen_top["historias"].astype(int).astype(str)

    chart_bar_with_labels(
        resumen_top,
        x="historias",
        y="CIUDAD",
        label_col="lbl_hist_pac",
        color="ENTORNO" if "ENTORNO" in resumen_top.columns else None,
        title="Historias únicas por ciudad y entorno",
    )

    with st.expander("Tabla y descarga"):
        st.dataframe(
            resumen.sort_values("historias", ascending=False),
            use_container_width=True,
        )
        xlsx = df_to_excel_bytes({"Ciudades": resumen.sort_values("historias", ascending=False)})
        st.download_button(
            "⬇️ Descargar resumen por ciudad (XLSX)",
            xlsx,
            file_name="resumen_ciudades_entorno.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


# =========================
# Comparativo mensual
# =========================
def comparativo_mensual(df: pd.DataFrame):
    if df.empty or "PERIODO" not in df.columns:
        st.info("No hay datos con PERIODO para comparar.")
        return

    st.subheader("📆 Comparativo mensual")

    tab_prof, tab_esp = st.tabs(
        ["👨‍⚕️ Por profesional (mensual)", "🩺 Por especialidad (mensual)"]
    )

    with tab_prof:
        group_cols = ["PERIODO"]
        if "NOMBRE_PROFESIONAL" in df.columns:
            group_cols.append("NOMBRE_PROFESIONAL")
            nombre_col = "NOMBRE_PROFESIONAL"
        elif "ID_PROFESIONAL" in df.columns:
            group_cols.append("ID_PROFESIONAL")
            nombre_col = "ID_PROFESIONAL"
        else:
            st.info("No hay columnas de profesional para comparar por mes.")
            return

        agg = (
            df.groupby(group_cols, dropna=False)
            .agg(
                historias=("ID ATENCION", "nunique") if "ID ATENCION" in df.columns else ("CEDULA_PACIENTE", "size"),
                atenciones=("ID ATENCION", "size") if "ID ATENCION" in df.columns else ("CEDULA_PACIENTE", "size"),
                pacientes_unicos=("CEDULA_PACIENTE", "nunique") if "CEDULA_PACIENTE" in df.columns else ("ID ATENCION", "nunique"),
            )
            .reset_index()
        )

        ult = agg["PERIODO"].max()
        top_ult = (
            agg[agg["PERIODO"] == ult]
            .sort_values("historias", ascending=False)
            .head(10)
        )
        nombres = sorted(agg[nombre_col].dropna().unique().tolist())
        default_sel = top_ult[nombre_col].tolist()

        sel_prof = st.multiselect("Profesionales a comparar", nombres, default=default_sel)
        metrica = st.selectbox("Métrica", ["historias", "pacientes_unicos", "atenciones"], index=0)

        if sel_prof:
            plot_df = agg[agg[nombre_col].isin(sel_prof)].copy()
        else:
            plot_df = agg[agg[nombre_col].isin(default_sel)].copy()

        if ALT_AVAILABLE:
            chart = (
                alt.Chart(plot_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("PERIODO:N"),
                    y=alt.Y(f"{metrica}:Q"),
                    color=f"{nombre_col}:N",
                    tooltip=list(plot_df.columns),
                )
                .properties(height=420)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Instala 'altair' para ver los gráficos de línea. Se muestra solo tabla.")

        with st.expander("Tabla y descarga"):
            st.dataframe(agg.sort_values([nombre_col, "PERIODO"]), use_container_width=True)
            xlsx = df_to_excel_bytes({"Comparativo_profesional": agg})
            st.download_button(
                "⬇️ Descargar comparativo mensual por profesional (XLSX)",
                xlsx,
                file_name="comparativo_mensual_profesional.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    with tab_esp:
        if "ESPECIALIDAD" not in df.columns:
            st.info("No existe ESPECIALIDAD para comparar.")
            return

        agg = (
            df.groupby(["PERIODO", "ESPECIALIDAD"], dropna=False)
            .agg(
                historias=("ID ATENCION", "nunique") if "ID ATENCION" in df.columns else ("CEDULA_PACIENTE", "size"),
                atenciones=("ID ATENCION", "size") if "ID ATENCION" in df.columns else ("CEDULA_PACIENTE", "size"),
                pacientes_unicos=("CEDULA_PACIENTE", "nunique") if "CEDULA_PACIENTE" in df.columns else ("ID ATENCION", "nunique"),
            )
            .reset_index()
        )

        ult = agg["PERIODO"].max()
        top_ult = (
            agg[agg["PERIODO"] == ult]
            .sort_values("historias", ascending=False)
            .head(10)
        )
        especialidades = sorted(agg["ESPECIALIDAD"].dropna().unique().tolist())
        default_sel = top_ult["ESPECIALIDAD"].tolist()

        sel_esp = st.multiselect("Especialidades a comparar", especialidades, default=default_sel)
        metrica_esp = st.selectbox(
            "Métrica (especialidad)",
            ["historias", "pacientes_unicos", "atenciones"],
            index=0,
            key="metrica_esp",
        )

        if sel_esp:
            plot_df = agg[agg["ESPECIALIDAD"].isin(sel_esp)].copy()
        else:
            plot_df = agg[agg["ESPECIALIDAD"].isin(default_sel)].copy()

        if ALT_AVAILABLE:
            chart = (
                alt.Chart(plot_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("PERIODO:N"),
                    y=alt.Y(f"{metrica_esp}:Q"),
                    color="ESPECIALIDAD:N",
                    tooltip=list(plot_df.columns),
                )
                .properties(height=420)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Instala 'altair' para ver los gráficos de línea. Se muestra solo tabla.")

        with st.expander("Tabla y descarga"):
            st.dataframe(agg.sort_values(["ESPECIALIDAD", "PERIODO"]), use_container_width=True)
            xlsx = df_to_excel_bytes({"Comparativo_especialidad": agg})
            st.download_button(
                "⬇️ Descargar comparativo mensual por especialidad (XLSX)",
                xlsx,
                file_name="comparativo_mensual_especialidad.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )


# =========================
# Carga de archivos
# =========================
st.sidebar.title("📂 Cargar historias (multi-mes)")
uploaded_files = st.sidebar.file_uploader(
    "Sube uno o varios archivos Excel con las historias registradas",
    type=["xlsx", "xls"],
    accept_multiple_files=True,
    help="Puedes cargar varios meses a la vez.",
)

if not uploaded_files:
    st.info(
        "Sube al menos un archivo Excel con las historias (por ejemplo: "
        "'Historias registradas_Octubre 2025.xlsx')."
    )
else:
    # Unimos todos los meses cargados
    dfs = [load_data(f) for f in uploaded_files]
    df_raw = pd.concat(dfs, ignore_index=True)

    df_filt = aplicar_filtros(df_raw)

    st.markdown("### 🔍 Resumen general de productividad")
    mostrar_kpis(df_filt)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "👨‍⚕️ Por profesional",
            "🩺 Por especialidad",
            "🌎 Ciudades y cárceles",
            "📆 Comparativo mensual",
            "📋 Detalle de datos",
        ]
    )

    with tab1:
        vista_por_profesional(df_filt)

    with tab2:
        vista_por_especialidad(df_filt)

    with tab3:
        vista_ciudades_carceles(df_filt)

    with tab4:
        comparativo_mensual(df_filt)

    with tab5:
        st.subheader("📋 Detalle filtrado")
        st.dataframe(df_filt, use_container_width=True)
        xlsx_det = df_to_excel_bytes({"Detalle_filtrado": df_filt})
        st.download_button(
            "⬇️ Descargar detalle filtrado (XLSX)",
            xlsx_det,
            file_name="detalle_filtrado.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
