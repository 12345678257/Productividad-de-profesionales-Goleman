import streamlit as st
import pandas as pd
import numpy as np

# Intentamos usar Altair para gráficos; si no está disponible, usamos los gráficos nativos de Streamlit
try:
    import altair as alt
    ALT_AVAILABLE = True
except ImportError:
    ALT_AVAILABLE = False


# -----------------------------
# Configuración general de la página
# -----------------------------
st.set_page_config(
    page_title="Productividad Profesionales - Historias Clínicas",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Monitor de Productividad de Profesionales")
st.caption(
    "Dashboard interactivo para analizar historias clínicas por profesional, "
    "especialidad, ciudad y contexto (PPL / comunidad)."
)


# -----------------------------
# Funciones auxiliares
# -----------------------------
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


@st.cache_data
def load_data(file) -> pd.DataFrame:
    """Carga un archivo Excel y aplica limpieza básica de columnas."""
    df = pd.read_excel(file)
    df = clean_columns(df)

    # Conversión de fechas (si existen)
    for col in ["FECHA ATENCIÓN", "FECHA REGISTRO", "FECHA NACIMIENTO", "FECHA PROGRAMADA"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Separar ID y nombre del profesional
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

    # Clasificación de entorno (PPL / Cárceles vs comunidad)
    dir_col = "DIRECCION" if "DIRECCION" in df.columns else None
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


def aplicar_filtros(df: pd.DataFrame) -> pd.DataFrame:
    """Crea filtros en la barra lateral y devuelve el DataFrame filtrado."""
    df = df.copy()

    # Rango de fechas
    if "FECHA ATENCIÓN" in df.columns:
        min_date = df["FECHA ATENCIÓN"].min()
        max_date = df["FECHA ATENCIÓN"].max()
    else:
        min_date = None
        max_date = None

    st.sidebar.header("🔎 Filtros")

    if min_date is not None and max_date is not None and not pd.isna(min_date) and not pd.isna(max_date):
        fecha_ini, fecha_fin = st.sidebar.date_input(
            "Rango de fecha de atención",
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date(),
        )
        # Convertimos a datetime para filtrar
        mask_fecha = (df["FECHA ATENCIÓN"].dt.date >= fecha_ini) & (df["FECHA ATENCIÓN"].dt.date <= fecha_fin)
        df = df[mask_fecha]

    # Filtro por sede
    if "SEDE" in df.columns:
        sedes = sorted(df["SEDE"].dropna().unique().tolist())
        sedes_sel = st.sidebar.multiselect("Sede", sedes, default=sedes)
        if sedes_sel:
            df = df[df["SEDE"].isin(sedes_sel)]

    # Filtro por programa
    if "PROGRAMA" in df.columns:
        programas = sorted(df["PROGRAMA"].dropna().unique().tolist())
        programas_sel = st.sidebar.multiselect("Programa", programas, default=programas)
        if programas_sel:
            df = df[df["PROGRAMA"].isin(programas_sel)]

    # Filtro por ciudad
    if "CIUDAD" in df.columns:
        ciudades = sorted(df["CIUDAD"].dropna().unique().tolist())
        ciudades_sel = st.sidebar.multiselect("Ciudad", ciudades)
        if ciudades_sel:
            df = df[df["CIUDAD"].isin(ciudades_sel)]

    # Filtro por entorno (PPL vs comunidad)
    if "ENTORNO" in df.columns:
        entornos = sorted(df["ENTORNO"].dropna().unique().tolist())
        entornos_sel = st.sidebar.multiselect("Entorno", entornos, default=entornos)
        if entornos_sel:
            df = df[df["ENTORNO"].isin(entornos_sel)]

    # Filtro por especialidad
    if "ESPECIALIDAD" in df.columns:
        especialidades = sorted(df["ESPECIALIDAD"].dropna().unique().tolist())
        especialidades_sel = st.sidebar.multiselect("Especialidad", especialidades)
        if especialidades_sel:
            df = df[df["ESPECIALIDAD"].isin(especialidades_sel)]

    # Filtro por profesional (nombre)
    if "NOMBRE_PROFESIONAL" in df.columns:
        profesionales = sorted(df["NOMBRE_PROFESIONAL"].dropna().unique().tolist())
        profesionales_sel = st.sidebar.multiselect("Profesional", profesionales)
        if profesionales_sel:
            df = df[df["NOMBRE_PROFESIONAL"].isin(profesionales_sel)]

    return df


def mostrar_kpis(df: pd.DataFrame):
    """Muestra tarjetas KPI principales."""
    if df.empty:
        st.warning("No hay datos para los filtros seleccionados.")
        return

    total_registros = len(df)
    historias_unicas = df["ID ATENCION"].nunique() if "ID ATENCION" in df.columns else total_registros
    pacientes_unicos = df["NUMERO PACIENTE"].nunique() if "NUMERO PACIENTE" in df.columns else np.nan
    profesionales_activos = df["ID_PROFESIONAL"].nunique() if "ID_PROFESIONAL" in df.columns else np.nan

    historias_por_paciente = None
    if pacientes_unicos and pacientes_unicos > 0:
        historias_por_paciente = historias_unicas / pacientes_unicos

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Historias únicas (ID ATENCIÓN)", f"{historias_unicas:,}")
    col2.metric(
        "Pacientes únicos valorados",
        f"{pacientes_unicos:,}" if not np.isnan(pacientes_unicos) else "N/D"
    )
    col3.metric(
        "Profesionales activos",
        f"{profesionales_activos:,}" if not np.isnan(profesionales_activos) else "N/D"
    )

    if historias_por_paciente is not None:
        col4.metric("Historias por paciente", f"{historias_por_paciente:.2f}")
    else:
        col4.metric("Historias por paciente", "N/D")

    # Información de calidad de datos: duplicados de ID ATENCION
    if "ID ATENCION" in df.columns:
        duplicados = total_registros - historias_unicas
        st.info(
            f"📌 Registros totales: **{total_registros:,}** · "
            f"Historias únicas: **{historias_unicas:,}** · "
            f"Posibles duplicados de ID ATENCION: **{duplicados:,}**"
        )


def chart_bar_with_labels(data: pd.DataFrame, x: str, y: str, color: str = None, title: str = ""):
    """Gráfico de barras horizontal con etiquetas en el extremo."""
    if data.empty:
        st.info("No hay datos para mostrar en este gráfico.")
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
            text=alt.Text(f"{x}:Q", format=".0f"),
        )

        st.altair_chart(
            (base + text).properties(title=title, height=500),
            use_container_width=True
        )
    else:
        st.write(f"*(Altair no está instalado; se muestra gráfico básico de barras para: {title})*")
        st.bar_chart(data.set_index(y)[x])


def vista_por_profesional(df: pd.DataFrame):
    """Vista de productividad por profesional."""
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

    resumen = (
        df.groupby(group_cols, dropna=False)
        .agg(
            historias=("ID ATENCION", "nunique")
            if "ID ATENCION" in df.columns
            else ("NOMBRE PACIENTE", "size"),
            registros=("ID ATENCION", "size"),
            pacientes_unicos=("NUMERO PACIENTE", "nunique")
            if "NUMERO PACIENTE" in df.columns
            else ("NOMBRE PACIENTE", "nunique"),
        )
        .reset_index()
    )

    # Evitamos división por cero
    resumen["hist_x_paciente"] = np.where(
        resumen["pacientes_unicos"] > 0,
        resumen["historias"] / resumen["pacientes_unicos"],
        np.nan,
    )

    # Ranking top N
    top_n = st.slider(
        "Número de profesionales a mostrar (Top N por historias)",
        5, 50, 20
    )
    resumen_top = resumen.sort_values("historias", ascending=False).head(top_n)

    st.subheader("🏆 Ranking de profesionales por historias únicas")
    chart_bar_with_labels(
        resumen_top,
        x="historias",
        y="NOMBRE_PROFESIONAL" if "NOMBRE_PROFESIONAL" in resumen_top.columns else group_cols[0],
        color="ENTORNO" if "ENTORNO" in resumen_top.columns else None,
        title="Historias únicas por profesional",
    )

    with st.expander("Ver tabla detallada de productividad por profesional"):
        st.dataframe(resumen.sort_values("historias", ascending=False), use_container_width=True)

        csv = resumen.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ Descargar resumen por profesional (CSV)",
            data=csv,
            file_name="resumen_productividad_profesional.csv",
            mime="text/csv",
        )


def vista_por_especialidad(df: pd.DataFrame):
    """Vista de productividad por especialidad."""
    if df.empty:
        st.warning("No hay datos para esta vista.")
        return

    if "ESPECIALIDAD" not in df.columns:
        st.warning("No existe la columna ESPECIALIDAD en el archivo.")
        return

    group_cols = ["ESPECIALIDAD"]
    if "ENTORNO" in df.columns:
        group_cols.append("ENTORNO")

    resumen = (
        df.groupby(group_cols, dropna=False)
        .agg(
            historias=("ID ATENCION", "nunique")
            if "ID ATENCION" in df.columns
            else ("NOMBRE PACIENTE", "size"),
            registros=("ID ATENCION", "size"),
            pacientes_unicos=("NUMERO PACIENTE", "nunique")
            if "NUMERO PACIENTE" in df.columns
            else ("NOMBRE PACIENTE", "nunique"),
            profesionales=("ID_PROFESIONAL", "nunique")
            if "ID_PROFESIONAL" in df.columns
            else ("PROFESIONAL ATIENDE", "nunique"),
        )
        .reset_index()
    )

    resumen["hist_x_paciente"] = np.where(
        resumen["pacientes_unicos"] > 0,
        resumen["historias"] / resumen["pacientes_unicos"],
        np.nan,
    )

    st.subheader("🩺 Productividad por especialidad")
    top_n = st.slider(
        "Número de especialidades a mostrar (Top N por historias)",
        5, 30, 15,
        key="slider_especialidades"
    )
    resumen_top = resumen.sort_values("historias", ascending=False).head(top_n)

    chart_bar_with_labels(
        resumen_top,
        x="historias",
        y="ESPECIALIDAD",
        color="ENTORNO" if "ENTORNO" in resumen_top.columns else None,
        title="Historias únicas por especialidad",
    )

    with st.expander("Ver tabla detallada de productividad por especialidad"):
        st.dataframe(resumen.sort_values("historias", ascending=False), use_container_width=True)
        csv = resumen.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ Descargar resumen por especialidad (CSV)",
            data=csv,
            file_name="resumen_productividad_especialidad.csv",
            mime="text/csv",
        )


def vista_ciudades_carceles(df: pd.DataFrame):
    """Vista por ciudad y entorno (PPL / Cárceles vs comunidad)."""
    if df.empty:
        st.warning("No hay datos para esta vista.")
        return

    if "CIUDAD" not in df.columns:
        st.warning("No existe la columna CIUDAD en el archivo.")
        return

    group_cols = ["CIUDAD"]
    if "ENTORNO" in df.columns:
        group_cols.append("ENTORNO")

    resumen = (
        df.groupby(group_cols, dropna=False)
        .agg(
            historias=("ID ATENCION", "nunique")
            if "ID ATENCION" in df.columns
            else ("NOMBRE PACIENTE", "size"),
            pacientes_unicos=("NUMERO PACIENTE", "nunique")
            if "NUMERO PACIENTE" in df.columns
            else ("NOMBRE PACIENTE", "nunique"),
            profesionales=("ID_PROFESIONAL", "nunique")
            if "ID_PROFESIONAL" in df.columns
            else ("PROFESIONAL ATIENDE", "nunique"),
        )
        .reset_index()
    )

    st.subheader("🌎 Mapeo por ciudad y entorno")
    top_n = st.slider(
        "Número de ciudades a mostrar (Top N por historias)",
        5, 40, 20,
        key="slider_ciudades"
    )
    resumen_top = resumen.sort_values("historias", ascending=False).head(top_n)

    chart_bar_with_labels(
        resumen_top,
        x="historias",
        y="CIUDAD",
        color="ENTORNO" if "ENTORNO" in resumen_top.columns else None,
        title="Historias únicas por ciudad y entorno",
    )

    with st.expander("Ver tabla detallada por ciudad"):
        st.dataframe(resumen.sort_values("historias", ascending=False), use_container_width=True)
        csv = resumen.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ Descargar resumen por ciudad (CSV)",
            data=csv,
            file_name="resumen_ciudades_entorno.csv",
            mime="text/csv",
        )


def vista_detalle(df: pd.DataFrame):
    """Vista de detalle de registros filtrados."""
    st.subheader("📋 Detalle de historias filtradas")
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇️ Descargar detalle de historias filtradas (CSV)",
        data=csv,
        file_name="historias_filtradas.csv",
        mime="text/csv",
    )


# -----------------------------
# Carga del archivo
# -----------------------------
st.sidebar.title("📂 Cargar archivo de historias")
uploaded_file = st.sidebar.file_uploader(
    "Sube el archivo Excel con las historias registradas",
    type=["xlsx", "xls"],
    help="Por ejemplo: 'Historias registradas_Octubre 2025.xlsx'",
)

if uploaded_file is None:
    st.info(
        "Sube en la barra lateral el archivo de historias (formato Excel) "
        "para comenzar a analizar la productividad."
    )
else:
    df_raw = load_data(uploaded_file)
    df_filt = aplicar_filtros(df_raw)

    st.markdown("### 🔍 Resumen general de productividad")
    mostrar_kpis(df_filt)

    # Tabs de análisis
    tab1, tab2, tab3, tab4 = st.tabs(
        ["👨‍⚕️ Por profesional", "🩺 Por especialidad", "🌎 Ciudades y cárceles", "📋 Detalle de datos"]
    )

    with tab1:
        vista_por_profesional(df_filt)

    with tab2:
        vista_por_especialidad(df_filt)

    with tab3:
        vista_ciudades_carceles(df_filt)

    with tab4:
        vista_detalle(df_filt)
