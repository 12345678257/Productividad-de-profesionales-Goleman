import streamlit as st
import pandas as pd
import numpy as np
import io

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
    layout="wide",
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

    # ----------------------------------------
    # Normalización de fechas (si existen)
    # ----------------------------------------
    for col in ["FECHA ATENCIÓN", "FECHA REGISTRO", "FECHA NACIMIENTO", "FECHA PROGRAMADA"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # ----------------------------------------
    # ID real del paciente -> usamos NUMERO PACIENTE
    # ----------------------------------------
    if "NUMERO PACIENTE" in df.columns:
        df["CEDULA_PACIENTE"] = df["NUMERO PACIENTE"].astype(str).str.strip()
    elif "IDENTIFICACION" in df.columns:
        df["CEDULA_PACIENTE"] = df["IDENTIFICACION"].astype(str).str.strip()
    else:
        df["CEDULA_PACIENTE"] = np.nan

    # ----------------------------------------
    # Profesional (separar ID y nombre)
    # ----------------------------------------
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

    # ----------------------------------------
    # Clasificación de entorno (PPL / Cárceles vs comunidad)
    # ----------------------------------------
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

    if (
        min_date is not None
        and max_date is not None
        and not pd.isna(min_date)
        and not pd.isna(max_date)
    ):
        fecha_ini, fecha_fin = st.sidebar.date_input(
            "Rango de fecha de atención",
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date(),
        )
        mask_fecha = (df["FECHA ATENCIÓN"].dt.date >= fecha_ini) & (
            df["FECHA ATENCIÓN"].dt.date <= fecha_fin
        )
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

    # Filtro por entorno
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

    # Filtro por profesional
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
    pacientes_unicos = df["CEDULA_PACIENTE"].nunique() if "CEDULA_PACIENTE" in df.columns else np.nan
    profesionales_activos = df["ID_PROFESIONAL"].nunique() if "ID_PROFESIONAL" in df.columns else np.nan

    historias_por_paciente = None
    if pacientes_unicos and pacientes_unicos > 0:
        historias_por_paciente = historias_unicas / pacientes_unicos

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Historias únicas (ID ATENCION)", f"{historias_unicas:,}")
    col2.metric(
        "Pacientes únicos (NUMERO PACIENTE)",
        f"{pacientes_unicos:,}" if not np.isnan(pacientes_unicos) else "N/D",
    )
    col3.metric(
        "Profesionales activos",
        f"{profesionales_activos:,}" if not np.isnan(profesionales_activos) else "N/D",
    )

    if historias_por_paciente is not None:
        col4.metric("Historias por paciente", f"{historias_por_paciente:.2f}")
    else:
        col4.metric("Historias por paciente", "N/D")

    if "ID ATENCION" in df.columns:
        duplicados = total_registros - historias_unicas
        st.info(
            f"📌 Registros totales: **{total_registros:,}** · "
            f"Historias únicas: **{historias_unicas:,}** · "
            f"Posibles duplicados de ID ATENCION: **{duplicados:,}**"
        )


def chart_bar_with_labels(
    data: pd.DataFrame,
    x: str,
    y: str,
    color: str = None,
    title: str = "",
    label_col: str = None,
):
    """Gráfico de barras horizontal con etiquetas de datos."""
    if data.empty:
        st.info("No hay datos para mostrar en este gráfico.")
        return

    if label_col is None:
        label_col = x

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
            (base + text).properties(title=title, height=500),
            use_container_width=True,
        )
    else:
        st.write(
            f"*(Altair no está instalado; se muestra gráfico básico de barras para: {title})*"
        )
        st.bar_chart(data.set_index(y)[x])


def chart_pie_with_labels(data: pd.DataFrame, category: str, value: str, title: str = ""):
    """Gráfico de torta (pie) con etiquetas de datos."""
    if data.empty or not ALT_AVAILABLE:
        return

    chart = alt.Chart(data).mark_arc(innerRadius=0).encode(
        theta=alt.Theta(f"{value}:Q"),
        color=alt.Color(f"{category}:N"),
        tooltip=[category, value],
    )

    text = alt.Chart(data).mark_text(radius=80, size=11).encode(
        theta=alt.Theta(f"{value}:Q"),
        text=alt.Text(f"{value}:Q", format=".0f"),
    )

    st.altair_chart(
        (chart + text).properties(title=title, height=400),
        use_container_width=True,
    )


def df_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Datos") -> bytes:
    """Convierte un DataFrame en un archivo Excel en memoria (bytes)."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    output.seek(0)
    return output.getvalue()


# -----------------------------
# Vistas
# -----------------------------
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
            else ("CEDULA_PACIENTE", "size"),
            atenciones=("ID ATENCION", "size"),
            pacientes_unicos=("CEDULA_PACIENTE", "nunique"),
        )
        .reset_index()
    )

    resumen["atenciones_por_paciente"] = np.where(
        resumen["pacientes_unicos"] > 0,
        resumen["atenciones"] / resumen["pacientes_unicos"],
        np.nan,
    )

    top_n = st.slider(
        "Número de profesionales a mostrar (Top N por historias)",
        5,
        50,
        20,
    )
    resumen_top = resumen.sort_values("historias", ascending=False).head(top_n)

    # Etiqueta con historias y pacientes
    resumen_top["lbl_hist_pac"] = resumen_top.apply(
        lambda r: f"H: {int(r['historias'])} | P: {int(r['pacientes_unicos'])}",
        axis=1,
    )

    st.subheader("🏆 Ranking de profesionales")

    chart_bar_with_labels(
        resumen_top,
        x="historias",
        y="NOMBRE_PROFESIONAL" if "NOMBRE_PROFESIONAL" in resumen_top.columns else group_cols[0],
        color="ENTORNO" if "ENTORNO" in resumen_top.columns else None,
        title="Historias únicas por profesional",
        label_col="lbl_hist_pac",
    )

    if ALT_AVAILABLE and "NOMBRE_PROFESIONAL" in resumen_top.columns:
        st.markdown("#### 🥧 Torta de participación de historias (Top N profesionales)")
        chart_pie_with_labels(
            resumen_top,
            category="NOMBRE_PROFESIONAL",
            value="historias",
            title="Distribución de historias en Top profesionales",
        )

    with st.expander("Ver tabla detallada de productividad por profesional"):
        st.dataframe(
            resumen.sort_values("historias", ascending=False),
            use_container_width=True,
        )

        excel_bytes = df_to_excel_bytes(
            resumen.sort_values("historias", ascending=False),
            sheet_name="Profesionales",
        )
        st.download_button(
            "⬇️ Descargar resumen por profesional (XLSX)",
            data=excel_bytes,
            file_name="resumen_productividad_profesional.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    # Pacientes repetidos por profesional
    st.markdown("### 👥 Pacientes repetidos por profesional (más de una atención)")

    if "CEDULA_PACIENTE" in df.columns and "ID_PROFESIONAL" in df.columns:
        rep_group_cols = ["ID_PROFESIONAL"]
        if "NOMBRE_PROFESIONAL" in df.columns:
            rep_group_cols.append("NOMBRE_PROFESIONAL")
        rep_group_cols.append("CEDULA_PACIENTE")

        agg_rep = {
            "atenciones": ("ID ATENCION", "size"),
            "historias": ("ID ATENCION", "nunique"),
        }
        if "NOMBRE PACIENTE" in df.columns:
            agg_rep["NOMBRE PACIENTE"] = ("NOMBRE PACIENTE", "first")

        pacientes_prof = df.groupby(rep_group_cols, dropna=False).agg(**agg_rep).reset_index()
        repetidos = pacientes_prof[pacientes_prof["atenciones"] > 1]

        if repetidos.empty:
            st.info("No se encontraron cédulas repetidas por profesional.")
        else:
            st.dataframe(
                repetidos.sort_values("atenciones", ascending=False),
                use_container_width=True,
            )

            excel_bytes_rep = df_to_excel_bytes(
                repetidos.sort_values("atenciones", ascending=False),
                sheet_name="Pacientes_repetidos",
            )
            st.download_button(
                "⬇️ Descargar pacientes repetidos por profesional (XLSX)",
                data=excel_bytes_rep,
                file_name="pacientes_repetidos_por_profesional.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
    else:
        st.info(
            "No es posible calcular pacientes repetidos: falta CEDULA_PACIENTE o ID_PROFESIONAL."
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
            else ("CEDULA_PACIENTE", "size"),
            atenciones=("ID ATENCION", "size"),
            pacientes_unicos=("CEDULA_PACIENTE", "nunique"),
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
        5,
        30,
        15,
        key="slider_especialidades",
    )
    resumen_top = resumen.sort_values("historias", ascending=False).head(top_n)

    resumen_top["lbl_hist_pac"] = resumen_top.apply(
        lambda r: f"H: {int(r['historias'])} | P: {int(r['pacientes_unicos'])}",
        axis=1,
    )

    chart_bar_with_labels(
        resumen_top,
        x="historias",
        y="ESPECIALIDAD",
        color="ENTORNO" if "ENTORNO" in resumen_top.columns else None,
        title="Historias únicas por especialidad",
        label_col="lbl_hist_pac",
    )

    with st.expander("Ver tabla detallada de productividad por especialidad"):
        st.dataframe(
            resumen.sort_values("historias", ascending=False),
            use_container_width=True,
        )

        excel_bytes = df_to_excel_bytes(
            resumen.sort_values("historias", ascending=False),
            sheet_name="Especialidades",
        )
        st.download_button(
            "⬇️ Descargar resumen por especialidad (XLSX)",
            data=excel_bytes,
            file_name="resumen_productividad_especialidad.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
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
            else ("CEDULA_PACIENTE", "size"),
            atenciones=("ID ATENCION", "size"),
            pacientes_unicos=("CEDULA_PACIENTE", "nunique"),
            profesionales=("ID_PROFESIONAL", "nunique")
            if "ID_PROFESIONAL" in df.columns
            else ("PROFESIONAL ATIENDE", "nunique"),
        )
        .reset_index()
    )

    st.subheader("🌎 Mapeo por ciudad y entorno")
    top_n = st.slider(
        "Número de ciudades a mostrar (Top N por historias)",
        5,
        40,
        20,
        key="slider_ciudades",
    )
    resumen_top = resumen.sort_values("historias", ascending=False).head(top_n)

    resumen_top["lbl_hist_pac"] = resumen_top.apply(
        lambda r: f"H: {int(r['historias'])} | P: {int(r['pacientes_unicos'])}",
        axis=1,
    )

    chart_bar_with_labels(
        resumen_top,
        x="historias",
        y="CIUDAD",
        color="ENTORNO" if "ENTORNO" in resumen_top.columns else None,
        title="Historias únicas por ciudad y entorno",
        label_col="lbl_hist_pac",
    )

    with st.expander("Ver tabla detallada por ciudad"):
        st.dataframe(
            resumen.sort_values("historias", ascending=False),
            use_container_width=True,
        )

        excel_bytes = df_to_excel_bytes(
            resumen.sort_values("historias", ascending=False),
            sheet_name="Ciudades",
        )
        st.download_button(
            "⬇️ Descargar resumen por ciudad (XLSX)",
            data=excel_bytes,
            file_name="resumen_ciudades_entorno.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


def vista_detalle(df: pd.DataFrame):
    """Vista de detalle de registros filtrados."""
    st.subheader("📋 Detalle de historias filtradas")
    st.dataframe(df, use_container_width=True)

    excel_bytes = df_to_excel_bytes(df, sheet_name="Detalle")
    st.download_button(
        "⬇️ Descargar detalle de historias filtradas (XLSX)",
        data=excel_bytes,
        file_name="historias_filtradas.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
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
