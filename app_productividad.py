import streamlit as st
import pandas as pd
import numpy as np
import io
import sqlite3
from pathlib import Path

# =========================
# Opcional: Altair para gráficos
# =========================
try:
    import altair as alt
    ALT_AVAILABLE = True
except Exception:
    ALT_AVAILABLE = False

# =========================
# Config de página
# =========================
st.set_page_config(
    page_title="Productividad Profesionales - Multi-mes (SQLite)",
    page_icon="📊",
    layout="wide",
)
st.title("📊 Productividad de Profesionales")
st.caption(
    "Base persistente en SQLite, filtros por EPS y más; KPIs, gráficas, tablas dinámicas y comparativos mensuales."
)

# =========================
# SQLite: paths y helpers
# =========================
DB_DIR = Path("data")
DB_DIR.mkdir(exist_ok=True)
DB_PATH = DB_DIR / "historias.db"

def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS historias (
            id_pk TEXT PRIMARY KEY,
            id_atencion TEXT,
            fecha_atencion TEXT,
            fecha_registro TEXT,
            numero_paciente TEXT,
            identificacion TEXT,
            nombre_paciente TEXT,
            profesional_atiende TEXT,
            id_profesional TEXT,
            nombre_profesional TEXT,
            sede TEXT,
            programa TEXT,
            ciudad TEXT,
            direccion TEXT,
            eps TEXT,
            especialidad TEXT,
            entorno TEXT,
            periodo TEXT,
            cedula_paciente TEXT
        )
    """)
    conn.commit()
    conn.close()

def clear_db():
    conn = get_connection()
    conn.execute("DELETE FROM historias")
    conn.commit()
    conn.close()

def load_all_from_db() -> pd.DataFrame:
    conn = get_connection()
    try:
        df = pd.read_sql_query("SELECT * FROM historias", conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()

    if df.empty:
        return df

    # Renombrar a nombres de análisis
    df = df.rename(columns={
        "id_atencion":"ID ATENCION",
        "fecha_atencion":"FECHA ATENCIÓN",
        "fecha_registro":"FECHA REGISTRO",
        "numero_paciente":"NUMERO PACIENTE",
        "identificacion":"IDENTIFICACION",
        "nombre_paciente":"NOMBRE PACIENTE",
        "profesional_atiende":"PROFESIONAL ATIENDE",
        "id_profesional":"ID_PROFESIONAL",
        "nombre_profesional":"NOMBRE_PROFESIONAL",
        "sede":"SEDE",
        "programa":"PROGRAMA",
        "ciudad":"CIUDAD",
        "direccion":"DIRECCION",
        "eps":"EPS",
        "especialidad":"ESPECIALIDAD",
        "entorno":"ENTORNO",
        "periodo":"PERIODO",
        "cedula_paciente":"CEDULA_PACIENTE",
    })
    # Fechas
    for col in ["FECHA ATENCIÓN", "FECHA REGISTRO"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def make_str_or_none(v):
    if pd.isna(v) or v is None:
        return None
    return str(v)

def upsert_df_to_db(df: pd.DataFrame):
    """Inserta/actualiza filas en SQLite. No borra. ON CONFLICT(id_pk) DO UPDATE."""
    if df.empty:
        return

    # Normalizar columnas requeridas
    needed = [
        "ID ATENCION","FECHA ATENCIÓN","FECHA REGISTRO","NUMERO PACIENTE","IDENTIFICACION",
        "NOMBRE PACIENTE","PROFESIONAL ATIENDE","ID_PROFESIONAL","NOMBRE_PROFESIONAL","SEDE",
        "PROGRAMA","CIUDAD","DIRECCION","EPS","ESPECIALIDAD","ENTORNO","PERIODO","CEDULA_PACIENTE"
    ]
    df = df.copy()
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan

    records = []
    for _, row in df.iterrows():
        # PK por ID ATENCION; si no existe, usar llave compuesta estable
        id_at = str(row.get("ID ATENCION", "") or "").strip()
        if id_at:
            id_pk = id_at
        else:
            parts = [
                str(row.get("CEDULA_PACIENTE","") or "").strip(),
                str(row.get("FECHA ATENCIÓN","") or "").strip(),
                str(row.get("ESPECIALIDAD","") or "").strip(),
                str(row.get("NOMBRE_PROFESIONAL","") or "").strip(),
                str(row.get("PROGRAMA","") or "").strip(),
                str(row.get("SEDE","") or "").strip(),
            ]
            id_pk = "|".join(parts)

        # fechas a texto ISO (YYYY-MM-DD)
        def ts2str(x):
            if pd.isna(x) or x is None:
                return None
            if isinstance(x, (pd.Timestamp,)):
                return x.strftime("%Y-%m-%d")
            return str(x)

        rec = (
            id_pk,
            make_str_or_none(row["ID ATENCION"]),
            ts2str(row["FECHA ATENCIÓN"]),
            ts2str(row["FECHA REGISTRO"]),
            make_str_or_none(row["NUMERO PACIENTE"]),
            make_str_or_none(row["IDENTIFICACION"]),
            make_str_or_none(row["NOMBRE PACIENTE"]),
            make_str_or_none(row["PROFESIONAL ATIENDE"]),
            make_str_or_none(row["ID_PROFESIONAL"]),
            make_str_or_none(row["NOMBRE_PROFESIONAL"]),
            make_str_or_none(row["SEDE"]),
            make_str_or_none(row["PROGRAMA"]),
            make_str_or_none(row["CIUDAD"]),
            make_str_or_none(row["DIRECCION"]),
            make_str_or_none(row["EPS"]),
            make_str_or_none(row["ESPECIALIDAD"]),
            make_str_or_none(row["ENTORNO"]),
            make_str_or_none(row["PERIODO"]),
            make_str_or_none(row["CEDULA_PACIENTE"]),
        )
        records.append(rec)

    conn = get_connection()
    cur = conn.cursor()
    cur.executemany("""
        INSERT INTO historias (
            id_pk, id_atencion, fecha_atencion, fecha_registro, numero_paciente,
            identificacion, nombre_paciente, profesional_atiende, id_profesional,
            nombre_profesional, sede, programa, ciudad, direccion, eps,
            especialidad, entorno, periodo, cedula_paciente
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(id_pk) DO UPDATE SET
            id_atencion=excluded.id_atencion,
            fecha_atencion=excluded.fecha_atencion,
            fecha_registro=excluded.fecha_registro,
            numero_paciente=excluded.numero_paciente,
            identificacion=excluded.identificacion,
            nombre_paciente=excluded.nombre_paciente,
            profesional_atiende=excluded.profesional_atiende,
            id_profesional=excluded.id_profesional,
            nombre_profesional=excluded.nombre_profesional,
            sede=excluded.sede,
            programa=excluded.programa,
            ciudad=excluded.ciudad,
            direccion=excluded.direccion,
            eps=excluded.eps,
            especialidad=excluded.especialidad,
            entorno=excluded.entorno,
            periodo=excluded.periodo,
            cedula_paciente=excluded.cedula_paciente
    """, records)
    conn.commit()
    conn.close()

# =========================
# Limpieza y features
# =========================
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    import re
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.upper()
        .str.replace("\n", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
    )
    return df

def ensure_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ["FECHA ATENCIÓN","FECHA REGISTRO","FECHA NACIMIENTO","FECHA PROGRAMADA"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def infer_period(df: pd.DataFrame) -> pd.Series:
    fecha = None
    if "FECHA ATENCIÓN" in df.columns:
        fecha = pd.to_datetime(df["FECHA ATENCIÓN"], errors="coerce")
    elif "FECHA REGISTRO" in df.columns:
        fecha = pd.to_datetime(df["FECHA REGISTRO"], errors="coerce")
    if fecha is None:
        return pd.Series(["SIN_FECHA"] * len(df), index=df.index)
    return fecha.dt.to_period("M").astype(str).fillna("SIN_FECHA")

def standardize_patient_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "NUMERO PACIENTE" in df.columns:
        df["CEDULA_PACIENTE"] = df["NUMERO PACIENTE"].astype(str).str.strip()
    elif "IDENTIFICACION" in df.columns:
        df["CEDULA_PACIENTE"] = df["IDENTIFICACION"].astype(str).str.strip()
    else:
        df["CEDULA_PACIENTE"] = np.nan
    return df

def split_profesional(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "PROFESIONAL ATIENDE" in df.columns:
        tmp = df["PROFESIONAL ATIENDE"].astype(str).str.split("-", n=1, expand=True)
        df["ID_PROFESIONAL"] = tmp[0].str.strip().replace({"nan": np.nan})
        df["NOMBRE_PROFESIONAL"] = (
            tmp[1].str.strip() if tmp.shape[1] > 1 else df["PROFESIONAL ATIENDE"]
        )
    else:
        df["ID_PROFESIONAL"] = np.nan
        df["NOMBRE_PROFESIONAL"] = np.nan
    return df

def classify_entorno(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dcol = "DIRECCION" if "DIRECCION" in df.columns else ("DIRECION" if "DIRECION" in df.columns else None)
    pcol = "PROGRAMA" if "PROGRAMA" in df.columns else None
    direccion_up = df[dcol].astype(str).str.upper() if dcol else pd.Series("", index=df.index)
    programa_up = df[pcol].astype(str).str.upper() if pcol else pd.Series("", index=df.index)
    df["ENTORNO"] = np.where(
        programa_up.str.contains("PPL", na=False)
        | direccion_up.str.contains("CPMS", na=False)
        | direccion_up.str.contains("EPMSC", na=False),
        "PPL / Cárceles",
        "Comunidad / Otros"
    )
    return df

@st.cache_data
def load_data(file) -> pd.DataFrame:
    df = pd.read_excel(file)
    df = clean_columns(df)
    df = ensure_dates(df)
    df = standardize_patient_id(df)
    df = split_profesional(df)
    df = classify_entorno(df)
    if "PERIODO" not in df.columns:
        df["PERIODO"] = infer_period(df)
    return df

def df_to_excel_bytes(dfs: dict) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet, data in dfs.items():
            data.to_excel(writer, index=False, sheet_name=sheet[:31])
    output.seek(0)
    return output.getvalue()

# =========================
# Filtros (incluye EPS)
# =========================
def aplicar_filtros(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    st.sidebar.header("🔎 Filtros")

    # Periodo (YYYY-MM)
    if "PERIODO" in df.columns:
        periodos = sorted(df["PERIODO"].dropna().unique().tolist())
        per_sel = st.sidebar.multiselect(
            "Periodo (YYYY-MM)", periodos, default=periodos, key="f_periodo"
        )
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
            key="f_rango_fecha",
        )
        mask_fecha = (df["FECHA ATENCIÓN"].dt.date >= fecha_ini) & (df["FECHA ATENCIÓN"].dt.date <= fecha_fin)
        df = df[mask_fecha]

    # Otros filtros categóricos (cada uno con key única)
    for col, label in [
        ("EPS", "EPS"),
        ("SEDE", "Sede"),
        ("PROGRAMA", "Programa"),
        ("CIUDAD", "Ciudad"),
        ("ENTORNO", "Entorno"),
        ("ESPECIALIDAD", "Especialidad"),
        ("NOMBRE_PROFESIONAL", "Profesional"),
    ]:
        if col in df.columns:
            vals = sorted([v for v in df[col].dropna().unique().tolist() if str(v).strip() != ""])
            sel = st.sidebar.multiselect(label, vals, key=f"f_{col.lower()}")
            if sel:
                df = df[df[col].isin(sel)]

    return df


# =========================
# KPIs
# =========================
def mostrar_kpis(df: pd.DataFrame):
    if df.empty:
        st.warning("No hay datos para los filtros.")
        return
    total_registros = len(df)
    historias = df["ID ATENCION"].nunique() if "ID ATENCION" in df.columns else total_registros
    pacientes = df["CEDULA_PACIENTE"].nunique() if "CEDULA_PACIENTE" in df.columns else np.nan
    profesionales = df["ID_PROFESIONAL"].nunique() if "ID_PROFESIONAL" in df.columns else np.nan
    eps_unicas = df["EPS"].nunique() if "EPS" in df.columns else np.nan
    historias_pac = (historias / pacientes) if pacientes and pacientes > 0 else np.nan

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Historias únicas (ID ATENCION)", f"{historias:,}")
    c2.metric("Pacientes únicos (NUMERO PACIENTE)", f"{pacientes:,}" if not np.isnan(pacientes) else "N/D")
    c3.metric("Profesionales activos", f"{profesionales:,}" if not np.isnan(profesionales) else "N/D")
    c4.metric("Historias por paciente", f"{historias_pac:.2f}" if not np.isnan(historias_pac) else "N/D")
    c5.metric("EPS únicas", f"{eps_unicas:,}" if not np.isnan(eps_unicas) else "N/D")

    if "ID ATENCION" in df.columns:
        duplicados = total_registros - historias
        st.info(
            f"📌 Registros: **{total_registros:,}** · "
            f"Historias únicas: **{historias:,}** · "
            f"Posibles duplicados (ID ATENCION): **{duplicados:,}**"
        )

def chart_barras(data: pd.DataFrame, x: str, y: str, label_col: str, color: str|None, title: str, height: int=460):
    if data.empty:
        st.info("Sin datos para el gráfico.")
        return
    if not ALT_AVAILABLE:
        st.bar_chart(data.set_index(y)[x])
        return
    base = alt.Chart(data).mark_bar().encode(
        x=alt.X(f"{x}:Q"),
        y=alt.Y(f"{y}:N", sort="-x"),
        tooltip=list(data.columns),
    )
    if color and color in data.columns:
        base = base.encode(color=alt.Color(f"{color}:N"))
    labels = alt.Chart(data).mark_text(align="left", baseline="middle", dx=3).encode(
        x=alt.X(f"{x}:Q"),
        y=alt.Y(f"{y}:N", sort="-x"),
        text=alt.Text(f"{label_col}:N"),
    )
    st.altair_chart((base + labels).properties(title=title, height=height), use_container_width=True)

def chart_torta(data: pd.DataFrame, cat_col: str, value_col: str, title: str, height:int=420):
    if data.empty or not ALT_AVAILABLE:
        st.dataframe(data, use_container_width=True)
        return
    chart = alt.Chart(data).mark_arc(innerRadius=60).encode(
        theta=alt.Theta(f"{value_col}:Q"),
        color=alt.Color(f"{cat_col}:N", legend=alt.Legend(title=cat_col)),
        tooltip=list(data.columns),
    ).properties(title=title, height=height)
    st.altair_chart(chart, use_container_width=True)

# =========================
# Vistas
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

    top_n = st.slider("Top profesionales por historias", 5, 50, 20, key="sl_top_prof")
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


def vista_por_especialidad(df: pd.DataFrame):
    if df.empty or "ESPECIALIDAD" not in df.columns:
        st.warning("No hay ESPECIALIDAD para esta vista.")
        return
    group_cols = ["ESPECIALIDAD"] + (["ENTORNO"] if "ENTORNO" in df.columns else [])
    agg = {
        "historias": ("ID ATENCION","nunique") if "ID ATENCION" in df.columns else ("CEDULA_PACIENTE","size"),
        "atenciones": ("ID ATENCION","size") if "ID ATENCION" in df.columns else ("CEDULA_PACIENTE","size"),
        "pacientes_unicos": ("CEDULA_PACIENTE","nunique")
    }
    res = df.groupby(group_cols, dropna=False).agg(**agg).reset_index()
    res["lbl"] = res.apply(lambda r: f"H: {int(r['historias'])} | P: {int(r['pacientes_unicos'])}", axis=1)
    top_n = st.slider("Top especialidades por historias", 5, 30, 15, key="top_esp")
    top = res.sort_values("historias", ascending=False).head(top_n)
    chart_barras(top, x="historias", y="ESPECIALIDAD", label_col="lbl",
                 color="ENTORNO" if "ENTORNO" in top.columns else None,
                 title="Historias únicas por especialidad")
    with st.expander("Tabla y descarga"):
        st.dataframe(res.sort_values("historias", ascending=False), use_container_width=True)
        xlsx = df_to_excel_bytes({"Especialidades": res.sort_values("historias", ascending=False)})
        st.download_button("⬇️ Descargar (XLSX)", xlsx, file_name="resumen_especialidades.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

def vista_ciudades_carceles(df: pd.DataFrame):
    if df.empty or "CIUDAD" not in df.columns:
        st.warning("No hay CIUDAD para esta vista.")
        return
    group_cols = ["CIUDAD"] + (["ENTORNO"] if "ENTORNO" in df.columns else [])
    agg = {
        "historias": ("ID ATENCION","nunique") if "ID ATENCION" in df.columns else ("CEDULA_PACIENTE","size"),
        "atenciones": ("ID ATENCION","size") if "ID ATENCION" in df.columns else ("CEDULA_PACIENTE","size"),
        "pacientes_unicos": ("CEDULA_PACIENTE","nunique")
    }
    res = df.groupby(group_cols, dropna=False).agg(**agg).reset_index()
    res["lbl"] = res.apply(lambda r: f"H: {int(r['historias'])} | P: {int(r['pacientes_unicos'])}", axis=1)
    top_n = st.slider("Top ciudades por historias", 5, 40, 20, key="top_ciudades")
    top = res.sort_values("historias", ascending=False).head(top_n)
    chart_barras(top, x="historias", y="CIUDAD", label_col="lbl",
                 color="ENTORNO" if "ENTORNO" in top.columns else None,
                 title="Historias únicas por ciudad y entorno")
    with st.expander("Tabla y descarga"):
        st.dataframe(res.sort_values("historias", ascending=False), use_container_width=True)
        xlsx = df_to_excel_bytes({"Ciudades": res.sort_values("historias", ascending=False)})
        st.download_button("⬇️ Descargar (XLSX)", xlsx, file_name="resumen_ciudades.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

def vista_por_eps(df: pd.DataFrame):
    st.subheader("🏥 Distribución por EPS")
    if df.empty or "EPS" not in df.columns:
        st.info("No existe columna EPS.")
        return
    # Top EPS por historias
    grp = df.groupby("EPS", dropna=False).agg(
        historias=("ID ATENCION","nunique") if "ID ATENCION" in df.columns else ("CEDULA_PACIENTE","size"),
        atenciones=("ID ATENCION","size") if "ID ATENCION" in df.columns else ("CEDULA_PACIENTE","size"),
        pacientes_unicos=("CEDULA_PACIENTE","nunique")
    ).reset_index()
    grp = grp.sort_values("historias", ascending=False)
    top_n = st.slider("Top EPS por historias", 3, 30, 10, key="top_eps")
    top = grp.head(top_n).copy()
    otros = pd.DataFrame([{
        "EPS":"OTRAS",
        "historias": grp["historias"][top_n:].sum(),
        "atenciones": grp["atenciones"][top_n:].sum(),
        "pacientes_unicos": grp["pacientes_unicos"][top_n:].sum()
    }]) if len(grp) > top_n else pd.DataFrame()
    show = pd.concat([top, otros], ignore_index=True) if not otros.empty else top
    show["lbl"] = show.apply(lambda r: f"H:{int(r['historias'])} P:{int(r['pacientes_unicos'])}", axis=1)

    c1, c2 = st.columns(2)
    with c1:
        chart_barras(show, x="historias", y="EPS", label_col="lbl", color=None,
                     title="Historias por EPS")
    with c2:
        chart_torta(show.rename(columns={"historias":"valor"}), cat_col="EPS", value_col="valor",
                    title="Participación de historias por EPS")

    with st.expander("Tabla y descarga"):
        st.dataframe(grp, use_container_width=True)
        xlsx = df_to_excel_bytes({"EPS": grp})
        st.download_button("⬇️ Descargar (XLSX)", xlsx, file_name="resumen_eps.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

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

        sel_prof = st.multiselect("Profesionales a comparar", nombres, default=default_sel, key="cmp_prof_sel")
        metrica = st.selectbox("Métrica", ["historias", "pacientes_unicos", "atenciones"], index=0, key="cmp_prof_metric")

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

        sel_esp = st.multiselect("Especialidades a comparar", especialidades, default=default_sel, key="cmp_esp_sel")
        metrica_esp = st.selectbox(
            "Métrica (especialidad)",
            ["historias", "pacientes_unicos", "atenciones"],
            index=0,
            key="met_esp",
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
# Tablas dinámicas / pivote
# =========================
def tabla_dinamica(df: pd.DataFrame):
    st.subheader("📊 Tabla dinámica (pivote)")
    cols_dim = [c for c in ["EPS","SEDE","PROGRAMA","CIUDAD","ENTORNO","ESPECIALIDAD","NOMBRE_PROFESIONAL","PERIODO"] if c in df.columns]
    if not cols_dim:
        st.info("No hay columnas para pivotear.")
        return

    c1, c2, c3 = st.columns(3)
    filas = c1.multiselect(
        "Filas", cols_dim,
        default=["ESPECIALIDAD"] if "ESPECIALIDAD" in cols_dim else [cols_dim[0]],
        key="pvt_rows"
    )
    columnas = c2.multiselect(
        "Columnas (opcional)", cols_dim,
        default=["EPS"] if "EPS" in cols_dim else [],
        key="pvt_cols"
    )
    metrica = c3.selectbox(
        "Métrica", ["historias","pacientes_unicos","atenciones"], index=0, key="pvt_metric"
    )

    if not filas:
        st.info("Selecciona al menos una dimensión para Filas.")
        return

    # Construir serie agregada
    if metrica == "historias":
        serie = df.groupby(filas + columnas, dropna=False)["ID ATENCION"].nunique() \
                 if "ID ATENCION" in df.columns \
                 else df.groupby(filas + columnas, dropna=False)["CEDULA_PACIENTE"].size()
    elif metrica == "pacientes_unicos":
        serie = df.groupby(filas + columnas, dropna=False)["CEDULA_PACIENTE"].nunique()
    else:  # atenciones
        base = "ID ATENCION" if "ID ATENCION" in df.columns else "CEDULA_PACIENTE"
        serie = df.groupby(filas + columnas, dropna=False)[base].size()

    pt = serie.unstack(columnas) if columnas else serie.to_frame(metrica)
    pt = pt.fillna(0).astype(int, errors="ignore")
    st.dataframe(pt, use_container_width=True)

    xlsx = df_to_excel_bytes({"Tabla_dinamica": pt.reset_index()})
    st.download_button(
        "⬇️ Descargar tabla dinámica (XLSX)",
        xlsx,
        file_name="tabla_dinamica.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="pvt_dl_btn"
    )


# =========================
# MAIN
# =========================
init_db()

st.sidebar.header("📂 Gestión de base (SQLite)")
files = st.sidebar.file_uploader(
    "Sube uno o varios Excel (meses). Se actualiza sin borrar.",
    type=["xlsx","xls"],
    accept_multiple_files=True
)
if st.sidebar.button("Cargar / actualizar base") and files:
    for f in files:
        df_new = load_data(f)
        upsert_df_to_db(df_new)
    st.sidebar.success("Datos cargados/actualizados en SQLite.")

with st.sidebar.expander("Opciones avanzadas (Admin)"):
    if st.button("Vaciar base"):
        clear_db()
        st.warning("Base vaciada. Vuelve a cargar archivos.")

# Cargar todo lo disponible en DB
df_db = load_all_from_db()
if df_db.empty:
    st.info("No hay datos en la base. Carga uno o varios archivos desde la barra lateral.")
    st.stop()

# Filtros (con EPS)
df_filt = aplicar_filtros(df_db)

# KPIs
st.markdown("### 🔍 Resumen general")
mostrar_kpis(df_filt)

# Tabs
t1, t2, t3, t4, t5, t6 = st.tabs([
    "👨‍⚕️ Profesional",
    "🩺 Especialidad",
    "🌎 Ciudades y entorno",
    "🏥 EPS",
    "📆 Comparativo mensual",
    "📊 Tabla dinámica"
])

with t1: vista_por_profesional(df_filt)
with t2: vista_por_especialidad(df_filt)
with t3: vista_ciudades_carceles(df_filt)
with t4: vista_por_eps(df_filt)
with t5: comparativo_mensual(df_filt)
with t6: tabla_dinamica(df_filt)

# Detalle y descarga
st.markdown("### 📋 Detalle filtrado")
st.dataframe(df_filt, use_container_width=True)
xlsx_det = df_to_excel_bytes({"Detalle_filtrado": df_filt})
st.download_button("⬇️ Descargar detalle filtrado (XLSX)", xlsx_det, file_name="detalle_filtrado.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

