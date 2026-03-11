import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings("ignore")

# ==========================================================
# CARGA Y AGREGACIÓN
# ==========================================================

@st.cache_data
def load_team_data(files, file_names=None):
    dfs = []
    for f in files:
        try:
            d = pd.read_csv(f, sep=';', skiprows=1, decimal=',')
            dfs.append(d)
        except Exception:
            pass
    if not dfs:
        return pd.DataFrame(), pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)

    # Agregar por equipo + partido
    partidos = df.groupby(
        ['equipo','jornada','fecha','rival','lado','resultado_equipo','resultado_rival']
    ).agg(
        goles              = ('goles',              'sum'),
        xG                 = ('xG',                 'sum'),
        asistencias        = ('asistencias',         'sum'),
        xA                 = ('xA',                 'sum'),
        tiros              = ('tiros_totales',       'sum'),
        tiros_puerta       = ('tiros_a_puerta',      'sum'),
        pases              = ('pases_totales',       'sum'),
        pases_precisos     = ('pases_precisos',      'sum'),
        pases_campo_rival  = ('pases_campo_rival',   'sum'),
        pases_campo_propio = ('pases_campo_propio',  'sum'),
        pases_largos       = ('pases_largos_totales','sum'),
        pases_largos_prec  = ('pases_largos_precisos','sum'),
        centros            = ('centros_totales',     'sum'),
        centros_precisos   = ('centros_precisos',    'sum'),
        entradas           = ('entradas_totales',    'sum'),
        entradas_ganadas   = ('entradas_ganadas',    'sum'),
        despejes           = ('despejes',            'sum'),
        duelos_ganados     = ('duelos_ganados',      'sum'),
        duelos_perdidos    = ('duelos_perdidos',     'sum'),
        duelos_aereos_g    = ('duelos_aereos_ganados','sum'),
        duelos_aereos_p    = ('duelos_aereos_perdidos','sum'),
        regates            = ('regates_exitosos',    'sum'),
        regates_int        = ('regates_intentados',  'sum'),
        regateado          = ('regateado',           'sum'),
        recuperaciones     = ('recuperaciones',      'sum'),
        perdidas           = ('perdidas',            'sum'),
        faltas_com         = ('faltas_cometidas',    'sum'),
        faltas_rec         = ('faltas_recibidas',    'sum'),
        toques             = ('toques',              'sum'),
        rating_medio       = ('rating_sofascore',    'mean'),
        jugadores          = ('jugador',             'nunique'),
    ).reset_index()

    partidos['puntos'] = partidos.apply(
        lambda r: 3 if r['resultado_equipo'] > r['resultado_rival']
                  else (1 if r['resultado_equipo'] == r['resultado_rival'] else 0), axis=1)
    partidos['goles_contra'] = partidos['resultado_rival']
    partidos['victoria'] = (partidos['puntos'] == 3).astype(int)
    partidos['empate']   = (partidos['puntos'] == 1).astype(int)
    partidos['derrota']  = (partidos['puntos'] == 0).astype(int)
    partidos['porteria_cero'] = (partidos['goles_contra'] == 0).astype(int)

    # Totales por equipo
    eq = partidos.groupby('equipo').agg(
        Partidos           = ('jornada',        'count'),
        Puntos             = ('puntos',         'sum'),
        Victorias          = ('victoria',       'sum'),
        Empates            = ('empate',         'sum'),
        Derrotas           = ('derrota',        'sum'),
        Goles_favor        = ('goles',          'sum'),
        Goles_contra       = ('goles_contra',   'sum'),
        xG_total           = ('xG',             'sum'),
        xA_total           = ('xA',             'sum'),
        Tiros_total        = ('tiros',          'sum'),
        Tiros_puerta_total = ('tiros_puerta',   'sum'),
        Pases_total        = ('pases',          'sum'),
        Pases_prec_total   = ('pases_precisos', 'sum'),
        Pases_rival_total  = ('pases_campo_rival','sum'),
        Pases_propio_total = ('pases_campo_propio','sum'),
        Centros_total      = ('centros',        'sum'),
        Centros_prec_total = ('centros_precisos','sum'),
        Entradas_total     = ('entradas',       'sum'),
        Entradas_g_total   = ('entradas_ganadas','sum'),
        Despejes_total     = ('despejes',       'sum'),
        Duelos_g_total     = ('duelos_ganados', 'sum'),
        Duelos_p_total     = ('duelos_perdidos','sum'),
        Duelos_aer_g       = ('duelos_aereos_g','sum'),
        Duelos_aer_p       = ('duelos_aereos_p','sum'),
        Regates_total      = ('regates',        'sum'),
        Regates_int_total  = ('regates_int',    'sum'),
        Regateado_total    = ('regateado',      'sum'),
        Recuperaciones_tot = ('recuperaciones', 'sum'),
        Perdidas_total     = ('perdidas',       'sum'),
        Faltas_com_total   = ('faltas_com',     'sum'),
        Faltas_rec_total   = ('faltas_rec',     'sum'),
        Toques_total       = ('toques',         'sum'),
        Porterias_cero     = ('porteria_cero',  'sum'),
        Rating_medio       = ('rating_medio',   'mean'),
    ).reset_index()

    p = eq['Partidos']

    # Métricas por partido
    eq['Goles/90']              = (eq['Goles_favor']        / p).round(2)
    eq['Goles_contra/90']       = (eq['Goles_contra']       / p).round(2)
    eq['xG/90']                 = (eq['xG_total']           / p).round(2)
    eq['xA/90']                 = (eq['xA_total']           / p).round(2)
    eq['Tiros/90']              = (eq['Tiros_total']        / p).round(2)
    eq['Tiros_puerta/90']       = (eq['Tiros_puerta_total'] / p).round(2)
    eq['Pases/90']              = (eq['Pases_total']        / p).round(2)
    eq['Pases_rival/90']        = (eq['Pases_rival_total']  / p).round(2)
    eq['Centros/90']            = (eq['Centros_total']      / p).round(2)
    eq['Entradas/90']           = (eq['Entradas_total']     / p).round(2)
    eq['Despejes/90']           = (eq['Despejes_total']     / p).round(2)
    eq['Recuperaciones/90']     = (eq['Recuperaciones_tot'] / p).round(2)
    eq['Perdidas/90']           = (eq['Perdidas_total']     / p).round(2)
    eq['Regates/90']            = (eq['Regates_total']      / p).round(2)
    eq['Regateado/90']          = (eq['Regateado_total']    / p).round(2)
    eq['Faltas_com/90']         = (eq['Faltas_com_total']   / p).round(2)
    eq['Faltas_rec/90']         = (eq['Faltas_rec_total']   / p).round(2)
    eq['Toques/90']             = (eq['Toques_total']       / p).round(2)
    eq['Duelos/90']             = ((eq['Duelos_g_total'] + eq['Duelos_p_total']) / p).round(2)
    eq['Duelos_perdidos/90']    = (eq['Duelos_p_total']     / p).round(2)
    eq['Duelos_aereos/90']      = ((eq['Duelos_aer_g'] + eq['Duelos_aer_p']) / p).round(2)
    eq['Regates_intentados/90'] = (eq['Regates_int_total']  / p).round(2)
    eq['Faltas_rec/90']         = (eq['Faltas_rec_total']   / p).round(2)
    eq['xG_contra/90']          = None  # no disponible directamente

    # Métricas porcentaje
    eq['Precision_pases_%']    = ((eq['Pases_prec_total']  / eq['Pases_total'].replace(0,np.nan)) * 100).fillna(0).round(1)
    eq['Precision_centros_%']  = ((eq['Centros_prec_total']/ eq['Centros_total'].replace(0,np.nan)) * 100).fillna(0).round(1)
    eq['Entradas_ganadas_%']   = ((eq['Entradas_g_total']  / eq['Entradas_total'].replace(0,np.nan)) * 100).fillna(0).round(1)
    eq['Duelos_ganados_%']     = ((eq['Duelos_g_total']    / (eq['Duelos_g_total']+eq['Duelos_p_total']).replace(0,np.nan)) * 100).fillna(0).round(1)
    eq['Duelos_aereos_%']      = ((eq['Duelos_aer_g']      / (eq['Duelos_aer_g']+eq['Duelos_aer_p']).replace(0,np.nan)) * 100).fillna(0).round(1)
    eq['Regates_exito_%']      = ((eq['Regates_total']     / eq['Regates_int_total'].replace(0,np.nan)) * 100).fillna(0).round(1)
    eq['Tiros_puerta_%']       = ((eq['Tiros_puerta_total']/ eq['Tiros_total'].replace(0,np.nan)) * 100).fillna(0).round(1)
    eq['Conversion_%']         = ((eq['Goles_favor']       / eq['Tiros_total'].replace(0,np.nan)) * 100).fillna(0).round(1)
    eq['Porterias_cero_%']     = ((eq['Porterias_cero']    / p) * 100).round(1)
    eq['Pases_rival_%']        = ((eq['Pases_rival_total'] / eq['Pases_total'].replace(0,np.nan)) * 100).fillna(0).round(1)

    # Diferencial
    eq['Dif_goles']            = eq['Goles_favor'] - eq['Goles_contra']
    eq['Dif_xG']               = (eq['xG_total'] - eq['Goles_favor']).round(2)
    eq['xG_overperformance']   = (eq['Goles_favor'] - eq['xG_total']).round(2)
    eq['Puntos/90']             = (eq['Puntos'] / p).round(2)

    eq['Rating_medio']         = eq['Rating_medio'].round(2)

    # ── Asignar liga desde nombre del archivo ───────────────────────────────
    import os as _os
    if file_names is None:
        file_names = [None] * len(files)
    liga_por_equipo = {}
    for f, d_orig, orig_name in zip(files, dfs, file_names):
        # Usar nombre original si está disponible, si no el del archivo
        fname_str = str(orig_name) if orig_name else str(f if isinstance(f, str) else getattr(f, 'name', str(f)))
        fname_str = _os.path.basename(fname_str)
        if '1rfef' in fname_str:           liga_nombre = '1a RFEF'
        elif 'laliga-2' in fname_str:      liga_nombre = 'LaLiga 2'
        elif 'serie-b' in fname_str:       liga_nombre = 'Serie B'
        elif 'ligue-2' in fname_str:       liga_nombre = 'Ligue 2'
        elif 'liga-portugal' in fname_str: liga_nombre = 'Liga Portugal 2'
        else:
            base = _os.path.splitext(fname_str)[0]
            liga_nombre = base if base else 'Otra'
        if 'equipo' in d_orig.columns:
            for eq_name in d_orig['equipo'].dropna().unique():
                liga_por_equipo[eq_name] = liga_nombre

    eq['Liga'] = eq['equipo'].map(liga_por_equipo).fillna('Otra')
    partidos['Liga'] = partidos['equipo'].map(liga_por_equipo).fillna('Otra')

    # ── Grupos solo para RFEF ────────────────────────────────────────────────
    GRUPO_1 = {
        'CD Tenerife', 'Celta Vigo B', 'Pontevedra CF', 'CD Lugo', 'Barakaldo CF',
        'Merida AD', 'Mérida AD', 'Racing de Ferrol', 'Athletic Club B U21', 'SD Ponferradina',
        'Zamora CF', 'Real Madrid Castilla U21', 'Unionistas de Salamanca CF',
        'Arenas Club', 'Real Aviles', 'Real Avilés', 'Ourense CF', 'CF Talavera',
        'CP Cacereno', 'CP Cacereño', 'Guadalajara', 'CD Arenteiro', 'Osasuna B',
    }
    def norm(s):
        return str(s).lower().replace('é','e').replace('á','a').replace('ó','o').replace('ú','u').replace('í','i').strip()

    GRUPO_1_NORM = {norm(x) for x in GRUPO_1}

    def asignar_grupo(equipo, liga):
        if liga != '1a RFEF':
            return liga
        if norm(equipo) in GRUPO_1_NORM:
            return 'Grupo 1'
        return 'Grupo 2'

    eq['Grupo'] = eq.apply(lambda r: asignar_grupo(r['equipo'], r['Liga']), axis=1)
    partidos['Grupo'] = partidos.apply(lambda r: asignar_grupo(r['equipo'], r['Liga']), axis=1)

    # Posición dentro del grupo
    eq = eq.sort_values('Puntos', ascending=False).reset_index(drop=True)
    eq['Pos_grupo'] = eq.groupby('Grupo').cumcount() + 1
    eq.insert(0, 'Pos', range(1, len(eq)+1))

    return eq, partidos

# ==========================================================
# MÉTRICAS DISPONIBLES POR CATEGORÍA
# ==========================================================

CATEGORIAS = {
    "📊 Clasificación":    ["Puntos", "Puntos/90", "Victorias", "Empates", "Derrotas",
                            "Dif_goles", "Rating_medio"],
    "⚽ Ataque":           ["Goles/90", "xG/90", "xG_overperformance", "Tiros/90",
                            "Tiros_puerta/90", "Tiros_puerta_%", "Conversion_%", "xA/90"],
    "🛡️ Defensa":          ["Goles_contra/90", "Porterias_cero_%", "Despejes/90",
                            "Entradas/90", "Entradas_ganadas_%", "Regateado/90"],
    "🎯 Pases":            ["Pases/90", "Precision_pases_%", "Pases_rival/90",
                            "Pases_rival_%", "Centros/90", "Precision_centros_%"],
    "💪 Duelos":           ["Duelos/90", "Duelos_ganados_%", "Duelos_perdidos/90", "Duelos_aereos_%",
                            "Duelos_aereos/90", "Recuperaciones/90", "Entradas_ganadas_%"],
    "🏃 Dinamismo":        ["Regates/90", "Regates_exito_%", "Regates_intentados/90",
                            "Faltas_rec/90", "Faltas_com/90", "Toques/90", "Perdidas/90"],
}

METRICAS_NEGATIVAS = {"Goles_contra/90", "Perdidas/90", "Regateado/90", "Faltas_com/90", "Derrotas", "Duelos_perdidos/90"}

ALL_METRICAS = [m for mlist in CATEGORIAS.values() for m in mlist]

# ==========================================================
# PERFILES TÁCTICOS
# ==========================================================

PERFILES = {
    "Posesión":    {"Pases/90": 0.25, "Precision_pases_%": 0.25, "Pases_rival_%": 0.20,
                    "Toques/90": 0.20, "Perdidas/90": 0.10},
    "Pressing":    {"Recuperaciones/90": 0.30, "Entradas/90": 0.20, "Entradas_ganadas_%": 0.20,
                    "Pases_rival/90": 0.15, "Duelos_ganados_%": 0.15},
    "Directo":     {"Pases_rival/90": 0.30, "Centros/90": 0.25, "Tiros/90": 0.20,
                    "Duelos_aereos_%": 0.15, "Duelos_aereos/90": 0.10},
    "Ofensivo":    {"Goles/90": 0.25, "xG/90": 0.25, "Tiros_puerta/90": 0.20,
                    "Conversion_%": 0.15, "Faltas_rec/90": 0.15},
    "Defensivo":   {"Goles_contra/90": 0.30, "Porterias_cero_%": 0.25, "Recuperaciones/90": 0.20,
                    "Entradas_ganadas_%": 0.15, "Duelos_ganados_%": 0.10},
    "Creatividad": {"xA/90": 0.30, "Centros/90": 0.15, "Precision_centros_%": 0.15,
                    "Regates/90": 0.20, "Regates_exito_%": 0.10, "Faltas_rec/90": 0.10},
}

def compute_team_percentiles(eq, per_group=True):
    df = eq.copy()
    for m in ALL_METRICAS:
        if m not in df.columns: continue
        neg = m in METRICAS_NEGATIVAS
        pct_col = f"{m}_pct"
        df[pct_col] = np.nan
        if per_group and 'Grupo' in df.columns:
            for grp, grp_df in df.groupby('Grupo'):
                vals = grp_df[m].dropna()
                if len(vals) < 2: continue
                def to_pct(v, rv=vals, n=neg):
                    if pd.isna(v): return np.nan
                    rank = (rv < v).sum() + (rv == v).sum() * 0.5
                    pct = rank / len(rv)
                    return float(np.clip(1 - pct if n else pct, 0, 1))
                df.loc[grp_df.index, pct_col] = grp_df[m].apply(to_pct)
        else:
            vals = df[m].dropna()
            if len(vals) < 2: continue
            def to_pct(v, rv=vals, n=neg):
                if pd.isna(v): return np.nan
                rank = (rv < v).sum() + (rv == v).sum() * 0.5
                pct = rank / len(rv)
                return float(np.clip(1 - pct if n else pct, 0, 1))
            df[pct_col] = df[m].apply(to_pct)
    return df

def compute_profile_scores(eq_pct):
    df = eq_pct.copy()
    for perfil, weights in PERFILES.items():
        scores = []
        for _, row in df.iterrows():
            available = {m: w for m, w in weights.items()
                         if f"{m}_pct" in df.columns and not pd.isna(row.get(f"{m}_pct", np.nan))}
            if not available:
                scores.append(0.0)
                continue
            tw = sum(available.values())
            s = sum(row[f"{m}_pct"] * (w/tw) for m, w in available.items())
            scores.append(round(s * 10, 2))
        df[f"Score_{perfil}"] = scores
    return df

# ==========================================================
# PLOTS
# ==========================================================

def radar_equipos(eq_pct, equipos, metricas):
    metricas = [m for m in metricas if m in eq_pct.columns]
    if not metricas or not equipos: return
    N      = len(metricas)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist() + [0]
    colors = ["#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6","#1abc9c"]
    fig, ax = plt.subplots(figsize=(7,7), subplot_kw=dict(polar=True))
    for i, eq in enumerate(equipos):
        row = eq_pct[eq_pct['equipo'] == eq]
        if row.empty: continue
        vals = [row.iloc[0].get(f"{m}_pct", 0) or 0 for m in metricas]
        vals += [vals[0]]
        c = colors[i % len(colors)]
        ax.plot(angles, vals, linewidth=2, label=eq, color=c)
        ax.fill(angles, vals, alpha=0.1, color=c)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metricas, fontsize=8)
    ax.set_ylim(0,1); ax.set_yticklabels([]); ax.grid(alpha=0.25)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5,-0.12),
               frameon=False, fontsize=9, ncol=2)
    plt.tight_layout()
    st.pyplot(fig)

def evolucion_equipo(partidos_df, equipo, metricas):
    df = partidos_df[partidos_df['equipo'] == equipo].sort_values('jornada')
    if df.empty: return
    fig = go.Figure()
    colors = ["#e74c3c","#3498db","#2ecc71","#f39c12"]
    for i, m in enumerate(metricas):
        if m not in df.columns: continue
        fig.add_trace(go.Scatter(
            x=df['jornada'], y=df[m], mode='lines+markers',
            name=m, line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=6),
            hovertemplate=f"J%{{x}} — {m}: %{{y:.2f}}<extra></extra>"
        ))
    fig.update_layout(
        title=f"{equipo} — Evolución por jornada",
        xaxis_title="Jornada", template="simple_white", height=400,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        margin=dict(l=50, r=30, t=50, b=80)
    )
    st.plotly_chart(fig, use_container_width=True)

def tabla_clasificacion(eq):
    cols = ['Pos','equipo','Partidos','Puntos','Victorias','Empates','Derrotas',
            'Goles_favor','Goles_contra','Dif_goles','xG/90','Rating_medio']
    df = eq[[c for c in cols if c in eq.columns]].copy()
    df.columns = [c.replace('_',' ') for c in df.columns]
    st.dataframe(df, use_container_width=True, hide_index=True)

# ==========================================================
# APP
# ==========================================================



with st.sidebar:
    st.header("📂 Datos")
    uploaded = st.file_uploader(
        "CSV partido a partido", type=["csv"], accept_multiple_files=True
    )

if not uploaded:
    st.info("👈 Sube uno o varios CSV de partido a partido para comenzar.")
    st.stop()

file_paths = []
file_names = []
import tempfile, os
for uf in uploaded:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    tmp.write(uf.read()); tmp.flush()
    file_paths.append(tmp.name)
    file_names.append(uf.name)  # nombre original del archivo

eq_raw, partidos = load_team_data(file_paths, file_names)
if eq_raw.empty:
    st.error("No se pudieron cargar los datos.")
    st.stop()

eq_pct  = compute_team_percentiles(eq_raw)
eq_prof = compute_profile_scores(eq_pct)

# Sidebar filtros
st.sidebar.divider()
st.sidebar.subheader("🔎 Filtros")

# Filtro de liga
ligas_disponibles = sorted(eq_raw['Liga'].unique().tolist()) if 'Liga' in eq_raw.columns else []
if len(ligas_disponibles) > 1:
    liga_sel = st.sidebar.multiselect("Liga", ligas_disponibles, default=ligas_disponibles, placeholder="Todas...")
    if not liga_sel:
        liga_sel = ligas_disponibles
else:
    liga_sel = ligas_disponibles

eq_raw_filt = eq_raw[eq_raw['Liga'].isin(liga_sel)] if 'Liga' in eq_raw.columns else eq_raw

# Filtro de grupo (solo si hay RFEF seleccionada)
hay_rfef = '1a RFEF' in liga_sel
grupos_disponibles = []
if hay_rfef and 'Grupo' in eq_raw_filt.columns:
    grupos_rfef = [g for g in eq_raw_filt['Grupo'].unique() if g in ['Grupo 1','Grupo 2']]
    grupos_disponibles = sorted(grupos_rfef)
grupo_sel = "Todos"
if grupos_disponibles:
    grupo_sel = st.sidebar.radio("Grupo RFEF", ["Todos"] + grupos_disponibles, key="grupo_sel")

equipos_en_grupo = sorted(
    eq_raw_filt[eq_raw_filt['Grupo'] == grupo_sel]['equipo'].tolist()
    if grupo_sel != "Todos" else eq_raw_filt['equipo'].tolist()
)
equipos_lista = sorted(eq_raw_filt['equipo'].tolist())

eq_sel_sidebar = st.sidebar.multiselect("Filtrar equipos", equipos_en_grupo, placeholder="Todos...")

# Base mask
eq_pct_filt = eq_pct[eq_pct['Liga'].isin(liga_sel)] if 'Liga' in eq_pct.columns else eq_pct
base_mask = eq_pct_filt['Grupo'] == grupo_sel if grupo_sel != "Todos" else pd.Series([True]*len(eq_pct_filt), index=eq_pct_filt.index)
if eq_sel_sidebar:
    base_mask = eq_pct_filt['equipo'].isin(eq_sel_sidebar)

eq_view      = eq_pct_filt[base_mask].copy()
_equipos_view = eq_view["equipo"].tolist()
eq_prof_filt  = compute_profile_scores(eq_pct_filt)
eq_prof_view  = eq_prof_filt[eq_prof_filt["equipo"].isin(_equipos_view)].copy()
partidos_view = partidos[partidos['equipo'].isin(eq_view['equipo'])].copy()
if 'Liga' in partidos_view.columns:
    partidos_view = partidos_view[partidos_view['Liga'].isin(liga_sel)]

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🏆 Clasificación", "📊 Ranking Métricas", "🕷 Radar",
    "🆚 Comparador", "📈 Evolución", "🔵 Scatter", "📝 Informe"
])

# ── TAB 1: CLASIFICACIÓN ─────────────────────────────────
with tab1:
    st.subheader("Clasificación")
    if 'Liga' in eq_view.columns and len(eq_view['Liga'].unique()) > 1 and grupo_sel == "Todos":
        # Varias ligas: mostrar por liga
        for lig in sorted(eq_view['Liga'].unique()):
            df_lig = eq_view[eq_view['Liga'] == lig].copy()
            # Si es RFEF, subdividir por grupos
            if lig == '1a RFEF' and 'Grupo' in df_lig.columns:
                for grp in sorted(df_lig['Grupo'].unique()):
                    st.markdown(f"#### {grp}")
                    df_grp = df_lig[df_lig['Grupo'] == grp].copy()
                    df_grp = df_grp.sort_values('Puntos', ascending=False).reset_index(drop=True)
                    df_grp['Pos'] = range(1, len(df_grp)+1)
                    tabla_clasificacion(df_grp)
            else:
                st.markdown(f"#### {lig}")
                df_lig = df_lig.sort_values('Puntos', ascending=False).reset_index(drop=True)
                df_lig['Pos'] = range(1, len(df_lig)+1)
                tabla_clasificacion(df_lig)
    elif 'Grupo' in eq_view.columns and grupo_sel == "Todos" and len(eq_view.get('Liga', pd.Series()).unique()) <= 1:
        for grp in sorted(eq_view['Grupo'].unique()):
            st.markdown(f"#### {grp}")
            df_grp = eq_view[eq_view['Grupo'] == grp].copy()
            df_grp = df_grp.sort_values('Puntos', ascending=False).reset_index(drop=True)
            df_grp['Pos'] = range(1, len(df_grp)+1)
            tabla_clasificacion(df_grp)
    else:
        df_show = eq_view.copy().sort_values('Puntos', ascending=False).reset_index(drop=True)
        df_show['Pos'] = range(1, len(df_show)+1)
        tabla_clasificacion(df_show)

    st.divider()
    st.subheader("🎭 Perfiles Tácticos")

    PERFILES_DESC = {
        "Posesión":    ("🔵 Posesión", "Equipos que construyen desde atrás con pases cortos y precisos, dominan el balón y buscan superar líneas por el interior. Se valora: pases/partido, precisión, % en campo rival y control de balón."),
        "Pressing":    ("🔴 Pressing", "Equipos que presionan alto y recuperan el balón rápido en campo rival. Intensidad defensiva y agresividad sin balón. Se valora: recuperaciones, entradas, duelos ganados y capacidad de ser regateado."),
        "Directo":     ("🟠 Directo", "Equipos que buscan la portería rival rápido con pases largos, centros y duelos aéreos. Poco rodeo, mucha verticalidad. Se valora: pases en campo rival, centros, tiros y duelos aéreos."),
        "Ofensivo":    ("🟢 Ofensivo", "Equipos con alto rendimiento goleador y capacidad de generar peligro real. Se valora: goles/partido, xG, tiros a puerta, conversión y si superan su xG esperado."),
        "Defensivo":   ("⚪ Defensivo", "Equipos sólidos atrás que conceden poco y mantienen porterías a cero. Se valora: goles encajados, porterías a cero, despejes, entradas ganadas y duelos defensivos."),
        "Creatividad": ("🟣 Creatividad", "Equipos que generan oportunidades con creatividad y desborde. Se valora: xA/partido, centros, precisión de centros, regates y éxito en el 1vs1."),
    }

    col_perf1, col_perf2 = st.columns([1, 2])
    with col_perf1:
        eq_perfil_sel = st.selectbox("Equipo", ["—"] + equipos_lista, key="eq_perfil_eq")

        if eq_perfil_sel != "—":
            row_perf = eq_prof[eq_prof['equipo'] == eq_perfil_sel]
            if not row_perf.empty:
                scores_eq = {p: row_perf.iloc[0].get(f"Score_{p}", 0) for p in PERFILES}
                perfil_top = max(scores_eq, key=scores_eq.get)
                icono, desc = PERFILES_DESC[perfil_top]

                st.markdown(f"### {icono}")
                st.markdown(f"**Perfil dominante: {perfil_top}**")
                st.info(desc)
                st.markdown("**Puntuaciones:**")
                for p, s in sorted(scores_eq.items(), key=lambda x: -x[1]):
                    bar = "█" * int(s) + "░" * (10 - int(s))
                    st.markdown(f"`{p:<12}` {bar} **{s:.1f}**")

    with col_perf2:
        if eq_perfil_sel != "—":
            row_perf = eq_prof[eq_prof['equipo'] == eq_perfil_sel]
            if not row_perf.empty:
                perfiles_list = list(PERFILES.keys())
                vals = [row_perf.iloc[0].get(f"Score_{p}", 0) / 10 for p in perfiles_list]
                vals_plot = vals + [vals[0]]
                N = len(perfiles_list)
                angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist() + [0]
                fig_prad, ax_prad = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))
                ax_prad.plot(angles, vals_plot, linewidth=2.5, color="#3498db")
                ax_prad.fill(angles, vals_plot, alpha=0.2, color="#3498db")
                ax_prad.set_xticks(angles[:-1])
                ax_prad.set_xticklabels(perfiles_list, fontsize=9)
                ax_prad.set_ylim(0, 1)
                ax_prad.set_yticklabels([])
                ax_prad.grid(alpha=0.3)
                ax_prad.set_title(eq_perfil_sel, fontsize=11, pad=15, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig_prad)
        else:
            st.info("Selecciona un equipo para ver su perfil táctico.")

    st.divider()
    st.subheader("📋 Todos los perfiles")
    with st.expander("ℹ️ ¿Cómo se calculan los perfiles?"):
        for p, (icono, desc) in PERFILES_DESC.items():
            st.markdown(f"**{icono}** — {desc}")
    score_cols = [f"Score_{p}" for p in PERFILES]
    df_perfiles = eq_prof_view[['equipo'] + score_cols].copy()
    df_perfiles.columns = ['Equipo'] + list(PERFILES.keys())
    df_perfiles['Perfil dominante'] = df_perfiles[list(PERFILES.keys())].idxmax(axis=1)
    df_perfiles = df_perfiles.sort_values('Posesión', ascending=False)
    st.dataframe(df_perfiles.round(2), use_container_width=True, hide_index=True)

# ── TAB 2: RANKING ───────────────────────────────────────
with tab2:
    subtab_met, subtab_global = st.tabs(["📊 Por métrica", "🏅 Ranking global"])

    with subtab_met:
        st.subheader("Ranking por Métrica Individual")
        col1, col2 = st.columns([1,2])
        with col1:
            cat_sel = st.selectbox("Categoría", list(CATEGORIAS.keys()), key="eq_rank_cat")
            metricas_cat = [m for m in CATEGORIAS[cat_sel] if m in eq_view.columns]
            metrica_sel = st.selectbox("Métrica", metricas_cat, key="eq_rank_met")
            top_n = st.slider("Top N", 5, len(eq_view), min(15, len(eq_view)), key="eq_rank_n")
        with col2:
            if metrica_sel:
                asc = metrica_sel in METRICAS_NEGATIVAS
                df_rank = eq_view[['equipo', metrica_sel]].sort_values(
                    metrica_sel, ascending=asc
                ).head(top_n)
                colors_rank = ["#e74c3c" if asc else "#2ecc71"] * len(df_rank)
                fig_rank = go.Figure(go.Bar(
                    x=df_rank[metrica_sel], y=df_rank['equipo'],
                    orientation='h',
                    marker_color=colors_rank,
                    text=df_rank[metrica_sel].round(2),
                    textposition='outside',
                ))
                fig_rank.update_layout(
                    title=metrica_sel, template='simple_white', height=max(350, top_n*30),
                    yaxis=dict(autorange='reversed'),
                    margin=dict(l=150, r=80, t=50, b=40)
                )
                st.plotly_chart(fig_rank, use_container_width=True)

    with subtab_global:
        st.subheader("🏅 Ranking Global por Perfil de Juego")
        st.caption("Score 0–10 por perfil táctico. Ordena por cualquier columna.")

        score_cols_g = [f"Score_{p}" for p in PERFILES]
        df_global = eq_prof_view[['equipo'] + score_cols_g].copy()
        df_global.columns = ['Equipo'] + list(PERFILES.keys())

        # Rank position per profile
        for p in PERFILES.keys():
            df_global[f"#{p}"] = df_global[p].rank(ascending=False, method='min').astype(int)

        df_global['Perfil dominante'] = df_global[list(PERFILES.keys())].idxmax(axis=1)
        df_global = df_global.sort_values('Posesión', ascending=False).reset_index(drop=True)
        df_global.insert(0, 'Pos', range(1, len(df_global)+1))

        # Sort selector
        sort_by = st.selectbox(
            "Ordenar por",
            list(PERFILES.keys()),
            key="eq_global_sort"
        )
        asc_global = False
        df_global_sorted = df_global.sort_values(sort_by, ascending=asc_global).reset_index(drop=True)
        df_global_sorted['Pos'] = range(1, len(df_global_sorted)+1)

        display_cols = ['Pos', 'Equipo'] + list(PERFILES.keys()) + ['Perfil dominante']
        df_display = df_global_sorted[display_cols].copy()
        for p in PERFILES.keys():
            df_display[p] = df_display[p].round(2)

        st.dataframe(df_display, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("🌡️ Mapa de calor — Perfiles por equipo")

        hl_heat = st.selectbox(
            "Resaltar equipo (opcional)",
            ["Ninguno"] + equipos_lista,
            key="eq_heat_hl"
        )

        # Ordenar de mayor a menor según el perfil seleccionado
        df_heat = df_global_sorted.set_index('Equipo')[list(PERFILES.keys())]
        df_heat = df_heat.sort_values(sort_by, ascending=False)

        equipo_list = df_heat.index.tolist()
        vals_heat   = df_heat.values.round(1)

        # Construir colores de texto: negro normal, rojo para equipo resaltado
        text_colors = []
        for eq_h in equipo_list:
            if hl_heat != "Ninguno" and eq_h == hl_heat:
                text_colors.append(["#e74c3c"] * len(PERFILES))
            else:
                text_colors.append(["#333333"] * len(PERFILES))

        # Opacidad: si hay equipo resaltado, bajar el resto
        if hl_heat != "Ninguno":
            opacity_vals = []
            for eq_h in equipo_list:
                row_op = [1.0 if eq_h == hl_heat else 0.35] * len(PERFILES)
                opacity_vals.append(row_op)
            # Plotly Heatmap no soporta opacidad por celda — usamos un segundo trace encima
            # para el resaltado con un rectángulo de borde
            pass

        fig_heat = go.Figure(go.Heatmap(
            z=vals_heat,
            x=list(PERFILES.keys()),
            y=equipo_list,
            colorscale='RdYlGn',
            zmin=0, zmax=10,
            text=vals_heat,
            texttemplate="%{text}",
            textfont=dict(size=9),
            colorbar=dict(title="Score"),
            opacity=1.0,
        ))

        # Si hay equipo resaltado, añadir capa semitransparente sobre el resto
        if hl_heat != "Ninguno" and hl_heat in equipo_list:
            # Capa gris sobre todos los equipos NO resaltados
            mask_z = []
            for eq_h in equipo_list:
                if eq_h != hl_heat:
                    mask_z.append([1] * len(PERFILES))
                else:
                    mask_z.append([None] * len(PERFILES))
            fig_heat.add_trace(go.Heatmap(
                z=mask_z,
                x=list(PERFILES.keys()),
                y=equipo_list,
                colorscale=[[0,"rgba(255,255,255,0.6)"],[1,"rgba(255,255,255,0.6)"]],
                showscale=False,
                hoverinfo='skip',
                zmin=0, zmax=1,
            ))


        fig_heat.update_layout(
            template='simple_white',
            height=max(400, len(df_heat) * 24),
            margin=dict(l=180, r=40, t=50, b=80),
            xaxis=dict(side='top'),
            yaxis=dict(autorange='reversed'),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        st.divider()
        with st.expander("📖 ¿Cómo se calcula cada score?"):
            st.markdown("""
Cada score (0–10) se calcula comparando al equipo con el resto de la liga.
Para cada métrica del perfil se obtiene el **percentil** del equipo (0 = peor, 1 = mejor),
se pondera según el peso de esa métrica y se multiplica por 10.

Las métricas **negativas** (⬇) se invierten: estar en percentil bajo es bueno
(ej. encajar pocos goles = percentil alto en Defensivo).
""")
            PERFILES_CALC = {
                "🔵 Posesión": [
                    ("Pases/90", "25%", "Volumen de pases por partido"),
                    ("Precision_pases_%", "25%", "% de pases completados"),
                    ("Pases_rival_%", "20%", "Proporción de pases en campo rival"),
                    ("Toques/90", "20%", "Contactos con el balón por partido"),
                    ("Perdidas/90 ⬇", "10%", "Pérdidas de balón (negativa: menos es mejor)"),
                ],
                "🔴 Pressing": [
                    ("Recuperaciones/90", "30%", "Recuperaciones de balón por partido"),
                    ("Entradas/90", "20%", "Entradas realizadas"),
                    ("Entradas_ganadas_%", "20%", "Efectividad en las entradas"),
                    ("Pases_rival/90", "15%", "Pases en campo rival — proxy de presión alta"),
                    ("Duelos_ganados_%", "15%", "% de duelos ganados"),
                ],
                "🟠 Directo": [
                    ("Pases_rival/90", "30%", "Pases en campo contrario"),
                    ("Centros/90", "25%", "Centros laterales por partido"),
                    ("Tiros/90", "20%", "Remates totales"),
                    ("Duelos_aereos_%", "15%", "% de duelos aéreos ganados"),
                    ("Duelos_aereos/90", "10%", "Volumen de duelos aéreos"),
                ],
                "🟢 Ofensivo": [
                    ("Goles/90", "25%", "Goles marcados por partido"),
                    ("xG/90", "25%", "Expected goals generados"),
                    ("Tiros_puerta/90", "20%", "Tiros entre los tres palos"),
                    ("Conversion_%", "15%", "% de tiros que acaban en gol"),
                    ("Faltas_rec/90", "15%", "Faltas recibidas — los equipos peligrosos las generan"),
                ],
                "⚪ Defensivo": [
                    ("Goles_contra/90 ⬇", "30%", "Goles encajados (negativa: menos es mejor)"),
                    ("Porterias_cero_%", "25%", "% de partidos sin encajar"),
                    ("Recuperaciones/90", "20%", "Recuperaciones defensivas"),
                    ("Entradas_ganadas_%", "15%", "Efectividad defensiva en entradas"),
                    ("Duelos_ganados_%", "10%", "% de duelos ganados"),
                ],
                "🟣 Creatividad": [
                    ("xA/90", "30%", "Expected assists por partido"),
                    ("Regates/90", "20%", "Regates exitosos"),
                    ("Centros/90", "15%", "Centros laterales"),
                    ("Precision_centros_%", "15%", "% de centros completados"),
                    ("Regates_exito_%", "10%", "% de regates exitosos"),
                    ("Faltas_rec/90", "10%", "Faltas recibidas por desequilibrio"),
                ],
            }
            for perfil, metricas in PERFILES_CALC.items():
                st.markdown(f"**{perfil}**")
                for m, peso, desc in metricas:
                    st.markdown(f"- `{m}` **{peso}** — {desc}")
                st.markdown("")

# ── TAB 3: RADAR ─────────────────────────────────────────
with tab3:
    st.subheader("🕷 Radar Comparativo")
    col_r1, col_r2 = st.columns([1,2])
    with col_r1:
        eq_radar = st.multiselect("Equipos (máx 6)", equipos_lista,
                                   max_selections=6, placeholder="Elige equipos...",
                                   key="eq_radar_eqs")
        cat_radar = st.selectbox("Categoría de métricas", list(CATEGORIAS.keys()),
                                  key="eq_radar_cat")
        mets_disponibles = [m for m in CATEGORIAS[cat_radar] if m in eq_pct.columns]
        mets_radar = st.multiselect("Métricas", mets_disponibles,
                                     default=mets_disponibles[:min(6, len(mets_disponibles))],
                                     key="eq_radar_mets")
    with col_r2:
        if eq_radar and mets_radar:
            radar_equipos(eq_pct, eq_radar, mets_radar)
        else:
            st.info("Selecciona equipos y métricas.")

# ── TAB 4: COMPARADOR ────────────────────────────────────
with tab4:
    st.subheader("🆚 Comparador de Equipos")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        eq_A = st.selectbox("Equipo A", ["—"] + equipos_lista, key="eq_comp_a")
    with col_c2:
        eq_B = st.selectbox("Equipo B", ["—"] + equipos_lista, key="eq_comp_b")

    if eq_A != "—" and eq_B != "—" and eq_A != eq_B:
        cat_comp = st.selectbox("Categoría", list(CATEGORIAS.keys()), key="eq_comp_cat")
        mets_comp = [m for m in CATEGORIAS[cat_comp] if m in eq_pct.columns]

        row_a = eq_pct[eq_pct['equipo'] == eq_A].iloc[0]
        row_b = eq_pct[eq_pct['equipo'] == eq_B].iloc[0]

        data_comp = []
        for m in mets_comp:
            va = row_a.get(m, np.nan)
            vb = row_b.get(m, np.nan)
            data_comp.append({"Métrica": m, eq_A: va, eq_B: vb})
        df_comp = pd.DataFrame(data_comp)

        # Barras dobles
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            name=eq_A, x=df_comp['Métrica'], y=df_comp[eq_A],
            marker_color="#e74c3c", opacity=0.85,
            text=df_comp[eq_A].round(2), textposition='outside'
        ))
        fig_comp.add_trace(go.Bar(
            name=eq_B, x=df_comp['Métrica'], y=df_comp[eq_B],
            marker_color="#3498db", opacity=0.85,
            text=df_comp[eq_B].round(2), textposition='outside'
        ))
        fig_comp.update_layout(
            barmode='group', template='simple_white', height=450,
            xaxis_tickangle=-35,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=40, r=40, t=60, b=100)
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        # Tabla comparativa con delta
        st.divider()
        df_comp['Δ (A−B)'] = (df_comp[eq_A] - df_comp[eq_B]).round(2)
        df_comp['Mejor'] = df_comp.apply(
            lambda r: eq_A if (
                (r['Δ (A−B)'] > 0 and r['Métrica'] not in METRICAS_NEGATIVAS) or
                (r['Δ (A−B)'] < 0 and r['Métrica'] in METRICAS_NEGATIVAS)
            ) else (eq_B if r['Δ (A−B)'] != 0 else "—"), axis=1
        )
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

        # Head to head
        st.divider()
        st.subheader(f"⚔️ Head to Head: {eq_A} vs {eq_B}")
        h2h = partidos[
            ((partidos['equipo'] == eq_A) & (partidos['rival'] == eq_B)) |
            ((partidos['equipo'] == eq_B) & (partidos['rival'] == eq_A))
        ].sort_values('jornada')
        if not h2h.empty:
            h2h_display = h2h[['jornada','equipo','rival','resultado_equipo','resultado_rival','puntos','xG']].copy()
            st.dataframe(h2h_display, use_container_width=True, hide_index=True)
        else:
            st.info("No se han enfrentado todavía en el dataset.")
    elif eq_A == eq_B and eq_A != "—":
        st.warning("Selecciona dos equipos distintos.")

# ── TAB 5: EVOLUCIÓN ─────────────────────────────────────
with tab5:
    st.subheader("📈 Evolución Temporal")
    eq_evol = st.selectbox("Equipo", ["—"] + equipos_lista, key="eq_evol_eq")

    if eq_evol != "—":
        # Posición acumulada por jornada
        st.subheader(f"📍 Posición en la clasificación — {eq_evol}")
        jornadas_unicas = sorted(partidos['jornada'].unique())
        posiciones = []
        for j in jornadas_unicas:
            df_j = partidos[partidos['jornada'] <= j].groupby('equipo').agg(
                pts=('puntos','sum'), gf=('goles','sum'), gc=('goles_contra','sum')
            ).reset_index()
            df_j['dif'] = df_j['gf'] - df_j['gc']
            df_j = df_j.sort_values(['pts','dif','gf'], ascending=False).reset_index(drop=True)
            df_j['pos'] = range(1, len(df_j)+1)
            row_j = df_j[df_j['equipo'] == eq_evol]
            if not row_j.empty:
                posiciones.append({'jornada': j, 'pos': row_j.iloc[0]['pos'],
                                   'pts': row_j.iloc[0]['pts']})

        if posiciones:
            df_pos = pd.DataFrame(posiciones)
            fig_pos = go.Figure()
            fig_pos.add_trace(go.Scatter(
                x=df_pos['jornada'], y=df_pos['pos'],
                mode='lines+markers',
                line=dict(color='#e74c3c', width=2.5),
                marker=dict(size=7, color='#e74c3c'),
                text=df_pos.apply(lambda r: f"J{int(r['jornada'])} — {int(r['pts'])} pts", axis=1),
                hovertemplate="%{text}<extra></extra>",
                name=eq_evol
            ))
            fig_pos.update_layout(
                yaxis=dict(autorange='reversed', title='Posición', dtick=1,
                           range=[len(equipos_lista)+0.5, 0.5]),
                xaxis=dict(title='Jornada', dtick=1),
                template='simple_white', height=380,
                margin=dict(l=50, r=30, t=30, b=50)
            )
            st.plotly_chart(fig_pos, use_container_width=True)

        # Tabla de resultados
        st.divider()
        st.subheader(f"📋 Resultados — {eq_evol}")
        df_res = partidos[partidos['equipo'] == eq_evol].sort_values('jornada')[
            ['jornada','rival','lado','resultado_equipo','resultado_rival','puntos','xG','rating_medio']
        ].copy()
        df_res['Resultado'] = df_res.apply(
            lambda r: f"{int(r['resultado_equipo'])}–{int(r['resultado_rival'])}", axis=1)
        df_res['R'] = df_res['puntos'].map({3:'✅ V', 1:'➖ E', 0:'❌ D'})
        # Puntos acumulados
        df_res['Pts acum'] = df_res['puntos'].cumsum()
        st.dataframe(
            df_res[['jornada','rival','lado','Resultado','R','puntos','Pts acum','xG','rating_medio']],
            use_container_width=True, hide_index=True
        )
    else:
        st.info("Selecciona un equipo para ver su evolución.")

# ── TAB 6: SCATTER ───────────────────────────────────────
with tab6:
    st.subheader("🔵 Scatter de Equipos")
    numeric_eq = sorted([
        c for c in eq_view.select_dtypes(include='number').columns
        if c not in ['Pos','Partidos','Victorias','Empates','Derrotas',
                     'Goles_favor','Goles_contra','xG_total','xA_total'] and '_pct' not in c
    ])

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        x_eq = st.selectbox("Eje X", numeric_eq, key="eq_sc_x",
                             index=None, placeholder="Métrica X...")
    with col_s2:
        y_eq = st.selectbox("Eje Y", numeric_eq, key="eq_sc_y",
                             index=None, placeholder="Métrica Y...")
    with col_s3:
        color_eq = st.selectbox("Color por", ["Puntos","Perfil táctico","Ninguno"],
                                 key="eq_sc_color")

    col_s4, col_s5 = st.columns(2)
    with col_s4:
        hl_eq = st.selectbox("Destacar equipo", ["Ninguno"] + equipos_lista,
                              key="eq_sc_hl")
    with col_s5:
        show_avg_eq = st.checkbox("Líneas de media", value=True, key="eq_sc_avg")

    if x_eq and y_eq:
        df_sc = eq_view[['equipo','Puntos', x_eq, y_eq]].dropna(subset=[x_eq, y_eq])

        if color_eq == "Puntos":
            mc = df_sc['Puntos']; cs = "RdYlGn"; ss = True; cb = dict(title="Puntos")
        else:
            mc = "#3498db"; cs = None; ss = False; cb = None

        mask_hl = df_sc['equipo'] == hl_eq
        fig_eq = go.Figure()

        # Puntos normales
        df_n = df_sc[~mask_hl]
        mc_n = mc[~mask_hl] if hasattr(mc, '__getitem__') and not isinstance(mc, str) else mc
        fig_eq.add_trace(go.Scatter(
            x=df_n[x_eq], y=df_n[y_eq], mode='markers+text',
            marker=dict(size=10, color=mc_n, colorscale=cs,
                        showscale=ss, colorbar=cb,
                        opacity=0.7, line=dict(width=0.5, color='white')),
            text=df_n['equipo'],
            textposition='top center',
            textfont=dict(size=8),
            hovertemplate=f"<b>%{{text}}</b><br>{x_eq}: %{{x:.2f}}<br>{y_eq}: %{{y:.2f}}<extra></extra>",
            showlegend=False
        ))

        # Equipo destacado
        if hl_eq != "Ninguno":
            df_h = df_sc[mask_hl]
            if not df_h.empty:
                fig_eq.add_trace(go.Scatter(
                    x=df_h[x_eq], y=df_h[y_eq], mode='markers+text',
                    marker=dict(size=18, color='#e74c3c', line=dict(width=2, color='black')),
                    text=df_h['equipo'], textposition='top center',
                    textfont=dict(size=10, color='#e74c3c'),
                    hovertemplate=f"<b>%{{text}}</b><br>{x_eq}: %{{x:.2f}}<br>{y_eq}: %{{y:.2f}}<extra></extra>",
                    name=hl_eq, showlegend=True
                ))

        # Líneas de media
        if show_avg_eq:
            mx = df_sc[x_eq].mean(); my = df_sc[y_eq].mean()
            fig_eq.add_vline(x=mx, line_dash='dash',
                             line_color='rgba(100,100,100,0.5)', line_width=1.5)
            fig_eq.add_hline(y=my, line_dash='dash',
                             line_color='rgba(100,100,100,0.5)', line_width=1.5)
            fig_eq.add_trace(go.Scatter(
                x=[mx], y=[df_sc[y_eq].max()], mode='text',
                text=[f"  μ={mx:.2f}"], textposition='middle right',
                textfont=dict(size=9, color='gray'), showlegend=False, hoverinfo='skip'
            ))
            fig_eq.add_trace(go.Scatter(
                x=[df_sc[x_eq].max()], y=[my], mode='text',
                text=[f"μ={my:.2f}  "], textposition='middle left',
                textfont=dict(size=9, color='gray'), showlegend=False, hoverinfo='skip'
            ))

        fig_eq.update_layout(
            title=f"{x_eq}  vs  {y_eq}", xaxis_title=x_eq, yaxis_title=y_eq,
            template='simple_white', height=650,
            margin=dict(l=60, r=40, t=60, b=60)
        )
        st.plotly_chart(fig_eq, use_container_width=True)
        st.caption(f"{len(df_sc)} equipos")
    else:
        st.info("Elige métricas para los ejes X e Y.")


# ── TAB 7: INFORME ───────────────────────────────────────
with tab7:
    st.subheader("📝 Informe de Equipo")
    eq_inf = st.selectbox("Equipo", ["—"] + equipos_lista, key="eq_inf_eq")

    if eq_inf != "—":
        row_inf  = eq_pct[eq_pct['equipo'] == eq_inf].iloc[0]
        row_prof = eq_prof[eq_prof['equipo'] == eq_inf].iloc[0]

        # Grupo del equipo y posición dentro del grupo
        grupo_eq  = row_inf.get('Grupo', '?')
        df_grupo  = eq_pct[eq_pct['Grupo'] == grupo_eq].copy()
        df_grupo  = df_grupo.sort_values('Puntos', ascending=False).reset_index(drop=True)
        df_grupo['_pos_grp'] = range(1, len(df_grupo)+1)
        pos_actual = int(df_grupo[df_grupo['equipo'] == eq_inf]['_pos_grp'].values[0])
        n_equipos  = len(df_grupo)

        puntos     = int(row_inf['Puntos'])
        partidos_j = int(row_inf['Partidos'])
        victorias  = int(row_inf['Victorias'])
        empates    = int(row_inf['Empates'])
        derrotas   = int(row_inf['Derrotas'])
        gf         = int(row_inf['Goles_favor'])
        gc         = int(row_inf['Goles_contra'])
        dif        = gf - gc

        scores_perf   = {p: round(float(row_prof[f'Score_{p}']), 2) for p in PERFILES}
        perfil_top    = max(scores_perf, key=scores_perf.get)
        perfil_bajo   = min(scores_perf, key=scores_perf.get)
        perfiles_ord  = sorted(scores_perf.items(), key=lambda x: -x[1])

        metricas_report = [
            'Goles/90', 'xG/90', 'Goles_contra/90', 'Porterias_cero_%',
            'Pases/90', 'Precision_pases_%', 'Pases_rival_%',
            'Recuperaciones/90', 'Duelos_ganados_%',
            'Regates/90', 'xA/90', 'Conversion_%', 'Tiros/90',
            'Entradas_ganadas_%', 'Perdidas/90', 'Faltas_com/90',
        ]
        top_fortalezas, top_debilidades = [], []
        for m in metricas_report:
            pct_col = f"{m}_pct"
            if pct_col in row_inf.index:
                pct_val = row_inf[pct_col]
                val     = row_inf.get(m, None)
                if pd.notna(pct_val) and pd.notna(val):
                    entry = (m, round(float(val), 2), round(float(pct_val) * 100, 1))
                    if pct_val >= 0.70:
                        top_fortalezas.append(entry)
                    elif pct_val <= 0.35:
                        top_debilidades.append(entry)

        top_fortalezas.sort(key=lambda x: -x[2])
        top_debilidades.sort(key=lambda x: x[2])

        df_rec   = partidos[partidos['equipo'] == eq_inf].sort_values('jornada', ascending=False).head(5)
        forma    = df_rec['puntos'].map({3: 'V', 1: 'E', 0: 'D'}).tolist()
        pts_rec  = int(df_rec['puntos'].sum())
        forma_txt = " · ".join(forma) if forma else "Sin datos"

        # ── Helpers narrativos ──────────────────────────────
        def zona_tabla(pos, n):
            tercio = n // 3
            if pos <= tercio:         return "zona alta de la tabla"
            elif pos <= 2 * tercio:   return "zona media de la tabla"
            else:                     return "zona baja de la tabla"

        def valorar_puntos(pts, pj):
            ppp = pts / pj if pj else 0
            if ppp >= 2.0:   return "un rendimiento excelente"
            elif ppp >= 1.5: return "un rendimiento sólido"
            elif ppp >= 1.0: return "un rendimiento discreto"
            else:            return "un rendimiento por debajo de lo esperado"

        def describir_goles(gf, gc, pj):
            gfp = round(gf / pj, 2) if pj else 0
            gcp = round(gc / pj, 2) if pj else 0
            ataque = "muy efectivo" if gfp >= 1.5 else ("efectivo" if gfp >= 1.0 else "poco productivo")
            defensa = "muy solida" if gcp <= 0.8 else ("solida" if gcp <= 1.2 else ("irregular" if gcp <= 1.6 else "vulnerable"))
            return gfp, gcp, ataque, defensa

        def describir_forma(forma, pts_rec):
            if pts_rec >= 12:   return "en un momento de forma excelente"
            elif pts_rec >= 9:  return "con una buena racha de resultados"
            elif pts_rec >= 6:  return "con resultados irregulares"
            elif pts_rec >= 3:  return "atravesando un bache de resultados"
            else:               return "en un momento de forma muy bajo"

        def describir_perfil(perfil):
            desc = {
                "Posesion":   "basado en la posesion del balon y el control del juego mediante pases precisos",
                "Pressing":   "caracterizado por una intensa presion sobre el rival y una rapida recuperacion del balon",
                "Directo":    "directo y vertical, buscando el area rival con rapidez mediante pases largos y centros",
                "Ofensivo":   "marcadamente ofensivo, con gran capacidad goleadora y generacion de ocasiones",
                "Defensivo":  "solido y organizado defensivamente, priorizando la seguridad atras",
                "Creatividad":"creativo y desequilibrante, apoyado en el regate y la asistencia",
            }
            return desc.get(perfil, perfil)

        gfp, gcp, ataque_desc, defensa_desc = describir_goles(gf, gc, partidos_j)
        zona = zona_tabla(pos_actual, n_equipos)
        rendimiento = valorar_puntos(puntos, partidos_j)
        forma_desc = describir_forma(forma, pts_rec)
        perfil_desc = describir_perfil(perfil_top)

        # ── Construir informe ───────────────────────────────
        seccion_resumen = (
            f"{eq_inf} ocupa la {pos_actual}a posicion de {n_equipos} equipos, "
            f"con {puntos} puntos en {partidos_j} partidos ({victorias}V · {empates}E · {derrotas}D), "
            f"lo que supone {rendimiento} ({round(puntos/partidos_j,2)} pts/partido). "
            f"Se encuentran en la {zona}, con un balance goleador de {gf} goles a favor y {gc} en contra "
            f"(diferencia: {dif:+d}). El equipo llega a este punto de la temporada {forma_desc}."
        )

        seccion_perfil = (
            f"Segun el analisis de perfiles tacticos, {eq_inf} presenta un estilo de juego "
            f"{perfil_desc}. Su perfil dominante es **{perfil_top}** "
            f"(score {scores_perf[perfil_top]}/10), mientras que su punto mas debil tactico "
            f"es el perfil de **{perfil_bajo}** (score {scores_perf[perfil_bajo]}/10). "
        )
        # Describe top 3 profiles
        p1, s1 = perfiles_ord[0]
        p2, s2 = perfiles_ord[1]
        p3, s3 = perfiles_ord[2]
        seccion_perfil += (
            f"El equipo combina elementos de {p1} ({s1}/10), {p2} ({s2}/10) y {p3} ({s3}/10), "
            f"lo que dibuja un perfil {('equilibrado' if s1 - s3 < 2 else 'con identidad tactica clara')}."
        )

        if top_fortalezas:
            items_fort = []
            for m, v, pct in top_fortalezas[:5]:
                items_fort.append(f"**{m}** ({v}, percentil {pct}%)")
            seccion_fortalezas = (
                f"{eq_inf} destaca especialmente en las siguientes metricas respecto al resto de la liga: "
                + ", ".join(items_fort[:-1]) + (f" y {items_fort[-1]}" if len(items_fort) > 1 else items_fort[0])
                + ". "
            )
            if any('Goles' in m or 'xG' in m or 'Tiros' in m for m,_,_ in top_fortalezas):
                seccion_fortalezas += "Su capacidad ofensiva es uno de sus activos mas destacados. "
            if any('Pases' in m or 'Posesion' in m for m,_,_ in top_fortalezas):
                seccion_fortalezas += "El control del balon es una de sus senyas de identidad. "
            if any('Recuper' in m or 'Entradas' in m for m,_,_ in top_fortalezas):
                seccion_fortalezas += "Su intensidad defensiva y capacidad de recuperacion destacan en la categoria. "
        else:
            seccion_fortalezas = (
                f"{eq_inf} no presenta metricas especialmente destacadas respecto a la media de la liga, "
                f"lo que sugiere un equipo equilibrado sin areas de excelencia clara."
            )

        if top_debilidades:
            items_deb = []
            for m, v, pct in top_debilidades[:5]:
                items_deb.append(f"**{m}** ({v}, percentil {pct}%)")
            seccion_debilidades = (
                f"Las principales areas de mejora del equipo se encuentran en: "
                + ", ".join(items_deb[:-1]) + (f" y {items_deb[-1]}" if len(items_deb) > 1 else items_deb[0])
                + ". "
            )
            if any('Goles_contra' in m or 'Porterias' in m for m,_,_ in top_debilidades):
                seccion_debilidades += "La solidez defensiva es un aspecto a reforzar. "
            if any('Perdidas' in m or 'Pases' in m for m,_,_ in top_debilidades):
                seccion_debilidades += "La gestion del balon y la precision en el pase pueden mejorar. "
        else:
            seccion_debilidades = (
                f"{eq_inf} no presenta debilidades estadisticas llamativas, "
                f"mostrando un perfil compensado en todas las areas analizadas."
            )

        tendencia = ""
        if pts_rec >= 10:
            tendencia = f"Los ultimos 5 partidos ({forma_txt}) han sido muy positivos, sumando {pts_rec} de 15 puntos posibles. El equipo llega en un estado de forma optimo."
        elif pts_rec >= 7:
            tendencia = f"La forma reciente es buena ({forma_txt}, {pts_rec}/15 pts), con mas victorias que derrotas en el tramo final."
        elif pts_rec >= 4:
            tendencia = f"Los ultimos resultados son irregulares ({forma_txt}, {pts_rec}/15 pts), alternando victorias y derrotas sin encontrar continuidad."
        else:
            tendencia = f"La racha reciente es preocupante ({forma_txt}, solo {pts_rec}/15 pts posibles), lo que puede comprometer los objetivos de la temporada."

        if pos_actual <= n_equipos // 4:
            conclusion_pos = f"{eq_inf} es uno de los equipos mas solidos de la categoria, con numeros que avalan sus aspiraciones."
        elif pos_actual <= n_equipos // 2:
            conclusion_pos = f"{eq_inf} se mantiene en la primera mitad de la tabla con un rendimiento aceptable."
        elif pos_actual <= 3 * n_equipos // 4:
            conclusion_pos = f"{eq_inf} se encuentra en la zona media-baja, necesitando mejorar su rendimiento para alejarse de los puestos de descenso."
        else:
            conclusion_pos = f"{eq_inf} se encuentra en puestos comprometidos y necesita una reaccion urgente."

        seccion_conclusion = (
            f"{conclusion_pos} "
            f"Su identidad tactica basada en {perfil_desc} "
            f"les ha permitido alcanzar {puntos} puntos, aunque "
        )
        if top_debilidades:
            debil_key = top_debilidades[0][0]
            seccion_conclusion += f"el apartado de **{debil_key}** (percentil {top_debilidades[0][2]}%) sigue siendo el area donde mas margen de mejora existe."
        else:
            seccion_conclusion += "el equipo muestra un perfil compensado sin debilidades estadisticas evidentes."

        # ── Generar PDF ──────────────────────────────────────
        def generar_pdf_informe():
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import cm
            from reportlab.lib import colors
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
            from reportlab.lib.enums import TA_LEFT, TA_CENTER
            import io, re

            buf = io.BytesIO()
            doc = SimpleDocTemplate(buf, pagesize=A4,
                                    leftMargin=2*cm, rightMargin=2*cm,
                                    topMargin=2*cm, bottomMargin=2*cm)

            styles = getSampleStyleSheet()
            style_title = ParagraphStyle('Title2', parent=styles['Title'],
                                         fontSize=20, textColor=colors.HexColor('#1a1a2e'),
                                         spaceAfter=4)
            style_sub = ParagraphStyle('Sub', parent=styles['Normal'],
                                       fontSize=10, textColor=colors.HexColor('#666666'),
                                       spaceAfter=12, alignment=TA_CENTER)
            style_h2 = ParagraphStyle('H2', parent=styles['Heading2'],
                                      fontSize=13, textColor=colors.HexColor('#2c3e50'),
                                      spaceBefore=14, spaceAfter=4,
                                      borderPad=2)
            style_body = ParagraphStyle('Body', parent=styles['Normal'],
                                        fontSize=10, leading=15, spaceAfter=8)
            style_kv = ParagraphStyle('KV', parent=styles['Normal'],
                                      fontSize=9, leading=13)

            def clean(txt):
                txt = re.sub(r'\*\*(.*?)\*\*', r'\1', txt)
                txt = txt.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                txt = txt.encode('latin-1', errors='replace').decode('latin-1')
                return txt

            def clean_metric(m):
                return m.replace('%', 'pct').replace('_', ' ').replace('/', ' / ')

            story = []

            # Cabecera
            story.append(Paragraph('Informe de Equipo', style_title))
            story.append(Paragraph(eq_inf, ParagraphStyle('Club', parent=styles['Title'],
                                   fontSize=16, textColor=colors.HexColor('#e74c3c'), spaceAfter=2)))
            story.append(Paragraph(f'Temporada 2025/26 - {grupo_eq} - Jornada {partidos_j}', style_sub))
            story.append(HRFlowable(width='100%', thickness=2, color=colors.HexColor('#e74c3c'), spaceAfter=10))

            # Ficha rapida como tabla
            ficha_data = [
                ['Posicion', f'{pos_actual}o / {n_equipos} ({grupo_eq})', 'Puntos', f'{puntos} pts'],
                ['Partidos', f'{partidos_j} ({victorias}V {empates}E {derrotas}D)', 'Goles', f'{gf} / {gc} ({dif:+d})'],
                ['Perfil dominante', perfil_top, 'Forma reciente', f'{forma_txt} ({pts_rec}/15)'],
            ]
            t = Table(ficha_data, colWidths=[3.5*cm, 6*cm, 3.5*cm, 4*cm])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8f9fa')),
                ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#2c3e50')),
                ('BACKGROUND', (2,0), (2,-1), colors.HexColor('#2c3e50')),
                ('TEXTCOLOR',  (0,0), (0,-1), colors.white),
                ('TEXTCOLOR',  (2,0), (2,-1), colors.white),
                ('FONTNAME',   (0,0), (-1,-1), 'Helvetica'),
                ('FONTSIZE',   (0,0), (-1,-1), 9),
                ('FONTNAME',   (0,0), (0,-1), 'Helvetica-Bold'),
                ('FONTNAME',   (2,0), (2,-1), 'Helvetica-Bold'),
                ('GRID',       (0,0), (-1,-1), 0.5, colors.HexColor('#dee2e6')),
                ('PADDING',    (0,0), (-1,-1), 6),
                ('ROWBACKGROUNDS', (0,0), (-1,-1), [colors.HexColor('#f8f9fa'), colors.HexColor('#ffffff')]),
            ]))
            story.append(t)
            story.append(Spacer(1, 12))

            # Scores tácticos
            story.append(Paragraph("Scores Tacticos (0-10 vs liga)", style_h2))
            scores_data = [["Perfil", "Score", "Nivel"]]
            for p, s in perfiles_ord:
                nivel = "Alto" if s >= 7 else ("Medio" if s >= 4 else "Bajo")
                scores_data.append([clean_metric(p), f"{s:.1f}", nivel])
            ts = Table(scores_data, colWidths=[5*cm, 3*cm, 3*cm])
            ts.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2c3e50')),
                ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
                ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTNAME',   (0,1), (-1,-1), 'Helvetica'),
                ('FONTSIZE',   (0,0), (-1,-1), 9),
                ('GRID',       (0,0), (-1,-1), 0.5, colors.HexColor('#dee2e6')),
                ('PADDING',    (0,0), (-1,-1), 5),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#f8f9fa'), colors.white]),
                ('ALIGN',      (1,0), (1,-1), 'CENTER'),
                ('ALIGN',      (2,0), (2,-1), 'CENTER'),
            ]))
            story.append(ts)
            story.append(Spacer(1, 8))

            # Secciones narrativas
            secciones = [
                ("1. Resumen Ejecutivo", seccion_resumen),
                ("2. Perfil Tactico",    seccion_perfil),
                ("3. Fortalezas",        seccion_fortalezas),
                ("4. Debilidades",       seccion_debilidades),
                ("5. Forma Reciente",    tendencia),
                ("6. Conclusion",        seccion_conclusion),
            ]
            for titulo, texto in secciones:
                story.append(HRFlowable(width="100%", thickness=0.5,
                                        color=colors.HexColor('#dee2e6'), spaceAfter=4))
                story.append(Paragraph(titulo, style_h2))
                story.append(Paragraph(clean(texto), style_body))

            # Fortalezas y debilidades detalladas
            if top_fortalezas or top_debilidades:
                story.append(HRFlowable(width="100%", thickness=0.5,
                                        color=colors.HexColor('#dee2e6'), spaceAfter=4))
                story.append(Paragraph("Metricas Detalladas", style_h2))
                met_data = [["Metrica", "Valor", "Percentil", "Tipo"]]
                for m, v, pct in top_fortalezas[:6]:
                    met_data.append([clean_metric(m), str(v), f"{pct}%", "Fortaleza"])
                for m, v, pct in top_debilidades[:6]:
                    met_data.append([clean_metric(m), str(v), f"{pct}%", "Debilidad"])
                tm = Table(met_data, colWidths=[5.5*cm, 2.5*cm, 2.5*cm, 2.5*cm])
                tm.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2c3e50')),
                    ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
                    ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
                    ('FONTNAME',   (0,1), (-1,-1), 'Helvetica'),
                    ('FONTSIZE',   (0,0), (-1,-1), 9),
                    ('GRID',       (0,0), (-1,-1), 0.5, colors.HexColor('#dee2e6')),
                    ('PADDING',    (0,0), (-1,-1), 5),
                    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#f8f9fa'), colors.white]),
                ]))
                story.append(tm)

            # Pie
            story.append(Spacer(1, 20))
            story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e74c3c')))
            story.append(Paragraph("Generado con AppScout · 1a RFEF 2025/26",
                                   ParagraphStyle('Footer', parent=styles['Normal'],
                                                  fontSize=8, textColor=colors.grey,
                                                  alignment=TA_CENTER, spaceBefore=6)))
            doc.build(story)
            buf.seek(0)
            return buf.read()

        pdf_bytes = generar_pdf_informe()
        st.download_button(
            label="📄 Descargar informe PDF",
            data=pdf_bytes,
            file_name=f"informe_{eq_inf.replace(' ','_')}.pdf",
            mime="application/pdf",
            type="primary",
        )
        st.divider()

        # ── Render ───────────────────────────────────────────
        col_inf1, col_inf2 = st.columns([2, 1])

        with col_inf1:
            st.markdown(f"## Informe — {eq_inf}")
            st.markdown(f"*Temporada 2025/26 · 1a RFEF · {grupo_eq} · Generado con datos hasta la jornada {partidos_j}*")
            st.divider()

            st.markdown("### 1. Resumen ejecutivo")
            st.markdown(seccion_resumen)

            st.markdown("### 2. Perfil tactico")
            st.markdown(seccion_perfil)

            st.markdown("### 3. Fortalezas")
            st.markdown(seccion_fortalezas)

            st.markdown("### 4. Debilidades")
            st.markdown(seccion_debilidades)

            st.markdown("### 5. Forma reciente")
            st.markdown(tendencia)

            st.markdown("### 6. Conclusion")
            st.markdown(seccion_conclusion)

        with col_inf2:
            st.markdown("### Ficha rapida")
            st.metric("Posicion", f"{pos_actual}o / {n_equipos}")
            st.metric("Puntos", f"{puntos} pts ({partidos_j} partidos)")
            st.metric("Balance goles", f"{gf} · {gc} ({dif:+d})")
            st.metric("Perfil dominante", perfil_top)
            st.metric("Forma (ult. 5)", f"{forma_txt} · {pts_rec}/15")
            st.divider()
            st.markdown("**Scores tacticos:**")
            for p, s in perfiles_ord:
                bar = "█" * int(s) + "░" * (10 - int(s))
                st.markdown(f"`{p:<12}` {bar} **{s:.1f}**")
            st.divider()
            if top_fortalezas:
                st.markdown("**Top fortalezas:**")
                for m, v, pct in top_fortalezas[:4]:
                    st.markdown(f"✅ `{m}` — P{pct}%")
            if top_debilidades:
                st.markdown("**Top debilidades:**")
                for m, v, pct in top_debilidades[:4]:
                    st.markdown(f"⚠️ `{m}` — P{pct}%")
    else:
        st.info("Selecciona un equipo para generar su informe.")

