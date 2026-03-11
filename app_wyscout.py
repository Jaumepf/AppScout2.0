import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import plotly.graph_objects as go
from matplotlib.patches import Arc, Rectangle, Circle
from sklearn.metrics.pairwise import cosine_similarity


warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="Scouting Wyscout Pro")
st.title("⚽ Scouting & Role Scoring Engine v1.2")

# ==========================================================
# CONFIGURACIÓN
# ==========================================================

NORMALIZATION_MODE = "role"  # "role" o "global"

# ==========================================================
# PESOS ROLES (MANTENGO LOS TUYOS SIN MODIFICAR)
# ==========================================================

pesos_roles_mejorados = {

    # ==============================
    # PORTERO
    # ==============================
    "Portero": {
        "Shot_Stop_Index": 0.16,
        "Goles evitados/90": 0.14,
        "Paradas, %": 0.12,
        "xG en contra/90": 0.10,
        "Goles recibidos/90": 0.10,
        "Salidas/90": 0.08,
        "Pases largos/90": 0.08,
        "Precisión pases largos, %": 0.08,
        "Precisión pases, %": 0.08,
        "Discipline_Index": 0.06
    },

    # ==============================
    # PORTERO AVANZADO
    # ==============================
    "Portero_Avanzado": {
        "Shot_Stop_Index": 0.14,
        "Goles evitados/90": 0.12,
        "Paradas, %": 0.10,
        "Pases largos/90": 0.12,
        "Precisión pases largos, %": 0.10,
        "Pases progresivos/90": 0.10,
        "Precisión pases, %": 0.10,
        "Salidas/90": 0.08,
        "Verticalidad": 0.08,
        "Discipline_Index": 0.06
    },

    # ==============================
    # LATERAL DEFENSIVO
    # ==============================
    "Lateral_Defensivo": {
        "Acciones defensivas realizadas/90": 0.14,
        "Duelos defensivos/90": 0.12,
        "Duelos defensivos ganados, %": 0.12,
        "Interceptaciones/90": 0.10,
        "Entradas/90": 0.10,
        "Recuperaciones": 0.10,
        "Duelo_Eficiencia": 0.08,
        "Faltas/90": 0.08,
        "Discipline_Index": 0.08,
        "Pases/90": 0.08
    },

    # ==============================
    # LATERAL OFENSIVO
    # ==============================
    "Lateral_Ofensivo": {
        "Carreras en progresión/90": 0.14,
        "Aceleraciones/90": 0.12,
        "Regates/90": 0.12,
        "Centros/90": 0.10,
        "Precisión centros, %": 0.10,
        "Pases progresivos/90": 0.10,
        "Verticalidad": 0.08,
        "Threat_Index": 0.08,
        "Recuperaciones": 0.08,
        "Discipline_Index": 0.08
    },

    # ==============================
    # CENTRAL STOPPER
    # ==============================
    "Central_Stopper": {
        "Duelos defensivos/90": 0.14,
        "Duelos defensivos ganados, %": 0.14,
        "Duelos aéreos en los 90": 0.12,
        "Duelos aéreos ganados, %": 0.12,
        "Interceptaciones/90": 0.10,
        "Tiros interceptados/90": 0.10,
        "Entradas/90": 0.08,
        "Recuperaciones": 0.08,
        "Faltas/90": 0.06,
        "Discipline_Index": 0.06
    },

    # ==============================
    # CENTRAL CLÁSICO
    # ==============================
    "Central_Clasico": {
        "Duelos defensivos ganados, %": 0.14,
        "Duelos aéreos ganados, %": 0.12,
        "Interceptaciones/90": 0.12,
        "Tiros interceptados/90": 0.10,
        "Pases hacia adelante/90": 0.10,
        "Pases largos/90": 0.10,
        "Precisión pases largos, %": 0.10,
        "Duelo_Eficiencia": 0.08,
        "Recuperaciones": 0.08,
        "Discipline_Index": 0.06
    },

    # ==============================
    # CENTRAL SALIDA
    # ==============================
    "Central_Salida": {
        "Pases/90": 0.14,
        "Precisión pases, %": 0.12,
        "Pases progresivos/90": 0.14,
        "Precisión pases progresivos, %": 0.10,
        "Pases largos/90": 0.10,
        "Precisión pases largos, %": 0.10,
        "Ratio_Pases_Adelante": 0.08,
        "Verticalidad": 0.08,
        "Duelo_Eficiencia": 0.08,
        "Discipline_Index": 0.06
    },

    # ==============================
    # PIVOTE DEFENSIVO
    # ==============================
    "Pivote_Defensivo": {
        "Acciones defensivas realizadas/90": 0.14,
        "Interceptaciones/90": 0.12,
        "Entradas/90": 0.12,
        "Recuperaciones": 0.10,
        "Duelos defensivos ganados, %": 0.10,
        "Pases/90": 0.10,
        "Precisión pases, %": 0.08,
        "Pases progresivos/90": 0.08,
        "Faltas/90": 0.08,
        "Discipline_Index": 0.08
    },

    # ==============================
    # INTERIOR
    # ==============================
    "Interior": {
        "Pases en el último tercio/90": 0.14,
        "Precisión pases en el último tercio, %": 0.12,
        "Pases progresivos/90": 0.12,
        "Jugadas claves/90": 0.10,
        "xA/90": 0.10,
        "Verticalidad": 0.10,
        "Threat_Index": 0.08,
        "Regates/90": 0.08,
        "Area_Involvement": 0.08,
        "Discipline_Index": 0.08
    },

    # ==============================
    # BOX TO BOX
    # ==============================
    "Box_to_Box": {
        "Carreras en progresión/90": 0.14,
        "Aceleraciones/90": 0.12,
        "Duelo_Eficiencia": 0.10,
        "Recuperaciones": 0.10,
        "Pases progresivos/90": 0.10,
        "Threat_Index": 0.10,
        "Area_Involvement": 0.08,
        "xG_Overperformance_90": 0.08,
        "Verticalidad": 0.10,
        "Discipline_Index": 0.08
    },

    # ==============================
    # MEDIAPUNTA
    # ==============================
    "Mediapunta": {
        "Jugadas claves/90": 0.16,
        "xA/90": 0.14,
        "xA_Overperformance": 0.12,
        "Pases al área de penalti/90": 0.10,
        "Pases progresivos/90": 0.10,
        "Regates/90": 0.10,
        "Threat_Index": 0.08,
        "Area_Involvement": 0.08,
        "Ratio_Pases_Adelante": 0.06,
        "Discipline_Index": 0.06
    },

    # ==============================
    # EXTREMO ASOCIATIVO
    # ==============================
    "Extremo_Asociativo": {
        "Asistencias/90": 0.14,
        "xA/90": 0.14,
        "Jugadas claves/90": 0.12,
        "Pases progresivos/90": 0.10,
        "Centros/90": 0.10,
        "Precisión centros, %": 0.10,
        "Threat_Index": 0.08,
        "Verticalidad": 0.08,
        "Area_Involvement": 0.08,
        "Discipline_Index": 0.06
    },

    # ==============================
    # EXTREMO PURO
    # ==============================
    "Extremo_Puro": {
        "Regates/90": 0.16,
        "Regates realizados, %": 0.12,
        "Duelos atacantes/90": 0.10,
        "Duelos atacantes ganados, %": 0.10,
        "Carreras en progresión/90": 0.12,
        "Aceleraciones/90": 0.10,
        "Threat_Index": 0.10,
        "Area_Involvement": 0.08,
        "xG_Overperformance_90": 0.06,
        "Discipline_Index": 0.06
    },

    # ==============================
    # DELANTERO GOLEADOR
    # ==============================
    "Delantero_Goleador": {
        "Goles/90": 0.16,
        "xG/90": 0.14,
        "Conversion_Gol_%": 0.12,
        "xG_Overperformance_90": 0.12,
        "Remates/90": 0.10,
        "Tiros a la portería, %": 0.10,
        "Area_Involvement": 0.10,
        "Threat_Index": 0.08,
        "Carreras en progresión/90": 0.04,
        "Discipline_Index": 0.04
    },

    # ==============================
    # DELANTERO MÓVIL
    # ==============================
    "Delantero_Movil": {
        "Goles/90": 0.14,
        "xG/90": 0.12,
        "Asistencias/90": 0.12,
        "xA/90": 0.10,
        "Regates/90": 0.10,
        "Carreras en progresión/90": 0.10,
        "Threat_Index": 0.10,
        "Area_Involvement": 0.08,
        "Verticalidad": 0.08,
        "Discipline_Index": 0.06
    }

}


metricas_negativas = [

    # -----------------------
    # DISCIPLINA
    # -----------------------
    'Faltas/90',
    'Tarjetas amarillas',
    'Tarjetas amarillas/90',
    'Tarjetas rojas',
    'Tarjetas rojas/90',

    # -----------------------
    # PORTEROS / DEFENSA GOL
    # -----------------------
    'Goles recibidos',
    'Goles recibidos/90',
    'Remates en contra',
    'Remates en contra/90',
    'xG en contra',
    'xG en contra/90',
]


roles_metrics = {
    rol: list(metrics.keys())
    for rol, metrics in pesos_roles_mejorados.items()
}
# ==========================================================
# POSICIONES — NORMALIZACIÓN
# ==========================================================

pos_equivalencias = {
    "GK": ["GK"],

    "CB": ["CB","RCB","LCB"],
    "RB": ["RB","RWB"],
    "LB": ["LB","LWB"],

    "DM": ["DMF","LDMF","RDMF"],
    "CM": ["CMF","LCMF","RCMF","MF"],
    "AM": ["AMF","LAMF","RAMF","CAM","AM"],

    "RW": ["RW","RWF"],
    "LW": ["LW","LWF"],

    "FW": ["FW","CF","ST","S"]
}

rol_pos_map = {
    "Portero": ["GK"],
    "Portero_Avanzado": ["GK"],

    "Lateral_Defensivo": ["RB","LB"],
    "Lateral_Ofensivo": ["RB","LB"],

    "Central_Clasico": ["CB"],
    "Central_Salida": ["CB"],
    "Central_Stopper": ["CB"],

    "Pivote_Defensivo": ["DM"],
    "Interior": ["CM","AM"],
    "Box_to_Box": ["CM"],
    "Mediapunta": ["AM"],

    "Extremo_Asociativo": ["RW","LW"],
    "Extremo_Puro": ["RW","LW"],

    "Delantero_Movil": ["FW"],
    "Delantero_Goleador": ["FW"]
}

def normalize_positions(pos_string):
    if pd.isna(pos_string):
        return []

    tokens = [p.strip().upper() for p in str(pos_string).split(",")]
    categorias = set()

    for token in tokens:
        for categoria, equivalencias in pos_equivalencias.items():
            if token in equivalencias:
                categorias.add(categoria)

    return list(categorias)

# ==========================================================
# NORMALIZACIÓN (RANK PERCENTILE ESTABLE)
# ==========================================================

def percentile_normalization(data, metrics):

    df = data.copy()

    for metric in metrics:

        if metric not in df.columns:
            continue

        if df[metric].dropna().shape[0] < 2:
            df[metric] = 0.5
            continue

        ranks = df[metric].rank(pct=True)

        if metric in metricas_negativas:
            df[metric] = 1 - ranks
        else:
            df[metric] = ranks

        df[metric] = df[metric].clip(0, 1)

    return df

# ==========================================================
# DERIVED METRICS
# ==========================================================

def safe_div(a, b):
    return np.where((b == 0) | (pd.isna(b)), 0, a / b)

def add_derived_metrics(df):

    df["Conversion_Gol_%"] = safe_div(df["Goles"], df["Remates"])
    df["xG_Overperformance_90"] = df["Goles/90"] - df["xG/90"]
    df["xA_Overperformance"] = df["Asistencias"] - df["xA"]

    df["Ratio_Pases_Adelante"] = safe_div(
        df["Pases hacia adelante/90"],
        df["Pases/90"]
    )

    df["Duelo_Eficiencia"] = (
        safe_div(df["Duelos ganados, %"], 100) *
        df["Duelos/90"]
    )

    df["Recuperaciones"] = (
        df["Entradas/90"] +
        df["Interceptaciones/90"]
    )

    df["Threat_Index"] = (
        df["Goles/90"] +
        df["xA/90"] +
        df["Regates/90"] +
        df["Pases al área de penalti/90"]
    )

    df["Verticalidad"] = (
        df["Carreras en progresión/90"] +
        df["Pases progresivos/90"]
    )

    df["Area_Involvement"] = (
        df["Toques en el área de penalti/90"] +
        df["Remates/90"]
    )

    df["Discipline_Index"] = (
        df["Tarjetas amarillas/90"] +
        2 * df["Tarjetas rojas/90"]
    )

    df["Shot_Stop_Index"] = (
        df["Paradas, %"] -
        df["xG en contra/90"]
    )

    return df
def compute_role_percentiles(players, min_minutes_base=600):

    players = players.copy()

    for rol, weights in pesos_roles_mejorados.items():

        allowed_positions = rol_pos_map.get(rol, [])
        metrics = list(weights.keys())

        df_role = players[
            (players["Minutos jugados"] >= min_minutes_base) &
            (players["Pos_norm"].apply(
                lambda x: any(p in allowed_positions for p in x)
            ))
        ].copy()

        if df_role.empty:
            continue

        df_norm = percentile_normalization(df_role, metrics)

        for metric in metrics:
            if metric in df_norm.columns:
                pct_col = f"{metric}_pct"
                players.loc[df_norm.index, pct_col] = df_norm[metric]

    return players

# ==========================================================
# SCORING CON PENALIZACIÓN POR MINUTOS
# ==========================================================

def compute_role_scores(players, min_minutes):

    role_scores = {}

    for rol, weights in pesos_roles_mejorados.items():

        df = players.copy()

        # 🔹 Filtro por minutos (solo afecta al ranking, no a percentiles)
        df = df[df["Minutos jugados"] >= min_minutes]

        # 🔹 Filtro posicional obligatorio
        allowed_positions = rol_pos_map.get(rol, [])

        if "Pos_norm" in df.columns:
            df = df[
                df["Pos_norm"].apply(
                    lambda lst: any(p in allowed_positions for p in lst)
                )
            ]

        if df.empty:
            continue

        metrics = [m for m in weights if m in df.columns]
        if not metrics:
            continue

        scores = []

        for _, row in df.iterrows():
            score = sum(
                row.get(f"{m}_pct", 0) * weights[m]
                for m in metrics
                if not pd.isna(row.get(f"{m}_pct", 0))
            )
            scores.append(score)

        df["Rating"] = np.round(np.array(scores) * 10, 2)

        role_scores[rol] = df.sort_values("Rating", ascending=False)

    return role_scores
    
# ==========================================================
# SIMILARIDAD (COSINE SIMILARITY)
# ==========================================================

def find_similar_players(players_df, role, player_name, min_minutes, top_n=5):

    # -------------------------------------------------
    # 1️⃣ Filtrar por minutos mínimos
    # -------------------------------------------------
    df = players_df[
        players_df["Minutos jugados"] >= min_minutes
    ].copy()

    # -------------------------------------------------
    # 2️⃣ Filtrar por posición del rol
    # -------------------------------------------------
    allowed_positions = rol_pos_map.get(role, [])

    df = df[
        df["Pos_norm"].apply(
            lambda x: any(p in allowed_positions for p in x)
        )
    ]

    if df.empty:
        return pd.DataFrame()

    # -------------------------------------------------
    # 3️⃣ Métricas del rol (percentiles estables)
    # -------------------------------------------------
    weights = pesos_roles_mejorados[role]

    pct_metrics = [
        f"{m}_pct"
        for m in weights
        if f"{m}_pct" in df.columns
    ]

    if not pct_metrics:
        return pd.DataFrame()

    # -------------------------------------------------
    # 4️⃣ Eliminar jugadores con NaN en esas métricas
    # -------------------------------------------------
    df = df.dropna(subset=pct_metrics)

    if df.empty:
        return pd.DataFrame()

    # -------------------------------------------------
    # 5️⃣ Seleccionar jugador target (solo 1 fila)
    # -------------------------------------------------
    target = df[df["Jugador"] == player_name]

    if target.empty:
        return pd.DataFrame()

    # 🔹 Si hay duplicados, usar el de más minutos
    target = target.sort_values(
        "Minutos jugados",
        ascending=False
    ).iloc[0]

    target_vector = target[pct_metrics].values.reshape(1, -1)

    # -------------------------------------------------
    # 6️⃣ Matriz universo
    # -------------------------------------------------
    matrix = df[pct_metrics].values

    # -------------------------------------------------
    # 7️⃣ Cosine similarity
    # -------------------------------------------------
    similarities = cosine_similarity(
        matrix,
        target_vector
    ).flatten()

    df["Similarity"] = similarities

    # -------------------------------------------------
    # 8️⃣ Eliminar el propio jugador
    # -------------------------------------------------
    df = df[df["Jugador"] != player_name]

    # -------------------------------------------------
    # 9️⃣ Ordenar y devolver
    # -------------------------------------------------
    return df.sort_values(
        "Similarity",
        ascending=False
    ).head(top_n)[
        [
            "Jugador",
            "Equipo durante el período seleccionado",
            "Posición específica",
            "Similarity"
        ]
    ]
# ==========================================================
# FUNCIONES
# ==========================================================

@st.cache_data
def load_data(files):

    dfs = [pd.read_excel(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    # -------------------------------------------------
    # 🔥 ELIMINAR DUPLICADOS
    # -------------------------------------------------

    # Si tienes columna Año → usarla
    if "Año" in df.columns:
        df = df.sort_values("Minutos jugados", ascending=False)
        df = df.drop_duplicates(
            subset=["Jugador", "Equipo durante el período seleccionado", "Año"],
            keep="first"
        )
    else:
        df = df.sort_values("Minutos jugados", ascending=False)
        df = df.drop_duplicates(
            subset=["Jugador", "Equipo durante el período seleccionado"],
            keep="first"
        )

    # -------------------------------------------------

    if "Posición específica" in df.columns:
        df["Pos_primary"] = (
            df["Posición específica"]
            .astype(str)
            .apply(lambda x: x.split(",")[0].strip().upper())
        )
        df["Pos_norm"] = df["Posición específica"].apply(normalize_positions)
    else:
        df["Pos_norm"] = [[] for _ in range(len(df))]

    return df

def best_roles_for_player(player_name, players, top_n=3):

    results = []

    for rol, weights in pesos_roles_mejorados.items():

        metrics = [m for m in weights if m in players.columns]
        if not metrics:
            continue

        df_norm = percentile_normalization(players, metrics)
        row = df_norm[df_norm["Jugador"] == player_name]

        if row.empty:
            continue

        row = row.iloc[0]
        score = sum(row[m] * weights[m] for m in metrics if not pd.isna(row[m]))

        results.append((rol, round(score * 10, 2)))

    results = sorted(results, key=lambda x: x[1], reverse=True)
    return results[:top_n]

def top_players_for_role(role_scores, role, top_n=3):

    if role not in role_scores or role_scores[role].empty:
        return []

    df = role_scores[role].head(top_n)

    result = []
    for _, row in df.iterrows():
        result.append((row["Jugador"], row["Rating"]))

    return result

def best_roles_for_player_smart(player_name, players, min_minutes, top_n=3):

    df = players[players["Minutos jugados"] >= min_minutes].copy()
    results = []

    player_row = df[df["Jugador"] == player_name]
    if player_row.empty:
        return []

    player_positions = player_row.iloc[0]["Pos_norm"]
    is_gk = "GK" in player_positions

    if is_gk:
        allowed_roles = ["Portero", "Portero_Avanzado"]
    else:
        allowed_roles = [
            rol for rol, pos_list in rol_pos_map.items()
            if any(p in player_positions for p in pos_list)
    ]

    for rol in allowed_roles:

        weights = pesos_roles_mejorados[rol]
        metrics = [m for m in weights if f"{m}_pct" in df.columns]

        if not metrics:
            continue

        row = df[df["Jugador"] == player_name]
        if row.empty:
            continue

        row = row.iloc[0]

        score = sum(
            row.get(f"{m}_pct", 0) * weights[m]
            for m in metrics
            if not pd.isna(row.get(f"{m}_pct", 0))
        )

        results.append((rol, round(score * 10, 2)))

    results = sorted(results, key=lambda x: x[1], reverse=True)

    if is_gk:
        return results[:2]

    return results[:top_n]

def radar_plot(df, role, players_selected):

    weights = pesos_roles_mejorados[role]
    pct_metrics = [f"{m}_pct" for m in weights if f"{m}_pct" in df.columns]

    if not pct_metrics:
        st.warning("No hay métricas disponibles para este rol.")
        return

    N = len(pct_metrics)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for player in players_selected:

        row = df[df["Jugador"] == player]
        if row.empty:
            continue

        values = row.iloc[0][pct_metrics].tolist()
        values += values[:1]

        ax.plot(angles, values, linewidth=2, label=player)
        ax.fill(angles, values, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace("_pct","") for m in pct_metrics], fontsize=9)
    ax.set_ylim(0,1)
    ax.set_yticklabels([])

    plt.legend(loc="upper right")
    st.pyplot(fig)
    
def radar_vs_role_top_player(players_df, player_name, role, min_minutes):

    df = players_df[players_df["Minutos jugados"] >= min_minutes].copy()

    # 🔹 Filtrar por posición del rol
    allowed_positions = rol_pos_map.get(role, [])
    df = df[df["Pos_norm"].apply(lambda x: any(p in allowed_positions for p in x))]

    if df.empty:
        return

    # 🔹 Usar percentiles ya calculados
    weights = pesos_roles_mejorados[role]
    pct_metrics = [f"{m}_pct" for m in weights if f"{m}_pct" in df.columns]

    if not pct_metrics:
        return

    player_row = df[df["Jugador"] == player_name]
    if player_row.empty:
        return

    # 🔹 Score estable
    scores = []
    for _, row in df.iterrows():
        s = sum(
            row.get(m, 0) * weights[m.replace("_pct","")]
            for m in pct_metrics
        )
        scores.append(s)

    df["Score"] = scores
    top_player = df.sort_values("Score", ascending=False).iloc[0]["Jugador"]

    top_row = df[df["Jugador"] == top_player]

    p_vals = player_row.iloc[0][pct_metrics].tolist()
    t_vals = top_row.iloc[0][pct_metrics].tolist()

    angles = np.linspace(0, 2*np.pi, len(pct_metrics), endpoint=False).tolist()
    angles += angles[:1]

    p_vals += p_vals[:1]
    t_vals += t_vals[:1]

    fig, ax = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))

    ax.plot(angles, p_vals, linewidth=2, label=player_name)
    ax.fill(angles, p_vals, alpha=0.15)

    ax.plot(angles, t_vals, linewidth=2, linestyle="--", label=f"Top {role}")
    ax.fill(angles, t_vals, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace("_pct","") for m in pct_metrics], fontsize=8)
    ax.set_ylim(0,1)
    ax.set_yticklabels([])

    ax.grid(alpha=0.25)

    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        frameon=False,
        fontsize=8
    )

    plt.tight_layout()
    st.pyplot(fig)
    
def radar_vs_role_best(players_df, player_name, role):
    metrics = roles_metrics[role]

    df_role = players_df.copy()

    # normalizamos todo el rol
    df_norm = percentile_normalization(df_role, metrics)

    # jugador seleccionado
    player_row = df_norm[df_norm["Jugador"] == player_name]
    if player_row.empty:
        st.warning("Jugador no encontrado")
        return

    player_values = player_row.iloc[0][metrics].tolist()

    # MEJOR VALOR POR MÉTRICA DEL ROL
    role_best = df_norm[metrics].max().tolist()

    N = len(metrics)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    player_values += player_values[:1]
    role_best += role_best[:1]

    fig, ax = plt.subplots(figsize=(7,7), subplot_kw=dict(polar=True))

    # jugador
    ax.plot(angles, player_values, linewidth=2, label=player_name)
    ax.fill(angles, player_values, alpha=0.25)

    # rol ideal
    ax.plot(angles, role_best, linewidth=2, linestyle="--", label="Ideal Rol")
    ax.fill(angles, role_best, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylim(0,1)

    plt.legend(loc="upper right")
    st.pyplot(fig)



def best_player_for_role(role_scores, role, used_players, side=None):

    if role not in role_scores:
        return "—"

    df_full = role_scores[role]

    # Excluir jugadores ya usados
    df_full = df_full[~df_full["Jugador"].isin(used_players)]

    if df_full.empty:
        return "—"

    # =========================
    # FILTRO POR LADO
    # =========================

    if side:

        # ---- EXTREMOS ----
        if role in ["Extremo_Puro", "Extremo_Asociativo"]:

            if side == "left":
                primary_filter = df_full["Pos_primary"].isin(["LW","LWF"])
                norm_filter = df_full["Pos_norm"].apply(lambda x: "LW" in x)

            elif side == "right":
                primary_filter = df_full["Pos_primary"].isin(["RW","RWF"])
                norm_filter = df_full["Pos_norm"].apply(lambda x: "RW" in x)

        # ---- LATERALES ----
        elif role in ["Lateral_Defensivo", "Lateral_Ofensivo"]:

            if side == "left":
                primary_filter = df_full["Pos_primary"].isin(["LB","LWB"])
                norm_filter = df_full["Pos_norm"].apply(lambda x: "LB" in x)

            elif side == "right":
                primary_filter = df_full["Pos_primary"].isin(["RB","RWB"])
                norm_filter = df_full["Pos_norm"].apply(lambda x: "RB" in x)

        else:
            primary_filter = None
            norm_filter = None

        # 🥇 1º intento → primaria correcta
        if primary_filter is not None:
            df_primary = df_full[primary_filter]
            if not df_primary.empty:
                return df_primary.iloc[0]["Jugador"]

        # 🥈 2º intento → posición secundaria válida
        if norm_filter is not None:
            df_secondary = df_full[norm_filter]
            if not df_secondary.empty:
                return df_secondary.iloc[0]["Jugador"]

    # 🥉 3º intento → mejor disponible del rol
    return df_full.iloc[0]["Jugador"]
def player_percentiles(players, player_name, role):

    weights = pesos_roles_mejorados.get(role, {})
    pct_metrics = [f"{m}_pct" for m in weights if f"{m}_pct" in players.columns]

    if not pct_metrics:
        return pd.DataFrame()

    row = players[players["Jugador"] == player_name]
    if row.empty:
        return pd.DataFrame()

    row = row.iloc[0]

    data = []

    for m in pct_metrics:
        percentile = row.get(m, 0) * 100
        data.append({
            "Métrica": m.replace("_pct", ""),
            "Percentil": round(percentile, 1)
        })

    return pd.DataFrame(data)

def percentile_color(p):

    if p >= 80:
        return "#2ecc71"   # verde
    elif p >= 60:
        return "#3498db"   # azul
    elif p >= 40:
        return "#f1c40f"   # amarillo
    elif p >= 20:
        return "#e67e22"   # naranja
    else:
        return "#e74c3c"   # rojo
    
def plot_percentiles(df_percent):

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = [percentile_color(p) for p in df_percent["Percentil"]]

    ax.barh(
        df_percent["Métrica"],
        df_percent["Percentil"],
        color=colors
    )

    ax.set_xlim(0, 100)
    ax.set_xlabel("Percentil")
    ax.set_title("Rendimiento por Métrica")

    ax.axvline(20, color="grey", linestyle="--", alpha=0.3)
    ax.axvline(40, color="grey", linestyle="--", alpha=0.3)
    ax.axvline(60, color="grey", linestyle="--", alpha=0.3)
    ax.axvline(80, color="grey", linestyle="--", alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)    


def draw_pitch():
    fig, ax = plt.subplots(figsize=(7, 11))

    # Fondo verde césped
    fig.patch.set_facecolor("#2E7D32")
    ax.set_facecolor("#2E7D32")

    line_color = "white"
    lw = 2

    # Bordes campo
    ax.plot([0, 0], [0, 100], color=line_color, lw=lw)
    ax.plot([0, 100], [100, 100], color=line_color, lw=lw)
    ax.plot([100, 100], [100, 0], color=line_color, lw=lw)
    ax.plot([100, 0], [0, 0], color=line_color, lw=lw)

    # Medio campo
    ax.plot([0, 100], [50, 50], color=line_color, lw=lw)

    # Círculo central
    ax.add_patch(Circle((50, 50), 9, fill=False, color=line_color, lw=lw))
    ax.plot(50, 50, 'o', color=line_color)

    # ÁREAS GRANDES
    ax.add_patch(Rectangle((30, 82), 40, 18, fill=False, ec=line_color, lw=lw))
    ax.add_patch(Rectangle((30, 0), 40, 18, fill=False, ec=line_color, lw=lw))

    # ÁREAS PEQUEÑAS
    ax.add_patch(Rectangle((40, 94), 20, 6, fill=False, ec=line_color, lw=lw))
    ax.add_patch(Rectangle((40, 0), 20, 6, fill=False, ec=line_color, lw=lw))

    # PUNTOS PENALTI
    ax.plot(50, 88, 'o', color=line_color)
    ax.plot(50, 12, 'o', color=line_color)

    # SEMICÍRCULOS ÁREA (LA D BIEN PROPORCIONADA)
    ax.add_patch(
        Arc((50, 84), 14, 14, theta1=200, theta2=340,
            color=line_color, lw=lw)
    )
    ax.add_patch(
        Arc((50, 16), 14, 14, theta1=20, theta2=160,
            color=line_color, lw=lw)
    )

    # CÓRNERS
    r = 3
    ax.add_patch(Arc((0, 0), r*2, r*2, theta1=0, theta2=90,
                     color=line_color, lw=lw))
    ax.add_patch(Arc((100, 0), r*2, r*2, theta1=90, theta2=180,
                     color=line_color, lw=lw))
    ax.add_patch(Arc((0, 100), r*2, r*2, theta1=270, theta2=360,
                     color=line_color, lw=lw))
    ax.add_patch(Arc((100, 100), r*2, r*2, theta1=180, theta2=270,
                     color=line_color, lw=lw))

    # PORTERÍAS
    ax.plot([45, 55], [100, 100], color=line_color, lw=lw)
    ax.plot([45, 55], [0, 0], color=line_color, lw=lw)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    return fig, ax

formation_coords = {
    "4-3-3": {
        "Portero": (50,10),
        "Lateral_Defensivo": (15,30),
        "Lateral_Ofensivo": (85,35),
        "Central_Clasico": (35,20),
        "Central_Salida": (65,20),
        "Pivote_Defensivo": (50,45),
        "Interior": (70,55),
        "Box_to_Box": (30,55),
        "Extremo_Puro": (15,75),
        "Extremo_Asociativo": (85,75),
        "Delantero_Goleador": (50,85)
    },

    "4-4-2": {
        "Portero": (50,10),
        "Lateral_Defensivo": [(15,30),(85,30)],
        "Central_Clasico": [(35,20),(65,20)],
        "Pivote_Defensivo": (40,45),
        "Interior": (60,55),
        "Extremo_Puro": [(15,75),(85,75)],
        "Delantero_Goleador": (40,90),
        "Delantero_Movil": (60,80)
    },

    "3-5-2": {
        "Portero": (50,10),
        "Central_Salida": (50,30),
        "Central_Clasico": [(30,20),(70,20)],
        "Lateral_Ofensivo": [(15,40),(85,40)],
        "Pivote_Defensivo": (50,50),
        "Interior": [(30,65),(70,65)],
        "Delantero_Goleador": (40,90),
        "Delantero_Movil": (60,80)
    },

    "5-3-2": {
        "Portero": (50,10),
        "Central_Clasico": [(30,20),(70,20)],
        "Central_Salida": (50,30),
        "Lateral_Defensivo": [(15,30),(85,30)],
        "Pivote_Defensivo": (50,50),
        "Interior": [(30,65),(70,65)],
        "Delantero_Goleador": (40,90),
        "Delantero_Movil": (60,80)
    },

    "4-5-1": {
        "Portero": (50,10),
        "Lateral_Defensivo": [(15,30),(85,30)],
        "Central_Clasico": [(35,20),(65,20)],
        "Pivote_Defensivo": (50,50),
        "Interior": [(30,65),(70,65)],
        "Extremo_Puro": [(15,75),(85,75)],
        "Delantero_Goleador": (50,90)
    },

    "3-4-3": {
        "Portero": (50,10),
        "Central_Salida": (50,30),
        "Central_Clasico": [(30,20),(70,20)],
        "Lateral_Ofensivo": [(15,50),(85,50)],
        "Pivote_Defensivo": (45,50),
        "Interior": [(55,60)],
        "Extremo_Puro": [(15,80),(85,80)],
        "Delantero_Goleador": (50,90)
    }
}
def plot_formation(formacion, alineacion, role_scores):

    fig, ax = draw_pitch()
    coords_map = formation_coords.get(formacion, {})

    role_counter = {}

    for rol_display, jugador, side in alineacion:

        rol_base = rol_display.split(" ")[0]
        role_counter[rol_base] = role_counter.get(rol_base, 0)

        coord = coords_map.get(rol_base)
        if coord is None:
            continue

        if isinstance(coord, list):
            if role_counter[rol_base] < len(coord):
                x, y = coord[role_counter[rol_base]]
            else:
                continue
        else:
            x, y = coord

        role_counter[rol_base] += 1

        # 🔹 TOP 3 jugadores por lado
        df_role = role_scores.get(rol_base)

        if df_role is None or df_role.empty:
            continue

        df_filtered = df_role.copy()

        # Filtrar por lado si aplica
        if side:

            if rol_base in ["Extremo_Puro", "Extremo_Asociativo"]:

                if side == "left":
                    df_filtered = df_filtered[
                        df_filtered["Pos_primary"].isin(["LW","LWF"])
                    ]
                else:
                    df_filtered = df_filtered[
                        df_filtered["Pos_primary"].isin(["RW","RWF"])
                    ]

            elif rol_base in ["Lateral_Defensivo", "Lateral_Ofensivo"]:

                if side == "left":
                    df_filtered = df_filtered[
                        df_filtered["Pos_primary"].isin(["LB","LWB"])
                    ]
                else:
                    df_filtered = df_filtered[
                        df_filtered["Pos_primary"].isin(["RB","RWB"])
                    ]

        # Si no hay del lado específico → fallback general
        if df_filtered.empty:
            df_filtered = df_role

        top_players = df_filtered.head(3)

        text_lines = [rol_display.replace("_", " ")]

        for i, (_, row) in enumerate(top_players.iterrows(), start=1):
            text_lines.append(f"{i}. {row['Jugador']} ({row['Rating']})")

        final_text = "\n".join(text_lines)

        ax.text(
            x, y, final_text,
            ha="center",
            va="center",
            fontsize=8,
            linespacing=1.1,
            bbox=dict(
                facecolor="white",
                alpha=0.85,
                boxstyle="round,pad=0.3",
                edgecolor="black"
            )
        )
    st.pyplot(fig)

def radar_vs_top_player(players_df, role_scores, player_name, role):

    weights = pesos_roles_mejorados[role]
    pct_metrics = [f"{m}_pct" for m in weights if f"{m}_pct" in players_df.columns]

    if not pct_metrics:
        return

    player_row = players_df[players_df["Jugador"] == player_name]
    if player_row.empty:
        return

    player_values = player_row.iloc[0][pct_metrics].tolist()

    top_df = role_scores[role]
    top_player_name = top_df.iloc[0]["Jugador"]

    top_row = players_df[players_df["Jugador"] == top_player_name]
    top_values = top_row.iloc[0][pct_metrics].tolist()

    N = len(pct_metrics)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    player_values += player_values[:1]
    top_values += top_values[:1]

    fig, ax = plt.subplots(figsize=(7,7), subplot_kw=dict(polar=True))

    ax.plot(angles, player_values, linewidth=2, label=player_name)
    ax.fill(angles, player_values, alpha=0.25)

    ax.plot(angles, top_values, linewidth=2, linestyle="--", label=f"Top {role}")
    ax.fill(angles, top_values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace("_pct","") for m in pct_metrics], fontsize=9)
    ax.set_ylim(0,1)

    plt.legend(loc="upper right")
    st.pyplot(fig)
    
def scatter_role_universe(df, x_metric, y_metric, highlight_player=None):

    if x_metric not in df.columns or y_metric not in df.columns:
        st.warning("Métrica no disponible")
        return

   
    fig = go.Figure()

    # 🔵 Todos los jugadores (mismo color)
    fig.add_trace(go.Scatter(
        x=df[x_metric],
        y=df[y_metric],
        mode='markers',
        marker=dict(
            size=10,
            color="#1f77b4",
            opacity=0.6
        ),
        text=df["Jugador"],
        hovertemplate=
            "<b>%{text}</b><br>" +
            f"{x_metric}: %{{x}}<br>" +
            f"{y_metric}: %{{y}}<extra></extra>",
        showlegend=False
    ))

    # 🔴 Jugador destacado
    if highlight_player:
        player_row = df[df["Jugador"] == highlight_player]
        if not player_row.empty:
            fig.add_trace(go.Scatter(
                x=player_row[x_metric],
                y=player_row[y_metric],
                mode='markers',
                marker=dict(
                    size=16,
                    color="#e74c3c",
                    line=dict(width=2, color="black")
                ),
                text=player_row["Jugador"],
                hovertemplate=
                    "<b>%{text}</b><br>" +
                    f"{x_metric}: %{{x}}<br>" +
                    f"{y_metric}: %{{y}}<extra></extra>",
                showlegend=False
            ))

    # Líneas medias
    fig.add_vline(x=df[x_metric].mean(), line_dash="dash", opacity=0.3)
    fig.add_hline(y=df[y_metric].mean(), line_dash="dash", opacity=0.3)

    fig.update_layout(
        title=f"{x_metric} vs {y_metric}",
        xaxis_title=x_metric,
        yaxis_title=y_metric,
        template="simple_white"
    )

    st.plotly_chart(fig, use_container_width=True)

def get_percentile_color(p):

    if p >= 80:
        return "#2ecc71"   # verde
    elif p >= 60:
        return "#3498db"   # azul
    elif p >= 40:
        return "#f1c40f"   # amarillo
    elif p >= 20:
        return "#e67e22"   # naranja
    else:
        return "#e74c3c"   # rojo



def percentile_to_color(p):
    # Gradiente rojo → amarillo → verde
    if p <= 50:
        r = 231
        g = int(76 + (p/50)*180)
        b = 60
    else:
        r = int(231 - ((p-50)/50)*150)
        g = 200
        b = 60
    return f"rgb({r},{g},{b})"



def stripplot_role_metrics(players_df, role, player_name, min_minutes):

    df = players_df[players_df["Minutos jugados"] >= min_minutes].copy()

    allowed_positions = rol_pos_map.get(role, [])
    df = df[df["Pos_norm"].apply(lambda x: any(p in allowed_positions for p in x))]

    if df.empty:
        st.warning("No hay jugadores en este rol.")
        return

    weights = pesos_roles_mejorados[role]

    # 🔹 Ordenar métricas por peso (importancia del rol)
    metrics = sorted(
        [m for m in weights if m in df.columns],
        key=lambda x: weights[x],
        reverse=True
    )

    player_row = df[df["Jugador"] == player_name]
    if player_row.empty:
        st.warning("Jugador no encontrado en este rol")
        return

    player_row = player_row.iloc[0]

    for metric in metrics:

        values = df[metric].dropna()
        if len(values) == 0:
            continue

        # Ordenamos valores
        values_sorted = values.sort_values().reset_index(drop=True)

        # Percentiles universo
        percentiles = values_sorted.rank(pct=True) * 100
        colors = [percentile_to_color(p) for p in percentiles]

        # 🔥 Beeswarm stacking simétrico
        stack_dict = {}
        y_positions = []

        for v in values_sorted:
            key = round(v, 3)  # agrupar valores cercanos

            if key not in stack_dict:
                stack_dict[key] = 0
            else:
                stack_dict[key] += 1

            level = stack_dict[key]

            # alternar arriba y abajo
            if level % 2 == 0:
                y_positions.append(level * 0.15)
            else:
                y_positions.append(-level * 0.15)

        fig = go.Figure()

        # Universo
        fig.add_trace(go.Scatter(
            x=values_sorted,
            y=y_positions,
            mode='markers',
            marker=dict(
                size=11,
                color=colors,
                opacity=0.9
            ),
            hovertemplate=f"{metric}: %{{x}}<extra></extra>",
            showlegend=False
        ))

        # 🔴 Jugador destacado
        player_value = player_row[metric]
        player_percentile = (values < player_value).mean() * 100

        fig.add_trace(go.Scatter(
            x=[player_value],
            y=[0],
            mode='markers',
            marker=dict(
                size=24,
                color="black",
                line=dict(width=2, color="white")
            ),
            hovertemplate=
                f"<b>{player_name}</b><br>"
                f"{metric}: %{{x}}<br>"
                f"Percentil: {player_percentile:.1f}"
                "<extra></extra>",
            showlegend=False
        ))

        fig.update_layout(
            title=dict(
                text=f"{metric}",
                x=0,
                xanchor="left"
            ),
            yaxis=dict(visible=False),
            xaxis=dict(
                showgrid=False,
                zeroline=False
            ),
            template="simple_white",
            height=180,
            margin=dict(l=40, r=20, t=40, b=30)
        )

        st.plotly_chart(fig, use_container_width=True)

def scatter_role_universe_export(df, x_metric, y_metric, highlight_player, output_path):

    if x_metric not in df.columns or y_metric not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(6,6))

    ax.scatter(
        df[x_metric],
        df[y_metric],
        alpha=0.4
    )

    player_row = df[df["Jugador"] == highlight_player]

    if not player_row.empty:
        ax.scatter(
            player_row[x_metric],
            player_row[y_metric],
            s=150
        )

    ax.set_xlabel(x_metric)
    ax.set_ylabel(y_metric)
    ax.set_title(f"{x_metric} vs {y_metric}")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

def generate_player_report(
    filename,
    player_name,
    role_fit_data,
    percentiles_df,
    similar_df,
    players_df=None,
    role=None,
    min_minutes=None
):
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import pagesizes
    from reportlab.lib.units import inch
    import os

    doc = SimpleDocTemplate(filename, pagesize=pagesizes.A4)
    elements = []
    styles = getSampleStyleSheet()

    # ================================
    # TÍTULO
    # ================================
    elements.append(Paragraph("<b>Informe Scouting</b>", styles["Title"]))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph(f"<b>Jugador:</b> {player_name}", styles["Heading2"]))
    elements.append(Spacer(1, 0.3 * inch))

    # ================================
    # ROLE FIT
    # ================================
    elements.append(Paragraph("<b>Role Fit</b>", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))

    if role_fit_data:
        role_table_data = [["Rol", "Rating"]]
        for rol, score in role_fit_data:
            role_table_data.append([rol, score])

        table = Table(role_table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('TEXTCOLOR',(0,0),(-1,0),colors.white),
            ('GRID', (0,0), (-1,-1), 0.5, colors.black)
        ]))
        elements.append(table)

    elements.append(Spacer(1, 0.4 * inch))

    # ================================
    # PERCENTILES
    # ================================
    elements.append(Paragraph("<b>Percentiles</b>", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))

    if not percentiles_df.empty:
        percent_table_data = [["Métrica", "Percentil"]]

        for _, row in percentiles_df.iterrows():
            percent_table_data.append([
                row["Métrica"],
                row["Percentil"]
            ])

        table = Table(percent_table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('TEXTCOLOR',(0,0),(-1,0),colors.white),
            ('GRID', (0,0), (-1,-1), 0.5, colors.black)
        ]))
        elements.append(table)

    elements.append(Spacer(1, 0.4 * inch))

    # ================================
    # SIMILARIDAD
    # ================================
    elements.append(Paragraph("<b>Jugadores Similares</b>", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))

    if not similar_df.empty:
        sim_table_data = [["Jugador", "Equipo", "Similarity"]]

        for _, row in similar_df.iterrows():
            sim_table_data.append([
                row["Jugador"],
                row["Equipo durante el período seleccionado"],
                round(row["Similarity"], 3)
            ])

        table = Table(sim_table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('TEXTCOLOR',(0,0),(-1,0),colors.white),
            ('GRID', (0,0), (-1,-1), 0.5, colors.black)
        ]))
        elements.append(table)

    elements.append(Spacer(1, 0.4 * inch))

    # ================================
    # SCATTER
    # ================================
    elements.append(Spacer(1, 0.4 * inch))
    elements.append(Paragraph("<b>Scatter Análisis</b>", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))

    weights = pesos_roles_mejorados[role]

    # métricas del rol que existan realmente
    available_metrics = [
        m for m in weights
        if m in players_df.columns
    ]

    if len(available_metrics) >= 2:

        # ordenar por peso
        sorted_metrics = sorted(
            available_metrics,
            key=lambda x: weights[x],
            reverse=True
        )

        x_metric = sorted_metrics[0]
        y_metric = sorted_metrics[1]

        df_role = players_df[
            players_df["Minutos jugados"] >= min_minutes
        ].copy()

        scatter_path = "scatter_temp.png"

        scatter_role_universe_export(
            df_role,
            x_metric,
            y_metric,
            player_name,
            scatter_path
        )

        if os.path.exists(scatter_path):
            elements.append(Image(scatter_path, width=4*inch, height=4*inch))
            os.remove(scatter_path)

# ==========================================================
# UI
# ==========================================================

st.sidebar.header("📂 Subir Excel")
files = st.sidebar.file_uploader(
    "Sube archivos",
    type=["xlsx"],
    accept_multiple_files=True
)

if files:

    players_master = load_data(files)
    players_master = add_derived_metrics(players_master)
    players_master = compute_role_percentiles(players_master, min_minutes_base=600)

    pct_cols = [col for col in players_master.columns if col.endswith("_pct")]
    players_master[pct_cols] = players_master[pct_cols].fillna(0)

    players = players_master.copy()
    # =========================
    # FILTROS
    # =========================

    st.sidebar.divider()
    st.sidebar.subheader("🔎 Filtros")

    # MINUTOS (siempre activo, sin checkbox)
    min_minutes = st.sidebar.slider(
        "Minutos mínimos",
        0,
        int(players["Minutos jugados"].max()),
        1000
    )

    # EDAD
    if "Edad" in players.columns:
        usar_edad = st.sidebar.checkbox("Filtrar por Edad", value=False)
        edad_min, edad_max = st.sidebar.slider(
            "Edad",
            int(players["Edad"].min()),
            int(players["Edad"].max()),
            (18, 35),
            disabled=not usar_edad
        )
        if usar_edad:
            players = players[
                (players["Edad"] >= edad_min) &
                (players["Edad"] <= edad_max)
            ]

    # VALOR DE MERCADO
    if "Valor de mercado (Transfermarkt)" in players.columns:

        players["Valor de mercado (Transfermarkt)"] = (
            players["Valor de mercado (Transfermarkt)"]
            .replace('[€,mM]', '', regex=True)
        )
        players["Valor de mercado (Transfermarkt)"] = pd.to_numeric(
            players["Valor de mercado (Transfermarkt)"],
            errors="coerce"
        )

        min_val = float(players["Valor de mercado (Transfermarkt)"].min())
        max_val = float(players["Valor de mercado (Transfermarkt)"].max())

        if min_val < max_val:
            usar_mercado = st.sidebar.checkbox("Filtrar por Valor de Mercado", value=False)
            market_min, market_max = st.sidebar.slider(
                "Valor de mercado (€ millones)",
                min_val,
                max_val,
                (min_val, max_val),
                disabled=not usar_mercado
            )
            if usar_mercado:
                players = players[
                    (players["Valor de mercado (Transfermarkt)"] >= market_min) &
                    (players["Valor de mercado (Transfermarkt)"] <= market_max)
                ]

    # VENCIMIENTO CONTRATO
    if "Vencimiento contrato" in players.columns:

        players["Vencimiento contrato"] = pd.to_datetime(
            players["Vencimiento contrato"],
            errors="coerce"
        )
        years_to_expiry = (
            players["Vencimiento contrato"] - pd.Timestamp.today()
        ).dt.days / 365

        usar_contrato = st.sidebar.checkbox("Filtrar por Contrato", value=False)
        max_years = st.sidebar.slider(
            "Contrato vence en ≤ años",
            0.0,
            5.0,
            2.0,
            disabled=not usar_contrato
        )
        if usar_contrato:
            players = players.loc[years_to_expiry[years_to_expiry <= max_years].index]

    # PIE
    if "Pie" in players.columns:
        pies = sorted(players["Pie"].dropna().unique())
        usar_pie = st.sidebar.checkbox("Filtrar por Pie dominante", value=False)
        if usar_pie:
            pie_sel = st.sidebar.multiselect("Pie dominante", pies)
            if pie_sel:
                players = players[players["Pie"].isin(pie_sel)]

    # COMPETICIÓN
    if "Competición" in players.columns:
        comps = sorted(players["Competición"].dropna().unique())
        usar_comp = st.sidebar.checkbox("Filtrar por Competición", value=False)
        if usar_comp:
            comp_sel = st.sidebar.multiselect("Competición", comps)
            if comp_sel:
                players = players[players["Competición"].isin(comp_sel)]

    # AÑO
    if "Año" in players.columns:
        years = sorted(players["Año"].dropna().unique())
        usar_año = st.sidebar.checkbox("Filtrar por Año", value=False)
        if usar_año:
            year_sel = st.sidebar.multiselect("Año", years)
            if year_sel:
                players = players[players["Año"].isin(year_sel)]


    # =========================
    # SCORING
    # =========================

    # players_master = universo completo para scoring y percentiles
    # players = subconjunto filtrado, solo para los selectboxes de jugador
    role_scores = compute_role_scores(
        players,
        min_minutes
    )
    # =========================
    # TABS
    # =========================
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "🏆 Rankings",
        "🕷 Radar",
        "📋 Alineación",
        "📊 Percentiles",
        "🆚 Comparador",
        "🎯 Role Fit",
        "📍 Strip Plot",
        "🔎 Similaridad",
        "🔵 Scatter"
        ])



    # ------------------------------------------------------
    # TAB 1 — RANKINGS
    # ------------------------------------------------------
    with tab1:

        st.subheader("Ranking por Rol")

        if role_scores:
            selected_role = st.selectbox("Rol", list(role_scores.keys()), key="rank", index=None, placeholder="Elige un rol...")

            if selected_role:
                df_role = role_scores[selected_role]

                # 🔥 Mostrar solo jugadores con Rating válido
                df_role = df_role[df_role["Rating"].notna()]

                st.dataframe(
                    df_role[[
                        "Jugador",
                        "Equipo durante el período seleccionado",
                        "Minutos jugados",
                        "Rating"
                    ]],
                    use_container_width=True
                )

    # ------------------------------------------------------
    # TAB 2 — RADAR
    # ------------------------------------------------------
    with tab2:

        st.subheader("Radar Comparativo")

        if role_scores:
            selected_role = st.selectbox("Rol Radar", list(role_scores.keys()), key="radar_role", index=None, placeholder="Elige un rol...")

            if selected_role:
                df_role = role_scores[selected_role]

                players_list = df_role["Jugador"].tolist()

                selected_players = st.multiselect("Jugadores", players_list, placeholder="Elige jugadores...")

                if selected_players:
                    radar_plot(df_role, selected_role, selected_players)

    # ------------------------------------------------------
    # TAB 3 — ALINEACIÓN
    # ------------------------------------------------------
    with tab3:

        st.subheader("Alineación Automática")

        formacion = st.selectbox(
            "Formación",
            ["4-3-3", "4-4-2","3-5-2","5-3-2","4-5-1","3-4-3"]
        )

        formaciones = {
            "4-3-3": {
                "Portero": 1,
                "Lateral_Defensivo": 1,
                "Lateral_Ofensivo": 1,
                "Central_Clasico": 1,
                "Central_Salida": 1,
                "Pivote_Defensivo": 1,
                "Interior": 1,
                "Box_to_Box": 1,
                "Extremo_Puro": 1,
                "Extremo_Asociativo": 1,
                "Delantero_Goleador": 1
            },
            "4-4-2": {
                "Portero": 1,
                "Lateral_Defensivo": 2,
                "Central_Clasico": 2,
                "Pivote_Defensivo": 1,
                "Interior": 1,
                "Extremo_Puro": 2,
                "Delantero_Goleador": 1,
                "Delantero_Movil": 1
            },
            "3-5-2": {
                "Portero": 1,
                "Central_Salida": 1,
                "Central_Clasico": 2,
                "Lateral_Ofensivo": 2,
                "Pivote_Defensivo": 1,
                "Interior": 2,
                "Delantero_Goleador": 1,
                "Delantero_Movil": 1
            },
            "5-3-2": {
                "Portero": 1,
                "Central_Clasico": 2,
                "Central_Salida": 1,
                "Lateral_Defensivo": 2,
                "Pivote_Defensivo": 1,
                "Interior": 2,
                "Delantero_Goleador": 1,
                "Delantero_Movil": 1
            },
            "4-5-1": {
                "Portero": 1,
                "Lateral_Defensivo": 2,
                "Central_Clasico": 2,
                "Pivote_Defensivo": 1,
                "Interior": 2,
                "Extremo_Puro": 2,
                "Delantero_Goleador": 1
            },
            "3-4-3": {
                "Portero": 1,
                "Central_Salida": 1,
                "Central_Clasico": 2,
                "Lateral_Ofensivo": 2,
                "Pivote_Defensivo": 1,
                "Interior": 1,
                "Extremo_Puro": 2,
                "Delantero_Goleador": 1
            }
        }

        # 🔹 Seguridad: si no existe
        if formacion not in formaciones:
            st.warning("Formación no disponible")
            st.stop()

        used_players = []
        alineacion = []

        for rol, cantidad in formaciones[formacion].items():

            for i in range(cantidad):

                rol_display = f"{rol} {i+1}"

                # 🔹 DEFINIR SIDE
                side = None

                if cantidad == 2:
                    side = "left" if i == 0 else "right"

                jugador = best_player_for_role(
                    role_scores,
                    rol,
                    used_players,
                    side=side
                )

                if jugador != "—":
                    used_players.append(jugador)

                alineacion.append((rol_display, jugador, side))

        st.divider()

        for rol, jugador, side in alineacion:
            st.write(f"**{rol}** → {jugador}")

        st.divider()

        # 🔹 Solo dibuja si hay coords
        if formacion in formation_coords:
            plot_formation(formacion, alineacion, role_scores)
        else:
            st.info("No hay coordenadas definidas para esta formación.")
    # ------------------------------------------------------
    # TAB 4 — PERCENTILES
    # ------------------------------------------------------
    with tab4:

        st.subheader("Percentiles por Jugador")

        if role_scores:

            all_players = players["Jugador"].unique().tolist()
            selected_player = st.selectbox(
                "Jugador",
                all_players,
                key="percent_player",
                index=None,
                placeholder="Elige un jugador..."
            )

            selected_role = st.selectbox(
                "Rol",
                list(role_scores.keys()),
                key="percent_role",
                index=None,
                placeholder="Elige un rol..."
            )

            if selected_player and selected_role:
                df_percent = player_percentiles(
                    players,
                    selected_player,
                    selected_role
                )

                if not df_percent.empty:
                    st.dataframe(df_percent, use_container_width=True)

                    # mini gráfico
                    plot_percentiles(df_percent)
                    
                else:
                    st.warning("Sin datos para este jugador.")
    # ------------------------------------------------------
    # TAB 5 — COMPARADOR
    # ------------------------------------------------------
    with tab5:

        st.subheader("Comparador de Jugadores")

        if role_scores:

            rol_comp = st.selectbox(
                "Rol",
                list(role_scores.keys()),
                key="comp_role",
                index=None,
                placeholder="Elige un rol..."
            )

            if rol_comp:
                df_role = role_scores[rol_comp]

                jugadores = df_role["Jugador"].tolist()

                jugadores_sel = st.multiselect(
                    "Jugadores a comparar",
                    jugadores,
                    max_selections=10,
                    placeholder="Elige jugadores..."
                )

                if len(jugadores_sel) >= 2:

                    metrics = [
                        m for m in roles_metrics[rol_comp]
                        if m in df_role.columns
                    ]

                    # 🔥 VALORES REALES (NO NORMALIZADOS)
                    tabla_real = df_role[
                        df_role["Jugador"].isin(jugadores_sel)
                    ][["Jugador"] + metrics]

                    st.write("### Métricas reales")
                    st.dataframe(tabla_real, use_container_width=True)

                else:
                    st.info("Selecciona mínimo 2 jugadores.")
   
    # ------------------------------------------------------
    # TAB 6 — ROLE FIT
    # ------------------------------------------------------
    with tab6:

        st.subheader("🎯 Encaje de jugador por rol")

        if role_scores:

            # SOLO jugadores con minutos mínimos
            eligible_players = players[
                players["Minutos jugados"] >= min_minutes
            ]["Jugador"].unique().tolist()

            player_choice = st.selectbox(
                "Seleccionar jugador",
                eligible_players,
                index=None,
                placeholder="Selecciona un jugador"
            )

            if player_choice:
                top_roles = best_roles_for_player_smart(
                    player_choice,
                    players,
                    min_minutes,
                    top_n=3
)

                st.markdown("### Mejores roles")

                for rol, score in top_roles:
                    st.write(f"**{rol}** — Rating: {score}")

                st.divider()
                st.markdown("### Comparativa vs Mejor Jugador del Rol")

                cols = st.columns(3)

                for i, (rol, score) in enumerate(top_roles):
                    with cols[i]:
                        st.markdown(f"#### {rol} — {score}")
                        radar_vs_role_top_player(
                            players,
                            player_choice,
                            rol,
                            min_minutes
                        )


        else:
            st.warning("Carga datos primero.")
# ------------------------------------------------------
# TAB 7 — STRIP PLOT POR ROL
# ------------------------------------------------------
    with tab7:

        st.subheader("📊 Distribución por Métricas (Strip Plot)")

        if role_scores:

            # 1️⃣ Primero elegir rol
            selected_role = st.selectbox(
                "Seleccionar rol",
                list(role_scores.keys()),
                index=None,
                placeholder="Elige un rol...",
                key="strip_role"
            )

            if selected_role:

                # 2️⃣ Todas las posiciones disponibles en los datos (sin limitar al rol)
                pos_disponibles = sorted(set(
                    pos
                    for pos_list in players["Pos_norm"]
                    for pos in pos_list
                ))

                selected_positions = st.multiselect(
                    "Filtrar por posición",
                    pos_disponibles,
                    key="strip_positions",
                    placeholder="Elige posiciones (vacío = todas)..."
                )

                # Si no se elige ninguna, usar todas
                filtro_pos = selected_positions if selected_positions else pos_disponibles

                # 3️⃣ Jugadores filtrados por posición y minutos
                eligible_players = players[
                    (players["Minutos jugados"] >= min_minutes) &
                    (players["Pos_norm"].apply(
                        lambda x: any(p in filtro_pos for p in x)
                    ))
                ]["Jugador"].unique().tolist()

                selected_player = st.selectbox(
                    "Seleccionar jugador",
                    eligible_players,
                    index=None,
                    placeholder="Elige un jugador...",
                    key="strip_player"
                )

                if selected_player:
                    stripplot_role_metrics(
                        players,
                        selected_role,
                        selected_player,
                        min_minutes
                    )
# ------------------------------------------------------
# TAB 8 — Similaridad
# ------------------------------------------------------    
    with tab8:

        st.subheader("🔎 Jugadores similares")

        eligible_players = players[
            players["Minutos jugados"] >= min_minutes
        ]["Jugador"].unique().tolist()

        selected_player = st.selectbox("Jugador", eligible_players, index=None, placeholder="Elige un jugador...")

        selected_role = st.selectbox("Rol", list(role_scores.keys()), index=None, placeholder="Elige un rol...")

        if selected_player and selected_role:

            similar_df = find_similar_players(
                players,
                selected_role,
                selected_player,
                min_minutes,
                top_n=5
            )

            if not similar_df.empty:
                st.dataframe(similar_df, use_container_width=True)
            else:
                st.warning("No se encontraron similares.")

# ------------------------------------------------------
# TAB 9 — SCATTER LIBRE
# ------------------------------------------------------
    with tab9:

        st.subheader("🔵 Scatter Plot")

        if role_scores:

            col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)

            with col_ctrl1:
                rol_scatter = st.selectbox(
                    "Filtrar por rol (opcional)",
                    ["Todos"] + list(role_scores.keys()),
                    key="scatter_rol"
                )

            if rol_scatter == "Todos":
                df_scatter = players[players["Minutos jugados"] >= min_minutes].copy()
            else:
                df_scatter = role_scores[rol_scatter].copy()

            numeric_cols = sorted([
                c for c in df_scatter.select_dtypes(include="number").columns
                if not c.endswith("_pct") and c not in ["Minutos jugados"]
            ])

            with col_ctrl2:
                x_metric = st.selectbox(
                    "Eje X", numeric_cols, key="scatter_x",
                    index=None, placeholder="Elige métrica..."
                )

            with col_ctrl3:
                y_metric = st.selectbox(
                    "Eje Y", numeric_cols, key="scatter_y",
                    index=None, placeholder="Elige métrica..."
                )

            col_ctrl4, col_ctrl5 = st.columns(2)

            with col_ctrl4:
                color_by = st.selectbox(
                    "Color por",
                    ["Rating", "Posición", "Ninguno"],
                    key="scatter_color"
                )

            with col_ctrl5:
                highlight_player = st.selectbox(
                    "Destacar jugador (opcional)",
                    ["Ninguno"] + df_scatter["Jugador"].tolist(),
                    key="scatter_highlight"
                )

            if x_metric and y_metric:

                keep_cols = ["Jugador", x_metric, y_metric]
                if "Rating" in df_scatter.columns:
                    keep_cols.append("Rating")
                if "Pos_primary" in df_scatter.columns:
                    keep_cols.append("Pos_primary")

                df_plot = df_scatter[keep_cols].dropna(subset=[x_metric, y_metric])

                fig = go.Figure()

                if color_by == "Rating" and "Rating" in df_plot.columns:
                    marker_color = df_plot["Rating"]
                    colorscale = "RdYlGn"
                    showscale = True
                    colorbar = dict(title="Rating")
                elif color_by == "Posición" and "Pos_primary" in df_plot.columns:
                    positions = df_plot["Pos_primary"].unique().tolist()
                    palette = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
                               "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]
                    pos_color_map = {p: palette[i % len(palette)] for i, p in enumerate(positions)}
                    marker_color = df_plot["Pos_primary"].map(pos_color_map)
                    showscale = False
                    colorscale = None
                    colorbar = None
                else:
                    marker_color = "#1f77b4"
                    showscale = False
                    colorscale = None
                    colorbar = None

                mask_highlight = (
                    df_plot["Jugador"] == highlight_player
                    if highlight_player != "Ninguno"
                    else pd.Series([False] * len(df_plot), index=df_plot.index)
                )

                df_normal = df_plot[~mask_highlight]
                colors_normal = (
                    marker_color[~mask_highlight]
                    if hasattr(marker_color, "__getitem__") and not isinstance(marker_color, str)
                    else marker_color
                )

                fig.add_trace(go.Scatter(
                    x=df_normal[x_metric],
                    y=df_normal[y_metric],
                    mode="markers",
                    marker=dict(
                        size=9,
                        color=colors_normal,
                        colorscale=colorscale,
                        showscale=showscale,
                        colorbar=colorbar,
                        opacity=0.7,
                        line=dict(width=0.5, color="white")
                    ),
                    text=df_normal["Jugador"],
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        + f"{x_metric}: %{{x:.2f}}<br>"
                        + f"{y_metric}: %{{y:.2f}}<extra></extra>"
                    ),
                    showlegend=False
                ))

                if highlight_player != "Ninguno":
                    df_hl = df_plot[mask_highlight]
                    if not df_hl.empty:
                        fig.add_trace(go.Scatter(
                            x=df_hl[x_metric],
                            y=df_hl[y_metric],
                            mode="markers+text",
                            marker=dict(
                                size=18,
                                color="#e74c3c",
                                line=dict(width=2, color="black")
                            ),
                            text=df_hl["Jugador"],
                            textposition="top center",
                            hovertemplate=(
                                "<b>%{text}</b><br>"
                                + f"{x_metric}: %{{x:.2f}}<br>"
                                + f"{y_metric}: %{{y:.2f}}<extra></extra>"
                            ),
                            showlegend=False
                        ))

                fig.add_vline(
                    x=df_plot[x_metric].mean(),
                    line_dash="dash", line_color="gray", opacity=0.4,
                    annotation_text="media X", annotation_position="top right"
                )
                fig.add_hline(
                    y=df_plot[y_metric].mean(),
                    line_dash="dash", line_color="gray", opacity=0.4,
                    annotation_text="media Y", annotation_position="top right"
                )

                fig.update_layout(
                    title=f"{x_metric}  vs  {y_metric}",
                    xaxis_title=x_metric,
                    yaxis_title=y_metric,
                    template="simple_white",
                    height=600,
                    margin=dict(l=60, r=40, t=60, b=60)
                )

                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"Mostrando {len(df_plot)} jugadores")

            else:
                st.info("Elige métricas para los ejes X e Y.")

