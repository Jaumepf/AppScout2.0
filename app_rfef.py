import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import os
import glob

warnings.filterwarnings("ignore")

# ==========================================================
# ROLES — SOFASCORE / 1ª RFEF  (40 métricas)
# ==========================================================

pesos_roles = {

    "Portero": {
        "Goles_evitados_%":         0.20,
        "Porterias_cero_%":         0.15,
        "Goles_encajados/90":       0.12,
        "Paradas/90":               0.12,
        "Paradas_area/90":          0.10,
        "Salidas/90":               0.08,
        "Error_disparo/90":         0.08,
        "Pases_largos/90":          0.07,
        "Precision_pases_%":        0.04,
        "Pases/90":                 0.04,
    },

    "Portero_Avanzado": {
        "Goles_evitados_%":         0.15,
        "Porterias_cero_%":         0.10,
        "Goles_encajados/90":       0.10,
        "Paradas/90":               0.08,
        "Salidas/90":               0.10,
        "Pases/90":                 0.12,
        "Precision_pases_%":        0.12,
        "Pases_largos/90":          0.10,
        "Ratio_pases_adelante_%":   0.08,
        "Error_disparo/90":         0.05,
    },

    "Lateral_Defensivo": {
        "Duelos_ganados_%":         0.14,
        "Entradas/90":              0.14,
        "Entradas_ganadas_%":       0.12,
        "Recuperaciones/90":        0.12,
        "Regateado/90":             0.10,
        "Duelos_aereos_ganados_%":  0.08,
        "Despejes/90":              0.08,
        "Pases_campo_rival/90":     0.08,
        "Perdidas/90":              0.07,
        "Faltas/90":                0.07,
    },

    "Lateral_Ofensivo": {
        "Centros/90":               0.16,
        "Precision_centros_%":      0.12,
        "Regates/90":               0.12,
        "Regates_exito_%":          0.10,
        "Pases_campo_rival/90":     0.10,
        "Asistencias/90":           0.10,
        "xA/90":                    0.10,
        "Faltas_recibidas/90":      0.08,
        "Recuperaciones/90":        0.07,
        "Entradas/90":              0.05,
    },

    "Central_Stopper": {
        "Duelos_ganados_%":         0.16,
        "Duelos_aereos_ganados_%":  0.14,
        "Duelos_aereos/90":         0.10,
        "Entradas/90":              0.10,
        "Entradas_ganadas_%":       0.10,
        "Despejes/90":              0.10,
        "Tiros_bloqueados_def/90":  0.10,
        "Recuperaciones/90":        0.08,
        "Regateado/90":             0.06,
        "Faltas/90":                0.06,
    },

    "Central_Clasico": {
        "Duelos_ganados_%":         0.14,
        "Duelos_aereos_ganados_%":  0.12,
        "Entradas_ganadas_%":       0.10,
        "Despejes/90":              0.10,
        "Recuperaciones/90":        0.10,
        "Pases/90":                 0.10,
        "Precision_pases_%":        0.10,
        "Pases_largos/90":          0.08,
        "Regateado/90":             0.08,
        "Perdidas/90":              0.08,
    },

    "Central_Salida": {
        "Pases/90":                 0.16,
        "Precision_pases_%":        0.14,
        "Pases_campo_rival/90":     0.12,
        "Ratio_pases_adelante_%":   0.12,
        "Pases_largos/90":          0.10,
        "Precision_pases_largos_%": 0.10,
        "Recuperaciones/90":        0.08,
        "Duelos_ganados_%":         0.08,
        "Regateado/90":             0.06,
        "Perdidas/90":              0.04,
    },

    "Pivote_Defensivo": {
        "Recuperaciones/90":        0.16,
        "Entradas/90":              0.14,
        "Entradas_ganadas_%":       0.12,
        "Duelos_ganados_%":         0.12,
        "Tiros_bloqueados_def/90":  0.10,
        "Pases/90":                 0.10,
        "Precision_pases_%":        0.08,
        "Perdidas/90":              0.08,
        "Faltas/90":                0.06,
        "Despejes/90":              0.04,
    },

    "Interior": {
        "Pases_campo_rival/90":     0.16,
        "Precision_pases_%":        0.12,
        "xA/90":                    0.12,
        "Asistencias/90":           0.10,
        "Faltas_recibidas/90":      0.10,
        "Regates/90":               0.10,
        "Toques/90":                0.08,
        "Duelos_ganados_%":         0.08,
        "Recuperaciones/90":        0.08,
        "Goles/90":                 0.06,
    },

    "Box_to_Box": {
        "Recuperaciones/90":        0.14,
        "Duelos_ganados_%":         0.12,
        "Regates/90":               0.10,
        "Pases_campo_rival/90":     0.10,
        "Goles/90":                 0.10,
        "xG/90":                    0.08,
        "Asistencias/90":           0.08,
        "xA/90":                    0.08,
        "Toques/90":                0.08,
        "Entradas/90":              0.06,
        "Faltas_recibidas/90":      0.06,
    },

    "Mediapunta": {
        "xA/90":                    0.18,
        "Asistencias/90":           0.14,
        "xA_Overperformance":       0.10,
        "Pases_campo_rival/90":     0.12,
        "Faltas_recibidas/90":      0.10,
        "Regates/90":               0.10,
        "Regates_exito_%":          0.08,
        "Toques/90":                0.08,
        "Goles/90":                 0.10,
    },

    "Extremo_Asociativo": {
        "Asistencias/90":           0.16,
        "xA/90":                    0.14,
        "Centros/90":               0.12,
        "Precision_centros_%":      0.10,
        "Faltas_recibidas/90":      0.10,
        "Pases_campo_rival/90":     0.10,
        "Regates/90":               0.10,
        "Goles/90":                 0.10,
        "xG/90":                    0.08,
    },

    "Extremo_Puro": {
        "Regates/90":               0.18,
        "Regates_exito_%":          0.14,
        "Faltas_recibidas/90":      0.12,
        "Duelos_ganados_%":         0.10,
        "Centros/90":               0.10,
        "Precision_centros_%":      0.08,
        "Goles/90":                 0.10,
        "xG/90":                    0.08,
        "Pases_campo_rival/90":     0.06,
        "Asistencias/90":           0.04,
    },

    "Delantero_Goleador": {
        "Goles/90":                 0.20,
        "xG/90":                    0.16,
        "Conversion_Gol_%":         0.14,
        "xG_Overperformance_90":    0.12,
        "Remates/90":               0.10,
        "Tiros_a_puerta_%":         0.10,
        "Duelos_aereos_ganados_%":  0.08,
        "Faltas_recibidas/90":      0.04,
        "Duelos_ganados_%":         0.04,
        "Toques/90":                0.02,
    },

    "Delantero_Movil": {
        "Goles/90":                 0.14,
        "xG/90":                    0.12,
        "Asistencias/90":           0.12,
        "xA/90":                    0.10,
        "Regates/90":               0.10,
        "xG_Overperformance_90":    0.08,
        "Faltas_recibidas/90":      0.10,
        "Pases_campo_rival/90":     0.08,
        "Duelos_ganados_%":         0.08,
        "Toques/90":                0.08,
    },
}

metricas_negativas = ["Faltas/90", "Perdidas/90", "Regateado/90", "Error_disparo/90", "Goles_encajados/90"]

pos_map = {"G": ["GK"], "D": ["CB","RB","LB"], "M": ["DM","CM","AM"], "F": ["RW","LW","FW"]}

rol_pos_map = {
    "Portero":            ["GK"],
    "Portero_Avanzado":   ["GK"],
    "Lateral_Defensivo":  ["CB","RB","LB"],
    "Lateral_Ofensivo":   ["CB","RB","LB"],
    "Central_Stopper":    ["CB","RB","LB"],
    "Central_Clasico":    ["CB","RB","LB"],
    "Central_Salida":     ["CB","RB","LB"],
    "Pivote_Defensivo":   ["DM","CM"],
    "Interior":           ["DM","CM","AM"],
    "Box_to_Box":         ["DM","CM","AM"],
    "Mediapunta":         ["CM","AM"],
    "Extremo_Asociativo": ["RW","LW","FW","AM"],
    "Extremo_Puro":       ["RW","LW","FW","AM"],
    "Delantero_Movil":    ["FW","RW","LW"],
    "Delantero_Goleador": ["FW","RW","LW"],
}

# ==========================================================
# CARGA Y TRANSFORMACIÓN
# ==========================================================

def normalize_positions(pos_string):
    if pd.isna(pos_string): return []
    return pos_map.get(str(pos_string).strip().upper(), [])


@st.cache_data
def load_squad_data(files):
    """Carga CSVs de jugadores (temporada) para enriquecer con datos de plantilla."""
    dfs = []
    for f in files:
        try:    d = pd.read_csv(f, sep=';', skiprows=1, decimal=',')
        except:
            try: d = pd.read_csv(f, sep=';', decimal=',')
            except: d = pd.read_csv(f, decimal=',')
        dfs.append(d)
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    # Normalizar nombres de columnas esperadas
    col_map = {
        'jugador': 'jugador', 'equipo': 'equipo',
        'altura_cm': 'altura_cm', 'pie_dominante': 'pie_dominante',
        'nacionalidad': 'nacionalidad', 'valor_mercado_eur': 'valor_mercado_eur',
        'contrato_hasta': 'contrato_hasta', 'posicion_detallada': 'posicion_detallada',
        'fecha_nacimiento': 'fecha_nacimiento',
    }
    df.columns = [c.strip().lower() for c in df.columns]
    keep = [c for c in col_map.values() if c in df.columns]
    if 'jugador' not in df.columns or 'equipo' not in df.columns:
        return pd.DataFrame()
    df = df[keep].drop_duplicates(subset=['jugador','equipo'])
    return df

@st.cache_data
def load_data(files):
    dfs = []
    for f in files:
        try:    d = pd.read_csv(f, sep=';', skiprows=1, decimal=',')
        except:
            try: d = pd.read_csv(f, sep=';', decimal=',')
            except: d = pd.read_csv(f, decimal=',')
        # Asignar liga desde nombre del archivo
        fname = f if isinstance(f, str) else getattr(f, 'name', '')
        fname_str = str(fname)
        if '1rfef' in fname_str:             liga_nombre = '1a RFEF'
        elif 'laliga-2' in fname_str:        liga_nombre = 'LaLiga 2'
        elif 'serie-b' in fname_str:         liga_nombre = 'Serie B'
        elif 'ligue-2' in fname_str:         liga_nombre = 'Ligue 2'
        elif 'liga-portugal' in fname_str:   liga_nombre = 'Liga Portugal 2'
        else:
            # Usar nombre del archivo sin extension como nombre de liga
            import os as _os
            base = _os.path.basename(fname_str)
            liga_nombre = _os.path.splitext(base)[0]
        d['liga'] = liga_nombre
        dfs.append(d)
    raw = pd.concat(dfs, ignore_index=True)

    # Asegurar columnas opcionales con valor 0 si no existen en el CSV
    for col in ['tiros_al_palo', 'tiros_bloqueados', 'xG', 'xA',
                'pases_largos_totales', 'pases_largos_precisos',
                'centros_totales', 'centros_precisos',
                'entradas_totales', 'entradas_ganadas',
                'duelos_aereos_ganados', 'duelos_aereos_perdidos',
                'regates_intentados', 'regates_exitosos', 'regateado',
                'paradas_dentro_area', 'salidas',
                'error_que_lleva_a_disparo', 'resultado_rival']:
        if col not in raw.columns:
            raw[col] = 0

    agg = raw.groupby(['jugador','equipo','liga']).agg(
        minutos_jugados        = ('minutos_jugados',           'sum'),
        partidos               = ('jornada',                   'count'),
        posicion               = ('posicion',                  lambda x: x.mode().iloc[0] if not x.mode().empty else ''),
        goles                  = ('goles',                     'sum'),
        tiros_totales          = ('tiros_totales',             'sum'),
        tiros_a_puerta         = ('tiros_a_puerta',            'sum'),
        tiros_al_palo          = ('tiros_al_palo',             'sum'),
        tiros_bloqueados       = ('tiros_bloqueados',          'sum'),
        xG                     = ('xG',                        'sum'),
        asistencias            = ('asistencias',               'sum'),
        xA                     = ('xA',                        'sum'),
        pases_totales          = ('pases_totales',             'sum'),
        pases_precisos         = ('pases_precisos',            'sum'),
        pases_campo_propio     = ('pases_campo_propio',        'sum'),
        pases_campo_rival      = ('pases_campo_rival',         'sum'),
        pases_largos_totales   = ('pases_largos_totales',      'sum'),
        pases_largos_precisos  = ('pases_largos_precisos',     'sum'),
        centros_totales        = ('centros_totales',           'sum'),
        centros_precisos       = ('centros_precisos',          'sum'),
        entradas_totales       = ('entradas_totales',          'sum'),
        entradas_ganadas       = ('entradas_ganadas',          'sum'),
        despejes               = ('despejes',                  'sum'),
        duelos_ganados         = ('duelos_ganados',            'sum'),
        duelos_perdidos        = ('duelos_perdidos',           'sum'),
        duelos_aereos_ganados  = ('duelos_aereos_ganados',     'sum'),
        duelos_aereos_perdidos = ('duelos_aereos_perdidos',    'sum'),
        regates_intentados     = ('regates_intentados',        'sum'),
        regates_exitosos       = ('regates_exitosos',          'sum'),
        regateado              = ('regateado',                 'sum'),
        faltas_cometidas       = ('faltas_cometidas',          'sum'),
        faltas_recibidas       = ('faltas_recibidas',          'sum'),
        recuperaciones         = ('recuperaciones',            'sum'),
        perdidas               = ('perdidas',                  'sum'),
        toques                 = ('toques',                    'sum'),
        paradas_totales        = ('paradas_totales',           'sum'),
        paradas_dentro_area    = ('paradas_dentro_area',       'sum'),
        salidas                = ('salidas',                   'sum'),
        error_disparo          = ('error_que_lleva_a_disparo', 'sum'),
        goles_encajados        = ('resultado_rival',           'sum'),
        porterias_cero         = ('resultado_rival',           lambda x: (x == 0).sum()),
    ).reset_index()

    m = agg['minutos_jugados'].replace(0, np.nan)

    agg['Goles/90']                 = agg['goles'] / m * 90
    agg['xG/90']                    = agg['xG'] / m * 90
    agg['xG_Overperformance_90']    = agg['Goles/90'] - agg['xG/90']
    agg['Conversion_Gol_%']         = (agg['goles'] / agg['tiros_totales'].replace(0,np.nan) * 100).fillna(0)
    agg['Remates/90']               = agg['tiros_totales'] / m * 90
    agg['Tiros_a_puerta_%']         = (agg['tiros_a_puerta'] / agg['tiros_totales'].replace(0,np.nan) * 100).fillna(0)
    agg['Tiros_al_palo/90']         = agg['tiros_al_palo'] / m * 90
    agg['Tiros_bloqueados_def/90']  = agg['tiros_bloqueados'] / m * 90
    agg['Asistencias/90']           = agg['asistencias'] / m * 90
    agg['xA/90']                    = agg['xA'] / m * 90
    agg['xA_Overperformance']       = agg['asistencias'] - agg['xA']
    agg['Faltas_recibidas/90']      = agg['faltas_recibidas'] / m * 90
    agg['Pases/90']                 = agg['pases_totales'] / m * 90
    agg['Precision_pases_%']        = (agg['pases_precisos'] / agg['pases_totales'].replace(0,np.nan) * 100).fillna(0)
    agg['Pases_campo_rival/90']     = agg['pases_campo_rival'] / m * 90
    agg['Pases_campo_propio/90']    = agg['pases_campo_propio'] / m * 90
    agg['Ratio_pases_adelante_%']   = (agg['pases_campo_rival'] / agg['pases_totales'].replace(0,np.nan) * 100).fillna(0)
    agg['Pases_largos/90']          = agg['pases_largos_totales'] / m * 90
    agg['Precision_pases_largos_%'] = (agg['pases_largos_precisos'] / agg['pases_largos_totales'].replace(0,np.nan) * 100).fillna(0)
    agg['Centros/90']               = agg['centros_totales'] / m * 90
    agg['Precision_centros_%']      = (agg['centros_precisos'] / agg['centros_totales'].replace(0,np.nan) * 100).fillna(0)
    agg['Entradas/90']              = agg['entradas_totales'] / m * 90
    agg['Entradas_ganadas_%']       = (agg['entradas_ganadas'] / agg['entradas_totales'].replace(0,np.nan) * 100).fillna(0)
    agg['Despejes/90']              = agg['despejes'] / m * 90
    agg['Duelos/90']                = (agg['duelos_ganados'] + agg['duelos_perdidos']) / m * 90
    agg['Duelos_ganados_%']         = (agg['duelos_ganados'] / (agg['duelos_ganados'] + agg['duelos_perdidos']).replace(0,np.nan) * 100).fillna(0)
    agg['Duelos_aereos/90']         = (agg['duelos_aereos_ganados'] + agg['duelos_aereos_perdidos']) / m * 90
    agg['Duelos_aereos_ganados_%']  = (agg['duelos_aereos_ganados'] / (agg['duelos_aereos_ganados'] + agg['duelos_aereos_perdidos']).replace(0,np.nan) * 100).fillna(0)
    agg['Regates/90']               = agg['regates_exitosos'] / m * 90
    agg['Regates_exito_%']          = (agg['regates_exitosos'] / agg['regates_intentados'].replace(0,np.nan) * 100).fillna(0)
    agg['Regateado/90']             = agg['regateado'] / m * 90
    agg['Faltas/90']                = agg['faltas_cometidas'] / m * 90
    agg['Recuperaciones/90']        = agg['recuperaciones'] / m * 90
    agg['Perdidas/90']              = agg['perdidas'] / m * 90
    agg['Toques/90']                = agg['toques'] / m * 90
    agg['Paradas/90']               = agg['paradas_totales'] / m * 90
    agg['Paradas_area/90']          = agg['paradas_dentro_area'] / m * 90
    agg['Salidas/90']               = agg['salidas'] / m * 90
    agg['Error_disparo/90']         = agg['error_disparo'] / m * 90
    agg['Goles_encajados/90']       = agg['goles_encajados'] / m * 90
    agg['Porterias_cero_%']         = agg['porterias_cero'] / agg['partidos'] * 100
    agg['Goles_evitados_%']         = (agg['paradas_totales'] / (agg['paradas_totales'] + agg['goles_encajados']).replace(0, np.nan) * 100).fillna(0)

    agg = agg.rename(columns={
        'jugador': 'Jugador', 'equipo': 'Equipo',
        'posicion': 'Posicion', 'minutos_jugados': 'Minutos jugados', 'partidos': 'Partidos',
    })
    agg['Pos_primary'] = agg['Posicion'].astype(str).str.strip().str.upper()
    agg['Pos_norm']    = agg['Posicion'].apply(normalize_positions)
    agg['Jugador_ID']  = agg['Jugador'] + ' (' + agg['Equipo'] + ')'
    return agg

# ==========================================================
# NORMALIZACIÓN Y SCORING
# ==========================================================

def percentile_normalization(data, metrics):
    df = data.copy()
    for metric in metrics:
        if metric not in df.columns: continue
        if df[metric].dropna().shape[0] < 2:
            df[metric] = 0.5; continue
        ranks = df[metric].rank(pct=True)
        df[metric] = (1 - ranks) if metric in metricas_negativas else ranks
        df[metric] = df[metric].clip(0, 1)
    return df

def compute_percentiles(players, min_minutes_base=450):
    players = players.copy()
    for rol, weights in pesos_roles.items():
        allowed = rol_pos_map.get(rol, [])
        metrics = list(weights.keys())
        # Poblacion de referencia: jugadores con minutos suficientes
        df_ref = players[
            (players["Minutos jugados"] >= min_minutes_base) &
            (players["Pos_norm"].apply(lambda x: any(p in allowed for p in x)))
        ].copy()
        if df_ref.empty: continue
        # Todos los jugadores de la posicion (incluidos pocos minutos)
        idx_all = players[
            players["Pos_norm"].apply(lambda x: any(p in allowed for p in x))
        ].index
        for metric in metrics:
            if metric not in df_ref.columns: continue
            ref_vals = df_ref[metric].dropna()
            if len(ref_vals) < 2: continue
            def to_pct(v, rv=ref_vals, neg=(metric in metricas_negativas)):
                if pd.isna(v): return np.nan
                n = len(rv)
                rank = (rv < v).sum() + (rv == v).sum() * 0.5
                pct = rank / n
                return float(np.clip(1 - pct if neg else pct, 0, 1))
            players.loc[idx_all, f"{metric}_pct"] = players.loc[idx_all, metric].apply(to_pct)
    return players

def compute_role_scores(players, min_minutes):
    role_scores = {}
    for rol, weights in pesos_roles.items():
        df = players[players["Minutos jugados"] >= min_minutes].copy()
        allowed = rol_pos_map.get(rol, [])
        df = df[df["Pos_norm"].apply(lambda x: any(p in allowed for p in x))]
        if df.empty: continue
        metrics = [m for m in weights if m in df.columns]
        if not metrics: continue
        pct_cols = [f"{m}_pct" for m in metrics if f"{m}_pct" in df.columns]
        df_score = df[pct_cols].fillna(0)
        w_vec = np.array([weights[m] for m in metrics if f"{m}_pct" in df.columns])
        scores = df_score.values @ w_vec
        df["Rating"] = np.round(scores * 10, 2)
        role_scores[rol] = df.sort_values("Rating", ascending=False)
    return role_scores

# ==========================================================
# FUNCIONES ANÁLISIS
# ==========================================================

def player_percentiles(players, player_name, role):
    weights  = pesos_roles.get(role, {})
    pct_cols = [f"{m}_pct" for m in weights if f"{m}_pct" in players.columns]
    row = players[players["Jugador_ID"] == player_name]
    if row.empty or not pct_cols: return pd.DataFrame()
    row = row.iloc[0]
    return pd.DataFrame([{
        "Métrica": m.replace("_pct",""),
        "Percentil": round(row.get(m, 0) * 100, 1),
        "Tipo": "⬇ negativa" if m.replace("_pct","") in metricas_negativas else ""
    } for m in pct_cols])

def best_roles_for_player(player_name, players, min_minutes, top_n=3):
    df = players[players["Minutos jugados"] >= min_minutes].copy()
    player_row = df[df["Jugador_ID"] == player_name]
    if player_row.empty: return []
    positions = player_row.iloc[0]["Pos_norm"]
    is_gk = "GK" in positions
    if is_gk:
        allowed_roles = [r for r in pesos_roles if "Portero" in r]
    else:
        allowed_roles = [r for r, pl in rol_pos_map.items()
                         if any(p in positions for p in pl) and "Portero" not in r]
    results = []
    for rol in allowed_roles:
        weights = pesos_roles[rol]
        metrics = [m for m in weights if f"{m}_pct" in df.columns]
        if not metrics: continue
        row = df[df["Jugador_ID"] == player_name].iloc[0]
        score = sum(row.get(f"{m}_pct", 0) * weights[m]
                    for m in metrics if not pd.isna(row.get(f"{m}_pct", 0)))
        results.append((rol, round(score * 10, 2)))
    return sorted(results, key=lambda x: x[1], reverse=True)[:top_n]

def find_similar_players(players, role, player_name, min_minutes, top_n=5):
    df = players[players["Minutos jugados"] >= min_minutes].copy()
    allowed = rol_pos_map.get(role, [])
    df = df[df["Pos_norm"].apply(lambda x: any(p in allowed for p in x))]
    if df.empty: return pd.DataFrame()
    weights  = pesos_roles[role]
    pct_cols = [f"{m}_pct" for m in weights if f"{m}_pct" in df.columns]
    if not pct_cols: return pd.DataFrame()
    df = df.dropna(subset=pct_cols)
    target = df[df["Jugador_ID"] == player_name]
    if target.empty: return pd.DataFrame()
    target_vec = target.iloc[0][pct_cols].values.reshape(1, -1)
    sims = cosine_similarity(df[pct_cols].values, target_vec).flatten()
    df["Similarity"] = sims
    df = df[df["Jugador_ID"] != player_name]
    return df.sort_values("Similarity", ascending=False).head(top_n)[["Jugador","Equipo","Posicion","Similarity"]]

# ==========================================================
# VISUALIZACIONES
# ==========================================================

def percentile_color(p):
    if p >= 80:   return "#2ecc71"
    elif p >= 60: return "#3498db"
    elif p >= 40: return "#f1c40f"
    elif p >= 20: return "#e67e22"
    else:         return "#e74c3c"

def plot_percentiles_bar(df_pct):
    fig, ax = plt.subplots(figsize=(8, max(4, len(df_pct) * 0.5)))
    colors = [percentile_color(p) for p in df_pct["Percentil"]]
    ax.barh(df_pct["Métrica"], df_pct["Percentil"], color=colors)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Percentil")
    for v in [20, 40, 60, 80]:
        ax.axvline(v, color="grey", linestyle="--", alpha=0.3)
    labels = ax.get_yticklabels()
    for label in labels:
        if label.get_text() in metricas_negativas:
            label.set_color("#e74c3c")
    plt.tight_layout()
    st.pyplot(fig)
    st.caption("🔴 Métricas en rojo = negativas (menor es mejor). El percentil ya está invertido.")

def radar_plot(df, role, players_selected):
    weights  = pesos_roles[role]
    pct_cols = [f"{m}_pct" for m in weights if f"{m}_pct" in df.columns]
    if not pct_cols: st.warning("Sin métricas disponibles."); return
    N      = len(pct_cols)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist() + [0]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for player in players_selected:
        row = df[df["Jugador_ID"] == player]
        if row.empty: continue
        vals = row.iloc[0][pct_cols].fillna(0).tolist() + [row.iloc[0][pct_cols].fillna(0).tolist()[0]]
        ax.plot(angles, vals, linewidth=2, label=player.split(" (")[0])
        ax.fill(angles, vals, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace("_pct","") for m in pct_cols], fontsize=8)
    ax.set_ylim(0, 1); ax.set_yticklabels([]); ax.grid(alpha=0.25)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5,-0.1), frameon=False, fontsize=9)
    plt.tight_layout(); st.pyplot(fig)

def radar_vs_top(players_df, player_name, role, min_minutes):
    df = players_df[players_df["Minutos jugados"] >= min_minutes].copy()
    allowed = rol_pos_map.get(role, [])
    df = df[df["Pos_norm"].apply(lambda x: any(p in allowed for p in x))]
    if df.empty: return
    weights  = pesos_roles[role]
    pct_cols = [f"{m}_pct" for m in weights if f"{m}_pct" in df.columns]
    if not pct_cols: return
    df["_score"] = [sum(row.get(m, 0) * weights[m.replace("_pct","")]
                        for m in pct_cols) for _, row in df.iterrows()]
    top_name   = df.sort_values("_score", ascending=False).iloc[0]["Jugador_ID"]
    player_row = df[df["Jugador_ID"] == player_name]
    top_row    = df[df["Jugador_ID"] == top_name]
    if player_row.empty: return
    N      = len(pct_cols)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist() + [0]
    p_vals = player_row.iloc[0][pct_cols].fillna(0).tolist() + [player_row.iloc[0][pct_cols].fillna(0).tolist()[0]]
    t_vals = top_row.iloc[0][pct_cols].fillna(0).tolist()    + [top_row.iloc[0][pct_cols].fillna(0).tolist()[0]]
    fig, ax = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))
    ax.plot(angles, p_vals, linewidth=2, label=player_name.split(" (")[0])
    ax.fill(angles, p_vals, alpha=0.15)
    ax.plot(angles, t_vals, linewidth=2, linestyle="--", label=f"Top {role}")
    ax.fill(angles, t_vals, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace("_pct","") for m in pct_cols], fontsize=7)
    ax.set_ylim(0, 1); ax.set_yticklabels([]); ax.grid(alpha=0.25)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5,-0.15), frameon=False, fontsize=8)
    plt.tight_layout(); st.pyplot(fig)

def stripplot_role(players_df, role, player_name, min_minutes):
    df = players_df[players_df["Minutos jugados"] >= min_minutes].copy()
    allowed = rol_pos_map.get(role, [])
    df = df[df["Pos_norm"].apply(lambda x: any(p in allowed for p in x))]
    if df.empty: st.warning("No hay jugadores en este rol."); return
    weights    = pesos_roles[role]
    metrics    = sorted([m for m in weights if m in df.columns], key=lambda x: weights[x], reverse=True)
    player_row = df[df["Jugador_ID"] == player_name]
    if player_row.empty: st.warning("Jugador no encontrado."); return
    player_row = player_row.iloc[0]
    for metric in metrics:
        values = df[metric].dropna()
        if len(values) == 0: continue
        p_val = player_row.get(metric, np.nan)
        if pd.isna(p_val): continue
        pct_rank = (values < p_val).sum() / len(values) * 100
        if metric in metricas_negativas: pct_rank = 100 - pct_rank
        fig, ax = plt.subplots(figsize=(8, 0.7))
        ax.scatter(values, np.zeros(len(values)), alpha=0.3, color="#aaaaaa", s=20, zorder=2)
        ax.axvline(values.mean(), color="steelblue", linestyle="--", alpha=0.5, linewidth=1)
        ax.scatter([p_val], [0], color=percentile_color(pct_rank), s=120, zorder=5)
        neg_tag = " ⬇" if metric in metricas_negativas else ""
        ax.set_xlabel(f"{metric}{neg_tag}  (pct: {pct_rank:.0f})", fontsize=9)
        ax.set_yticks([]); ax.spines[["top","right","left"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig)

# ==========================================================
# UI
# ==========================================================

st.sidebar.header("📂 Subir CSV(s)")
files_partidos = st.sidebar.file_uploader(
    "CSV partido a partido", type=["csv"], accept_multiple_files=True, key="partidos"
)
files_jugadores = st.sidebar.file_uploader(
    "CSV jugadores (opcional, añade altura, pie, nacionalidad...)", type=["csv"], accept_multiple_files=True, key="jugadores"
)

if not files_partidos:
    st.info("👈 Sube uno o varios CSV de partido a partido para comenzar.")
    st.stop()

files = files_partidos

players_master = load_data(files)

# Enriquecer con datos de plantilla si se han subido
if files_jugadores:
    squad_df = load_squad_data(files_jugadores)
    if not squad_df.empty:
        # Cruzar por jugador+equipo
        rename_squad = {}
        if 'altura_cm'           in squad_df.columns: rename_squad['altura_cm']           = 'Altura_cm'
        if 'pie_dominante'       in squad_df.columns: rename_squad['pie_dominante']       = 'Pie_dominante'
        if 'nacionalidad'        in squad_df.columns: rename_squad['nacionalidad']        = 'Nacionalidad'
        if 'valor_mercado_eur'   in squad_df.columns: rename_squad['valor_mercado_eur']   = 'Valor_mercado_EUR'
        if 'contrato_hasta'      in squad_df.columns: rename_squad['contrato_hasta']      = 'Contrato_hasta'
        if 'posicion_detallada'  in squad_df.columns: rename_squad['posicion_detallada']  = 'Posicion_detallada'
        if 'fecha_nacimiento'    in squad_df.columns: rename_squad['fecha_nacimiento']    = 'Fecha_nacimiento'
        squad_df = squad_df.rename(columns=rename_squad)
        squad_df['_key'] = squad_df['jugador'].str.strip().str.lower() + '|' + squad_df['equipo'].str.strip().str.lower()
        squad_df = squad_df.drop(columns=['jugador','equipo']).set_index('_key')
        players_master['_key'] = players_master['Jugador'].str.strip().str.lower() + '|' + players_master['Equipo'].str.strip().str.lower()
        for col in squad_df.columns:
            players_master[col] = players_master['_key'].map(squad_df[col])
        players_master = players_master.drop(columns=['_key'])

st.sidebar.divider()
st.sidebar.subheader("🔎 Filtros")

# Filtro de liga
if "liga" in players_master.columns:
    ligas_en_datos = sorted(players_master["liga"].dropna().unique())
    if len(ligas_en_datos) > 1:
        ligas_sel = st.sidebar.multiselect("Liga", ligas_en_datos, default=ligas_en_datos, placeholder="Todas...")
        if ligas_sel:
            players_master = players_master[players_master["liga"].isin(ligas_sel)]

players_master = compute_percentiles(players_master, min_minutes_base=450)
pct_cols = [c for c in players_master.columns if c.endswith("_pct")]

min_minutes = st.sidebar.slider("Minutos mínimos", 0, int(players_master["Minutos jugados"].max()), 450)
players = players_master[players_master["Minutos jugados"] >= min_minutes].copy()

equipos = sorted(players["Equipo"].dropna().unique())
if len(equipos) > 1:
    usar_equipo = st.sidebar.checkbox("Filtrar por Equipo", value=False)
    if usar_equipo:
        equipo_sel = st.sidebar.multiselect("Equipos", equipos, placeholder="Todos...")
        if equipo_sel:
            players = players[players["Equipo"].isin(equipo_sel)]

pos_disp = sorted(set(p for pl in players["Pos_norm"] for p in pl))
if pos_disp:
    usar_pos = st.sidebar.checkbox("Filtrar por Posición", value=False)
    if usar_pos:
        pos_sel = st.sidebar.multiselect("Posiciones", pos_disp, placeholder="Todas...")
        if pos_sel:
            players = players[players["Pos_norm"].apply(lambda x: any(p in pos_sel for p in x))]

# Filtro edad (solo si hay Fecha_nacimiento)
if "Fecha_nacimiento" in players.columns:
    from datetime import datetime as _dt
    def _calc_edad(fn):
        if pd.isna(fn): return None
        for fmt in ["%d/%m/%Y", "%Y-%m-%d"]:
            try: return (_dt.now() - _dt.strptime(str(fn), fmt)).days // 365
            except: pass
        return None
    players["_edad"] = players["Fecha_nacimiento"].apply(_calc_edad)
    edades_validas = players["_edad"].dropna()
    if not edades_validas.empty:
        edad_min_v = int(edades_validas.min())
        edad_max_v = int(edades_validas.max())
        if edad_min_v < edad_max_v:
            usar_edad = st.sidebar.checkbox("Filtrar por Edad", value=False)
            if usar_edad:
                edad_sel = st.sidebar.slider("Rango de edad", edad_min_v, edad_max_v, (edad_min_v, edad_max_v))
                players = players[players["_edad"].between(edad_sel[0], edad_sel[1], inclusive="both") | players["_edad"].isna()]
    players = players.drop(columns=["_edad"], errors="ignore")

# Filtro pie dominante (solo si hay Pie_dominante)
if "Pie_dominante" in players.columns:
    pies_disp = sorted(players["Pie_dominante"].dropna().unique())
    if len(pies_disp) > 1:
        usar_pie = st.sidebar.checkbox("Filtrar por Pie", value=False)
        if usar_pie:
            pie_sel = st.sidebar.multiselect("Pie dominante", pies_disp, placeholder="Todos...")
            if pie_sel:
                players = players[players["Pie_dominante"].isin(pie_sel)]

role_scores = compute_role_scores(players, min_minutes)

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "🏆 Rankings", "🕷 Radar", "📋 Alineación", "📊 Percentiles",
    "🆚 Comparador", "🎯 Role Fit", "📍 Strip Plot", "🔎 Similaridad", "🔵 Scatter", "📝 Informe",
])

with tab1:
    st.subheader("Ranking por Rol")
    if role_scores:
        sel = st.selectbox("Rol", list(role_scores.keys()), index=None, placeholder="Elige un rol...", key="rfef_rank_role")
        if sel:
            df_r = role_scores[sel][role_scores[sel]["Rating"].notna()]
            st.dataframe(df_r[["Jugador","Equipo","Posicion","Minutos jugados","Partidos","Rating"]], use_container_width=True)
            with st.expander("ℹ️ Métricas de este rol"):
                w = pesos_roles[sel]
                rows = [{"Métrica": m, "Peso": f"{v*100:.0f}%",
                         "Tipo": "⬇ negativa" if m in metricas_negativas else ""}
                        for m, v in sorted(w.items(), key=lambda x: -x[1])]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Radar Comparativo")
    if role_scores:
        sel = st.selectbox("Rol", list(role_scores.keys()), index=None, placeholder="Elige un rol...", key="rfef_radar_role")
        if sel:
            jugadores = role_scores[sel]["Jugador_ID"].tolist()
            sel_p = st.multiselect("Jugadores", jugadores, placeholder="Elige jugadores...")
            if sel_p: radar_plot(role_scores[sel], sel, sel_p)

with tab3:
    st.subheader("📋 Alineación Automática")

    # ----------------------------------------------------------
    # Coordenadas por rol y formación (igual que Wyscout)
    # ----------------------------------------------------------
    formation_coords_rfef = {
        "4-3-3": {
            "Portero":            (50, 8),
            "Lateral_Defensivo":  [(18, 28), (82, 28)],
            "Central_Stopper":    (35, 20),
            "Central_Clasico":    (65, 20),
            "Pivote_Defensivo":   (50, 44),
            "Box_to_Box":         (28, 54),
            "Interior":           (72, 54),
            "Extremo_Puro":       (15, 76),
            "Extremo_Asociativo": (85, 76),
            "Delantero_Goleador": (50, 87),
        },
        "4-4-2": {
            "Portero":            (50, 8),
            "Lateral_Defensivo":  [(18, 25), (82, 25)],
            "Central_Clasico":    [(35, 18), (65, 18)],
            "Pivote_Defensivo":   (35, 48),
            "Interior":           (65, 48),
            "Extremo_Puro":       [(15, 72), (85, 72)],
            "Delantero_Goleador": (38, 87),
            "Delantero_Movil":    (62, 87),
        },
        "3-5-2": {
            "Portero":            (50, 8),
            "Central_Salida":     (50, 22),
            "Central_Clasico":    [(28, 18), (72, 18)],
            "Lateral_Ofensivo":   [(12, 44), (88, 44)],
            "Pivote_Defensivo":   (50, 50),
            "Interior":           [(28, 62), (72, 62)],
            "Delantero_Goleador": (38, 87),
            "Delantero_Movil":    (62, 87),
        },
        "5-3-2": {
            "Portero":            (50, 8),
            "Central_Clasico":    [(28, 18), (72, 18)],
            "Central_Salida":     (50, 22),
            "Lateral_Defensivo":  [(10, 28), (90, 28)],
            "Pivote_Defensivo":   (50, 50),
            "Interior":           [(28, 62), (72, 62)],
            "Delantero_Goleador": (38, 87),
            "Delantero_Movil":    (62, 87),
        },
        "4-5-1": {
            "Portero":            (50, 8),
            "Lateral_Defensivo":  [(18, 25), (82, 25)],
            "Central_Clasico":    [(35, 18), (65, 18)],
            "Pivote_Defensivo":   (50, 48),
            "Interior":           [(28, 58), (72, 58)],
            "Extremo_Puro":       [(12, 72), (88, 72)],
            "Delantero_Goleador": (50, 87),
        },
        "3-4-3": {
            "Portero":            (50, 8),
            "Central_Salida":     (50, 22),
            "Central_Clasico":    [(28, 18), (72, 18)],
            "Lateral_Ofensivo":   [(12, 48), (88, 48)],
            "Mediapunta":         (50, 58),
            "Extremo_Puro":       [(18, 78), (82, 78)],
            "Delantero_Goleador": (50, 87),
        },
        "4-3-3 Asimetrico": {
            "Portero":            (50, 8),
            "Lateral_Defensivo":  (15, 27),
            "Central_Clasico":    (35, 18),
            "Central_Stopper":    (65, 18),
            "Lateral_Ofensivo":   (85, 30),
            "Pivote_Defensivo":   (50, 44),
            "Box_to_Box":         (30, 54),
            "Mediapunta":         (68, 60),
            "Extremo_Puro":       (15, 76),
            "Extremo_Asociativo": (82, 78),
            "Delantero_Goleador": (50, 88),
        },
        "4-2-3-1": {
            "Portero":            (50, 8),
            "Lateral_Defensivo":  (15, 27),
            "Central_Clasico":    (35, 18),
            "Central_Stopper":    (65, 18),
            "Lateral_Ofensivo":   (85, 30),
            "Pivote_Defensivo":   (70, 44),
            "Box_to_Box":         (30, 50),
            "Mediapunta":         (50, 70),
            "Extremo_Puro":       (15, 70),
            "Extremo_Asociativo": (82, 70),
            "Delantero_Goleador": (50, 88),
        },
    }

    # Cuántos jugadores necesita cada rol por formación
    formation_roles = {
        "4-3-3": {
            "Portero": 1, "Lateral_Defensivo": 2,
            "Central_Stopper": 1, "Central_Clasico": 1,
            "Pivote_Defensivo": 1, "Box_to_Box": 1, "Interior": 1,
            "Extremo_Puro": 1, "Extremo_Asociativo": 1, "Delantero_Goleador": 1,
        },
        "4-4-2": {
            "Portero": 1, "Lateral_Defensivo": 2, "Central_Clasico": 2,
            "Pivote_Defensivo": 1, "Interior": 1,
            "Extremo_Puro": 2, "Delantero_Goleador": 1, "Delantero_Movil": 1,
        },
        "3-5-2": {
            "Portero": 1, "Central_Salida": 1, "Central_Clasico": 2,
            "Lateral_Ofensivo": 2, "Pivote_Defensivo": 1, "Interior": 2,
            "Delantero_Goleador": 1, "Delantero_Movil": 1,
        },
        "5-3-2": {
            "Portero": 1, "Central_Clasico": 2, "Central_Salida": 1,
            "Lateral_Defensivo": 2, "Pivote_Defensivo": 1, "Interior": 2,
            "Delantero_Goleador": 1, "Delantero_Movil": 1,
        },
        "4-5-1": {
            "Portero": 1, "Lateral_Defensivo": 2, "Central_Clasico": 2,
            "Pivote_Defensivo": 1, "Interior": 2,
            "Extremo_Puro": 2, "Delantero_Goleador": 1,
        },
        "3-4-3": {
            "Portero": 1, "Central_Salida": 1, "Central_Clasico": 2,
            "Lateral_Ofensivo": 2, "Mediapunta": 1,
            "Extremo_Puro": 2, "Delantero_Goleador": 1,
        },
        "4-3-3 Asimetrico": {
            "Portero": 1,
            "Lateral_Defensivo": 1,
            "Central_Clasico": 1,
            "Central_Stopper": 1,
            "Lateral_Ofensivo": 1,
            "Pivote_Defensivo": 1,
            "Box_to_Box": 1,
            "Mediapunta": 1,
            "Extremo_Puro": 1,
            "Extremo_Asociativo": 1,
            "Delantero_Goleador": 1,
        },
        "4-2-3-1": {
            "Portero": 1,
            "Lateral_Defensivo": 1,
            "Central_Clasico": 1,
            "Central_Stopper": 1,
            "Lateral_Ofensivo": 1,
            "Pivote_Defensivo": 1,
            "Box_to_Box": 1,
            "Mediapunta": 1,
            "Extremo_Puro": 1,
            "Extremo_Asociativo": 1,
            "Delantero_Goleador": 1,
        },
    }

    def best_player_for_role_rfef(role_scores, role, used_players):
        if role not in role_scores:
            return "—"
        df = role_scores[role]
        df = df[~df["Jugador_ID"].isin(used_players)]
        if df.empty:
            return "—"
        return df.iloc[0]["Jugador_ID"]

    formacion = st.selectbox("Formación", list(formation_roles.keys()), key="rfef_alin_formacion")

    roles_needed = formation_roles[formacion]
    coords_map   = formation_coords_rfef[formacion]

    # Construir alineación: titular por rol sin repetir jugador
    used_players = []
    alineacion   = []   # lista de (rol_display, jugador, side)

    for rol, cantidad in roles_needed.items():
        for i in range(cantidad):
            side = None
            if cantidad == 2:
                side = "left" if i == 0 else "right"
            rol_display = f"{rol} {i+1}" if cantidad > 1 else rol
            jugador = best_player_for_role_rfef(role_scores, rol, used_players)
            if jugador != "—":
                used_players.append(jugador)
            alineacion.append((rol_display, jugador, side))

    # ----------------------------------------------------------
    # Dibujar campo
    # ----------------------------------------------------------
    from matplotlib.patches import Arc, Rectangle, Circle as MCircle

    fig_p, ax_p = plt.subplots(figsize=(8, 13))
    fig_p.patch.set_facecolor("#1a5c1a")
    ax_p.set_facecolor("#1a5c1a")
    lc, lw2 = "white", 2

    # Líneas campo
    ax_p.plot([0,0],[0,100], color=lc, lw=lw2)
    ax_p.plot([0,100],[100,100], color=lc, lw=lw2)
    ax_p.plot([100,100],[100,0], color=lc, lw=lw2)
    ax_p.plot([100,0],[0,0], color=lc, lw=lw2)
    ax_p.plot([0,100],[50,50], color=lc, lw=lw2)
    ax_p.add_patch(MCircle((50,50), 9, fill=False, color=lc, lw=lw2))
    ax_p.plot(50, 50, 'o', color=lc)
    ax_p.add_patch(Rectangle((30,82), 40, 18, fill=False, ec=lc, lw=lw2))
    ax_p.add_patch(Rectangle((30,0),  40, 18, fill=False, ec=lc, lw=lw2))
    ax_p.add_patch(Rectangle((40,94), 20, 6,  fill=False, ec=lc, lw=lw2))
    ax_p.add_patch(Rectangle((40,0),  20, 6,  fill=False, ec=lc, lw=lw2))
    ax_p.plot(50, 88, 'o', color=lc); ax_p.plot(50, 12, 'o', color=lc)
    ax_p.add_patch(Arc((50,84), 14, 14, theta1=200, theta2=340, color=lc, lw=lw2))
    ax_p.add_patch(Arc((50,16), 14, 14, theta1=20,  theta2=160, color=lc, lw=lw2))
    for cx, cy, t1, t2 in [(0,0,0,90),(100,0,90,180),(0,100,270,360),(100,100,180,270)]:
        ax_p.add_patch(Arc((cx,cy), 6, 6, theta1=t1, theta2=t2, color=lc, lw=lw2))
    ax_p.plot([45,55],[100,100], color=lc, lw=lw2)
    ax_p.plot([45,55],[0,0], color=lc, lw=lw2)
    ax_p.set_xlim(0,100); ax_p.set_ylim(0,100); ax_p.axis("off")

    # Colocar jugadores: top 3 por rol en cada posición
    role_counter = {}
    for rol_display, jugador, side in alineacion:
        rol_base = rol_display.rsplit(" ", 1)[0] if rol_display[-1].isdigit() else rol_display
        role_counter[rol_base] = role_counter.get(rol_base, 0)

        coord = coords_map.get(rol_base)
        if coord is None:
            role_counter[rol_base] += 1
            continue

        if isinstance(coord, list):
            if role_counter[rol_base] < len(coord):
                x, y = coord[role_counter[rol_base]]
            else:
                role_counter[rol_base] += 1
                continue
        else:
            x, y = coord

        role_counter[rol_base] += 1

        # Top 3 de ese rol
        df_rol = role_scores.get(rol_base)
        if df_rol is None or df_rol.empty:
            continue

        top3 = df_rol.head(3)
        label_lines = [rol_base.replace("_", " ")]
        for idx, (_, row) in enumerate(top3.iterrows(), 1):
            label_lines.append(f"{idx}. {row['Jugador']} ({row['Rating']})")

        ax_p.text(x, y, "\n".join(label_lines),
                  ha="center", va="center", fontsize=7, linespacing=1.35,
                  color="black", fontweight="bold",
                  bbox=dict(facecolor="white", alpha=0.88,
                            boxstyle="round,pad=0.4",
                            edgecolor="#333333", linewidth=1.2))

    st.pyplot(fig_p)

    # Lista debajo
    st.divider()
    st.markdown("**Alineación titular**")
    for rol_display, jugador, _ in alineacion:
        rol_base = rol_display.rsplit(" ", 1)[0] if rol_display[-1].isdigit() else rol_display
        df_rol = role_scores.get(rol_base)
        if df_rol is not None and not df_rol.empty:
            top3 = df_rol.head(3)
            alts = " · ".join(
                f"{i+1}. {r['Jugador']} ({r['Rating']})"
                for i, (_, r) in enumerate(top3.iterrows())
            )
            st.write(f"**{rol_display.replace('_',' ')}** → {alts}")
        else:
            st.write(f"**{rol_display.replace('_',' ')}** → —")


with tab4:
    st.subheader("Percentiles por Jugador")
    if role_scores:
        sel_p = st.selectbox("Jugador", sorted(players["Jugador_ID"].unique().tolist()), index=None, placeholder="Elige un jugador...", key="rfef_pct_player")
        sel_r = st.selectbox("Rol", list(role_scores.keys()), index=None, placeholder="Elige un rol...", key="rfef_pct_role")
        if sel_p and sel_r:
            df_pct = player_percentiles(players, sel_p, sel_r)
            if not df_pct.empty:
                st.dataframe(df_pct[["Métrica","Percentil","Tipo"]], use_container_width=True)
                plot_percentiles_bar(df_pct)
            else:
                st.warning("Sin datos para este jugador en este rol.")

with tab5:
    st.subheader("🆚 Comparador de Jugadores")
    if role_scores:
        sel_r = st.selectbox("Rol", list(role_scores.keys()), index=None, placeholder="Elige un rol...", key="rfef_comp_role")
        if sel_r:
            j_sel = st.multiselect("Jugadores", role_scores[sel_r]["Jugador_ID"].tolist(), max_selections=6, placeholder="Elige jugadores...")
            if len(j_sel) >= 2:
                df_comp = role_scores[sel_r][role_scores[sel_r]["Jugador_ID"].isin(j_sel)].copy()
                metrics = [m for m in pesos_roles[sel_r] if f"{m}_pct" in df_comp.columns]
                pct_cols_comp = [f"{m}_pct" for m in metrics]

                # ── Tarjetas de jugador ──────────────────────────────────
                st.markdown("### Perfil")
                cols_cards = st.columns(len(j_sel))
                for ci, jid in enumerate(j_sel):
                    row_c = df_comp[df_comp["Jugador_ID"] == jid].iloc[0]
                    rating = round(float(role_scores[sel_r][role_scores[sel_r]["Jugador_ID"] == jid]["Rating"].iloc[0]), 2)
                    with cols_cards[ci]:
                        st.markdown(f"**{row_c['Jugador']}**")
                        st.caption(f"{row_c['Equipo']}")
                        # datos plantilla si disponibles
                        _meta = []
                        if "Altura_cm" in row_c and pd.notna(row_c.get("Altura_cm")):
                            try: _meta.append(f"📏 {int(float(str(row_c['Altura_cm']).replace(',','.')))} cm")
                            except: pass
                        if "Pie_dominante" in row_c and pd.notna(row_c.get("Pie_dominante")):
                            pie_icon = "🦶" 
                            _meta.append(f"{pie_icon} {row_c['Pie_dominante']}")
                        if "Fecha_nacimiento" in row_c and pd.notna(row_c.get("Fecha_nacimiento")):
                            try:
                                from datetime import datetime
                                _fn = str(row_c["Fecha_nacimiento"])
                                for fmt in ["%d/%m/%Y", "%Y-%m-%d"]:
                                    try:
                                        edad = (datetime.now() - datetime.strptime(_fn, fmt)).days // 365
                                        _meta.append(f"🎂 {edad} años")
                                        break
                                    except: pass
                            except: pass
                        if _meta:
                            st.markdown("  ".join(_meta))
                        st.metric("Rating rol", f"{rating}/10")
                        st.progress(min(rating/10, 1.0))
                        mins = int(row_c.get("Minutos jugados", 0))
                        parts = int(row_c.get("Partidos", max(1, mins//90)))
                        st.caption(f"⏱ {mins} min · {parts} partidos")

                st.divider()

                # ── Radar comparativo ────────────────────────────────────
                st.markdown("### Radar comparativo")
                metrics_radar = metrics[:8]
                pct_radar = [f"{m}_pct" for m in metrics_radar]
                labels_radar = [m.replace("/90","").replace("_"," ").replace("%","").strip() for m in metrics_radar]
                N = len(labels_radar)
                if N >= 3:
                    angles = [n / float(N) * 2 * np.pi for n in range(N)]
                    angles += angles[:1]
                    fig_rad, ax_rad = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
                    ax_rad.set_facecolor("#1a1a2e")
                    fig_rad.patch.set_facecolor("#1a1a2e")
                    colors_rad = ["#3498db","#e74c3c","#2ecc71","#f39c12","#9b59b6","#1abc9c"]
                    for ci2, jid in enumerate(j_sel):
                        row_r = df_comp[df_comp["Jugador_ID"] == jid]
                        if row_r.empty: continue
                        vals_r = [float(row_r.iloc[0].get(p, 0) or 0) * 100 for p in pct_radar]
                        vals_r += vals_r[:1]
                        col_r = colors_rad[ci2 % len(colors_rad)]
                        ax_rad.plot(angles, vals_r, color=col_r, linewidth=2, label=row_r.iloc[0]["Jugador"])
                        ax_rad.fill(angles, vals_r, color=col_r, alpha=0.15)
                    ax_rad.set_xticks(angles[:-1])
                    ax_rad.set_xticklabels(labels_radar, color="white", size=9)
                    ax_rad.set_ylim(0, 100)
                    ax_rad.set_yticks([25,50,75,100])
                    ax_rad.set_yticklabels(["25","50","75","100"], color="grey", size=7)
                    ax_rad.tick_params(colors="white")
                    ax_rad.spines["polar"].set_color("#444")
                    ax_rad.grid(color="#333", linewidth=0.8)
                    ax_rad.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), labelcolor="white",
                                  framealpha=0, fontsize=9)
                    st.pyplot(fig_rad, use_container_width=False)
                    plt.close(fig_rad)

                st.divider()

                # ── Heatmap de percentiles ───────────────────────────────
                st.markdown("### Percentiles por métrica")
                pct_data = {}
                for jid in j_sel:
                    row_h = df_comp[df_comp["Jugador_ID"] == jid]
                    if row_h.empty: continue
                    nombre_h = row_h.iloc[0]["Jugador"]
                    pct_data[nombre_h] = [round(float(row_h.iloc[0].get(p, 0) or 0) * 100, 1) for p in pct_cols_comp]
                if pct_data:
                    df_heat = pd.DataFrame(pct_data, index=[m.replace("/90","").replace("_"," ") for m in metrics])
                    fig_h, ax_h = plt.subplots(figsize=(max(5, len(j_sel)*1.5), max(4, len(metrics)*0.5)))
                    fig_h.patch.set_facecolor("#1a1a2e")
                    ax_h.set_facecolor("#1a1a2e")
                    import matplotlib.colors as mcolors
                    cmap = mcolors.LinearSegmentedColormap.from_list("rfef", ["#e74c3c","#f39c12","#2ecc71"])
                    im = ax_h.imshow(df_heat.values, aspect="auto", cmap=cmap, vmin=0, vmax=100)
                    ax_h.set_xticks(range(len(df_heat.columns)))
                    ax_h.set_xticklabels(df_heat.columns, color="white", fontsize=10)
                    ax_h.set_yticks(range(len(df_heat.index)))
                    ax_h.set_yticklabels(df_heat.index, color="white", fontsize=9)
                    for i in range(len(df_heat.index)):
                        for j in range(len(df_heat.columns)):
                            val_h = df_heat.iloc[i, j]
                            ax_h.text(j, i, f"{val_h:.0f}", ha="center", va="center",
                                      color="white" if val_h < 60 else "black", fontsize=9, fontweight="bold")
                    plt.tight_layout()
                    st.pyplot(fig_h, use_container_width=True)
                    plt.close(fig_h)
            else:
                st.info("Selecciona mínimo 2 jugadores.")

with tab6:
    st.subheader("🎯 Encaje por Rol")
    if role_scores:
        pc = st.selectbox("Jugador", sorted(players["Jugador_ID"].unique().tolist()), index=None, placeholder="Selecciona un jugador...", key="rfef_rolefit_player")
        if pc:
            top_roles = best_roles_for_player(pc, players, min_minutes, top_n=3)
            if top_roles:
                st.markdown("### Mejores roles")
                for rol, score in top_roles:
                    st.write(f"**{rol}** — Rating: {score}")
                st.divider()
                st.markdown("### Comparativa vs mejor del rol")
                cols = st.columns(len(top_roles))
                for i, (rol, score) in enumerate(top_roles):
                    with cols[i]:
                        st.markdown(f"#### {rol} — {score}")
                        radar_vs_top(players, pc, rol, min_minutes)
            else:
                st.warning("No se encontraron roles.")

with tab7:
    st.subheader("📍 Distribución por Métricas")
    if role_scores:
        sel_r = st.selectbox("Rol", list(role_scores.keys()), index=None, placeholder="Elige un rol...", key="rfef_strip_role")
        if sel_r:
            sel_p = st.selectbox("Jugador", role_scores[sel_r]["Jugador_ID"].tolist(), index=None, placeholder="Elige un jugador...", key="rfef_strip_player")
            if sel_p: stripplot_role(players, sel_r, sel_p, min_minutes)

with tab8:
    st.subheader("🔎 Jugadores Similares")
    if role_scores:
        sel_p = st.selectbox("Jugador", sorted(players["Jugador_ID"].unique().tolist()), index=None, placeholder="Elige un jugador...", key="rfef_sim_player")
        sel_r = st.selectbox("Rol", list(role_scores.keys()), index=None, placeholder="Elige un rol...", key="rfef_sim_role")
        if sel_p and sel_r:
            sim_df = find_similar_players(players, sel_r, sel_p, min_minutes, top_n=6)
            if not sim_df.empty:
                # Añadir datos extra
                sim_df2 = sim_df.merge(
                    players[["Jugador","Equipo","Minutos jugados","Partidos"] +
                             [c for c in ["Altura_cm","Pie_dominante","Fecha_nacimiento","Nacionalidad"] if c in players.columns]],
                    on=["Jugador","Equipo"], how="left"
                )
                # Referencia: el jugador seleccionado
                ref_row = players[players["Jugador_ID"] == sel_p].iloc[0]
                ref_name = ref_row["Jugador"]
                ref_equipo = ref_row["Equipo"]

                # ── Tarjetas de similares ────────────────────────────────
                st.markdown(f"### Jugadores similares a **{ref_name}** como *{sel_r.replace('_',' ')}*")
                n_cols = min(3, len(sim_df2))
                rows_cards = [sim_df2.iloc[i:i+n_cols] for i in range(0, len(sim_df2), n_cols)]
                for row_group in rows_cards:
                    cols_s = st.columns(n_cols)
                    for ci, (_, sr) in enumerate(row_group.iterrows()):
                        sim_pct = round(float(sr["Similarity"]) * 100, 1)
                        with cols_s[ci]:
                            # barra de similitud con color
                            color_sim = "#2ecc71" if sim_pct >= 80 else "#3498db" if sim_pct >= 65 else "#f39c12"
                            st.markdown(
                                f"""<div style='background:#1e1e2e;border-radius:10px;padding:12px;border-left:4px solid {color_sim}'>
                                <b style='font-size:15px;color:#ffffff'>{sr['Jugador']}</b><br>
                                <span style='color:#cccccc;font-size:12px'>{sr['Equipo']}</span><br>
                                <span style='color:#aaaaaa;font-size:11px'>{sr.get('Posicion','')}</span>
                                </div>""", unsafe_allow_html=True
                            )
                            st.progress(sim_pct / 100)
                            st.caption(f"Similitud: **{sim_pct}%**")
                            # Datos plantilla
                            _meta_s = []
                            if "Altura_cm" in sr and pd.notna(sr.get("Altura_cm")):
                                try: _meta_s.append(f"📏 {int(float(str(sr['Altura_cm']).replace(',','.')))} cm")
                                except: pass
                            if "Pie_dominante" in sr and pd.notna(sr.get("Pie_dominante")):
                                _meta_s.append(f"🦶 {sr['Pie_dominante']}")
                            if "Fecha_nacimiento" in sr and pd.notna(sr.get("Fecha_nacimiento")):
                                try:
                                    from datetime import datetime
                                    _fn2 = str(sr["Fecha_nacimiento"])
                                    for fmt2 in ["%d/%m/%Y", "%Y-%m-%d"]:
                                        try:
                                            edad2 = (datetime.now() - datetime.strptime(_fn2, fmt2)).days // 365
                                            _meta_s.append(f"🎂 {edad2} años")
                                            break
                                        except: pass
                                except: pass
                            if "Nacionalidad" in sr and pd.notna(sr.get("Nacionalidad")):
                                _meta_s.append(f"🌍 {sr['Nacionalidad']}")
                            if _meta_s:
                                st.caption("  ·  ".join(_meta_s))
                            mins_s = int(sr.get("Minutos jugados", 0) or 0)
                            parts_s = int(sr.get("Partidos", max(1, mins_s // 90)) or 1)
                            st.caption(f"⏱ {mins_s} min · {parts_s} partidos")

                st.divider()

                # ── Radar de comparación con el referente ───────────────
                st.markdown("### Radar: referente vs similares")
                weights_s = pesos_roles[sel_r]
                metrics_s = [m for m in weights_s if f"{m}_pct" in players.columns][:8]
                pct_s = [f"{m}_pct" for m in metrics_s]
                labels_s = [m.replace("/90","").replace("_"," ").strip() for m in metrics_s]
                N_s = len(labels_s)
                if N_s >= 3:
                    angles_s = [n / float(N_s) * 2 * np.pi for n in range(N_s)]
                    angles_s += angles_s[:1]
                    fig_s, ax_s = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
                    ax_s.set_facecolor("#1a1a2e")
                    fig_s.patch.set_facecolor("#1a1a2e")
                    colors_s = ["#f1c40f","#3498db","#e74c3c","#2ecc71","#9b59b6","#e67e22","#1abc9c"]
                    # Primero el referente
                    ref_data = players[players["Jugador_ID"] == sel_p]
                    if not ref_data.empty:
                        vals_ref = [float(ref_data.iloc[0].get(p, 0) or 0) * 100 for p in pct_s]
                        vals_ref += vals_ref[:1]
                        ax_s.plot(angles_s, vals_ref, color="#f1c40f", linewidth=2.5, label=ref_name, linestyle="--")
                        ax_s.fill(angles_s, vals_ref, color="#f1c40f", alpha=0.1)
                    # Luego los similares
                    for ci3, (_, sr3) in enumerate(sim_df2.head(3).iterrows()):
                        sim_row = players[(players["Jugador"] == sr3["Jugador"]) & (players["Equipo"] == sr3["Equipo"])]
                        if sim_row.empty: continue
                        vals_s3 = [float(sim_row.iloc[0].get(p, 0) or 0) * 100 for p in pct_s]
                        vals_s3 += vals_s3[:1]
                        col_s3 = colors_s[ci3 + 1]
                        ax_s.plot(angles_s, vals_s3, color=col_s3, linewidth=1.8, label=sr3["Jugador"])
                        ax_s.fill(angles_s, vals_s3, color=col_s3, alpha=0.1)
                    ax_s.set_xticks(angles_s[:-1])
                    ax_s.set_xticklabels(labels_s, color="white", size=9)
                    ax_s.set_ylim(0, 100)
                    ax_s.set_yticks([25,50,75,100])
                    ax_s.set_yticklabels(["25","50","75","100"], color="grey", size=7)
                    ax_s.spines["polar"].set_color("#444")
                    ax_s.grid(color="#333", linewidth=0.8)
                    ax_s.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), labelcolor="white",
                                framealpha=0, fontsize=9)
                    st.pyplot(fig_s, use_container_width=False)
                    plt.close(fig_s)
            else:
                st.warning("No se encontraron jugadores similares.")

with tab9:
    st.subheader("🔵 Scatter Plot")
    if role_scores:
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            rol_scatter = st.selectbox(
                "Filtrar por rol (opcional)",
                ["Todos"] + list(role_scores.keys()),
                key="rfef_scatter_rol"
            )
        df_scatter = (players[players["Minutos jugados"] >= min_minutes].copy()
                      if rol_scatter == "Todos" else role_scores[rol_scatter].copy())

        numeric_cols = sorted([
            c for c in df_scatter.select_dtypes(include="number").columns
            if not c.endswith("_pct") and c not in ["Minutos jugados", "Partidos"]
        ])

        with col_c2:
            x_metric = st.selectbox("Eje X", numeric_cols, key="rfef_scatter_x",
                                    index=None, placeholder="Elige métrica...")
        with col_c3:
            y_metric = st.selectbox("Eje Y", numeric_cols, key="rfef_scatter_y",
                                    index=None, placeholder="Elige métrica...")

        col_c4, col_c5 = st.columns(2)
        with col_c4:
            color_by = st.selectbox("Color por", ["Rating", "Posición", "Equipo", "Ninguno"], key="rfef_scatter_color")
        with col_c5:
            highlight_player = st.selectbox(
                "Destacar jugador (opcional)",
                ["Ninguno"] + df_scatter["Jugador_ID"].tolist(),
                key="rfef_scatter_highlight"
            )

        equipos_scatter = sorted(df_scatter["Equipo"].dropna().unique().tolist())
        highlight_team = st.selectbox(
            "Destacar equipo (opcional)",
            ["Ninguno"] + equipos_scatter,
            key="rfef_scatter_team"
        )

        show_avg_lines = st.checkbox("Mostrar líneas de media", value=True, key="rfef_scatter_avglines")

        if x_metric and y_metric:
            keep_cols = ["Jugador", "Jugador_ID", "Equipo", x_metric, y_metric]
            if "Rating" in df_scatter.columns:      keep_cols.append("Rating")
            if "Pos_primary" in df_scatter.columns: keep_cols.append("Pos_primary")
            df_plot = df_scatter[keep_cols].dropna(subset=[x_metric, y_metric])

            if color_by == "Rating" and "Rating" in df_plot.columns:
                marker_color = df_plot["Rating"]; colorscale = "RdYlGn"; showscale = True
                colorbar = dict(title="Rating")
            elif color_by == "Posición" and "Pos_primary" in df_plot.columns:
                palette = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd"]
                pos_color_map = {p: palette[i % len(palette)]
                                 for i, p in enumerate(df_plot["Pos_primary"].unique())}
                marker_color = df_plot["Pos_primary"].map(pos_color_map)
                showscale = False; colorscale = None; colorbar = None
            elif color_by == "Equipo":
                equipos_uniq = df_plot["Equipo"].unique()
                import hashlib
                def eq_color(eq):
                    h = int(hashlib.md5(eq.encode()).hexdigest()[:6], 16)
                    r = (h >> 16) & 0xFF; g = (h >> 8) & 0xFF; b = h & 0xFF
                    return f"rgb({r},{g},{b})"
                marker_color = df_plot["Equipo"].apply(eq_color)
                showscale = False; colorscale = None; colorbar = None
            else:
                marker_color = "#1f77b4"; showscale = False; colorscale = None; colorbar = None

            # Masks: highlighted player + highlighted team
            mask_hl_player = (df_plot["Jugador_ID"] == highlight_player
                              if highlight_player != "Ninguno"
                              else pd.Series([False]*len(df_plot), index=df_plot.index))
            mask_hl_team   = (df_plot["Equipo"] == highlight_team
                              if highlight_team != "Ninguno"
                              else pd.Series([False]*len(df_plot), index=df_plot.index))
            mask_hl        = mask_hl_player | mask_hl_team

            fig_sc = go.Figure()

            # Normal points
            df_normal = df_plot[~mask_hl]
            colors_normal = (marker_color[~mask_hl]
                             if hasattr(marker_color, "__getitem__") and not isinstance(marker_color, str)
                             else marker_color)
            fig_sc.add_trace(go.Scatter(
                x=df_normal[x_metric], y=df_normal[y_metric], mode="markers",
                marker=dict(size=8, color=colors_normal, colorscale=colorscale,
                            showscale=showscale, colorbar=colorbar,
                            opacity=0.5, line=dict(width=0.5, color="white")),
                text=df_normal["Jugador"],
                hovertemplate=(f"<b>%{{text}}</b><br>{x_metric}: %{{x:.2f}}<br>"
                               + f"{y_metric}: %{{y:.2f}}<extra></extra>"),
                showlegend=False
            ))

            # Highlighted team
            if highlight_team != "Ninguno":
                df_ht = df_plot[mask_hl_team & ~mask_hl_player]
                if not df_ht.empty:
                    fig_sc.add_trace(go.Scatter(
                        x=df_ht[x_metric], y=df_ht[y_metric], mode="markers+text",
                        marker=dict(size=12, color="#f39c12",
                                    line=dict(width=1.5, color="black")),
                        text=df_ht["Jugador"], textposition="top center",
                        hovertemplate=(f"<b>%{{text}}</b> ({highlight_team})<br>"
                                       + f"{x_metric}: %{{x:.2f}}<br>{y_metric}: %{{y:.2f}}<extra></extra>"),
                        name=highlight_team, showlegend=True
                    ))

            # Highlighted player
            if highlight_player != "Ninguno":
                df_hl = df_plot[mask_hl_player]
                if not df_hl.empty:
                    fig_sc.add_trace(go.Scatter(
                        x=df_hl[x_metric], y=df_hl[y_metric], mode="markers+text",
                        marker=dict(size=18, color="#e74c3c", line=dict(width=2, color="black")),
                        text=df_hl["Jugador"], textposition="top center",
                        hovertemplate=(f"<b>%{{text}}</b><br>{x_metric}: %{{x:.2f}}<br>"
                                       + f"{y_metric}: %{{y:.2f}}<extra></extra>"),
                        name=highlight_player.split(" (")[0], showlegend=True
                    ))

            # Mean lines
            if show_avg_lines:
                mean_x = df_plot[x_metric].mean()
                mean_y = df_plot[y_metric].mean()
                fig_sc.add_vline(x=mean_x, line_dash="dash", line_color="rgba(100,100,100,0.6)",
                                 line_width=1.5)
                fig_sc.add_hline(y=mean_y, line_dash="dash", line_color="rgba(100,100,100,0.6)",
                                 line_width=1.5)
                # Anotaciones como scatter invisible para que queden dentro del área
                fig_sc.add_trace(go.Scatter(
                    x=[mean_x], y=[df_plot[y_metric].max()],
                    mode="text",
                    text=[f"  μ {x_metric}={mean_x:.2f}"],
                    textposition="middle right",
                    textfont=dict(size=10, color="gray"),
                    showlegend=False, hoverinfo="skip"
                ))
                fig_sc.add_trace(go.Scatter(
                    x=[df_plot[x_metric].max()], y=[mean_y],
                    mode="text",
                    text=[f"μ {y_metric}={mean_y:.2f}  "],
                    textposition="middle left",
                    textfont=dict(size=10, color="gray"),
                    showlegend=False, hoverinfo="skip"
                ))

            fig_sc.update_layout(
                title=f"{x_metric}  vs  {y_metric}",
                xaxis_title=x_metric, yaxis_title=y_metric,
                template="simple_white", height=600,
                margin=dict(l=60, r=40, t=60, b=60),
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_sc, use_container_width=True)
            st.caption(f"Mostrando {len(df_plot)} jugadores")
        else:
            st.info("Elige métricas para los ejes X e Y.")

with tab10:
    st.subheader("📝 Informe de Jugador")

    if not role_scores:
        st.info("Carga datos para generar informes.")
    else:
        jugadores_inf = sorted(players[players["Minutos jugados"] >= min_minutes]["Jugador_ID"].unique().tolist())
        sel_inf = st.selectbox("Jugador", ["—"] + jugadores_inf, key="rfef_inf_player")

        if sel_inf != "—":
            row_j = players[players["Jugador_ID"] == sel_inf].iloc[0]

            nombre    = row_j["Jugador"]
            equipo    = row_j["Equipo"]
            posicion  = row_j.get("Posicion", row_j.get("Pos_primary", "?"))
            minutos   = int(row_j["Minutos jugados"])
            partidos  = int(row_j.get("Partidos", max(1, minutos // 90)))
            goles     = round(float(row_j.get("Goles/90", 0) or 0) * minutos / 90, 1)
            asist     = round(float(row_j.get("Asistencias/90", 0) or 0) * minutos / 90, 1)

            # Mejores roles
            roles_player_list = best_roles_for_player(sel_inf, players, min_minutes, top_n=5)
            if not roles_player_list:
                st.warning("No hay suficientes datos para generar el informe.")
                st.stop()

            # Convertir a DataFrame para uso uniforme
            roles_player = pd.DataFrame(roles_player_list, columns=["Rol", "Rating"])
            mejor_rol   = roles_player.iloc[0]["Rol"]
            mejor_score = round(float(roles_player.iloc[0]["Rating"]), 2)

            # Percentiles del mejor rol
            df_pct_inf = player_percentiles(players, sel_inf, mejor_rol)
            # Métricas que no tienen sentido como "debilidad" individual
            EXCLUIR_DEBILIDADES = {
                "Faltas_recibidas/90",  # recibir faltas es positivo
                "Toques/90",            # depende del sistema, no del jugador
                "Pases_campo_propio/90", # posicional, no habilidad
                "Pases_campo_rival/90",  # posicional
            }
            top_fort = df_pct_inf[df_pct_inf["Percentil"] >= 70].sort_values("Percentil", ascending=False)
            met_col = "Métrica" if "Métrica" in df_pct_inf.columns else "Metrica"
            top_deb  = df_pct_inf[
                (df_pct_inf["Percentil"] <= 35) &
                (~df_pct_inf[met_col].isin(EXCLUIR_DEBILIDADES))
            ].sort_values("Percentil")

            # Jugadores similares (top 5 del mejor rol)
            sim_df = find_similar_players(players, mejor_rol, sel_inf, min_minutes, top_n=5)

            # ── Texto narrativo ──────────────────────────────
            def desc_minutos(m):
                if m >= 2000:   return "con una presencia muy destacada en el equipo"
                elif m >= 1200: return "con una participacion regular"
                elif m >= 600:  return "con una participacion limitada"
                else:           return "con pocos minutos esta temporada"

            def desc_rol(rol):
                d = {
                    "Portero":             "portero clasico, valorado por su seguridad bajo palos",
                    "Portero_Avanzado":    "portero con buen juego de pies y participacion en la construccion",
                    "Lateral_Defensivo":   "lateral con perfil defensivo y solidez en el uno contra uno",
                    "Lateral_Ofensivo":    "lateral ofensivo con proyeccion hacia el ataque",
                    "Central_Stopper":     "central fisico y dominador del juego aereo",
                    "Central_Clasico":     "central equilibrado entre defensa y salida de balon",
                    "Central_Salida":      "central con capacidad para iniciar el juego desde atras",
                    "Pivote_Defensivo":    "pivote con perfil defensivo, recuperador de balones",
                    "Interior":            "interior con llegada al area y buen juego entre lineas",
                    "Box_to_Box":          "mediocentro total, capaz de cubrir todo el campo",
                    "Mediapunta":          "mediapunta creativo, enlace entre lineas",
                    "Extremo_Asociativo":  "extremo asociativo, con participacion en combinaciones y asistencias",
                    "Extremo_Puro":        "extremo desequilibrante, con velocidad y regate",
                    "Delantero_Goleador":  "delantero goleador, referencia ofensiva del equipo",
                    "Delantero_Movil":     "delantero movil, con participacion en el juego combinativo",
                }
                return d.get(rol, rol.replace("_"," "))

            def valorar_score(s):
                if s >= 7.5:   return "excepcional"
                elif s >= 6.5: return "muy bueno"
                elif s >= 5.5: return "notable"
                elif s >= 4.5: return "aceptable"
                else:          return "con margen de mejora"

            resumen = (
                f"{nombre} es un jugador de {posicion} del {equipo}, "
                f"{desc_minutos(minutos)} ({minutos} minutos en {partidos} partidos esta temporada). "
                f"Su perfil estadistico lo posiciona mejor como **{mejor_rol.replace('_',' ')}** "
                f"(rating {mejor_score}/10), un {desc_rol(mejor_rol)}."
            )

            seccion_roles = f"Analizando todos los roles compatibles con su posicion, {nombre} obtiene los siguientes ratings:\n"
            for _, r in roles_player.iterrows():
                barra = "█" * int(r['Rating']) + "░" * (10 - int(r['Rating']))
                seccion_roles += f"\n- **{r['Rol'].replace('_',' ')}**: {r['Rating']:.1f}/10"

            if not top_fort.empty:
                items_f = [f"**{r['Metrica'] if 'Metrica' in r else r.get('Métrica','')}** (P{r['Percentil']}%)"
                           for _, r in top_fort.head(5).iterrows()]
                seccion_fort = (
                    f"{nombre} destaca especialmente en: "
                    + ", ".join(items_f[:-1]) + (f" y {items_f[-1]}" if len(items_f) > 1 else (items_f[0] if items_f else ""))
                    + ". Estas son sus principales armas dentro del esquema tactico."
                )
            else:
                seccion_fort = f"{nombre} no presenta metricas especialmente destacadas respecto a su posicion."

            if not top_deb.empty:
                items_d = [f"**{r['Metrica'] if 'Metrica' in r else r.get('Métrica','')}** (P{r['Percentil']}%)"
                           for _, r in top_deb.head(5).iterrows()]
                seccion_deb = (
                    f"Las areas donde {nombre} tiene mayor margen de mejora son: "
                    + ", ".join(items_d[:-1]) + (f" y {items_d[-1]}" if len(items_d) > 1 else (items_d[0] if items_d else ""))
                    + ". Son aspectos a trabajar para aumentar su rendimiento."
                )
            else:
                seccion_deb = f"{nombre} no presenta debilidades estadisticas llamativas en su posicion."

            val = valorar_score(mejor_score)
            conclusion = (
                f"{nombre} es un jugador {val} para el rol de {mejor_rol.replace('_',' ')}, "
                f"con un rating de {mejor_score}/10 en la 1a RFEF esta temporada. "
            )
            if not top_fort.empty:
                mejor_met = top_fort.iloc[0]
                met_nombre = mejor_met.get('Métrica', mejor_met.get('Metrica',''))
                conclusion += f"Su mayor activo es {met_nombre} (percentil {mejor_met['Percentil']}%). "
            if roles_player.shape[0] >= 2:
                segundo_rol = roles_player.iloc[1]
                conclusion += (
                    f"Tambien puede desempenarse como {segundo_rol['Rol'].replace('_',' ')} "
                    f"({segundo_rol['Rating']:.1f}/10), lo que le da versatilidad tactica."
                )

            # ── RENDER ──────────────────────────────────────
            col_i1, col_i2 = st.columns([2, 1])

            with col_i1:
                st.markdown(f"## {nombre}")
                # Datos de plantilla si están disponibles
                _info_parts = [posicion, equipo, f"{minutos} min", "Temporada 2025/26"]
                if 'Altura_cm' in row_j and pd.notna(row_j.get('Altura_cm')):
                    _info_parts.insert(2, f"{int(row_j['Altura_cm'])} cm")
                if 'Pie_dominante' in row_j and pd.notna(row_j.get('Pie_dominante')):
                    _info_parts.insert(3, f"Pie: {row_j['Pie_dominante']}")
                if 'Nacionalidad' in row_j and pd.notna(row_j.get('Nacionalidad')):
                    _info_parts.insert(1, row_j['Nacionalidad'])
                st.markdown(f"*{' · '.join(_info_parts)}*")
                # Valor de mercado y contrato
                _extras = []
                if 'Valor_mercado_EUR' in row_j and pd.notna(row_j.get('Valor_mercado_EUR')):
                    try: _extras.append(f"💶 Valor: {int(float(str(row_j['Valor_mercado_EUR']).replace(',','.'))):,} €")
                    except: pass
                if 'Contrato_hasta' in row_j and pd.notna(row_j.get('Contrato_hasta')):
                    _extras.append(f"📅 Contrato hasta: {row_j['Contrato_hasta']}")
                if 'Posicion_detallada' in row_j and pd.notna(row_j.get('Posicion_detallada')):
                    _extras.append(f"📌 Posición: {row_j['Posicion_detallada']}")
                if _extras:
                    st.markdown("  ".join(_extras))
                st.divider()

                st.markdown("### 1. Resumen")
                st.markdown(resumen)

                st.markdown("### 2. Roles y ratings")
                st.markdown(seccion_roles)
                # Tabla visual
                st.dataframe(
                    roles_player[["Rol","Rating"]].rename(columns={"Rol":"Rol","Rating":"Rating /10"}),
                    use_container_width=True, hide_index=True
                )

                st.markdown("### 3. Fortalezas")
                st.markdown(seccion_fort)
                if not top_fort.empty:
                    for _, r in top_fort.head(6).iterrows():
                        met = r.get('Métrica', r.get('Metrica',''))
                        pct = r['Percentil']
                        bar = "█" * int(pct/10) + "░" * (10 - int(pct/10))
                        st.markdown(f"`{met:<30}` {bar} **{pct}%**")

                st.markdown("### 4. Debilidades")
                st.markdown(seccion_deb)
                if not top_deb.empty:
                    for _, r in top_deb.head(5).iterrows():
                        met = r.get('Métrica', r.get('Metrica',''))
                        pct = r['Percentil']
                        bar = "█" * int(pct/10) + "░" * (10 - int(pct/10))
                        st.markdown(f"`{met:<30}` {bar} **{pct}%**")

                st.markdown("### 5. Conclusion")
                st.markdown(conclusion)

            with col_i2:
                st.markdown("### Ficha")
                st.metric("Jugador", nombre)
                st.metric("Equipo", equipo)
                st.metric("Posicion", posicion)
                st.metric("Minutos", f"{minutos} min")
                st.metric("Mejor rol", mejor_rol.replace("_"," "))
                st.metric("Rating", f"{mejor_score}/10")
                st.divider()
                st.markdown("**Todos los roles:**")
                for _, r in roles_player.iterrows():
                    bar = "█" * int(r['Rating']) + "░" * (10 - int(r['Rating']))
                    st.markdown(f"`{r['Rol'].replace('_',' '):<18}` **{r['Rating']:.1f}**")

            # ── JUGADORES SIMILARES ──────────────────────────
            st.divider()
            st.subheader(f"🔎 Jugadores similares a {nombre} como {mejor_rol.replace('_',' ')}")
            if not sim_df.empty:
                sim_df_show = sim_df.copy()
                sim_df_show["Similitud"] = (sim_df_show["Similarity"] * 100).round(1).astype(str) + "%"
                st.dataframe(
                    sim_df_show[["Jugador","Equipo","Posicion","Similitud"]],
                    use_container_width=True, hide_index=True
                )

                # Mini radar comparativo
                st.markdown("#### Comparativa percentiles (mejor rol)")
                weights_rol = pesos_roles.get(mejor_rol, {})
                mets_radar  = [m for m in list(weights_rol.keys())[:8] if f"{m}_pct" in players.columns]
                if mets_radar:
                    jugadores_comp = [sel_inf] + sim_df["Jugador"].head(3).tolist()
                    # Map Jugador → Jugador_ID for lookup
                    jid_map = players.set_index("Jugador")["Jugador_ID"].to_dict()
                    N      = len(mets_radar)
                    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist() + [0]
                    colors_r = ["#e74c3c","#3498db","#2ecc71","#f39c12"]
                    fig_r, ax_r = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
                    for ci, jug in enumerate(jugadores_comp):
                        jid = jid_map.get(jug, jug)
                        row_r = players[players["Jugador_ID"] == jid]
                        if row_r.empty:
                            row_r = players[players["Jugador_ID"] == sel_inf]
                        vals_r = [float(row_r.iloc[0].get(f"{m}_pct", 0) or 0) for m in mets_radar]
                        vals_r += [vals_r[0]]
                        c = colors_r[ci % len(colors_r)]
                        lw = 2.5 if ci == 0 else 1.5
                        ax_r.plot(angles, vals_r, linewidth=lw, label=jug.split(" (")[0], color=c)
                        ax_r.fill(angles, vals_r, alpha=0.08 if ci > 0 else 0.15, color=c)
                    ax_r.set_xticks(angles[:-1])
                    ax_r.set_xticklabels(mets_radar, fontsize=7)
                    ax_r.set_ylim(0,1); ax_r.set_yticklabels([]); ax_r.grid(alpha=0.25)
                    plt.legend(loc="upper center", bbox_to_anchor=(0.5,-0.12),
                               frameon=False, fontsize=8, ncol=2)
                    plt.tight_layout()
                    st.pyplot(fig_r)
            else:
                st.info("No se encontraron jugadores similares con los filtros actuales.")

            # ── PDF ─────────────────────────────────────────
            st.divider()
            def generar_pdf_jugador():
                from reportlab.lib.pagesizes import A4
                from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                from reportlab.lib.units import cm
                from reportlab.lib import colors
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
                from reportlab.lib.enums import TA_CENTER
                import io, re

                buf = io.BytesIO()
                doc = SimpleDocTemplate(buf, pagesize=A4,
                                        leftMargin=2*cm, rightMargin=2*cm,
                                        topMargin=2*cm, bottomMargin=2*cm)
                styles = getSampleStyleSheet()

                def enc(txt):
                    txt = re.sub(r'\*\*(.*?)\*\*', r'\1', str(txt))
                    txt = txt.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
                    return txt.encode('latin-1', errors='replace').decode('latin-1')

                def clean_m(m):
                    return str(m).replace('%','pct').replace('_',' ').replace('/','/')

                style_title = ParagraphStyle('T', parent=styles['Title'], fontSize=20,
                                             textColor=colors.HexColor('#1a1a2e'), spaceAfter=4)
                style_sub   = ParagraphStyle('S', parent=styles['Normal'], fontSize=10,
                                             textColor=colors.HexColor('#666'), spaceAfter=10, alignment=TA_CENTER)
                style_h2    = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=12,
                                             textColor=colors.HexColor('#2c3e50'), spaceBefore=12, spaceAfter=4)
                style_body  = ParagraphStyle('B', parent=styles['Normal'], fontSize=10, leading=15, spaceAfter=8)

                story = []
                story.append(Paragraph("Informe de Jugador", style_title))
                story.append(Paragraph(enc(nombre), ParagraphStyle('Club', parent=styles['Title'],
                             fontSize=16, textColor=colors.HexColor('#e74c3c'), spaceAfter=2)))
                story.append(Paragraph(enc(f"{posicion} - {equipo} - {minutos} min - 1a RFEF 2025/26"), style_sub))
                story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#e74c3c'), spaceAfter=10))

                # Ficha tabla
                ficha = [
                    ["Posicion", enc(posicion), "Equipo", enc(equipo)],
                    ["Minutos", str(minutos), "Partidos", str(partidos)],
                    ["Mejor rol", enc(mejor_rol.replace('_',' ')), "Rating", f"{mejor_score}/10"],
                ]
                t = Table(ficha, colWidths=[3.5*cm, 5.5*cm, 3.5*cm, 4*cm])
                t.setStyle(TableStyle([
                    ('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#f8f9fa')),
                    ('BACKGROUND',(0,0),(0,-1),colors.HexColor('#2c3e50')),
                    ('BACKGROUND',(2,0),(2,-1),colors.HexColor('#2c3e50')),
                    ('TEXTCOLOR',(0,0),(0,-1),colors.white),
                    ('TEXTCOLOR',(2,0),(2,-1),colors.white),
                    ('FONTNAME',(0,0),(-1,-1),'Helvetica'),
                    ('FONTNAME',(0,0),(0,-1),'Helvetica-Bold'),
                    ('FONTNAME',(2,0),(2,-1),'Helvetica-Bold'),
                    ('FONTSIZE',(0,0),(-1,-1),9),
                    ('GRID',(0,0),(-1,-1),0.5,colors.HexColor('#dee2e6')),
                    ('PADDING',(0,0),(-1,-1),6),
                ]))
                story.append(t); story.append(Spacer(1,10))

                # Roles
                story.append(Paragraph("Ratings por Rol", style_h2))
                roles_data = [["Rol", "Rating /10", "Nivel"]]
                for _, r in roles_player.iterrows():
                    nivel = "Excepcional" if r['Rating']>=7.5 else ("Muy bueno" if r['Rating']>=6.5 else ("Notable" if r['Rating']>=5.5 else "Aceptable"))
                    roles_data.append([enc(r['Rol'].replace('_',' ')), f"{r['Rating']:.1f}", nivel])
                tr = Table(roles_data, colWidths=[6*cm, 3*cm, 4*cm])
                tr.setStyle(TableStyle([
                    ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#2c3e50')),
                    ('TEXTCOLOR',(0,0),(-1,0),colors.white),
                    ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
                    ('FONTNAME',(0,1),(-1,-1),'Helvetica'),
                    ('FONTSIZE',(0,0),(-1,-1),9),
                    ('GRID',(0,0),(-1,-1),0.5,colors.HexColor('#dee2e6')),
                    ('PADDING',(0,0),(-1,-1),5),
                    ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f8f9fa'),colors.white]),
                ]))
                story.append(tr); story.append(Spacer(1,8))

                # Secciones narrativas
                for titulo, texto in [
                    ("Resumen", resumen),
                    ("Fortalezas", seccion_fort),
                    ("Debilidades", seccion_deb),
                    ("Conclusion", conclusion),
                ]:
                    story.append(HRFlowable(width="100%", thickness=0.5,
                                            color=colors.HexColor('#dee2e6'), spaceAfter=4))
                    story.append(Paragraph(titulo, style_h2))
                    story.append(Paragraph(enc(texto), style_body))

                # Fortalezas tabla
                if not top_fort.empty or not top_deb.empty:
                    story.append(HRFlowable(width="100%", thickness=0.5,
                                            color=colors.HexColor('#dee2e6'), spaceAfter=4))
                    story.append(Paragraph("Metricas Detalladas", style_h2))
                    met_data = [["Metrica", "Percentil", "Tipo"]]
                    for _, r in top_fort.head(6).iterrows():
                        met_data.append([clean_m(r.get('Métrica', r.get('Metrica',''))),
                                         f"{r['Percentil']}%", "Fortaleza"])
                    for _, r in top_deb.head(5).iterrows():
                        met_data.append([clean_m(r.get('Métrica', r.get('Metrica',''))),
                                         f"{r['Percentil']}%", "Debilidad"])
                    tm = Table(met_data, colWidths=[7*cm, 3*cm, 3*cm])
                    tm.setStyle(TableStyle([
                        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#2c3e50')),
                        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
                        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
                        ('FONTNAME',(0,1),(-1,-1),'Helvetica'),
                        ('FONTSIZE',(0,0),(-1,-1),9),
                        ('GRID',(0,0),(-1,-1),0.5,colors.HexColor('#dee2e6')),
                        ('PADDING',(0,0),(-1,-1),5),
                        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f8f9fa'),colors.white]),
                    ]))
                    story.append(tm)

                # Jugadores similares
                if not sim_df.empty:
                    story.append(HRFlowable(width="100%", thickness=0.5,
                                            color=colors.HexColor('#dee2e6'), spaceAfter=4))
                    story.append(Paragraph(f"Jugadores Similares (rol: {enc(mejor_rol.replace('_',' '))})", style_h2))
                    sim_data = [["Jugador", "Equipo", "Posicion", "Similitud"]]
                    for _, r in sim_df.iterrows():
                        sim_data.append([enc(r['Jugador']), enc(r['Equipo']),
                                         enc(r['Posicion']), f"{r['Similarity']*100:.1f}%"])
                    ts2 = Table(sim_data, colWidths=[5*cm, 4*cm, 2.5*cm, 2.5*cm])
                    ts2.setStyle(TableStyle([
                        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#2c3e50')),
                        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
                        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
                        ('FONTNAME',(0,1),(-1,-1),'Helvetica'),
                        ('FONTSIZE',(0,0),(-1,-1),9),
                        ('GRID',(0,0),(-1,-1),0.5,colors.HexColor('#dee2e6')),
                        ('PADDING',(0,0),(-1,-1),5),
                        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f8f9fa'),colors.white]),
                    ]))
                    story.append(ts2)

                # Pie
                story.append(Spacer(1,20))
                story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e74c3c')))
                story.append(Paragraph("Generado con AppScout · 1a RFEF 2025/26",
                             ParagraphStyle('F', parent=styles['Normal'], fontSize=8,
                                            textColor=colors.grey, alignment=TA_CENTER, spaceBefore=6)))
                doc.build(story)
                buf.seek(0)
                return buf.read()

            pdf_bytes = generar_pdf_jugador()
            st.download_button(
                label="📄 Descargar informe PDF",
                data=pdf_bytes,
                file_name=f"informe_{nombre.replace(' ','_')}.pdf",
                mime="application/pdf",
                type="primary",
            )
        else:
            st.info("Selecciona un jugador para generar su informe.")
