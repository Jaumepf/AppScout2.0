import streamlit as st
import contextlib
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

    # ── PORTEROS ─────────────────────────────────────────────────────────────
    "Portero_Clasico": {
        "Goles_evitados_%":   0.22,
        "Paradas/90":         0.18,
        "Porterias_cero_%":   0.15,
        "Goles_encajados/90": 0.15,
        "Pases_largos/90":    0.12,
        "Precision_pases_%":  0.08,
        "Error_disparo/90":   0.06,
        "Salidas/90":         0.04,
    },
    "Portero_Moderno": {
        "Salidas/90":         0.20,
        "Goles_evitados_%":   0.14,
        "Pases/90":           0.14,
        "Precision_pases_%":  0.12,
        "Pases_largos/90":    0.12,
        "Precision_largos_%": 0.10,
        "Ratio_pases_adelante_%": 0.10,
        "Goles_encajados/90": 0.08,
    },

    # ── CENTRALES ─────────────────────────────────────────────────────────────
    "Central_Stopper": {
        "Duelos_aereos_ganados_%": 0.18,
        "Duelos_aereos/90":        0.14,
        "Duelos_ganados_%":        0.14,
        "Despejes/90":             0.12,
        "Intercepciones/90":       0.12,
        "Entradas/90":             0.12,
        "Recuperaciones/90":       0.10,
        "Regateado/90":            0.08,
    },
    "Central_Clasico": {
        "Duelos_ganados_%":        0.16,
        "Duelos_aereos_ganados_%": 0.12,
        "Entradas_ganadas_%":      0.12,
        "Despejes/90":             0.12,
        "Recuperaciones/90":       0.12,
        "Pases/90":                0.12,
        "Precision_pases_%":       0.12,
        "Regateado/90":            0.12,
    },
    "Central_Salida": {
        "Pases/90":                0.18,
        "Precision_pases_%":       0.16,
        "Pases_campo_rival/90":    0.14,
        "Ratio_pases_adelante_%":  0.12,
        "Pases_largos/90":         0.12,
        "Precision_largos_%":      0.10,
        "Recuperaciones/90":       0.10,
        "Perdidas/90":             0.08,
    },

    # ── LATERALES ─────────────────────────────────────────────────────────────
    "Lateral_Defensivo": {
        "Duelos_ganados_%":        0.16,
        "Entradas/90":             0.14,
        "Entradas_ganadas_%":      0.14,
        "Recuperaciones/90":       0.14,
        "Intercepciones/90":       0.12,
        "Despejes/90":             0.10,
        "Duelos_aereos_ganados_%": 0.10,
        "Regateado/90":            0.10,
    },
    "Lateral_Ofensivo": {
        "xA/90":                   0.18,
        "Asistencias/90":          0.14,
        "Centros/90":              0.12,
        "Pases_campo_rival/90":    0.12,
        "Regates/90":              0.12,
        "Faltas_recibidas/90":     0.10,
        "Recuperaciones/90":       0.10,
        "Entradas/90":             0.12,
    },
    "Carrilero": {
        "Pases_campo_rival/90":    0.18,
        "xA/90":                   0.14,
        "Recuperaciones/90":       0.14,
        "Entradas/90":             0.12,
        "Pases/90":                0.12,
        "Centros/90":              0.10,
        "Duelos_ganados_%":        0.10,
        "Perdidas/90":             0.10,
    },

    # ── CENTROCAMPISTAS ───────────────────────────────────────────────────────
    "Pivote_Defensivo": {
        "Recuperaciones/90":       0.20,
        "Intercepciones/90":       0.18,
        "Entradas/90":             0.14,
        "Duelos_ganados_%":        0.14,
        "Entradas_ganadas_%":      0.10,
        "Pases/90":                0.10,
        "Precision_pases_%":       0.08,
        "Perdidas/90":             0.06,
    },
    "Mediocentro_Org": {
        "Pases/90":                0.20,
        "Precision_pases_%":       0.18,
        "Pases_campo_rival/90":    0.14,
        "Ratio_pases_adelante_%":  0.12,
        "Pases_largos/90":         0.12,
        "Precision_largos_%":      0.10,
        "Recuperaciones/90":       0.08,
        "Perdidas/90":             0.06,
    },
    "Box_to_Box": {
        "Recuperaciones/90":       0.16,
        "Goles/90":                0.14,
        "xG/90":                   0.12,
        "xA/90":                   0.12,
        "Duelos_ganados_%":        0.12,
        "Regates/90":              0.10,
        "Entradas/90":             0.10,
        "Faltas_recibidas/90":     0.14,
    },
    "Interior_Mediapunta": {
        "xA/90":                   0.18,
        "Pases_campo_rival/90":    0.16,
        "Asistencias/90":          0.12,
        "Faltas_recibidas/90":     0.12,
        "Regates/90":              0.12,
        "Goles/90":                0.12,
        "xG/90":                   0.10,
        "Toques/90":               0.08,
    },

    # ── EXTREMOS ──────────────────────────────────────────────────────────────
    "Extremo_Puro": {
        "Regates/90":              0.22,
        "Regates_exito_%":         0.16,
        "Faltas_recibidas/90":     0.14,
        "Duelos_ganados_%":        0.12,
        "xA/90":                   0.10,
        "Goles/90":                0.10,
        "xG/90":                   0.08,
        "Pases_campo_rival/90":    0.08,
    },
    "Extremo_Asociativo": {
        "xA/90":                   0.18,
        "Asistencias/90":          0.14,
        "Centros/90":              0.14,
        "Precision_centros_%":     0.12,
        "Pases_campo_rival/90":    0.12,
        "Faltas_recibidas/90":     0.10,
        "Goles/90":                0.10,
        "xG/90":                   0.10,
    },
    "Interior_Banda": {
        "xG/90":                   0.22,
        "Goles/90":                0.18,
        "xG_Overperformance_90":   0.12,
        "Remates/90":              0.12,
        "Regates/90":              0.12,
        "Faltas_recibidas/90":     0.10,
        "xA/90":                   0.08,
        "Centros/90":              0.06,
    },

    # ── DELANTEROS ────────────────────────────────────────────────────────────
    "Delantero_Goleador": {
        "Goles/90":                0.22,
        "xG/90":                   0.18,
        "xG_Overperformance_90":   0.14,
        "Conversion_Gol_%":        0.14,
        "Remates/90":              0.12,
        "Tiros_a_puerta_%":        0.10,
        "Duelos_aereos_ganados_%": 0.06,
        "Faltas_recibidas/90":     0.04,
    },
    "Delantero_Movil": {
        "Goles/90":                0.16,
        "xG/90":                   0.14,
        "xA/90":                   0.14,
        "Asistencias/90":          0.12,
        "Regates/90":              0.12,
        "Faltas_recibidas/90":     0.10,
        "xG_Overperformance_90":   0.10,
        "Pases_campo_rival/90":    0.12,
    },
    "Segundo_Delantero": {
        "Duelos_aereos/90":        0.24,
        "Duelos_aereos_ganados_%": 0.18,
        "Faltas_recibidas/90":     0.14,
        "Toques/90":               0.12,
        "xG/90":                   0.12,
        "Duelos_ganados_%":        0.10,
        "Faltas/90":               0.06,
        "Regates/90":              0.04,
    },
}

metricas_negativas = [
    "Faltas/90", "Perdidas/90", "Regateado/90",
    "Error_disparo/90", "Goles_encajados/90",
]

# Métricas negativas específicas por rol (solo se aplican para ese rol concreto)
metricas_negativas_por_rol = {
    "Interior_Banda": ["Centros/90"],
    "Carrilero":      ["Perdidas/90"],
    "Central_Salida": ["Perdidas/90"],
    "Pivote_Defensivo": ["Perdidas/90"],
    "Mediocentro_Org":  ["Perdidas/90"],
}

def es_negativa(metrica, rol=None):
    """Devuelve True si la métrica es negativa para el rol dado."""
    if metrica in metricas_negativas:
        return True
    if rol and metrica in metricas_negativas_por_rol.get(rol, []):
        return True
    return False

pos_map = {"G": ["GK"], "D": ["CB","RB","LB"], "M": ["DM","CM","AM"], "F": ["RW","LW","FW","ST"]}

pos_detallada_map = {
    "GK": "GK", "G":  "GK",
    "DC": "CB", "D":  "CB",
    "DR": "RB", "DL": "LB",
    "MC": "CM", "M":  "CM",
    "DM": "DM", "AM": "AM",
    "MR": "RM", "ML": "LM",
    "ST": "ST", "F":  "ST",
    "RW": "RW", "LW": "LW",
    "CB": "CB", "RB": "RB", "LB": "LB",
    "CM": "CM", "FW": "ST", "CF": "ST", "SS": "ST",
}

def normalize_pos_detallada(pos_det):
    if pd.isna(pos_det) or str(pos_det).strip() == '': return []
    result = []
    partes = str(pos_det).strip().upper().split('/')
    for parte in partes:
        key = parte.strip()
        mapped = pos_detallada_map.get(key)
        if mapped:
            if mapped not in result:
                result.append(mapped)
        else:
            for k, v in pos_detallada_map.items():
                if k in key and v not in result:
                    result.append(v)
                    break
    return result

rol_pos_map = {
    "Portero_Clasico":         ["GK"],
    "Portero_Moderno":         ["GK"],
    "Lateral_Defensivo":       ["RB","LB"],
    "Lateral_Ofensivo":        ["RB","LB"],
    "Carrilero":               ["RB","LB","RM","LM"],
    "Central_Stopper":         ["CB"],
    "Central_Clasico":         ["CB"],
    "Central_Salida":          ["CB"],
    "Pivote_Defensivo":        ["DM","CM"],
    "Mediocentro_Org":         ["DM","CM"],
    "Box_to_Box":              ["DM","CM","RM","LM"],
    "Interior_Mediapunta":     ["DM","CM","AM","RM","LM"],
    "Extremo_Asociativo":      ["RW","LW","AM","RM","LM"],
    "Extremo_Puro":            ["RW","LW"],
    "Interior_Banda":          ["RW","LW","RM","LM"],
    "Delantero_Movil":         ["ST","RW","LW"],
    "Delantero_Goleador":      ["ST"],
    "Segundo_Delantero":       ["ST"],
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

    # Columnas numéricas que deben ser 0 si no existen o son NaN
    # (si un jugador no hizo regates en un partido, es 0, no desconocido)
    stat_cols = [
        'goles','tiros_totales','tiros_a_puerta','tiros_al_palo','tiros_bloqueados',
        'xG','xA','asistencias','pases_totales','pases_precisos',
        'pases_campo_propio','pases_campo_rival',
        'pases_largos_totales','pases_largos_precisos',
        'centros_totales','centros_precisos',
        'entradas_totales','entradas_ganadas','despejes',
        'duelos_ganados','duelos_perdidos',
        'duelos_aereos_ganados','duelos_aereos_perdidos',
        'regates_intentados','regates_exitosos','regateado',
        'faltas_cometidas','faltas_recibidas',
        'recuperaciones','perdidas','toques',
        'paradas_totales','paradas_dentro_area','salidas',
        'error_que_lleva_a_disparo','resultado_rival',
    ]
    for col in stat_cols:
        if col not in raw.columns:
            raw[col] = 0
        else:
            raw[col] = pd.to_numeric(raw[col], errors='coerce').fillna(0)

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

@st.cache_data(show_spinner=False)
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
            def to_pct(v, rv=ref_vals, neg=es_negativa(metric, rol)):
                if pd.isna(v): return np.nan
                n = len(rv)
                rank = (rv < v).sum() + (rv == v).sum() * 0.5
                pct = rank / n
                return float(np.clip(1 - pct if neg else pct, 0, 1))
            players.loc[idx_all, f"{metric}_pct"] = players.loc[idx_all, metric].apply(to_pct)
    return players

@st.cache_data(show_spinner=False)
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
        "Tipo": "⬇ negativa" if es_negativa(m.replace("_pct",""), role) else ""
    } for m in pct_cols])

def best_roles_for_player(player_name, players, min_minutes, top_n=3):
    df = players[players["Minutos jugados"] >= min_minutes].copy()
    player_row = df[df["Jugador_ID"] == player_name]
    if player_row.empty: return []
    positions = player_row.iloc[0]["Pos_norm"]
    is_gk = "GK" in positions
    if is_gk:
        allowed_roles = [r for r in pesos_roles if "Portero_Clasico" in r]
    else:
        allowed_roles = [r for r, pl in rol_pos_map.items()
                         if any(p in positions for p in pl) and "Portero_Clasico" not in r]
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

def pos_display(row):
    """Devuelve Posicion_detallada si existe, si no Posicion genérica."""
    pd_val = row.get("Posicion_detallada", None)
    if pd_val is not None and pd.notna(pd_val) and str(pd_val).strip():
        return str(pd_val).strip()
    return row.get("Posicion", "")

@st.cache_data(show_spinner=False)
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
    df_out = df.sort_values("Similarity", ascending=False).head(top_n).copy()
    df_out["Posicion"] = df_out.apply(pos_display, axis=1)
    return df_out[["Jugador","Equipo","Posicion","Similarity"]]

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
        if es_negativa(label.get_text()):
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
        if es_negativa(metric, role): pct_rank = 100 - pct_rank
        fig, ax = plt.subplots(figsize=(8, 0.7))
        ax.scatter(values, np.zeros(len(values)), alpha=0.3, color="#aaaaaa", s=20, zorder=2)
        ax.axvline(values.mean(), color="steelblue", linestyle="--", alpha=0.5, linewidth=1)
        ax.scatter([p_val], [0], color=percentile_color(pct_rank), s=120, zorder=5)
        neg_tag = " ⬇" if es_negativa(metric, role) else ""
        ax.set_xlabel(f"{metric}{neg_tag}  (pct: {pct_rank:.0f})", fontsize=9)
        ax.set_yticks([]); ax.spines[["top","right","left"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig)

# ==========================================================
# UI
# ==========================================================

st.sidebar.header("📂 Subir CSV(s)")
all_files = st.sidebar.file_uploader(
    "Sube los CSV (partidos y/o jugadores)", type=["csv"], accept_multiple_files=True, key="all_files"
)

# Separar automáticamente por nombre de archivo
files_partidos = [f for f in all_files if 'partidos' in getattr(f, 'name', '').lower()]
files_jugadores = [f for f in all_files if 'jugadores' in getattr(f, 'name', '').lower()]

if not files_partidos:
    st.info("👈 Sube los CSV de partidos (y opcionalmente los de jugadores) para comenzar.")
    if all_files:
        st.warning(f"⚠️ Archivos subidos: {[getattr(f,'name','') for f in all_files]}. Asegúrate que los nombres contienen 'partidos' o 'jugadores'.")
    st.stop()

players_master = load_data(files_partidos)

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
        # Recalcular Pos_norm usando posicion_detallada si está disponible
        if 'Posicion_detallada' in players_master.columns:
            def _recalc_pos_norm(row):
                pos_det = row.get('Posicion_detallada')
                if pd.notna(pos_det) and str(pos_det).strip():
                    result = normalize_pos_detallada(str(pos_det))
                    if result:
                        return result
                return row['Pos_norm']
            players_master['Pos_norm'] = players_master.apply(_recalc_pos_norm, axis=1)

st.sidebar.divider()
st.sidebar.subheader("🔎 Filtros")

# Filtro de liga
if "liga" in players_master.columns:
    ligas_en_datos = sorted(players_master["liga"].dropna().unique())
    if len(ligas_en_datos) > 1:
        ligas_sel = st.sidebar.multiselect("Liga", ligas_en_datos, default=ligas_en_datos, placeholder="Todas...")
        if ligas_sel:
            players_master = players_master[players_master["liga"].isin(ligas_sel)]

# Percentiles siempre sobre el master completo (independiente del slider)
with st.spinner("Calculando percentiles...") if "players_master_pct" not in st.session_state else contextlib.nullcontext():
    players_master = compute_percentiles(players_master, min_minutes_base=450)
pct_cols = [c for c in players_master.columns if c.endswith("_pct")]

min_minutes = st.sidebar.slider("Minutos mínimos", 0, int(players_master["Minutos jugados"].max()), 450)

# Filtrar por minutos (operación ligera, sin recalcular percentiles)
players = players_master[players_master["Minutos jugados"] >= min_minutes].copy()

# Calcular edad una sola vez sobre players_master para uso en tabs
from datetime import datetime as _dt
@st.cache_data(show_spinner=False)
def calc_edades(fechas_series):
    def _edad(fn):
        if pd.isna(fn): return None
        for fmt in ["%d/%m/%Y", "%Y-%m-%d"]:
            try: return (_dt.now() - _dt.strptime(str(fn), fmt)).days // 365
            except: pass
        return None
    return fechas_series.apply(_edad)

if "Fecha_nacimiento" in players.columns:
    players["_edad"] = calc_edades(players["Fecha_nacimiento"])
else:
    players["_edad"] = None

role_scores = compute_role_scores(players, min_minutes)

# ── Helper: filtros locales por tab ─────────────────────────────────────────
def filtros_tab(df, key_prefix, mostrar_pos=True, mostrar_pie=True, mostrar_edad=True, mostrar_equipo=False):
    """Aplica filtros locales de posición, pie, edad y equipo. Devuelve df filtrado."""
    resultado = df  # no copiar hasta que haya filtro real
    cols = st.columns(4 if mostrar_equipo else 3)
    ci = 0
    # Posición
    if mostrar_pos:
        pos_disp = sorted(set(p for pl in resultado["Pos_norm"] for p in pl))
        if pos_disp:
            with cols[ci]:
                pos_sel = st.multiselect("Posición", pos_disp, placeholder="Todas...", key=f"{key_prefix}_pos")
                if pos_sel:
                    resultado = resultado[resultado["Pos_norm"].apply(lambda x: any(p in pos_sel for p in x))]
            ci += 1
    # Pie dominante
    if mostrar_pie and "Pie_dominante" in resultado.columns:
        pies_disp = sorted(resultado["Pie_dominante"].dropna().unique())
        if len(pies_disp) > 1:
            with cols[ci]:
                pie_sel = st.multiselect("Pie", pies_disp, placeholder="Todos...", key=f"{key_prefix}_pie")
                if pie_sel:
                    resultado = resultado[resultado["Pie_dominante"].isin(pie_sel)]
            ci += 1
    # Edad
    if mostrar_edad and "_edad" in resultado.columns:
        edades = resultado["_edad"].dropna()
        if not edades.empty and int(edades.min()) < int(edades.max()):
            with cols[ci]:
                edad_min_v, edad_max_v = int(edades.min()), int(edades.max())
                edad_sel = st.slider("Edad", edad_min_v, edad_max_v, (edad_min_v, edad_max_v), key=f"{key_prefix}_edad")
                if edad_sel != (edad_min_v, edad_max_v):
                    resultado = resultado[resultado["_edad"].between(edad_sel[0], edad_sel[1]) | resultado["_edad"].isna()]
            ci += 1
    # Equipo
    if mostrar_equipo:
        equipos_disp = sorted(resultado["Equipo"].dropna().unique())
        if len(equipos_disp) > 1:
            with cols[ci]:
                eq_sel = st.multiselect("Equipo", equipos_disp, placeholder="Todos...", key=f"{key_prefix}_equipo")
                if eq_sel:
                    resultado = resultado[resultado["Equipo"].isin(eq_sel)]
    return resultado

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "🏆 Rankings", "🕷 Radar", "📋 Alineación", "📊 Percentiles",
    "🆚 Comparador", "🎯 Role Fit", "📍 Strip Plot", "🔎 Similaridad", "🔵 Scatter", "📝 Informe",
])

with tab1:
    st.subheader("Ranking por Rol")
    if role_scores:
        sel = st.selectbox("Rol", list(role_scores.keys()), index=None, placeholder="Elige un rol...", key="rfef_rank_role")
        if sel:
            df_r = role_scores[sel][role_scores[sel]["Rating"].notna()].copy()
            # Filtros locales
            df_r = filtros_tab(df_r, "rank", mostrar_pos=False, mostrar_pie=True, mostrar_edad=True, mostrar_equipo=True)
            df_r_show = df_r.copy()
            df_r_show["Posicion"] = df_r_show.apply(
                lambda r: r["Posicion_detallada"]
                if "Posicion_detallada" in r.index and pd.notna(r.get("Posicion_detallada")) and str(r.get("Posicion_detallada","")).strip() != ""
                else r["Posicion"], axis=1
            )
            st.dataframe(df_r_show[["Jugador","Equipo","Posicion","Minutos jugados","Partidos","Rating"]], use_container_width=True)
            with st.expander("ℹ️ Métricas de este rol"):
                w = pesos_roles[sel]
                rows = [{"Métrica": m, "Peso": f"{v*100:.0f}%",
                         "Tipo": "⬇ negativa" if es_negativa(m, sel) else ""}
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
    st.subheader("📋 Alineación")

    # ── Selector de líneas ───────────────────────────────────────────
    col_l1, col_l2, col_l3, col_l4 = st.columns(4)
    with col_l1: st.markdown("**Portero:** 1")
    with col_l2: n_def = st.selectbox("Defensas",       [3,4,5],    index=1, key="alin_def")
    with col_l3: n_med = st.selectbox("Mediocampistas", [2,3,4,5],  index=1, key="alin_med")
    with col_l4: n_del = st.selectbox("Delanteros",     [1,2,3],    index=1, key="alin_del")

    total = 1 + n_def + n_med + n_del
    if total != 11:
        st.warning(f"La suma es {total}/11. Ajusta las líneas para completar el equipo.")

    # ── Calcular slots ────────────────────────────────────────────────
    def calcular_slots(n_def, n_med, n_del):
        slots = []
        lineas = [
            (1,    "Portero",        8),
            (n_def,"Defensa",        28),
            (n_med,"Mediocampista",  52),
            (n_del,"Delantero",      76),
        ]
        idx = 0
        for n, etiqueta, y in lineas:
            for i in range(n):
                x = 50 if n == 1 else 10 + (80 / (n - 1)) * i if n > 1 else 50
                lbl = etiqueta if n == 1 else f"{etiqueta} {i+1}"
                slots.append({"id": idx, "label": lbl, "x": x, "y": y})
                idx += 1
        return slots

    SLOTS = calcular_slots(n_def, n_med, n_del)
    n_slots = len(SLOTS)

    # ── Session state ─────────────────────────────────────────────────
    key_slots = f"alin_slots_{n_def}_{n_med}_{n_del}"
    if key_slots not in st.session_state:
        st.session_state[key_slots] = {
            s["id"]: {"rol": None, "jugador": None, "rating": None, "pos_det": None, "equipo": None}
            for s in SLOTS
        }
    if "alin_editing" not in st.session_state:
        st.session_state["alin_editing"] = None

    alin = st.session_state[key_slots]
    if len(alin) != n_slots:
        st.session_state[key_slots] = {
            s["id"]: {"rol": None, "jugador": None, "rating": None, "pos_det": None, "equipo": None}
            for s in SLOTS
        }
        alin = st.session_state[key_slots]

    # ── Botón reset + info ────────────────────────────────────────────
    col_r, col_i = st.columns([1, 5])
    with col_r:
        if st.button("🔄 Limpiar", key="alin_reset"):
            st.session_state[key_slots] = {
                s["id"]: {"rol": None, "jugador": None, "rating": None, "pos_det": None, "equipo": None}
                for s in SLOTS
            }
            st.session_state["alin_editing"] = None
            st.rerun()
    with col_i:
        ocupadas = sum(1 for v in alin.values() if v["jugador"])
        st.caption(f"{ocupadas}/{n_slots} posiciones cubiertas  ·  Formación: 1-{n_def}-{n_med}-{n_del}")

    editing_id = st.session_state.get("alin_editing")

    # ── Botones de selección de posición (ARRIBA del campo) ───────────
    st.markdown("**Selecciona una posición:**")
    n_cols = min(n_slots, 6)
    btn_cols = st.columns(n_cols)
    for i, s in enumerate(SLOTS):
        sid   = s["id"]
        datos = alin[sid]
        lbl   = datos["jugador"].split(" (")[0][:12] if datos["jugador"] else s["label"]
        pref  = "🟡 " if editing_id == sid else ("✓ " if datos["jugador"] else "")
        with btn_cols[i % n_cols]:
            if st.button(f"{pref}{lbl}", key=f"alin_btn_{sid}", use_container_width=True):
                st.session_state["alin_editing"] = None if editing_id == sid else sid
                st.rerun()

    # ── Panel de selección (cuando hay posición activa) ───────────────
    if editing_id is not None and editing_id in alin:
        s_info  = next(s for s in SLOTS if s["id"] == editing_id)
        datos_s = alin[editing_id]

        with st.container():
            st.divider()
            col_rol, col_top3 = st.columns([1, 2])

            with col_rol:
                st.markdown(f"**{s_info['label']}**")
                if datos_s["jugador"]:
                    st.caption(f"Actual: {datos_s['jugador'].split(' (')[0]}")
                    if st.button("✕ Quitar", key="alin_quitar"):
                        st.session_state[key_slots][editing_id] = {"rol": None, "jugador": None, "rating": None, "pos_det": None, "equipo": None}
                        st.rerun()

                rol_opts    = ["— Elige un rol —"] + list(pesos_roles.keys())
                idx_default = 0
                if datos_s["rol"] and datos_s["rol"] in pesos_roles:
                    idx_default = rol_opts.index(datos_s["rol"])
                rol_sel = st.selectbox("Rol", rol_opts, index=idx_default, key=f"alin_rol_{editing_id}")

            with col_top3:
                if rol_sel != "— Elige un rol —" and rol_sel in role_scores:
                    df_rol = role_scores[rol_sel].copy()
                    usados = [v["jugador"] for k,v in alin.items() if v["jugador"] and k != editing_id]
                    df_rol = df_rol[~df_rol["Jugador_ID"].isin(usados)]
                    top3   = df_rol.head(3)

                    if top3.empty:
                        st.warning("No hay jugadores disponibles.")
                    else:
                        st.markdown("**Top 3:**")
                        cols_top = st.columns(3)
                        for ci, (_, row) in enumerate(top3.iterrows()):
                            nombre  = row["Jugador"]
                            equipo  = row["Equipo"]
                            rating  = round(float(row["Rating"]), 1)
                            pos_det = row.get("Posicion_detallada") or row.get("Posicion") or ""
                            if pd.isna(pos_det): pos_det = ""
                            es_act  = (datos_s["jugador"] == row["Jugador_ID"])
                            with cols_top[ci]:
                                st.markdown(f"**{nombre}**")
                                st.caption(f"{equipo}  ·  {pos_det}")
                                st.metric("Rating", f"{rating}")
                                if st.button(
                                    "✓" if es_act else "Elegir",
                                    key=f"alin_pick_{editing_id}_{row['Jugador_ID']}",
                                    type="primary" if es_act else "secondary",
                                    use_container_width=True
                                ):
                                    st.session_state[key_slots][editing_id] = {
                                        "rol":     rol_sel,
                                        "jugador": row["Jugador_ID"],
                                        "rating":  rating,
                                        "pos_det": str(pos_det),
                                        "equipo":  equipo,
                                    }
                                    st.session_state["alin_editing"] = None
                                    st.rerun()
                elif rol_sel != "— Elige un rol —":
                    st.warning(f"Sin datos para {rol_sel}.")
        st.divider()

    # ── Campo HORIZONTAL ─────────────────────────────────────────────
    from matplotlib.patches import Arc, Rectangle, Circle as MCircle

    # Campo horizontal: x=largo (0-100), y=ancho (0-60)
    fig_a, ax_a = plt.subplots(figsize=(13, 8))
    fig_a.patch.set_facecolor("#1a5c1a")
    ax_a.set_facecolor("#1a5c1a")
    lc, lw2 = "white", 1.5

    # Líneas del campo (horizontal: largo en x, ancho en y)
    ax_a.plot([0,0],   [0,60],  color=lc, lw=lw2)
    ax_a.plot([0,100], [60,60], color=lc, lw=lw2)
    ax_a.plot([100,100],[60,0], color=lc, lw=lw2)
    ax_a.plot([100,0], [0,0],   color=lc, lw=lw2)
    ax_a.plot([50,50], [0,60],  color=lc, lw=lw2)
    ax_a.add_patch(MCircle((50,30), 9, fill=False, color=lc, lw=lw2))
    ax_a.plot(50, 30, "o", color=lc, ms=3)
    # Área grande izquierda
    ax_a.add_patch(Rectangle((0, 13.5), 16, 33, fill=False, ec=lc, lw=lw2))
    # Área grande derecha
    ax_a.add_patch(Rectangle((84, 13.5), 16, 33, fill=False, ec=lc, lw=lw2))
    # Área pequeña izquierda
    ax_a.add_patch(Rectangle((0, 22), 5.5, 16, fill=False, ec=lc, lw=lw2))
    # Área pequeña derecha
    ax_a.add_patch(Rectangle((94.5, 22), 5.5, 16, fill=False, ec=lc, lw=lw2))
    # Puntos de penalti
    ax_a.plot(11, 30, "o", color=lc, ms=3)
    ax_a.plot(89, 30, "o", color=lc, ms=3)
    # Arcos de penalti
    ax_a.add_patch(Arc((11, 30), 18, 18, theta1=307, theta2=53, color=lc, lw=lw2))
    ax_a.add_patch(Arc((89, 30), 18, 18, theta1=127, theta2=233, color=lc, lw=lw2))
    # Esquinas
    for cx, cy, t1, t2 in [(0,0,0,90),(100,0,90,180),(0,60,270,360),(100,60,180,270)]:
        ax_a.add_patch(Arc((cx,cy), 3, 3, theta1=t1, theta2=t2, color=lc, lw=lw2))

    # Dibujar jugadores — convertir coordenadas (x=0-100 vertical, y=0-100 campo)
    # al sistema horizontal (campo_x = slot_y, campo_y = (100-slot_x)*0.6)
    for s in SLOTS:
        sid  = s["id"]
        # Convertir: slot_y (0-100, portero=8 arriba) → campo_x (0-100, portero=8 izq)
        # slot_x (0-100, centro=50) → campo_y (0-60, centro=30)
        cx = s["y"]           # y del slot → x del campo horizontal
        cy = s["x"] * 0.60   # x del slot → y del campo horizontal (escalado a 60)
        datos = alin[sid]
        es_ed = (editing_id == sid)

        if datos["jugador"]:
            fc = "#f39c12" if es_ed else "#2c3e50"
            ec = "#f1c40f" if es_ed else "white"
            ax_a.add_patch(plt.Rectangle(
                (cx - 6, cy - 4), 12, 8,
                facecolor=fc, edgecolor=ec, lw=1.2,
                transform=ax_a.transData, clip_on=False, zorder=3,
                joinstyle="round"
            ))
            nombre = datos["jugador"].split(" (")[0]
            if len(nombre) > 13: nombre = nombre[:12] + "."
            ax_a.text(cx, cy+1.5, nombre, ha="center", va="center",
                      fontsize=6.5, color="white", fontweight="bold", zorder=4)
            ax_a.text(cx, cy-0.5, datos.get("pos_det") or "", ha="center", va="center",
                      fontsize=5.5, color="#cccccc", zorder=4)
            ax_a.text(cx, cy-2.5, f"★{datos['rating']}" if datos["rating"] else "",
                      ha="center", va="center", fontsize=6, color="#f1c40f", zorder=4)
        else:
            fc    = "#f39c12" if es_ed else "#ffffff"
            alpha = 1.0 if es_ed else 0.15
            ax_a.add_patch(plt.Circle(
                (cx, cy), 4,
                facecolor=fc, edgecolor="white", lw=1,
                alpha=alpha, transform=ax_a.transData, zorder=3
            ))
            lbl_corto = s["label"].replace("Mediocampista","MC").replace("Delantero","Del").replace("Defensa","Def")
            ax_a.text(cx, cy+0.4, lbl_corto, ha="center", va="center",
                      fontsize=6, color="white", zorder=4)

    ax_a.set_xlim(0, 100)
    ax_a.set_ylim(0, 60)
    ax_a.axis("off")
    plt.tight_layout(pad=0.3)
    st.pyplot(fig_a)
    plt.close(fig_a)

    # ── Resumen ───────────────────────────────────────────────────────
    st.divider()
    ocupados = [(s, alin[s["id"]]) for s in SLOTS if alin[s["id"]]["jugador"]]
    if ocupados:
        st.markdown(f"**Alineación 1-{n_def}-{n_med}-{n_del}:**")
        cols_l = st.columns(4)
        for i, (s, datos) in enumerate(ocupados):
            with cols_l[i % 4]:
                st.markdown(f"**{s['label']}** — {datos['jugador'].split(' (')[0]}")
                st.caption(f"{(datos['rol'] or '').replace('_',' ')}  ·  ★ {datos['rating']}")

with tab4:
    st.subheader("Percentiles por Jugador")
    if role_scores:
        _df_pct_pool = filtros_tab(players, "pct", mostrar_pos=True, mostrar_pie=True, mostrar_edad=True, mostrar_equipo=False)
        sel_p = st.selectbox("Jugador", sorted(_df_pct_pool["Jugador_ID"].unique().tolist()), index=None, placeholder="Elige un jugador...", key="rfef_pct_player")
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
            st.markdown("**Filtrar pool de jugadores:**")
            _df_comp_pool = role_scores[sel_r].copy()
            _df_comp_pool = filtros_tab(_df_comp_pool, "comp", mostrar_pos=False, mostrar_pie=True, mostrar_edad=True, mostrar_equipo=True)
            j_sel = st.multiselect("Jugadores", _df_comp_pool["Jugador_ID"].tolist(), max_selections=6, placeholder="Elige jugadores...")
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

                    # ── Tabla de valores reales ──────────────────────
                    st.markdown("### Valores reales por métrica")
                    # Construir tabla con métricas como filas y jugadores como columnas
                    tabla_rows = []
                    for m in metrics:
                        fila = {"Métrica": m.replace("/90","").replace("_"," ")}
                        for jid in j_sel:
                            row_t = df_comp[df_comp["Jugador_ID"] == jid]
                            if row_t.empty:
                                fila[row_t.iloc[0]["Jugador"] if not row_t.empty else jid] = "-"
                                continue
                            nombre_t = row_t.iloc[0]["Jugador"]
                            val_real = row_t.iloc[0].get(m, None)
                            pct_val  = row_t.iloc[0].get(f"{m}_pct", None)
                            if val_real is not None and not pd.isna(val_real):
                                val_str = f"{float(str(val_real).replace(',','.')):.2f}".rstrip('0').rstrip('.')
                                fila[nombre_t] = val_str
                            else:
                                fila[nombre_t] = "-"
                        tabla_rows.append(fila)
                    df_tabla = pd.DataFrame(tabla_rows).set_index("Métrica")
                    st.dataframe(df_tabla, use_container_width=True)
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
        col_sim1, col_sim2 = st.columns(2)
        with col_sim1:
            sel_p = st.selectbox("Jugador", sorted(players["Jugador_ID"].unique().tolist()), index=None, placeholder="Elige un jugador...", key="rfef_sim_player")
        with col_sim2:
            sel_r = st.selectbox("Rol", list(role_scores.keys()), index=None, placeholder="Elige un rol...", key="rfef_sim_role")
        # Filtros de edad y pie — se aplican SOLO a los candidatos, nunca al referente
        st.markdown("**Filtrar resultados:**")
        players_sim_filtrado = filtros_tab(players, "sim", mostrar_pos=False, mostrar_pie=True, mostrar_edad=True, mostrar_equipo=False)
        # Asegurar que el jugador referente siempre está en el pool aunque no pase el filtro
        if sel_p:
            ref_row_sim = players[players["Jugador_ID"] == sel_p]
            players_sim_pool = pd.concat([players_sim_filtrado, ref_row_sim]).drop_duplicates(subset=["Jugador_ID"])
        else:
            players_sim_pool = players_sim_filtrado
        if sel_p and sel_r:
            sim_df = find_similar_players(players_sim_pool, sel_r, sel_p, min_minutes, top_n=6)
            # Filtrar resultados por el pool filtrado (edad/pie), excluyendo el referente
            if not sim_df.empty and "Jugador" in sim_df.columns:
                ids_filtrados = set(players_sim_filtrado["Jugador_ID"].tolist())
                sim_df = sim_df[sim_df.apply(
                    lambda r: (r["Jugador"] + " (" + r["Equipo"] + ")") in ids_filtrados, axis=1
                )]
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
            posicion  = (
                str(row_j.get("Posicion_detallada", "") or "").strip()
                or row_j.get("Posicion", row_j.get("Pos_primary", "?"))
            )
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
                    "Portero_Clasico":         "portero clasico, seguridad bajo palos y buen juego aereo",
                    "Portero_Moderno":    "portero con buen juego de pies y participacion en la construccion",
                    "Lateral_Defensivo":   "lateral con perfil defensivo y solidez en el uno contra uno",
                    "Lateral_Ofensivo":    "lateral ofensivo con proyeccion hacia el ataque",
                    "Central_Stopper":     "central fisico y dominador del juego aereo",
                    "Central_Clasico":     "central equilibrado entre defensa y salida de balon",
                    "Central_Salida":      "central con capacidad para iniciar el juego desde atras",
                    "Pivote_Defensivo":    "pivote con perfil defensivo, recuperador de balones",
                    "Interior_Mediapunta":            "interior con llegada al area y buen juego entre lineas",
                    "Box_to_Box":          "mediocentro total, capaz de cubrir todo el campo",
                    "Interior_Mediapunta":          "mediapunta creativo, enlace entre lineas",
                    "Extremo_Asociativo":  "extremo asociativo, con participacion en combinaciones y asistencias",
                    "Extremo_Puro":        "extremo desequilibrante, con velocidad y regate",
                    "Delantero_Goleador":  "delantero goleador, referencia ofensiva del equipo",
                    "Delantero_Movil":         "delantero movil, con participacion en el juego combinativo",
                    "Segundo_Delantero":       "delantero fisico, gana duelos aereos y retiene el balon",
                    "Carrilero":               "carrilero que sube y baja con impacto ofensivo y defensivo",
                    "Mediocentro_Org":         "mediocentro organizador, domina el juego posicional",
                    "Interior_Banda":          "interior de banda que corta hacia dentro y busca el remate",
                    "Portero_Moderno":         "portero moderno, con mucho juego de pies y salidas al area",
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
