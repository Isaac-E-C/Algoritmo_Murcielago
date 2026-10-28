"""Evacuación independiente de los bloques G y H mediante Bat Algorithm.

Requiere: numpy, opencv-python, matplotlib y pillow.
Ejecutar: python simulacion_evacuacion_bat.py --imagen campus_espe.png
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image, ImageDraw, ImageFont

AREA_REAL_M2, PERIMETRO_REAL_M = 1649.39, 211.84
VELOCIDAD_M_S, ALTURA_ENTREPISO_M = 1.2, 3.0
NIVELES = {"S": -1, "PB": 0, "P1": 1, "P2": 2}

# Coordenadas normalizadas sobre campus_espe.png (1476 x 880), escalables.
PUNTOS = {
    "Bloque G": {"origen": (0.476, 0.529), "gradas": (0.505, 0.449), "salida": (0.535, 0.230), "encuentro": (0.641, 0.105), "color": "#00d9ff"},
    "Bloque H": {"origen": (0.365, 0.515), "gradas": (0.414, 0.645), "salida": (0.518, 0.690), "encuentro": (0.575, 0.857), "color": "#ffb000"},
}
POLIGONO_ESCALA = np.array([
    (.458, .278), (.440, .330), (.410, .341), (.382, .389), (.359, .460),
    (.358, .579), (.405, .635), (.440, .599), (.450, .565), (.474, .550),
    (.469, .510), (.500, .477), (.522, .512), (.548, .458), (.525, .413), (.497, .423),
], dtype=float)


def px(puntos: Sequence[Sequence[float]], ancho: int, alto: int) -> np.ndarray:
    return np.asarray(puntos, dtype=float) * np.array((ancho - 1, alto - 1))


@dataclass
class Escala:
    pixeles_metro: float
    por_area: float
    por_perimetro: float

    @classmethod
    def crear(cls, poligono: np.ndarray) -> "Escala":
        area_px = abs(cv2.contourArea(poligono.astype(np.float32)))
        per_px = cv2.arcLength(poligono.astype(np.float32), True)
        a, p = math.sqrt(area_px / AREA_REAL_M2), per_px / PERIMETRO_REAL_M
        return cls((a + p) / 2, a, p)


class Campus:
    def __init__(self, imagen: str):
        self.bgr = cv2.imread(imagen)
        if self.bgr is None:
            raise FileNotFoundError(f"No existe o no se puede abrir: {imagen}")
        self.alto, self.ancho = self.bgr.shape[:2]
        self.rgb = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2RGB)
        self.escala = Escala.crear(px(POLIGONO_ESCALA, self.ancho, self.alto))

    def metros(self, distancia_px: float) -> float:
        return distancia_px / self.escala.pixeles_metro


def ruta_ortogonal(a: np.ndarray, b: np.ndarray, esquina: np.ndarray) -> np.ndarray:
    """Ruta de Manhattan: todos los tramos son verticales u horizontales."""
    return np.array([a, [esquina[0], a[1]], esquina, [b[0], esquina[1]], b], dtype=float)


def longitud_ortogonal(ruta: np.ndarray) -> float:
    return float(np.abs(np.diff(ruta, axis=0)).sum())


@dataclass
class Resultado:
    ruta: np.ndarray
    distancia_m: float
    fitness: float
    historia: List[dict]


class BatAlgorithmEvacuation:
    """Bat Algorithm de Yang aplicado a una salida individual de cada bloque."""
    def __init__(self, campus: Campus, origen: np.ndarray, gradas: np.ndarray,
                 salida: np.ndarray, encuentro: np.ndarray, penalizacion_nivel: float, semilla: int):
        self.campus, self.origen, self.gradas = campus, origen, gradas
        self.salida, self.encuentro = salida, encuentro
        self.penalizacion_nivel = penalizacion_nivel
        self.rng = np.random.default_rng(semilla)
        self.n_bats, self.max_iter = 50, 200
        self.f_min, self.f_max, self.alpha, self.gamma, self.a0, self.r0 = 0., 2., .90, .90, 1., .5
        # Una esquina optimizable por cada tramo: inicio→gradas→salida→encuentro.
        self.limite = np.tile([campus.ancho - 1, campus.alto - 1], 3)

    def ruta(self, x: np.ndarray) -> np.ndarray:
        # Cada bloque usa sus propias gradas, salida y punto de encuentro.
        anclas = (self.origen, self.gradas, self.salida, self.encuentro)
        ruta = [anclas[0]]
        for i in range(3):
            ruta.extend(ruta_ortogonal(anclas[i], anclas[i + 1], x[2*i:2*i+2])[1:])
        return np.asarray(ruta)

    def evaluar(self, x: np.ndarray) -> Tuple[float, float]:
        distancia = self.campus.metros(longitud_ortogonal(self.ruta(x)))
        return distancia + self.penalizacion_nivel, distancia

    def ejecutar(self) -> Resultado:
        x = self.rng.uniform(0, self.limite, size=(self.n_bats, 6))
        v = np.zeros_like(x)
        A, r = np.full(self.n_bats, self.a0), np.full(self.n_bats, self.r0)
        f = np.array([self.evaluar(q)[0] for q in x])
        idx = int(np.argmin(f)); gbest, mejor = x[idx].copy(), float(f[idx])
        historia: List[dict] = []
        for it in range(1, self.max_iter + 1):
            for i in range(self.n_bats):
                beta = self.rng.random()
                frecuencia = self.f_min + (self.f_max - self.f_min) * beta
                v[i] = v[i] + (x[i] - gbest) * frecuencia
                candidato = np.clip(x[i] + v[i], 0, self.limite)
                if self.rng.random() > r[i]:
                    candidato = np.clip(gbest + .01 * self.rng.normal(size=6) * A.mean() * self.limite, 0, self.limite)
                nuevo, _ = self.evaluar(candidato)
                if nuevo <= f[i] and self.rng.random() < A[i]:
                    x[i], f[i] = candidato, nuevo
                    A[i] *= self.alpha
                    r[i] = self.r0 * (1 - math.exp(-self.gamma * it))
                if f[i] < mejor:
                    gbest, mejor = x[i].copy(), float(f[i])
            _, dist = self.evaluar(gbest)
            historia.append({"bats": x.copy(), "ruta": self.ruta(gbest), "fitness": mejor,
                             "distancia": dist, "A": A.mean(), "r": r.mean()})
        return Resultado(self.ruta(gbest), self.evaluar(gbest)[1], mejor, historia)


def dibujar_mapa_mejorado(origen: Path) -> Path:
    """Genera una copia clara y no destructiva del mapa con los seis puntos."""
    salida = origen.with_name("campus_espe_mejorado.png")
    imagen = Image.open(origen).convert("RGB")
    d = ImageDraw.Draw(imagen); w, h = imagen.size
    try:
        fuente = ImageFont.truetype("arialbd.ttf", 23)
    except OSError:
        fuente = ImageFont.load_default()
    for bloque, info in PUNTOS.items():
        for tipo, etiqueta, radio in (("origen", f"INICIO {bloque}", 15), ("gradas", f"GRADAS {bloque}", 13), ("salida", f"SALIDA {bloque}", 15), ("encuentro", f"PUNTO DE ENCUENTRO {bloque}", 17)):
            x, y = px([info[tipo]], w, h)[0]; x, y = int(x), int(y)
            d.ellipse((x-radio, y-radio, x+radio, y+radio), fill=info["color"], outline="black", width=3)
            if tipo == "gradas":
                d.rectangle((x-7, y-7, x+7, y+7), fill="white", outline="black", width=2)
            d.text((x + 18, y - 12), etiqueta, fill="white", stroke_width=3, stroke_fill="black", font=fuente)
    imagen.save(salida)
    return salida


def animar(campus: Campus, resultados: Dict[str, Resultado], nivel: str, mapa_anotado: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 8), dpi=110); fig.subplots_adjust(right=.75)
    ax.imshow(campus.rgb); ax.set_axis_off()
    ax.set_title("Evacuación independiente: Bloque G y Bloque H — rutas ortogonales")
    plots, rutas = {}, {}
    for bloque, info in PUNTOS.items():
        origen, gradas, salida, encuentro = px([info["origen"], info["gradas"], info["salida"], info["encuentro"]], campus.ancho, campus.alto)
        ax.scatter(*origen, s=105, color=info["color"], edgecolor="black", zorder=5)
        ax.scatter(*gradas, s=110, marker="s", color="white", edgecolor="black", zorder=5)
        ax.scatter(*salida, s=200, marker="*", color="#39ff14", edgecolor="black", zorder=5)
        ax.scatter(*encuentro, s=170, marker="X", color="#39ff14", edgecolor="black", zorder=5)
        plots[bloque] = ax.scatter([], [], s=16, color=info["color"], alpha=.55)
        line, = ax.plot([], [], color=info["color"], lw=3, zorder=4)
        line.set_path_effects([patheffects.Stroke(linewidth=6, foreground="black", alpha=.55), patheffects.Normal()])
        rutas[bloque] = line
    hud = fig.text(.77, .79, "", va="top", color="white", fontsize=10,
                   bbox=dict(boxstyle="round,pad=.7", fc="#101820", ec="#39ff14", alpha=.94))
    transicion = abs(NIVELES[nivel]) * ALTURA_ENTREPISO_M

    def actualizar(k: int):
        artistas, texto = [], [f"NIVEL: {nivel}", "Rutas independientes", f"Iteración: {k+1}/200", ""]
        for bloque, resultado in resultados.items():
            e = resultado.historia[k]
            plots[bloque].set_offsets(e["bats"].reshape(-1, 2))
            ruta = e["ruta"]; rutas[bloque].set_data(ruta[:, 0], ruta[:, 1])
            real = e["distancia"] + transicion
            texto += [bloque, f"  Fitness: {e['fitness']:.2f} m", f"  Distancia: {real:.2f} m", f"  Tiempo: {real/VELOCIDAD_M_S:.1f} s", f"  A={e['A']:.3f}  r={e['r']:.3f}", ""]
            artistas.extend((plots[bloque], rutas[bloque]))
        hud.set_text("\n".join(texto)); return (*artistas, hud)

    anim = FuncAnimation(fig, actualizar, frames=200, interval=45, blit=False, repeat=False)
    anim.save("evacuacion_bat_algorithm.gif", writer=PillowWriter(fps=18), dpi=100)
    actualizar(199); fig.savefig("ruta_optima_evacuacion.png", dpi=300, bbox_inches="tight")
    print(f"Mapa anotado: {mapa_anotado}")
    print("Resultados: evacuacion_bat_algorithm.gif y ruta_optima_evacuacion.png")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Bat Algorithm para evacuación G-H")
    parser.add_argument("--imagen", default="campus_espe.png")
    parser.add_argument("--nivel", choices=NIVELES, default="P2")
    args = parser.parse_args(); imagen = Path(args.imagen)
    mapa_anotado = dibujar_mapa_mejorado(imagen)
    campus = Campus(str(mapa_anotado)); penalizacion = abs(NIVELES[args.nivel]) * ALTURA_ENTREPISO_M
    resultados: Dict[str, Resultado] = {}
    for semilla, (bloque, info) in enumerate(PUNTOS.items(), 17):
        origen, gradas, salida, encuentro = px([info["origen"], info["gradas"], info["salida"], info["encuentro"]], campus.ancho, campus.alto)
        resultados[bloque] = BatAlgorithmEvacuation(campus, origen, gradas, salida, encuentro, penalizacion, semilla).ejecutar()
    print(f"Escala combinada: {campus.escala.pixeles_metro:.3f} px/m")
    for bloque, r in resultados.items():
        total = r.distancia_m + penalizacion
        print(f"{bloque}: {total:.2f} m | {total / VELOCIDAD_M_S:.1f} s")
    animar(campus, resultados, args.nivel, str(mapa_anotado))


if __name__ == "__main__":
    main()
