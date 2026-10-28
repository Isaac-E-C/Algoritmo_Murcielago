# Simulación de evacuación de los bloques G y H

## Propósito

`simulacion_evacuacion_bat.py` es una simulación académica de rutas de evacuación con el **Bat Algorithm** de Xin-She Yang. Las evacuaciones son independientes; G y H no se consideran un solo origen.

| Origen | Escalera obligatoria | Salida asignada | Punto final |
|---|---|---|---|
| Bloque G | Gradas del Bloque G | Salida norte del Bloque G | Punto de encuentro del Bloque G |
| Bloque H | Gradas del Bloque H | Salida sureste del Bloque H | Punto de encuentro del Bloque H |

Los seis puntos se encuentran en el diccionario `PUNTOS`. Fueron ubicados usando el mapa proporcionado y se guardan como coordenadas normalizadas, por lo que se ajustan a la resolución de la imagen. Si se cambia el encuadre de la ortofoto, deben validarse nuevamente.

## Restricción geométrica

Las rutas se dibujan con geometría Manhattan: cada tramo es horizontal o vertical. No hay diagonales. La secuencia obligatoria para cada bloque es:

```text
Inicio → gradas de su bloque → salida de su bloque → punto de encuentro asignado
```

Para los niveles S, P1 y P2 se agrega una transición vertical de `3 m × número de pisos`; en PB es cero.

## Algoritmo

Se ejecutan dos poblaciones independientes de 50 murciélagos por 200 iteraciones. Los parámetros son `f_min=0`, `f_max=2`, `alpha=0.90`, `gamma=0.90`, `A0=1` y `r0=0.5`.

Cada murciélago optimiza las esquinas de tres recorridos ortogonales: inicio–gradas, gradas–salida y salida–punto de encuentro. La aptitud es la distancia métrica ortogonal más la transición de nivel. Las actualizaciones de frecuencia, velocidad, posición, caminata aleatoria, sonoridad `A` y tasa de pulsos `r` corresponden al Bat Algorithm.

## Escala y productos

La escala combina las mediciones de Google Earth: área **1 649.39 m²** y perímetro **211.84 m**. La distancia se expresa en metros y el tiempo se estima con una velocidad de **1.2 m/s**.

Al ejecutar se crean:

- `campus_espe_mejorado.png`: copia no destructiva del mapa con inicio, gradas y salidas claramente identificados.
- `evacuacion_bat_algorithm.gif`: animación de ambas poblaciones y rutas.
- `ruta_optima_evacuacion.png`: imagen final de alta resolución.

## Ejecución

```powershell
python -m pip install numpy opencv-python matplotlib pillow
python simulacion_evacuacion_bat.py --imagen campus_espe.png --nivel P2
```

> El resultado no reemplaza una inspección de seguridad. La accesibilidad, las puertas, la capacidad de escaleras, señalética y punto de encuentro deben ser confirmados por responsables de seguridad de la Universidad.
