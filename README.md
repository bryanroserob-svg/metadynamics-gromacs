# Well-Tempered Metadynamics Pipeline (GROMACS + PLUMED)

Pipeline automatizado para simulaciones de **Well-Tempered Metadynamics (WTMetaD)**
usando GROMACS y PLUMED. Diseñado para explorar paisajes de energía libre (FES)
y detectar estados conformacionales metaestables.

---

## Requisitos

| Dependencia   | Versión mínima | Notas                               |
|---------------|----------------|-------------------------------------|
| GROMACS       | 2022+          | Compilado con PLUMED (patch)        |
| PLUMED        | 2.8+           | `gmx mdrun -plumed` debe funcionar  |
| Python        | 3.8+           | —                                   |
| numpy         | 1.20+          | `pip install numpy`                 |
| matplotlib    | 3.5+           | `pip install matplotlib`            |
| scipy         | 1.7+           | Opcional (detección de mínimos FES) |
| bc            | cualquiera     | Usualmente preinstalado en Linux    |

---

## Estructura del Proyecto

```
metadinamycs_gromacs/
├── run_metad_pipeline.sh       # Orquestador principal
├── scripts/
│   ├── generate_plumed.py      # Generador de plumed.dat
│   ├── analyze_convergence.py  # Análisis FES + convergencia
│   └── plot_metad.py           # Visualización (CV, FES, ΔG)
├── mdp/
│   ├── ions.mdp
│   ├── em.mdp
│   ├── nvt.mdp
│   ├── npt.mdp
│   └── md_metad.mdp
├── proteins/                   # PDBs de entrada
├── ligands/                    # Topologías y PDBs del ligando
└── README.md
```

### Directorio de Simulación (generado automáticamente)

```
METAD_RUN_<Protein>_<Ligand>/
├── config.txt                  # Configuración guardada (para --resume)
├── logs/                       # Logs de mdrun, grompp, análisis
├── 01_minimization/
├── 02_nvt/
├── 03_npt/
├── 04_metadynamics/
│   ├── HILLS                   # Gaussianas depositadas por PLUMED
│   ├── COLVAR                  # Evolución de CVs en el tiempo
│   └── plumed.dat              # Configuración PLUMED generada
├── 05_analysis/
│   ├── metad_fes.dat           # Superficie de Energía Libre (FES)
│   ├── gaussian_heights.dat    # Alturas de gaussianas vs tiempo
│   ├── deltaG_vs_time.dat      # Convergencia de ΔG
│   └── convergence_report.txt  # Reporte completo con mínimos del FES
├── 06_qc/                      # Quality Control post-producción
│   ├── energy.xvg              # Temperatura, Presión, Potencial
│   └── rmsd_backbone.xvg       # RMSD del backbone proteico
└── 07_plots/                   # Gráficos generados
```

---

## Uso

### Modo interactivo (primera vez)

```bash
chmod +x run_metad_pipeline.sh
./run_metad_pipeline.sh
```

El menú interactivo solicitará:
1. **Proteína** — archivo PDB en `proteins/`
2. **Ligando** — opcional, topología en `ligands/`
3. **Force field** — local o del sistema (CHARMM36, OPLS, Amber…)
4. **Box, agua, iones** — parámetros de solvación
5. **Tiempo de producción** y **timestep `dt`** (en ps, default 0.002)
6. **CVs** — tipo (DISTANCE, RMSD, TORSION, COORDINATION), átomos, sigma
7. **WTMetaD** — HEIGHT, PACE, BIASFACTOR, temperatura
8. **Walls** — confinamiento opcional del muestreo por CV
9. **Grid** — opción recomendada para optimizar lookup de gaussianas O(N)→O(1)

### Reanudar desde checkpoint

```bash
./run_metad_pipeline.sh --resume METAD_RUN_<Protein>_<Ligand>/
```

La configuración se restaura automáticamente desde `config.txt`.
Si hay `HILLS` previo, PLUMED arranca en modo `RESTART` para no duplicar gaussianas.

### Limpiar archivos temporales

```bash
./run_metad_pipeline.sh --cleanup METAD_RUN_<Protein>_<Ligand>/
```

---

## Variables Colectivas Soportadas

| Tipo          | Descripción                                   | Átomos requeridos        |
|---------------|-----------------------------------------------|--------------------------|
| `DISTANCE`    | Distancia COM entre dos grupos                | 2 grupos                 |
| `RMSD`        | RMSD respecto a estructura de referencia PDB  | 1 grupo + `reference.pdb`|
| `TORSION`     | Ángulo dihedral                               | 4 átomos                 |
| `COORDINATION`| Número de coordinación (función switching)    | 2 grupos                 |

Se pueden combinar hasta 3 CVs simultáneamente.

---

## Parámetros WTMetaD Recomendados

| Parámetro     | Rango           | Notas                                          |
|---------------|-----------------|------------------------------------------------|
| `HEIGHT`      | 0.5–2.0 kJ/mol  | Inicio. Se reduce automáticamente (WTMetaD)    |
| `SIGMA`       | 0.1–0.5 nm      | ~1/10 del rango esperado del CV                |
| `PACE`        | 500–1000 steps  | Deposición cada 1–2 ps                         |
| `BIASFACTOR`  | 10–20           | Mayor → mayor exploración, convergencia lenta  |
| `dt`          | 0.002 ps (2 fs) | Estándar; ajustable en el menú                 |
| Tiempo total  | 100–500 ns      | Depende del sistema y barreras a superar       |

---

## Etapas del Pipeline

```
Entrada PDB
    │
    ▼
Preparación del sistema
  (solvación, iones, topología, índices)
    │
    ▼
01 Minimización de energía (EM)
    │
    ▼
02 Equilibración NVT (100 ps)
    │
    ▼
03 Equilibración NPT (100 ps)
    │
    ▼
04 Producción WTMetaD  ◄─── plumed.dat (generado automáticamente)
   HILLS + COLVAR
    │
    ▼
06 QC automatizado
   gmx energy  →  energy.xvg
   gmx rms     →  rmsd_backbone.xvg
    │
    ▼
05 Análisis de convergencia
   plumed sum_hills  →  FES
   ΔG vs tiempo
   Mínimos locales del FES (detección automática)
   convergence_report.txt
    │
    ▼
07 Visualización
   CV vs tiempo | Gaussianas | FES 2D/1D | ΔG convergencia
```

---

## Análisis de Convergencia

`analyze_convergence.py` implementa:

1. **Decay de gaussianas** — ratio `h_final/h_inicial` < 0.01 indica convergencia.
2. **ΔG vs tiempo** — variación en los últimos 3 bloques < 2 kJ/mol indica estabilidad.
3. **Detección automática de mínimos locales** — identifica estados metaestables en el FES y calcula ΔG entre cada par de estados (barrera mínima configurable, default 2.5 kJ/mol).
4. **Block averaging** — estimación de errores estadísticos en las alturas.

---

## Quality Control Automatizado (etapa 06_qc)

Después de la producción, el pipeline ejecuta automáticamente:

```bash
# Temperatura, Presión, Potencial (verifica estabilidad termodinámica)
gmx energy → 06_qc/energy.xvg

# RMSD backbone proteico (verifica integridad estructural bajo el bias)
gmx rms    → 06_qc/rmsd_backbone.xvg
```

Si el RMSD crece monotónicamente > 5 Å, puede indicar desplegamiento inducido
por el bias. Considere reducir `HEIGHT` o agregar `WALLS`.

---

## Mejoras Futuras

- **OPES** (On-the-fly Probability Enhanced Sampling): método más robusto y eficiente.
- **PT-MetaD**: combinar Parallel Tempering con Metadinámica.
- **Multi-Walker**: exploración paralela del espacio conformacional.
- **Funnel Metadynamics**: para calcular ΔG de unión proteína–ligando.
- **Reweighting Tiwary-Parrinello**: calcular observables no sesgados del FES.
- **MDAnalysis**: análisis de trayectoria avanzado (clustering, contactos, SASA).

---

## Nota de Migración

Si actualizas desde una versión anterior del pipeline:

- **Renombrar carpeta de gráficos**: las versiones anteriores creaban `06_plots/`
  dentro de cada directorio de simulación. Ahora la carpeta correcta es `07_plots/`.
  Si tienes simulaciones previas, renombra:

  ```bash
  # Para cada directorio de simulación existente:
  mv METAD_RUN_*/06_plots METAD_RUN_*/07_plots 2>/dev/null || true
  ```

- **Timestep configurable**: el argumento `--dt` ahora se pasa al generador de PLUMED.
  Si usas `dt ≠ 0.002 ps`, los reportes de PACE en ps serán ahora correctos.
