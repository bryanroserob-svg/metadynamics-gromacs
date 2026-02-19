#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_convergence.py — Análisis de convergencia y reconstrucción FES
para Well-Tempered Metadynamics.

Funcionalidades:
  1. Reconstrucción del FES via plumed sum_hills
  2. Convergencia difusiva: ΔG entre estados vs. tiempo
  3. Block averaging para errores estadísticos
  4. Reweighting Tiwary-Parrinello (opcional)

Uso:
  python3 analyze_convergence.py --hills HILLS --colvar COLVAR \\
      --temp 300 --biasfactor 15 --output-dir analysis/

Llamado por run_metad_pipeline.sh durante la etapa de post-procesamiento.
"""

import argparse
import os
import sys
import subprocess
import numpy as np
from pathlib import Path


def load_plumed_file(filepath, comment='#'):
    """Carga archivo PLUMED (HILLS o COLVAR) ignorando comentarios.
    
    Returns:
        numpy.ndarray: Datos del archivo.
    """
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(comment) or line.startswith('@'):
                continue
            try:
                values = [float(x) for x in line.split()]
                data.append(values)
            except ValueError:
                continue
    
    if not data:
        print(f"ERROR: No se encontraron datos en {filepath}", file=sys.stderr)
        sys.exit(1)
    
    return np.array(data)


def run_sum_hills(hills_file, output_prefix, args, time_blocks=None):
    """Ejecuta plumed sum_hills para reconstruir el FES.
    
    Args:
        hills_file: Ruta al archivo HILLS.
        output_prefix: Prefijo para archivos de salida.
        args: Argumentos CLI.
        time_blocks: Lista de tiempos máximos para análisis de convergencia.
    
    Returns:
        Lista de archivos FES generados.
    """
    fes_files = []
    
    if time_blocks is None:
        # FES completo
        cmd = [
            "plumed", "sum_hills",
            "--hills", hills_file,
            "--outfile", f"{output_prefix}_fes.dat",
            "--mintozero",
        ]
        
        if args.kt:
            cmd.extend(["--kt", str(args.kt)])
        
        print(f"  Ejecutando: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"WARNING: sum_hills falló: {result.stderr}", file=sys.stderr)
            return fes_files
        
        fes_files.append(f"{output_prefix}_fes.dat")
        print(f"  ✓ FES generado: {output_prefix}_fes.dat")
    else:
        # FES por bloques temporales (convergencia)
        for t_max in time_blocks:
            out_file = f"{output_prefix}_fes_t{t_max:.0f}.dat"
            cmd = [
                "plumed", "sum_hills",
                "--hills", hills_file,
                "--outfile", out_file,
                "--mintozero",
                "--stride", str(int(t_max)),
            ]
            
            if args.kt:
                cmd.extend(["--kt", str(args.kt)])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                fes_files.append(out_file)
        
        print(f"  ✓ {len(fes_files)} FES parciales generados")
    
    return fes_files


def analyze_hills_convergence(hills_file):
    """Analiza la convergencia de las gaussianas depositadas.
    
    En WTMetaD, la altura de las gaussianas debe decaer exponencialmente:
      h(t) = h₀ × exp(-V(s,t) / (kB × ΔT))
    
    Un decaimiento estable indica que el sistema está convergiendo difusivamente.
    
    Returns:
        dict: Datos de convergencia (tiempo, alturas, estadísticas).
    """
    hills = load_plumed_file(hills_file)
    
    # Columnas HILLS: time, cv1, [cv2, ...], sigma1, [sigma2, ...], height, biasfactor
    time = hills[:, 0]
    
    # Buscar la columna de height (asumimos que es la penúltima o antepenúltima)
    # En PLUMED: time, cv1, sigma1, height, biasfactor (para 1 CV)
    # O: time, cv1, cv2, sigma1, sigma2, height, biasfactor (para 2 CVs)
    n_cols = hills.shape[1]
    
    # La columna de height suele ser la antepenúltima
    # Formato: time | cv1 [cv2] | sigma1 [sigma2] | height | biasfactor
    height_col = n_cols - 2  # Penúltima columna
    heights = hills[:, height_col]
    
    # Si las alturas son todas iguales, probablemente no es WT
    if np.std(heights) < 1e-6:
        print("  WARNING: Alturas constantes — ¿MetaD estándar (no WTMetaD)?")
    
    # Estadísticas de convergencia
    n_gaussians = len(heights)
    h_initial = heights[0] if n_gaussians > 0 else 0
    h_final = heights[-1] if n_gaussians > 0 else 0
    h_ratio = h_final / h_initial if h_initial > 0 else 0
    
    # Calcular decay rate (ajuste exponencial)
    result = {
        'time': time,
        'heights': heights,
        'n_gaussians': n_gaussians,
        'h_initial': h_initial,
        'h_final': h_final,
        'h_ratio': h_ratio,
        'total_time_ps': time[-1] if n_gaussians > 0 else 0,
    }
    
    return result


def compute_deltaG_vs_time(colvar_file, hills_file, args, n_blocks=10):
    """Calcula ΔG entre dos estados metaestables en función del tiempo.
    
    Divide la simulación en bloques y calcula ΔG acumulativo.
    Convergencia se indica cuando ΔG se estabiliza.
    
    Args:
        colvar_file: Ruta al archivo COLVAR.
        hills_file: Ruta al archivo HILLS.
        args: Argumentos CLI.
        n_blocks: Número de bloques temporales.
    
    Returns:
        dict: Tiempos y valores de ΔG.
    """
    colvar = load_plumed_file(colvar_file)
    
    time = colvar[:, 0]
    total_time = time[-1]
    
    # Definir bloques temporales
    block_times = np.linspace(total_time / n_blocks, total_time, n_blocks)
    
    print(f"  Tiempo total: {total_time:.1f} ps ({total_time/1000:.1f} ns)")
    print(f"  Calculando ΔG en {n_blocks} bloques temporales...")
    
    # Para cada bloque, reconstruir FES parcial
    deltaG_values = []
    
    for t_block in block_times:
        # Filtrar HILLS hasta t_block
        hills = load_plumed_file(hills_file)
        mask = hills[:, 0] <= t_block
        if not np.any(mask):
            continue
        
        hills_block = hills[mask]
        
        # Escribir HILLS filtrado temporalmente
        temp_hills = f"/tmp/hills_block_{t_block:.0f}.dat"
        np.savetxt(temp_hills, hills_block, fmt='%.6f')
        
        # sum_hills parcial
        temp_fes = f"/tmp/fes_block_{t_block:.0f}.dat"
        cmd = [
            "plumed", "sum_hills",
            "--hills", temp_hills,
            "--outfile", temp_fes,
            "--mintozero",
        ]
        if args.kt:
            cmd.extend(["--kt", str(args.kt)])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and os.path.exists(temp_fes):
            try:
                fes = load_plumed_file(temp_fes)
                # ΔG = max(FES) - min(FES) o diferencia entre mínimos
                fes_values = fes[:, -1]  # Última columna = energía libre
                delta_g = np.max(fes_values) - np.min(fes_values)
                deltaG_values.append((t_block, delta_g))
            except Exception:
                pass
        
        # Limpiar temporales
        for f in [temp_hills, temp_fes]:
            if os.path.exists(f):
                os.remove(f)
    
    return {
        'times': [dg[0] for dg in deltaG_values],
        'deltaG': [dg[1] for dg in deltaG_values],
    }


def block_averaging(data, n_blocks=5):
    """Calcula estadísticas por block averaging.
    
    Divide los datos en bloques y calcula media y error estándar
    entre bloques para estimar errores estadísticos.
    
    Args:
        data: Array 1D de datos.
        n_blocks: Número de bloques.
    
    Returns:
        tuple: (mean, stderr, block_means)
    """
    block_size = len(data) // n_blocks
    if block_size == 0:
        return np.mean(data), np.std(data), [np.mean(data)]
    
    block_means = []
    for i in range(n_blocks):
        start = i * block_size
        end = start + block_size
        block_means.append(np.mean(data[start:end]))
    
    block_means = np.array(block_means)
    mean = np.mean(block_means)
    stderr = np.std(block_means) / np.sqrt(n_blocks)
    
    return mean, stderr, block_means.tolist()


def find_fes_minima(fes_file, energy_window_kj=2.5):
    """Detecta minimos locales en un FES 1D y calcula DeltaG entre ellos.

    Args:
        fes_file: Ruta al archivo FES generado por plumed sum_hills.
        energy_window_kj: Barrera minima (kJ/mol) para no descartar un minimo.

    Returns:
        dict con 'cv', 'energies', 'deltaG_pairs', 'n_minima', o None.
    """
    if not os.path.exists(fes_file):
        return None
    try:
        data = load_plumed_file(fes_file)
    except SystemExit:
        return None
    if data.ndim < 2 or data.shape[1] < 2:
        return None

    cv_vals = data[:, 0]
    fes_vals = data[:, 1] - np.min(data[:, 1])   # normalizar a 0

    window = max(3, len(fes_vals) // 50)
    try:
        from scipy.signal import argrelmin
        local_min_idx = list(argrelmin(fes_vals, order=window)[0])
    except ImportError:
        local_min_idx = [
            i for i in range(window, len(fes_vals) - window)
            if fes_vals[i] == np.min(fes_vals[max(0, i-window):i+window+1])
        ]

    if not local_min_idx:
        return None

    # Filtrar minimos separados por una barrera >= energy_window_kj
    filtered = [local_min_idx[0]]
    for idx in local_min_idx[1:]:
        barrier = np.max(fes_vals[filtered[-1]:idx+1]) - fes_vals[idx]
        if barrier >= energy_window_kj:
            filtered.append(idx)

    min_cv = cv_vals[filtered]
    min_e = fes_vals[filtered]

    pairs = [
        (i, j, float(min_e[j] - min_e[i]))
        for i in range(len(filtered))
        for j in range(i + 1, len(filtered))
    ]

    return {
        'cv': min_cv.tolist(),
        'energies': min_e.tolist(),
        'deltaG_pairs': pairs,
        'n_minima': len(filtered)
    }


def save_convergence_report(output_dir, hills_conv, deltaG_data, args):
    """Guarda reporte de convergencia en formato texto."""
    report_path = os.path.join(output_dir, "convergence_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("REPORTE DE CONVERGENCIA — Well-Tempered Metadynamics\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("PARÁMETROS DE SIMULACIÓN:\n")
        f.write(f"  Temperatura:  {args.temp} K\n")
        f.write(f"  Biasfactor:   {args.biasfactor}\n")
        f.write(f"  kT:           {args.kt:.4f} kJ/mol\n")
        f.write(f"  ΔT:           {args.temp * (args.biasfactor - 1):.1f} K\n")
        f.write(f"  T_eff:        {args.temp * args.biasfactor:.1f} K\n\n")
        
        f.write("ANÁLISIS DE GAUSSIANAS:\n")
        f.write(f"  Total depositadas:  {hills_conv['n_gaussians']}\n")
        f.write(f"  Altura inicial:     {hills_conv['h_initial']:.4f} kJ/mol\n")
        f.write(f"  Altura final:       {hills_conv['h_final']:.4f} kJ/mol\n")
        f.write(f"  Ratio h_f/h_i:      {hills_conv['h_ratio']:.4f}\n")
        f.write(f"  Tiempo total:       {hills_conv['total_time_ps']:.1f} ps "
                f"({hills_conv['total_time_ps']/1000:.2f} ns)\n\n")
        
        # Evaluación de convergencia
        f.write("EVALUACIÓN DE CONVERGENCIA:\n")
        if hills_conv['h_ratio'] < 0.01:
            f.write("  ✓ CONVERGIDO: Gaussianas prácticamente nulas\n")
            f.write("    El FES está bien definido.\n")
        elif hills_conv['h_ratio'] < 0.1:
            f.write("  ~ PARCIALMENTE CONVERGIDO: Gaussianas aún apreciables\n")
            f.write("    Considere extender la simulación.\n")
        else:
            f.write("  ✗ NO CONVERGIDO: Gaussianas aún significativas\n")
            f.write("    Se requiere más tiempo de simulación.\n")
        
        if deltaG_data['deltaG']:
            f.write(f"\n  DeltaG (ultimo bloque): {deltaG_data['deltaG'][-1]:.2f} kJ/mol\n")
            if len(deltaG_data['deltaG']) > 3:
                last_3 = deltaG_data['deltaG'][-3:]
                variation = np.std(last_3)
                f.write(f"  Variacion ultimos 3 bloques: +/-{variation:.2f} kJ/mol\n")
                if variation < 2.0:
                    f.write("  OK DeltaG estable (variacion < 2 kJ/mol)\n")
                else:
                    f.write("  WARN DeltaG inestable (variacion >= 2 kJ/mol)\n")

        # Minimos locales
        minima = deltaG_data.get('minima')
        if minima:
            f.write(f"\nMINIMOS LOCALES ({minima['n_minima']} detectados):\n")
            for k, (cv_v, e_v) in enumerate(zip(minima['cv'], minima['energies'])):
                f.write(f"  Estado {k+1}: CV={cv_v:.4f}  E={e_v:.2f} kJ/mol\n")
            if minima['deltaG_pairs']:
                f.write("  DeltaG entre estados:\n")
                for i, j, dG in minima['deltaG_pairs']:
                    f.write(f"    Estado {i+1} -> Estado {j+1}: {dG:+.2f} kJ/mol\n")

        f.write("\n" + "=" * 60 + "\n")
    
    print(f"  ✓ Reporte: {report_path}")
    return report_path


def save_data_files(output_dir, hills_conv, deltaG_data):
    """Guarda datos procesados en formato CSV/DAT."""
    
    # Alturas de gaussianas vs tiempo
    heights_path = os.path.join(output_dir, "gaussian_heights.dat")
    with open(heights_path, 'w') as f:
        f.write("# time(ps)  height(kJ/mol)\n")
        for t, h in zip(hills_conv['time'], hills_conv['heights']):
            f.write(f"{t:.4f}  {h:.6f}\n")
    print(f"  ✓ Alturas: {heights_path}")
    
    # ΔG vs tiempo
    if deltaG_data['deltaG']:
        dg_path = os.path.join(output_dir, "deltaG_vs_time.dat")
        with open(dg_path, 'w') as f:
            f.write("# time(ps)  deltaG(kJ/mol)\n")
            for t, dg in zip(deltaG_data['times'], deltaG_data['deltaG']):
                f.write(f"{t:.4f}  {dg:.4f}\n")
        print(f"  ✓ ΔG vs tiempo: {dg_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Análisis de convergencia para WT-MetaD',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Análisis básico
  %(prog)s --hills HILLS --colvar COLVAR --temp 300 --biasfactor 15

  # Especificar directorio de salida
  %(prog)s --hills HILLS --colvar COLVAR --temp 300 --biasfactor 15 \\
           --output-dir analysis/ --n-blocks 20

  # Sin reconstrucción FES (solo análisis de gaussianas)
  %(prog)s --hills HILLS --temp 300 --biasfactor 15 --no-fes
        """
    )
    
    parser.add_argument('--hills', required=True, help='Archivo HILLS de PLUMED')
    parser.add_argument('--colvar', default=None, help='Archivo COLVAR de PLUMED')
    parser.add_argument('--temp', type=float, default=300, help='Temperatura (K)')
    parser.add_argument('--biasfactor', type=float, default=15, help='Bias factor γ')
    parser.add_argument('--output-dir', default='analysis', dest='output_dir',
                        help='Directorio de salida')
    parser.add_argument('--n-blocks', type=int, default=10, dest='n_blocks',
                        help='Número de bloques para convergencia')
    parser.add_argument('--no-fes', action='store_true', dest='no_fes',
                        help='No reconstruir FES (solo analizar HILLS)')
    
    args = parser.parse_args()
    
    # Calcular kT
    kB = 0.0083144621  # kJ/(mol·K)
    args.kt = kB * args.temp
    
    # Crear directorio de salida
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("ANÁLISIS DE CONVERGENCIA — WT-MetaD")
    print("=" * 60)
    print(f"  HILLS:      {args.hills}")
    print(f"  COLVAR:     {args.colvar or 'N/A'}")
    print(f"  Temp:       {args.temp} K")
    print(f"  Biasfactor: {args.biasfactor}")
    print(f"  kT:         {args.kt:.4f} kJ/mol")
    print(f"  Output:     {args.output_dir}/")
    print()
    
    # 1. Analizar gaussianas
    print("─── Análisis de Gaussianas ───")
    hills_conv = analyze_hills_convergence(args.hills)
    print(f"  Gaussianas depositadas: {hills_conv['n_gaussians']}")
    print(f"  h_inicial = {hills_conv['h_initial']:.4f} kJ/mol")
    print(f"  h_final   = {hills_conv['h_final']:.4f} kJ/mol")
    print(f"  ratio     = {hills_conv['h_ratio']:.4f}")
    print()
    
    # 2. Reconstruir FES
    deltaG_data = {'times': [], 'deltaG': [], 'minima': None}
    
    if not args.no_fes:
        print("─── Reconstrucción FES ───")
        plumed_available = subprocess.run(
            ["plumed", "--help"], capture_output=True
        ).returncode == 0
        
        if plumed_available:
            fes_files = run_sum_hills(
                args.hills, 
                os.path.join(args.output_dir, "metad"),
                args
            )
            
            # 3. Convergencia ΔG vs tiempo
            if args.colvar:
                print()
                print("─── Convergencia ΔG vs Tiempo ───")
                deltaG_data = compute_deltaG_vs_time(
                    args.colvar, args.hills, args, args.n_blocks
                )
                deltaG_data['minima'] = None
                if deltaG_data['deltaG']:
                    print(f"  ΔG final: {deltaG_data['deltaG'][-1]:.2f} kJ/mol")

            # 4. Mínimos locales del FES
            fes_main = os.path.join(args.output_dir, "metad_fes.dat")
            print()
            print("─── Mínimos Locales del FES ───")
            minima = find_fes_minima(fes_main, energy_window_kj=2.5)
            if minima:
                deltaG_data['minima'] = minima
                print(f"  Detectados {minima['n_minima']} mínimos:")
                for k, (cv_v, e_v) in enumerate(zip(minima['cv'], minima['energies'])):
                    print(f"    Estado {k+1}: CV={cv_v:.4f}  E={e_v:.2f} kJ/mol")
                for i, j, dG in minima['deltaG_pairs']:
                    print(f"    ΔG estado {i+1}→{j+1}: {dG:+.2f} kJ/mol")
            else:
                print("  FES unimodal (sin mínimos secundarios significativos)")
        else:
            print("  WARNING: 'plumed' no encontrado en PATH")
            print("  Saltando reconstrucción FES.")
            print("  Para instalar: conda install -c conda-forge plumed")

    
    # 4. Guardar resultados
    print()
    print("─── Guardando resultados ───")
    save_data_files(args.output_dir, hills_conv, deltaG_data)
    save_convergence_report(args.output_dir, hills_conv, deltaG_data, args)
    
    # 5. Block averaging de las alturas
    if hills_conv['n_gaussians'] > 10:
        mean_h, stderr_h, _ = block_averaging(hills_conv['heights'])
        print(f"\n  Block averaging alturas: {mean_h:.4f} ± {stderr_h:.4f} kJ/mol")
    
    print()
    print("✓ Análisis completado")
    print(f"  Resultados en: {args.output_dir}/")


if __name__ == '__main__':
    main()
