#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_metad.py — Visualización de resultados de Well-Tempered Metadynamics.

Genera los siguientes gráficos:
  1. Evolución de CVs vs Tiempo
  2. Altura de Gaussianas vs Tiempo (verificación WTMetaD)
  3. Mapa de calor 2D del FES con isolíneas (para 2 CVs)
  4. FES 1D para cada CV individual
  5. Convergencia: ΔG vs Tiempo

Uso:
  python3 plot_metad.py --colvar COLVAR --hills HILLS \\
      --fes fes.dat --output-dir plots/

Llamado por run_metad_pipeline.sh durante la etapa de visualización.
"""

import argparse
import os
import sys
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # Backend no interactivo para servidores
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize
except ImportError:
    print("ERROR: matplotlib no instalado. Instalar: pip install matplotlib", file=sys.stderr)
    sys.exit(1)

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def setup_plot_style():
    """Configura estilo global de los gráficos."""
    plt.rcParams.update({
        'figure.figsize': (10, 7),
        'figure.dpi': 150,
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 1.5,
    })
    if HAS_SEABORN:
        sns.set_palette("deep")


def load_plumed_file(filepath, comment='#'):
    """Carga archivo PLUMED ignorando comentarios y líneas FIELDS."""
    data = []
    headers = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#! FIELDS'):
                # Extraer nombres de columnas
                headers = line.replace('#! FIELDS', '').strip().split()
            if line.startswith('#') or line.startswith('@'):
                continue
            try:
                values = [float(x) for x in line.split()]
                data.append(values)
            except ValueError:
                continue
    return np.array(data) if data else np.array([]), headers


def plot_cv_evolution(colvar_file, output_dir):
    """Gráfico 1: Evolución temporal de las CVs.
    
    Muestra cómo las variables colectivas exploran el espacio de configuración
    durante la simulación. En WTMetaD exitosa, se espera ver transiciones
    repetidas entre estados metaestables.
    """
    data, headers = load_plumed_file(colvar_file)
    if data.size == 0:
        print("  WARNING: COLVAR vacío, saltando plot_cv_evolution")
        return
    
    time = data[:, 0]
    n_cvs = data.shape[1] - 2  # Excluir time y bias (último)
    
    if n_cvs <= 0:
        print("  WARNING: No se encontraron CVs en COLVAR")
        return
    
    # Convertir a nanosegundos si el tiempo es grande
    time_unit = "ps"
    time_plot = time
    if time[-1] > 10000:
        time_plot = time / 1000
        time_unit = "ns"
    
    fig, axes = plt.subplots(n_cvs, 1, figsize=(12, 4 * n_cvs), sharex=True)
    if n_cvs == 1:
        axes = [axes]
    
    colors = plt.cm.Set2(np.linspace(0, 1, max(n_cvs, 3)))
    
    for i in range(n_cvs):
        cv_data = data[:, i + 1]
        cv_label = headers[i + 1] if len(headers) > i + 1 else f"CV{i+1}"
        
        axes[i].plot(time_plot, cv_data, color=colors[i], alpha=0.8, linewidth=0.5)
        axes[i].set_ylabel(cv_label, fontweight='bold')
        axes[i].set_title(f"Evolución de {cv_label}")
        
        # Media móvil para tendencia
        window = max(1, len(cv_data) // 200)
        if window > 1:
            cv_smooth = np.convolve(cv_data, np.ones(window)/window, mode='valid')
            t_smooth = time_plot[:len(cv_smooth)]
            axes[i].plot(t_smooth, cv_smooth, color='black', linewidth=2, 
                        alpha=0.7, label='Media móvil')
            axes[i].legend(loc='upper right')
    
    axes[-1].set_xlabel(f"Tiempo ({time_unit})")
    fig.suptitle("Evolución de Variables Colectivas", fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, "cv_evolution.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out_path}")


def plot_gaussian_heights(hills_file, output_dir):
    """Gráfico 2: Altura de Gaussianas vs Tiempo.
    
    En WTMetaD, las alturas deben decaer exponencialmente:
      h(t) ∝ exp(-V(s,t) / (kB × ΔT))
    
    Si el decaimiento se detiene → convergencia alcanzada.
    Si las alturas no decaen → problema con parámetros.
    """
    data, headers = load_plumed_file(hills_file)
    if data.size == 0:
        print("  WARNING: HILLS vacío, saltando plot_gaussian_heights")
        return
    
    time = data[:, 0]
    n_cols = data.shape[1]
    height_col = n_cols - 2  # Penúltima columna
    heights = data[:, height_col]
    
    # Convertir tiempo
    time_unit = "ps"
    time_plot = time
    if time[-1] > 10000:
        time_plot = time / 1000
        time_unit = "ns"
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Panel superior: Alturas lineales
    ax1.plot(time_plot, heights, color='#e74c3c', alpha=0.7, linewidth=0.5)
    ax1.set_ylabel("Altura (kJ/mol)")
    ax1.set_title("Altura de Gaussianas — Escala Lineal")
    ax1.axhline(y=heights[0] * 0.01, color='green', linestyle='--', alpha=0.5,
                label=f'1% de h₀ ({heights[0]*0.01:.4f})')
    ax1.legend()
    
    # Panel inferior: Alturas log
    ax2.semilogy(time_plot, heights, color='#3498db', alpha=0.7, linewidth=0.5)
    ax2.set_ylabel("Altura (kJ/mol) [log]")
    ax2.set_xlabel(f"Tiempo ({time_unit})")
    ax2.set_title("Altura de Gaussianas — Escala Logarítmica")
    
    # Indicador de convergencia
    ratio = heights[-1] / heights[0] if heights[0] > 0 else 1
    if ratio < 0.01:
        status = "✓ CONVERGIDO"
        status_color = 'green'
    elif ratio < 0.1:
        status = "~ PARCIALMENTE"
        status_color = 'orange'
    else:
        status = "✗ NO CONVERGIDO"
        status_color = 'red'
    
    ax2.text(0.02, 0.95, f"{status} (h_f/h_i = {ratio:.4f})",
             transform=ax2.transAxes, fontsize=13, fontweight='bold',
             color=status_color, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    fig.suptitle("Decaimiento de Gaussianas (Well-Tempered)", fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, "gaussian_heights.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out_path}")


def plot_fes_2d(fes_file, output_dir):
    """Gráfico 3: Mapa de calor 2D del FES con isolíneas.
    
    Solo se genera si hay 2 CVs en el FES.
    Muestra el paisaje de energía libre con mínimos y barreras.
    """
    data, headers = load_plumed_file(fes_file)
    if data.size == 0:
        print("  WARNING: FES vacío, saltando plot_fes_2d")
        return
    
    n_cols = data.shape[1]
    
    if n_cols < 3:
        print("  INFO: FES tiene < 3 columnas, solo se generará FES 1D")
        return
    
    # Para 2D FES: columnas son cv1, cv2, fes
    cv1 = data[:, 0]
    cv2 = data[:, 1]
    fes = data[:, 2]
    
    # Determinar grid
    cv1_unique = np.unique(cv1)
    cv2_unique = np.unique(cv2)
    
    if len(cv1_unique) < 3 or len(cv2_unique) < 3:
        print("  WARNING: Grid FES demasiado pequeño para 2D")
        return
    
    # Reshape a 2D
    try:
        n1 = len(cv1_unique)
        n2 = len(cv2_unique)
        FES = fes.reshape(n1, n2)
        CV1 = cv1_unique
        CV2 = cv2_unique
    except ValueError:
        print("  WARNING: No se pudo reshape FES a 2D")
        return
    
    # Cap FES para mejor visualización (no mostrar > 50 kJ/mol sobre mínimo)
    fes_min = np.min(FES)
    fes_max = min(np.max(FES), fes_min + 50)
    FES_plot = np.clip(FES, fes_min, fes_max)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Mapa de calor
    cv1_label = headers[0] if headers else "CV1"
    cv2_label = headers[1] if len(headers) > 1 else "CV2"
    
    im = ax.contourf(CV1, CV2, FES_plot.T, levels=30, cmap='RdYlBu_r')
    contours = ax.contour(CV1, CV2, FES_plot.T, levels=15, colors='black', 
                          linewidths=0.5, alpha=0.5)
    ax.clabel(contours, inline=True, fontsize=8, fmt='%.1f')
    
    cbar = plt.colorbar(im, ax=ax, label='Energía Libre (kJ/mol)')
    
    # Marcar mínimos
    min_idx = np.unravel_index(np.argmin(FES), FES.shape)
    ax.plot(CV1[min_idx[0]], CV2[min_idx[1]], 'w*', markersize=15, 
            markeredgecolor='black', markeredgewidth=1, label='Mínimo global')
    
    ax.set_xlabel(cv1_label, fontweight='bold')
    ax.set_ylabel(cv2_label, fontweight='bold')
    ax.set_title("Paisaje de Energía Libre (FES) 2D", fontsize=18, fontweight='bold')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, "fes_2d.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out_path}")


def plot_fes_1d(fes_file, output_dir):
    """Gráfico 4: FES 1D para cada CV.
    
    Si el FES es 1D (1 CV), plotea directamente.
    Si es 2D, proyecta a cada eje integrando Boltzmann.
    """
    data, headers = load_plumed_file(fes_file)
    if data.size == 0:
        return
    
    n_cols = data.shape[1]
    
    if n_cols == 2:
        # FES 1D directo
        cv = data[:, 0]
        fes = data[:, 1]
        cv_label = headers[0] if headers else "CV1"
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(cv, fes, color='#2c3e50', linewidth=2)
        ax.fill_between(cv, fes, alpha=0.3, color='#3498db')
        ax.set_xlabel(cv_label, fontweight='bold')
        ax.set_ylabel("Energía Libre (kJ/mol)", fontweight='bold')
        ax.set_title("Perfil de Energía Libre 1D", fontsize=18, fontweight='bold')
        
        # Marcar mínimos
        min_idx = np.argmin(fes)
        ax.axvline(x=cv[min_idx], color='red', linestyle='--', alpha=0.5,
                   label=f'Mínimo: {cv[min_idx]:.2f}')
        ax.legend()
        
        plt.tight_layout()
        out_path = os.path.join(output_dir, "fes_1d.png")
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
        print(f"  ✓ {out_path}")
    
    elif n_cols >= 3:
        # FES 2D → proyectar a 1D por cada eje
        cv1_unique = np.unique(data[:, 0])
        cv2_unique = np.unique(data[:, 1])
        
        try:
            FES = data[:, 2].reshape(len(cv1_unique), len(cv2_unique))
        except ValueError:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Proyección sobre CV1 (integrar sobre CV2)
        fes_cv1 = -np.log(np.sum(np.exp(-FES / 2.494), axis=1)) * 2.494  # kT=2.494 a 300K
        fes_cv1 -= np.min(fes_cv1)
        cv1_label = headers[0] if headers else "CV1"
        
        ax1.plot(cv1_unique, fes_cv1, color='#2c3e50', linewidth=2)
        ax1.fill_between(cv1_unique, fes_cv1, alpha=0.3, color='#3498db')
        ax1.set_xlabel(cv1_label, fontweight='bold')
        ax1.set_ylabel("Energía Libre (kJ/mol)", fontweight='bold')
        ax1.set_title(f"FES 1D — {cv1_label}")
        
        # Proyección sobre CV2
        fes_cv2 = -np.log(np.sum(np.exp(-FES / 2.494), axis=0)) * 2.494
        fes_cv2 -= np.min(fes_cv2)
        cv2_label = headers[1] if len(headers) > 1 else "CV2"
        
        ax2.plot(cv2_unique, fes_cv2, color='#2c3e50', linewidth=2)
        ax2.fill_between(cv2_unique, fes_cv2, alpha=0.3, color='#e74c3c')
        ax2.set_xlabel(cv2_label, fontweight='bold')
        ax2.set_ylabel("Energía Libre (kJ/mol)", fontweight='bold')
        ax2.set_title(f"FES 1D — {cv2_label}")
        
        fig.suptitle("Proyecciones 1D del FES", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        out_path = os.path.join(output_dir, "fes_1d_projections.png")
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
        print(f"  ✓ {out_path}")


def plot_deltaG_convergence(dg_file, output_dir):
    """Gráfico 5: ΔG vs Tiempo (convergencia).
    
    Un ΔG que se estabiliza indica convergencia difusiva.
    """
    if not os.path.exists(dg_file):
        print("  INFO: No hay datos de ΔG, saltando plot")
        return
    
    data = np.loadtxt(dg_file, comments='#')
    if data.size == 0:
        return
    
    time = data[:, 0]
    deltaG = data[:, 1]
    
    time_unit = "ps"
    time_plot = time
    if time[-1] > 10000:
        time_plot = time / 1000
        time_unit = "ns"
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(time_plot, deltaG, 'o-', color='#8e44ad', markersize=8, linewidth=2)
    
    # Banda de convergencia (últimos 3 puntos)
    if len(deltaG) > 3:
        mean_last = np.mean(deltaG[-3:])
        std_last = np.std(deltaG[-3:])
        ax.axhspan(mean_last - std_last, mean_last + std_last, 
                   alpha=0.2, color='green', label=f'Rango convergencia: {mean_last:.1f} ± {std_last:.1f}')
        ax.axhline(y=mean_last, color='green', linestyle='--', alpha=0.5)
    
    ax.set_xlabel(f"Tiempo ({time_unit})", fontweight='bold')
    ax.set_ylabel("ΔG (kJ/mol)", fontweight='bold')
    ax.set_title("Convergencia: ΔG vs Tiempo", fontsize=18, fontweight='bold')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, "deltaG_convergence.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out_path}")


def plot_cv_correlation(colvar_file, output_dir):
    """Gráfico bonus: Scatter plot de CV1 vs CV2 coloreado por tiempo.
    
    Útil para visualizar la exploración del espacio de CVs.
    Solo se genera si hay al menos 2 CVs.
    """
    data, headers = load_plumed_file(colvar_file)
    if data.size == 0 or data.shape[1] < 4:
        return  # Necesita time, cv1, cv2, bias al mínimo
    
    time = data[:, 0]
    cv1 = data[:, 1]
    cv2 = data[:, 2]
    
    cv1_label = headers[1] if len(headers) > 1 else "CV1"
    cv2_label = headers[2] if len(headers) > 2 else "CV2"
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(cv1, cv2, c=time, cmap='viridis', s=1, alpha=0.5)
    cbar = plt.colorbar(scatter, ax=ax)
    
    t_unit = "ps"
    if time[-1] > 10000:
        t_unit = "ns"
        cbar.set_label(f"Tiempo ({t_unit})")
        # Actualizar ticks del colorbar
        ticks = cbar.get_ticks()
        cbar.set_ticklabels([f"{t/1000:.0f}" for t in ticks])
    else:
        cbar.set_label(f"Tiempo ({t_unit})")
    
    ax.set_xlabel(cv1_label, fontweight='bold')
    ax.set_ylabel(cv2_label, fontweight='bold')
    ax.set_title("Exploración del espacio de CVs", fontsize=18, fontweight='bold')
    
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, "cv_exploration.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualización de resultados de WT-MetaD',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--colvar', help='Archivo COLVAR')
    parser.add_argument('--hills', help='Archivo HILLS')
    parser.add_argument('--fes', help='Archivo FES (de sum_hills)')
    parser.add_argument('--deltaG', default=None, help='Archivo deltaG_vs_time.dat')
    parser.add_argument('--output-dir', default='plots', dest='output_dir',
                        help='Directorio de salida para gráficos')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    setup_plot_style()
    
    print("=" * 50)
    print("VISUALIZACIÓN — WT-MetaD")
    print("=" * 50)
    
    if args.colvar:
        print("\n─── CVs vs Tiempo ───")
        plot_cv_evolution(args.colvar, args.output_dir)
        
        print("\n─── Exploración CV1 vs CV2 ───")
        plot_cv_correlation(args.colvar, args.output_dir)
    
    if args.hills:
        print("\n─── Gaussianas ───")
        plot_gaussian_heights(args.hills, args.output_dir)
    
    if args.fes:
        print("\n─── FES 2D ───")
        plot_fes_2d(args.fes, args.output_dir)
        
        print("\n─── FES 1D ───")
        plot_fes_1d(args.fes, args.output_dir)
    
    if args.deltaG:
        print("\n─── Convergencia ΔG ───")
        plot_deltaG_convergence(args.deltaG, args.output_dir)
    elif os.path.exists(os.path.join(os.path.dirname(args.output_dir or '.'), 'analysis', 'deltaG_vs_time.dat')):
        dg_path = os.path.join(os.path.dirname(args.output_dir or '.'), 'analysis', 'deltaG_vs_time.dat')
        print("\n─── Convergencia ΔG ───")
        plot_deltaG_convergence(dg_path, args.output_dir)
    
    print(f"\n✓ Gráficos generados en: {args.output_dir}/")


if __name__ == '__main__':
    main()
