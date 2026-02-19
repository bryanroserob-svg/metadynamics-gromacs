#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_plumed.py — Generador dinámico de archivos plumed.dat
para Well-Tempered Metadynamics (WTMetaD).

Soporta 4 tipos de Variables Colectivas (CVs):
  - DISTANCE: Distancia entre centros de masa de dos grupos
  - RMSD:     RMSD respecto a una estructura de referencia
  - TORSION:  Ángulo dihedral (4 átomos)
  - COORDINATION: Número de coordinación entre dos grupos

Uso típico:
  python3 generate_plumed.py \\
      --cv-type distance --cv-atoms "1,100" --cv-atoms "200,300" \\
      --sigma 0.3 0.3 --height 1.2 --pace 500 \\
      --biasfactor 15 --temp 300 --output plumed.dat

Llamado por run_metad_pipeline.sh durante la etapa de generación de input PLUMED.
"""

import argparse
import os
import sys
import json
from datetime import datetime


def validate_atom_selection(atoms_str):
    """Valida formato de selección de átomos (ej: '1,100' o '1-50')."""
    atoms_str = atoms_str.strip()
    # Formato rango: 1-50
    if '-' in atoms_str and ',' not in atoms_str:
        parts = atoms_str.split('-')
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            return True
    # Formato lista: 1,2,3,100
    for part in atoms_str.split(','):
        if not part.strip().isdigit():
            return False
    return True


def atoms_to_plumed(atoms_str):
    """Convierte selección de átomos a formato PLUMED.
    
    '1,100' -> '1,100'
    '1-50'  -> '1-50'
    """
    return atoms_str.strip()


def generate_cv_block(cv_configs):
    """Genera el bloque de definición de CVs en formato PLUMED.
    
    Args:
        cv_configs: Lista de diccionarios con configuración de cada CV.
        
    Returns:
        Tuple: (lines, cv_names) - Líneas PLUMED y nombres de los CVs.
    """
    lines = []
    cv_names = []
    
    lines.append("# ==============================================")
    lines.append("# DEFINICIÓN DE VARIABLES COLECTIVAS (CVs)")
    lines.append("# ==============================================")
    lines.append("")
    
    for i, cv in enumerate(cv_configs):
        cv_name = f"cv{i+1}"
        cv_names.append(cv_name)
        cv_type = cv['type'].upper()
        
        if cv_type == 'DISTANCE':
            # Distancia entre centros de masa
            # cv_atoms debe ser una lista de 2 elementos: grupo1, grupo2
            atoms = cv['atoms']
            if len(atoms) < 2:
                print(f"ERROR: DISTANCE requiere al menos 2 grupos de átomos, recibidos: {len(atoms)}", file=sys.stderr)
                sys.exit(1)
            
            # Si son grupos COM, usamos DISTANCE con ATOMS
            group1 = atoms_to_plumed(atoms[0])
            group2 = atoms_to_plumed(atoms[1])
            
            # Definir centros de masa si hay rangos
            if '-' in group1 or ',' in group1:
                lines.append(f"# CV{i+1}: Distancia COM entre grupos de átomos")
                lines.append(f"com1: COM ATOMS={group1}")
                lines.append(f"com2: COM ATOMS={group2}")
                lines.append(f"{cv_name}: DISTANCE ATOMS=com1,com2 NOPBC")
            else:
                lines.append(f"# CV{i+1}: Distancia entre átomos {group1} y {group2}")
                lines.append(f"{cv_name}: DISTANCE ATOMS={group1},{group2} NOPBC")
                
        elif cv_type == 'RMSD':
            # RMSD respecto a estructura de referencia
            ref_file = cv.get('reference', 'reference.pdb')
            atom_selection = cv['atoms'][0] if cv['atoms'] else '1-1000'
            lines.append(f"# CV{i+1}: RMSD respecto a {ref_file}")
            lines.append(f"# IMPORTANT: El archivo de referencia debe estar en formato PDB")
            lines.append(f"# y contener las mismas coordenadas de átomos que el sistema.")
            if atom_selection != '1-1000' and atom_selection != ref_file:
                lines.append(f"{cv_name}: RMSD REFERENCE={ref_file} TYPE=OPTIMAL")
            else:
                lines.append(f"{cv_name}: RMSD REFERENCE={ref_file} TYPE=OPTIMAL")
            
        elif cv_type == 'TORSION':
            # Ángulo dihedral (4 átomos)
            atoms = cv['atoms']
            if len(atoms) < 1:
                print(f"ERROR: TORSION requiere 4 átomos (ej: '5,7,9,15')", file=sys.stderr)
                sys.exit(1)
            atom_list = atoms_to_plumed(atoms[0])
            lines.append(f"# CV{i+1}: Ángulo dihedral")
            lines.append(f"{cv_name}: TORSION ATOMS={atom_list}")
            
        elif cv_type == 'COORDINATION':
            # Número de coordinación
            atoms = cv['atoms']
            if len(atoms) < 2:
                print(f"ERROR: COORDINATION requiere 2 grupos", file=sys.stderr)
                sys.exit(1)
            group_a = atoms_to_plumed(atoms[0])
            group_b = atoms_to_plumed(atoms[1])
            r_0 = cv.get('r0', '0.3')
            lines.append(f"# CV{i+1}: Número de coordinación")
            lines.append(f"{cv_name}: COORDINATION GROUPA={group_a} GROUPB={group_b} R_0={r_0}")
        else:
            print(f"ERROR: Tipo de CV no soportado: {cv_type}", file=sys.stderr)
            print("  Tipos válidos: distance, rmsd, torsion, coordination", file=sys.stderr)
            sys.exit(1)
        
        lines.append("")
    
    return lines, cv_names


def generate_metad_block(cv_names, args):
    """Genera el bloque METAD (bias) en formato PLUMED.
    
    Well-Tempered MetaD deposita gaussianas con altura decreciente:
      h(t) = h₀ × exp(-V(s,t) / (kB × ΔT))
    donde ΔT = T × (biasfactor - 1)
    
    Args:
        cv_names: Lista de nombres de CVs.
        args: Argumentos de línea de comandos.
    """
    lines = []
    lines.append("# ==============================================")
    lines.append("# WELL-TEMPERED METADYNAMICS")
    lines.append("# ==============================================")
    lines.append("# Parámetros clave:")
    lines.append(f"#   BIASFACTOR = {args.biasfactor}")
    lines.append(f"#     → ΔT = T × (γ-1) = {args.temp} × ({args.biasfactor}-1) = {args.temp * (args.biasfactor - 1):.1f} K")
    lines.append(f"#     → Temperatura efectiva = γ × T = {args.biasfactor * args.temp:.1f} K")
    lines.append(f"#   SIGMA = {args.sigma}")
    lines.append(f"#     → Ancho de gaussianas (debe reflejar fluctuaciones de CV en eq.)")
    lines.append(f"#   HEIGHT = {args.height} kJ/mol")
    lines.append(f"#     → Altura INICIAL de gaussianas (decrece con WT)")
    lines.append(f"#   PACE = {args.pace}")
    lines.append(f"#     → Deposición cada {args.pace} steps ({args.pace * args.dt:.1f} ps)")
    lines.append("")
    
    # Construir ARG y SIGMA
    arg_str = ",".join(cv_names)
    sigma_str = ",".join(str(s) for s in args.sigma)
    
    # Línea METAD principal
    metad_parts = [
        f"metad: METAD ARG={arg_str}",
        f"  SIGMA={sigma_str}",
        f"  HEIGHT={args.height}",
        f"  PACE={args.pace}",
        f"  BIASFACTOR={args.biasfactor}",
        f"  TEMP={args.temp}",
        f"  GRID_MIN={args.grid_min}" if args.grid_min else None,
        f"  GRID_MAX={args.grid_max}" if args.grid_max else None,
        f"  GRID_BIN={args.grid_bin}" if args.grid_bin else None,
        f"  FILE=HILLS",
    ]
    
    # Multi-walker si se especifica
    if args.walkers and args.walkers > 1:
        metad_parts.append(f"  WALKERS_N={args.walkers}")
        metad_parts.append(f"  WALKERS_DIR=./")
        metad_parts.append(f"  WALKERS_RSTRIDE={args.pace}")
    
    # Filtrar None y unir
    metad_parts = [p for p in metad_parts if p is not None]
    
    # Formato multi-línea con ...
    for j, part in enumerate(metad_parts):
        if j < len(metad_parts) - 1:
            lines.append(f"{part} ...")
            if j == 0:
                continue
        else:
            lines.append(part)
    
    lines.append("")
    
    return lines


def generate_walls_block(cv_names, args):
    """Genera restricciones UPPER/LOWER WALL opcionales."""
    lines = []
    
    if not args.walls:
        return lines
    
    lines.append("# ==============================================")
    lines.append("# WALLS (restricciones suaves)")
    lines.append("# Previenen exploración de regiones no físicas")
    lines.append("# ==============================================")
    
    for wall_spec in args.walls:
        # Formato: "cv_name,type,at,kappa"
        # Ejemplo: "cv1,upper,3.0,150"
        parts = wall_spec.split(',')
        if len(parts) < 4:
            print(f"WARNING: Wall spec incompleto: {wall_spec} (formato: cv,type,at,kappa)", file=sys.stderr)
            continue
        
        cv_target, wall_type, at_val, kappa = parts[0], parts[1], parts[2], parts[3]
        
        if cv_target not in cv_names:
            print(f"WARNING: CV '{cv_target}' no definido, ignorando wall", file=sys.stderr)
            continue
        
        wall_label = f"wall_{cv_target}_{wall_type}"
        
        if wall_type.lower() == 'upper':
            lines.append(f"{wall_label}: UPPER_WALLS ARG={cv_target} AT={at_val} KAPPA={kappa} EXP=2 EPS=1 OFFSET=0")
        elif wall_type.lower() == 'lower':
            lines.append(f"{wall_label}: LOWER_WALLS ARG={cv_target} AT={at_val} KAPPA={kappa} EXP=2 EPS=1 OFFSET=0")
        else:
            print(f"WARNING: Tipo de wall '{wall_type}' no reconocido (usar upper/lower)", file=sys.stderr)
    
    lines.append("")
    return lines


def generate_print_block(cv_names, args):
    """Genera bloque PRINT para escritura de COLVAR."""
    lines = []
    lines.append("# ==============================================")
    lines.append("# OUTPUT")
    lines.append("# ==============================================")
    
    # PRINT: CVs + bias
    all_args = cv_names + ["metad.bias"]
    arg_str = ",".join(all_args)
    
    lines.append(f"PRINT ARG={arg_str} STRIDE={args.print_stride} FILE=COLVAR")
    lines.append("")
    
    # Flush periódico para evitar pérdida de datos
    lines.append(f"FLUSH STRIDE={args.print_stride * 10}")
    lines.append("")
    
    return lines


def write_plumed_file(output_path, all_lines, args):
    """Escribe el archivo plumed.dat con header informativo."""
    with open(output_path, 'w') as f:
        # Header
        f.write("# =====================================================\n")
        f.write(f"# plumed.dat — Well-Tempered Metadynamics\n")
        f.write(f"# Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Generador: generate_plumed.py\n")
        f.write("# =====================================================\n")
        f.write(f"# Temperatura: {args.temp} K\n")
        f.write(f"# Biasfactor:  {args.biasfactor}\n")
        f.write(f"# Sigma:       {args.sigma}\n")
        f.write(f"# Height:      {args.height} kJ/mol\n")
        f.write(f"# Pace:        {args.pace} steps\n")
        if args.walkers and args.walkers > 1:
            f.write(f"# Walkers:     {args.walkers}\n")
        f.write("# =====================================================\n\n")
        
        for line in all_lines:
            f.write(line + "\n")


def generate_from_json(json_path, args):
    """Genera plumed.dat desde un archivo JSON de configuración.
    
    Formato JSON esperado:
    {
        "cvs": [
            {"type": "distance", "atoms": ["1-50", "100-150"]},
            {"type": "torsion",  "atoms": ["5,7,9,15"]}
        ],
        "sigma": [0.3, 0.15],
        "height": 1.2,
        "pace": 500,
        "biasfactor": 15,
        "temp": 300,
        "walls": ["cv1,upper,3.0,150"]
    }
    """
    with open(json_path, 'r') as f:
        config = json.load(f)
    
    # Sobreescribir args con valores del JSON
    if 'sigma' in config:
        args.sigma = config['sigma']
    if 'height' in config:
        args.height = config['height']
    if 'pace' in config:
        args.pace = config['pace']
    if 'biasfactor' in config:
        args.biasfactor = config['biasfactor']
    if 'temp' in config:
        args.temp = config['temp']
    if 'walls' in config:
        args.walls = config['walls']
    if 'walkers' in config:
        args.walkers = config['walkers']
    
    return config.get('cvs', [])


def main():
    parser = argparse.ArgumentParser(
        description='Generador de plumed.dat para Well-Tempered Metadynamics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Distancia COM entre dos grupos
  %(prog)s --cv-type distance --cv-atoms "1-50" --cv-atoms "100-150" \\
           --sigma 0.3 --height 1.2 --pace 500 --biasfactor 15

  # 2 CVs: distancia + torsión
  %(prog)s --cv-type distance --cv-atoms "1-50" --cv-atoms "100-150" \\
           --cv-type torsion --cv-atoms "5,7,9,15" \\
           --sigma 0.3 0.15 --height 1.2 --pace 500 --biasfactor 15

  # Desde archivo JSON
  %(prog)s --from-json metad_config.json --output plumed.dat

  # Con walls para confinar el muestreo
  %(prog)s --cv-type distance --cv-atoms "1,100" --cv-atoms "200,300" \\
           --sigma 0.35 --height 1.5 --pace 500 --biasfactor 20 \\
           --wall "cv1,lower,0.2,150" --wall "cv1,upper,4.0,150"
        """
    )
    
    # CV configuration
    parser.add_argument('--cv-type', action='append', dest='cv_types',
                        help='Tipo de CV: distance, rmsd, torsion, coordination. '
                             'Se puede repetir para múltiples CVs.')
    parser.add_argument('--cv-atoms', action='append', dest='cv_atoms',
                        help='Átomos para el CV (formato: "1,100" o "1-50"). '
                             'Para DISTANCE: 2 --cv-atoms por CV. '
                             'Para TORSION: 1 --cv-atoms con 4 átomos.')
    parser.add_argument('--cv-reference', default=None,
                        help='Archivo de referencia para RMSD (formato PDB)')
    parser.add_argument('--cv-r0', default='0.3',
                        help='R_0 para COORDINATION (default: 0.3 nm)')
    
    # MetaD parameters
    parser.add_argument('--sigma', nargs='+', type=float, required=False, default=[0.3],
                        help='Ancho de gaussianas por CV (un valor por CV)')
    parser.add_argument('--height', type=float, default=1.2,
                        help='Altura INICIAL de gaussianas en kJ/mol (default: 1.2)')
    parser.add_argument('--pace', type=int, default=500,
                        help='Deposición cada N steps (default: 500 = 1 ps)')
    parser.add_argument('--biasfactor', type=float, default=15,
                        help='Bias factor γ para WTMetaD (default: 15). '
                             'γ grande → más exploración, convergencia lenta. '
                             'Recomendado: 10-20 para la mayoría de sistemas.')
    parser.add_argument('--temp', type=float, default=300,
                        help='Temperatura del sistema en K (default: 300)')
    parser.add_argument('--dt', type=float, default=0.002,
                        help='Timestep en ps (default: 0.002). Usado para reportar PACE en ps.')
    
    # Grid
    parser.add_argument('--grid-min', dest='grid_min', default=None,
                        help='Mínimo del grid para bias (formato: "0,0" para 2 CVs)')
    parser.add_argument('--grid-max', dest='grid_max', default=None,
                        help='Máximo del grid para bias')
    parser.add_argument('--grid-bin', dest='grid_bin', default=None,
                        help='Número de bins del grid')
    
    # Walls
    parser.add_argument('--wall', action='append', dest='walls',
                        help='Wall: "cv_name,type,at,kappa" (ej: "cv1,upper,3.0,150")')
    
    # Multi-walker
    parser.add_argument('--walkers', type=int, default=None,
                        help='Número de walkers para Multi-Walker MetaD')
    
    # Output
    parser.add_argument('--print-stride', type=int, default=500, dest='print_stride',
                        help='Frecuencia de escritura de COLVAR (default: 500)')
    parser.add_argument('--output', '-o', default='plumed.dat',
                        help='Archivo de salida (default: plumed.dat)')
    
    # JSON config
    parser.add_argument('--from-json', dest='from_json', default=None,
                        help='Cargar configuración desde archivo JSON')
    
    args = parser.parse_args()
    
    # ── Construir configuraciones de CVs ──
    cv_configs = []
    
    if args.from_json:
        # Modo JSON
        json_cvs = generate_from_json(args.from_json, args)
        for cv_spec in json_cvs:
            cv_configs.append({
                'type': cv_spec['type'],
                'atoms': cv_spec.get('atoms', []),
                'reference': cv_spec.get('reference', 'reference.pdb'),
                'r0': cv_spec.get('r0', '0.3'),
            })
    elif args.cv_types:
        # Modo CLI
        atom_idx = 0
        all_atoms = args.cv_atoms or []
        
        for cv_type in args.cv_types:
            cv_type_upper = cv_type.upper()
            cv_atoms_list = []
            
            if cv_type_upper == 'DISTANCE' or cv_type_upper == 'COORDINATION':
                # Necesita 2 grupos de átomos
                if atom_idx + 1 < len(all_atoms):
                    atom1 = all_atoms[atom_idx]
                    atom2 = all_atoms[atom_idx + 1]
                    if not validate_atom_selection(atom1):
                        print(f"ERROR: Formato de átomos inválido para {cv_type} grupo 1: '{atom1}'"
                              f"\n  Formatos válidos: '1-50' (rango) o '1,2,3' (lista)", file=sys.stderr)
                        sys.exit(1)
                    if not validate_atom_selection(atom2):
                        print(f"ERROR: Formato de átomos inválido para {cv_type} grupo 2: '{atom2}'"
                              f"\n  Formatos válidos: '1-50' (rango) o '1,2,3' (lista)", file=sys.stderr)
                        sys.exit(1)
                    cv_atoms_list = [atom1, atom2]
                    atom_idx += 2
                else:
                    print(f"ERROR: {cv_type} requiere 2 --cv-atoms, faltan átomos", file=sys.stderr)
                    sys.exit(1)
            elif cv_type_upper == 'TORSION':
                # Necesita 1 grupo con 4 átomos
                if atom_idx < len(all_atoms):
                    torsion_val = all_atoms[atom_idx]
                    if not validate_atom_selection(torsion_val):
                        print(f"ERROR: Formato de átomos inválido para TORSION: '{torsion_val}'"
                              f"\n  Formato esperado: '5,7,9,15' (4 átomos)", file=sys.stderr)
                        sys.exit(1)
                    cv_atoms_list = [torsion_val]
                    atom_idx += 1
                else:
                    print(f"ERROR: TORSION requiere 1 --cv-atoms con 4 átomos", file=sys.stderr)
                    sys.exit(1)
            elif cv_type_upper == 'RMSD':
                if atom_idx < len(all_atoms):
                    cv_atoms_list = [all_atoms[atom_idx]]
                    atom_idx += 1
                else:
                    cv_atoms_list = []
            
            cv_configs.append({
                'type': cv_type,
                'atoms': cv_atoms_list,
                'reference': args.cv_reference or 'reference.pdb',
                'r0': args.cv_r0,
            })
    else:
        print("ERROR: Debe especificar al menos un --cv-type o usar --from-json", file=sys.stderr)
        parser.print_help()
        sys.exit(1)
    
    # Validar que haya suficientes sigmas
    if len(args.sigma) == 1 and len(cv_configs) > 1:
        # Expandir sigma para todos los CVs
        args.sigma = args.sigma * len(cv_configs)
    elif len(args.sigma) != len(cv_configs):
        print(f"ERROR: Número de SIGMA ({len(args.sigma)}) != número de CVs ({len(cv_configs)})",
              file=sys.stderr)
        sys.exit(1)
    
    # ── Generar bloques ──
    all_lines = []
    
    cv_lines, cv_names = generate_cv_block(cv_configs)
    all_lines.extend(cv_lines)
    
    metad_lines = generate_metad_block(cv_names, args)
    all_lines.extend(metad_lines)
    
    if args.walls:
        wall_lines = generate_walls_block(cv_names, args)
        all_lines.extend(wall_lines)
    
    print_lines = generate_print_block(cv_names, args)
    all_lines.extend(print_lines)
    
    # ── Escribir ──
    write_plumed_file(args.output, all_lines, args)
    
    print(f"✓ plumed.dat generado: {args.output}")
    print(f"  CVs definidos: {len(cv_names)} ({', '.join(cv_names)})")
    print(f"  Parámetros WTMetaD:")
    print(f"    σ = {args.sigma}")
    print(f"    h₀ = {args.height} kJ/mol")
    print(f"    pace = {args.pace} steps ({args.pace * args.dt:.1f} ps)")
    print(f"    γ = {args.biasfactor}")
    print(f"    T_eff = {args.biasfactor * args.temp:.0f} K")
    if args.walkers and args.walkers > 1:
        print(f"    Multi-Walker: {args.walkers} walkers")


if __name__ == '__main__':
    main()
