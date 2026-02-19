#!/bin/bash
set -euo pipefail

###############################################################################
#  METADYNAMICS PIPELINE v1 — GROMACS + PLUMED
#  Well-Tempered Metadynamics (WTMetaD)
#  Requiere: GROMACS patcheado con PLUMED, Python 3.8+
###############################################################################

#==========================================
# CONFIGURACIÓN
#==========================================
if command -v gmx &> /dev/null; then
    readonly GMX=gmx; readonly USE_MPI=false
elif command -v gmx_mpi &> /dev/null; then
    readonly GMX=gmx_mpi; readonly USE_MPI=true
else
    echo "ERROR: No se encontró GROMACS (gmx o gmx_mpi)" >&2; exit 1
fi

readonly NT=${METAD_NT:-$(nproc --all 2>/dev/null || echo 4)}
if [ "$USE_MPI" = true ]; then
    readonly MDRUN="$GMX mdrun"
else
    readonly MDRUN="$GMX mdrun -nt $NT"
fi

readonly BASE_PROT=proteins
readonly BASE_LIG=ligands
readonly BASE_MDP=mdp
readonly WORKDIR=METAD_RUN
readonly INITIAL_DIR="$(pwd)"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts"

FF_IS_LOCAL=false

#==========================================
# COLORES Y LOGGING
#==========================================
readonly RED='\033[0;31m' GREEN='\033[0;32m' YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m' CYAN='\033[0;36m' MAGENTA='\033[0;35m' NC='\033[0m'
CURRENT_STAGE="inicialización"

log_step() {
    CURRENT_STAGE="$1"
    echo -e "\n${BLUE}=========================================${NC}"
    echo -e "${BLUE}>>> $1${NC}"
    echo -e "${BLUE}=========================================${NC}\n"
}
log_success() { echo -e "${GREEN}✓${NC} $1"; }
log_error()   { echo -e "${RED}❌ ERROR:${NC} $1" >&2; }
log_warning() { echo -e "${YELLOW}⚠${NC} $1"; }
log_info()    { echo -e "${CYAN}ℹ${NC} $1"; }

create_dir() { [ ! -d "$1" ] && mkdir -p "$1" && log_success "Carpeta: $1"; }
run_gmx() { "$GMX" "$@"; }

mark_stage_done() { echo "$1" >> "$RUNDIR/.completed_stages"; }
is_stage_done() { [ -f "$RUNDIR/.completed_stages" ] && grep -qxF "$1" "$RUNDIR/.completed_stages"; }

backup_mdp() {
    local mdp_file="$1"
    if [ -f "$mdp_file" ] && [ ! -f "${mdp_file}.original" ]; then
        cp "$mdp_file" "${mdp_file}.original"
    fi
}

#==========================================
# TRAP
#==========================================
cleanup_on_error() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo ""; log_error "Interrumpido durante: ${CURRENT_STAGE}"
        log_info "Reanudar con: $0 --resume ${RUNDIR:-}"
    fi
}
trap cleanup_on_error EXIT

#==========================================
# AYUDA
#==========================================
show_help() {
    cat <<EOF
${BLUE}  METADYNAMICS PIPELINE v1 — GROMACS + PLUMED${NC}

${CYAN}Uso:${NC}
  $0                    Ejecución interactiva
  $0 --resume <DIR>     Reanudar desde checkpoint
  $0 --cleanup <DIR>    Limpiar archivos temporales
  $0 --help             Ayuda

${CYAN}Requisitos:${NC}
  • GROMACS >= 2020 patcheado con PLUMED >= 2.7
  • Python 3.8+ con numpy, matplotlib
  • plumed CLI (para sum_hills)

${CYAN}CVs soportados:${NC}
  distance, rmsd, torsion, coordination
EOF
    exit 0
}

#==========================================
# ADAPTAR MDP SEGÚN FORCE FIELD
#==========================================
adapt_mdp_for_forcefield() {
    local mdp_dir="$1"
    local is_charmm=false
    case "${FF_DIR:-}" in charmm*) is_charmm=true ;; esac

    if [ "$is_charmm" = true ]; then
        log_info "Force field CHARMM detectado: MDP sin modificaciones"
        return 0
    fi

    log_step "Adaptando MDPs para ${FF_DIR:-desconocido} (no-CHARMM)"
    for mdp_file in "$mdp_dir"/*.mdp; do
        [ -f "$mdp_file" ] || continue
        if grep -q 'force-switch' "$mdp_file"; then
            sed -i 's/vdw-modifier.*=.*force-switch/vdw-modifier             = Potential-shift/' "$mdp_file"
            sed -i '/^rvdw-switch/d' "$mdp_file"
            sed -i 's/^DispCorr.*=.*no/DispCorr                 = EnerPres/' "$mdp_file"
            log_success "$(basename "$mdp_file") adaptado"
        fi
    done
}

#==========================================
# VALIDACIÓN DE DEPENDENCIAS
#==========================================
validate_dependencies() {
    log_step "Validando dependencias"
    log_success "GROMACS: $(which $GMX) $([ "$USE_MPI" = true ] && echo "(MPI)" || echo "(serial)")"

    # Verificar PLUMED
    if $GMX mdrun -plumed /dev/null 2>&1 | grep -q "PLUMED is not available"; then
        log_error "GROMACS NO está patcheado con PLUMED"
        log_info "Recompila GROMACS con: plumed patch -p"
        exit 1
    fi
    log_success "PLUMED integrado con GROMACS"

    # Python
    if ! command -v python3 &> /dev/null; then
        log_error "python3 no encontrado"; exit 1
    fi
    log_success "Python3: $(python3 --version 2>&1)"

    # Verificar módulos Python
    for mod in numpy matplotlib; do
        if python3 -c "import $mod" 2>/dev/null; then
            log_success "Python: $mod ✓"
        else
            log_warning "Python: $mod no instalado (pip install $mod)"
        fi
    done

    # plumed CLI (para sum_hills)
    if command -v plumed &> /dev/null; then
        log_success "PLUMED CLI: $(plumed --version 2>&1 | head -1)"
    else
        log_warning "plumed CLI no encontrado (análisis FES limitado)"
    fi

    # bc — requerido para comparaciones de flotantes (concentración iónica)
    if ! command -v bc &> /dev/null; then
        log_error "'bc' no encontrado. Instalar: sudo apt install bc"
        exit 1
    fi
    log_success "bc: $(which bc)"
}

#==========================================
# INPUT DEL USUARIO
#==========================================
get_user_input() {
    echo -e "${MAGENTA}═══════════════════════════════════════════${NC}"
    echo -e "${MAGENTA}  WELL-TEMPERED METADYNAMICS — CONFIG${NC}"
    echo -e "${MAGENTA}═══════════════════════════════════════════${NC}\n"

    # ── Proteína ──
    echo -e "${CYAN}═══ ESTRUCTURA PROTEICA ═══${NC}\n"
    local PROT_DIRS=() PROT_PDBS=()
    if [ ! -d "$INITIAL_DIR/$BASE_PROT" ]; then
        log_error "Carpeta '$BASE_PROT/' no encontrada"; exit 1
    fi

    while IFS= read -r -d '' pdb_found; do
        local dir_name rel_path
        dir_name=$(basename "$(dirname "$pdb_found")")
        rel_path=$(realpath --relative-to="$INITIAL_DIR" "$pdb_found")
        PROT_DIRS+=("$dir_name")
        PROT_PDBS+=("$rel_path")
    done < <(find "$INITIAL_DIR/$BASE_PROT" -name "*.pdb" -type f -print0 | sort -z)

    if [ ${#PROT_DIRS[@]} -eq 0 ]; then
        log_error "No se encontraron .pdb en '$BASE_PROT/'"; exit 1
    fi

    echo "  Proteínas disponibles:"
    for i in "${!PROT_DIRS[@]}"; do
        echo "   $((i+1))) ${PROT_DIRS[$i]}  (${PROT_PDBS[$i]})"
    done
    echo -e "\n  ${YELLOW}Seleccione [1-${#PROT_DIRS[@]}]:${NC}"
    read -r PROT_CHOICE
    if [[ "$PROT_CHOICE" =~ ^[0-9]+$ ]] && [ "$PROT_CHOICE" -ge 1 ] && [ "$PROT_CHOICE" -le ${#PROT_DIRS[@]} ]; then
        PROT="${PROT_DIRS[$((PROT_CHOICE-1))]}"
    else
        log_error "Selección inválida"; exit 1
    fi
    log_success "Proteína: $PROT"

    # ── Ligando ──
    echo -e "\n${CYAN}═══ LIGANDO ═══${NC}\n"
    HAS_LIGAND=false
    local LIG_DIRS=() LIG_FILES=()

    if [ -d "$INITIAL_DIR/$BASE_LIG" ]; then
        for lig_dir in "$INITIAL_DIR/$BASE_LIG"/*/; do
            if [ -d "$lig_dir" ] && [ -f "$lig_dir/ligando.itp" ] && [ -f "$lig_dir/ligando.gro" ]; then
                local dir_name extras=""
                dir_name=$(basename "$lig_dir")
                [ -f "$lig_dir/ligando.prm" ] && extras=" +prm"
                LIG_DIRS+=("$dir_name")
                LIG_FILES+=("ligando.itp, ligando.gro${extras}")
            fi
        done
    fi

    if [ ${#LIG_DIRS[@]} -gt 0 ]; then
        echo "  Ligandos disponibles:"
        for i in "${!LIG_DIRS[@]}"; do
            echo "   $((i+1))) ${LIG_DIRS[$i]}  (${LIG_FILES[$i]})"
        done
        echo "   0) Sin ligando"
        echo -e "\n  ${YELLOW}Seleccione [0-${#LIG_DIRS[@]}]:${NC}"
        read -r LIG_CHOICE
        if [[ "$LIG_CHOICE" =~ ^[1-9][0-9]*$ ]] && [ "$LIG_CHOICE" -le ${#LIG_DIRS[@]} ]; then
            LIG="${LIG_DIRS[$((LIG_CHOICE-1))]}"
            HAS_LIGAND=true
            log_success "Ligando: $LIG"
        else
            log_info "Sin ligando (solo proteína)"
            LIG=""
        fi
    else
        log_info "No se encontraron ligandos parametrizados"
        LIG=""
    fi

    # ── Force Field ──
    echo -e "\n${CYAN}═══ FORCE FIELD ═══${NC}\n"
    local LOCAL_FF_DIRS=() LOCAL_FF_NAMES=()
    for ff_dir in "$INITIAL_DIR"/*.ff; do
        if [ -d "$ff_dir" ]; then
            local dir_bn ff_desc
            dir_bn=$(basename "$ff_dir")
            ff_desc="$dir_bn"
            [ -f "$ff_dir/forcefield.doc" ] && ff_desc=$(grep -m1 '[^ ]' "$ff_dir/forcefield.doc" 2>/dev/null || echo "$dir_bn")
            LOCAL_FF_DIRS+=("$dir_bn")
            LOCAL_FF_NAMES+=("$ff_desc")
        fi
    done

    local NATIVE_FF_DIRS=() NATIVE_FF_NAMES=() gmx_topdir=""
    local gmx_datadir=""
    gmx_datadir=$($GMX -h 2>&1 | grep -oP 'Data prefix:\s*\K\S+' || true)
    if [ -n "$gmx_datadir" ]; then gmx_topdir="$gmx_datadir/share/gromacs/top"
    elif [ -d "/usr/local/gromacs/share/gromacs/top" ]; then gmx_topdir="/usr/local/gromacs/share/gromacs/top"
    elif [ -d "/usr/share/gromacs/top" ]; then gmx_topdir="/usr/share/gromacs/top"; fi

    if [ -n "$gmx_topdir" ] && [ -d "$gmx_topdir" ]; then
        for ff_dir in "$gmx_topdir"/*.ff; do
            if [ -d "$ff_dir" ]; then
                local dir_bn ff_desc is_dup=false
                dir_bn=$(basename "$ff_dir")
                for local_ff in "${LOCAL_FF_DIRS[@]+"${LOCAL_FF_DIRS[@]}"}"; do
                    [ "$local_ff" = "$dir_bn" ] && is_dup=true && break
                done
                [ "$is_dup" = true ] && continue
                ff_desc="$dir_bn"
                [ -f "$ff_dir/forcefield.doc" ] && ff_desc=$(grep -m1 '[^ ]' "$ff_dir/forcefield.doc" 2>/dev/null || echo "$dir_bn")
                NATIVE_FF_DIRS+=("$dir_bn")
                NATIVE_FF_NAMES+=("$ff_desc")
            fi
        done
    fi

    local total_ff=$(( ${#LOCAL_FF_DIRS[@]} + ${#NATIVE_FF_DIRS[@]} ))
    if [ "$total_ff" -eq 0 ]; then log_error "No se encontraron force fields"; exit 1; fi

    local ff_idx=1
    if [ ${#LOCAL_FF_DIRS[@]} -gt 0 ]; then
        echo -e "  ${GREEN}── Force fields locales ──${NC}"
        for i in "${!LOCAL_FF_DIRS[@]}"; do
            echo "   ${ff_idx}) ${LOCAL_FF_NAMES[$i]}  ${CYAN}[${LOCAL_FF_DIRS[$i]}]${NC}"
            ff_idx=$((ff_idx + 1))
        done; echo ""
    fi
    if [ ${#NATIVE_FF_DIRS[@]} -gt 0 ]; then
        echo -e "  ${YELLOW}── Force fields nativos ──${NC}"
        for i in "${!NATIVE_FF_DIRS[@]}"; do
            echo "   ${ff_idx}) ${NATIVE_FF_NAMES[$i]}  ${CYAN}[${NATIVE_FF_DIRS[$i]}]${NC}"
            ff_idx=$((ff_idx + 1))
        done; echo ""
    fi

    echo -e "  ${YELLOW}Seleccione [1-${total_ff}]:${NC}"
    read -r FF_CHOICE
    if [[ "$FF_CHOICE" =~ ^[0-9]+$ ]] && [ "$FF_CHOICE" -ge 1 ] && [ "$FF_CHOICE" -le "$total_ff" ]; then
        if [ "$FF_CHOICE" -le ${#LOCAL_FF_DIRS[@]} ]; then
            SELECTED_FF_DIR="${LOCAL_FF_DIRS[$((FF_CHOICE-1))]}"; SELECTED_FF_LOCAL=true
        else
            local sel=$((FF_CHOICE - ${#LOCAL_FF_DIRS[@]} - 1))
            SELECTED_FF_DIR="${NATIVE_FF_DIRS[$sel]}"; SELECTED_FF_LOCAL=false
        fi
    else
        log_error "Selección inválida"; exit 1
    fi
    log_success "Force field: $SELECTED_FF_DIR (local=$SELECTED_FF_LOCAL)"

    # ── Parámetros de simulación ──
    echo -e "\n${CYAN}═══ PARÁMETROS DE SIMULACIÓN ═══${NC}\n"

    echo "  Tipo de caja:"
    echo "   1) cubic  2) triclinic  3) dodecahedron (recomendado)  4) octahedron"
    echo -e "  ${YELLOW}[1-4] (default: 3):${NC}"
    read -r BOX_CHOICE
    case $BOX_CHOICE in
        1) BOX_TYPE="cubic" ;; 2) BOX_TYPE="triclinic" ;; 4) BOX_TYPE="octahedron" ;;
        *) BOX_TYPE="dodecahedron" ;;
    esac

    echo -e "\n  Distancia a bordes (nm) [default: 1.2]:"
    read -r BOX_DIST; BOX_DIST=${BOX_DIST:-1.2}

    echo -e "\n  Modelo de agua: 1) tip3p  2) spc  3) spce  4) tip4p"
    echo -e "  ${YELLOW}[1-4] (default: 1):${NC}"
    read -r WATER_CHOICE
    case $WATER_CHOICE in
        2) WATER_MODEL="spc"; WATER_FILE="spc216.gro" ;;
        3) WATER_MODEL="spce"; WATER_FILE="spc216.gro" ;;
        4) WATER_MODEL="tip4p"; WATER_FILE="tip4p.gro" ;;
        *) WATER_MODEL="tip3p"; WATER_FILE="spc216.gro" ;;
    esac

    echo -e "\n  Concentración NaCl (mol/L) [default: 0.15]:"
    read -r ION_CONC; ION_CONC=${ION_CONC:-0.15}

    echo -e "\n  Tiempo de metadinámica (ns) [default: 100]:"
    read -r METAD_NS; METAD_NS=${METAD_NS:-100}

    echo -e "  Timestep dt (ps) [default: 0.002]:"
    read -r DT; DT=${DT:-0.002}
    METAD_NSTEPS=$(echo "$METAD_NS * 1000 / $DT" | bc)
    log_info "Producción: $METAD_NS ns | dt=$DT ps | $METAD_NSTEPS steps"

    # ── Variables Colectivas ──
    echo -e "\n${CYAN}═══ VARIABLES COLECTIVAS (CVs) ═══${NC}\n"
    echo "  Tipos disponibles:"
    echo "   1) DISTANCE  — Distancia COM entre dos grupos"
    echo "   2) RMSD      — RMSD respecto a referencia"
    echo "   3) TORSION   — Ángulo dihedral (4 átomos)"
    echo "   4) COORDINATION — Número de coordinación"
    echo ""
    echo -e "  ${YELLOW}Número de CVs [1-3] (default: 2):${NC}"
    read -r N_CVS; N_CVS=${N_CVS:-2}

    CV_TYPES=()
    CV_ATOMS=()
    SIGMAS=()

    for (( cv_i=1; cv_i<=N_CVS; cv_i++ )); do
        echo -e "\n  ${MAGENTA}── CV $cv_i ──${NC}"
        echo -e "  ${YELLOW}Tipo [1-4]:${NC}"
        read -r cv_type_choice
        case $cv_type_choice in
            1) cv_type="distance" ;;
            2) cv_type="rmsd" ;;
            3) cv_type="torsion" ;;
            4) cv_type="coordination" ;;
            *) cv_type="distance" ;;
        esac
        CV_TYPES+=("$cv_type")

        if [ "$cv_type" = "distance" ] || [ "$cv_type" = "coordination" ]; then
            echo -e "  ${YELLOW}Átomos grupo 1 (ej: 1-50 o 1,2,3):${NC}"
            read -r grp1
            echo -e "  ${YELLOW}Átomos grupo 2 (ej: 100-150):${NC}"
            read -r grp2
            CV_ATOMS+=("$grp1")
            CV_ATOMS+=("$grp2")
        elif [ "$cv_type" = "torsion" ]; then
            echo -e "  ${YELLOW}4 átomos del dihedral (ej: 5,7,9,15):${NC}"
            read -r torsion_atoms
            CV_ATOMS+=("$torsion_atoms")
        elif [ "$cv_type" = "rmsd" ]; then
            echo -e "  ${YELLOW}Archivo de referencia PDB [default: reference.pdb]:${NC}"
            read -r ref_file; ref_file=${ref_file:-reference.pdb}
            # Validar que el archivo PDB de referencia existe
            if [ ! -f "$INITIAL_DIR/$ref_file" ] && [ ! -f "$ref_file" ]; then
                log_warning "reference.pdb '$ref_file' no encontrado — asegúrate de copiarlo antes de la producción"
            fi
            CV_ATOMS+=("$ref_file")
        fi

        echo -e "  ${YELLOW}SIGMA para CV $cv_i (ancho de gaussiana) [default: 0.3]:${NC}"
        read -r sigma_val; sigma_val=${sigma_val:-0.3}
        SIGMAS+=("$sigma_val")
    done

    # ── Parámetros WTMetaD ──
    echo -e "\n${CYAN}═══ PARÁMETROS WELL-TEMPERED METAD ═══${NC}\n"
    echo -e "  ${YELLOW}HEIGHT — Altura inicial de gaussianas (kJ/mol) [default: 1.2]:${NC}"
    read -r METAD_HEIGHT; METAD_HEIGHT=${METAD_HEIGHT:-1.2}

    echo -e "  ${YELLOW}PACE — Deposición cada N steps [default: 500 = 1 ps]:${NC}"
    read -r METAD_PACE; METAD_PACE=${METAD_PACE:-500}

    echo -e "  ${YELLOW}BIASFACTOR γ — Factor de temperado [default: 15]:${NC}"
    echo -e "  ${CYAN}(10-20 para la mayoría de sistemas. Más alto = más exploración)${NC}"
    read -r METAD_BIASFACTOR; METAD_BIASFACTOR=${METAD_BIASFACTOR:-15}

    echo -e "  ${YELLOW}Temperatura (K) [default: 300]:${NC}"
    read -r METAD_TEMP; METAD_TEMP=${METAD_TEMP:-300}

    # ── Walls opcionales ──
    echo -e "\n  ${YELLOW}¿Añadir WALLS para confinar exploración? [y/N]:${NC}"
    read -r ADD_WALLS
    WALL_SPECS=()
    if [[ "$ADD_WALLS" =~ ^[yYsS] ]]; then
        for (( cv_i=1; cv_i<=N_CVS; cv_i++ )); do
            echo -e "  ${YELLOW}Lower wall para cv${cv_i}? (valor AT, 0=no) [default: 0]:${NC}"
            read -r lw; lw=${lw:-0}
            if [ "$lw" != "0" ]; then
                WALL_SPECS+=("cv${cv_i},lower,${lw},150")
            fi
            echo -e "  ${YELLOW}Upper wall para cv${cv_i}? (valor AT, 0=no) [default: 0]:${NC}"
            read -r uw; uw=${uw:-0}
            if [ "$uw" != "0" ]; then
                WALL_SPECS+=("cv${cv_i},upper,${uw},150")
            fi
        done
    fi

    # ── Grid para METAD (acelera 5-10x en sims largas) ──
    echo -e "\n${CYAN}═══ GRID PARA METAD (opcional, recomendado) ═══${NC}"
    echo -e "  ${CYAN}Un grid optimiza el lookup de gaussianas de O(N) a O(1).${NC}"
    echo -e "  ${YELLOW}¿Usar grid? [Y/n]:${NC}"
    read -r USE_GRID; USE_GRID=${USE_GRID:-Y}
    GRID_MIN="" GRID_MAX="" GRID_BIN=""
    if [[ ! "$USE_GRID" =~ ^[nN] ]]; then
        echo -e "  ${YELLOW}GRID_MIN por CV (ej: '0.0,0.0' para 2 CVs):${NC}"
        read -r GRID_MIN
        echo -e "  ${YELLOW}GRID_MAX por CV (ej: '5.0,3.14'):${NC}"
        read -r GRID_MAX
        echo -e "  ${YELLOW}GRID_BIN por CV (ej: '200,200'):${NC}"
        read -r GRID_BIN
        log_info "Grid: min=[$GRID_MIN] max=[$GRID_MAX] bin=[$GRID_BIN]"
    fi

    log_success "Configuración completada"
}

#==========================================
# PERSISTENCIA DE CONFIGURACIÓN
#==========================================
save_config() {
    local config_file="$RUNDIR/config.txt"
    cat > "$config_file" <<EOF
# Configuración guardada — $(date)
PROT="$PROT"
LIG="${LIG:-}"
HAS_LIGAND=$HAS_LIGAND
FF_DIR="${FF_DIR:-}"
FF_IS_LOCAL=$FF_IS_LOCAL
SELECTED_FF_DIR="${SELECTED_FF_DIR:-}"
SELECTED_FF_LOCAL=${SELECTED_FF_LOCAL:-false}
BOX_TYPE="$BOX_TYPE"
BOX_DIST="$BOX_DIST"
WATER_MODEL="$WATER_MODEL"
WATER_FILE="$WATER_FILE"
ION_CONC="$ION_CONC"
METAD_NS=$METAD_NS
DT="$DT"
METAD_NSTEPS=$METAD_NSTEPS
METAD_HEIGHT="$METAD_HEIGHT"
METAD_PACE=$METAD_PACE
METAD_BIASFACTOR="$METAD_BIASFACTOR"
METAD_TEMP="$METAD_TEMP"
N_CVS=$N_CVS
EOF
    # Arrays: guardar como string con separador
    declare -p CV_TYPES >> "$config_file"
    declare -p CV_ATOMS >> "$config_file"
    declare -p SIGMAS   >> "$config_file"
    declare -p WALL_SPECS >> "$config_file" 2>/dev/null || true
    log_success "Config guardada: $config_file"
}

#==========================================
# VALIDACIÓN MDP
#==========================================
validate_mdp_files() {
    log_step "Verificando archivos MDP"
    [ ! -d "$BASE_MDP" ] && { log_error "Carpeta '$BASE_MDP/' no encontrada"; exit 1; }
    for mdp in ions.mdp em.mdp nvt.mdp npt.mdp md_metad.mdp; do
        [ ! -f "$BASE_MDP/$mdp" ] && { log_error "$BASE_MDP/$mdp no encontrado"; exit 1; }
        log_success "$mdp ✓"
    done
}

#==========================================
# ESTRUCTURA DE CARPETAS
#==========================================
setup_directory_structure() {
    log_step "Creando estructura de carpetas"
    local name_suffix="${PROT}"
    [ "$HAS_LIGAND" = true ] && name_suffix="${PROT}_${LIG}"
    RUNDIR="$INITIAL_DIR/$WORKDIR/${name_suffix}_$(date +%Y%m%d_%H%M%S)"

    for d in 00_setup 01_minimization 02_nvt 03_npt 04_metadynamics \
             05_analysis 07_plots logs mdp_used; do
        create_dir "$RUNDIR/$d"
    done
    cp "$INITIAL_DIR/$BASE_MDP"/*.mdp "$RUNDIR/mdp_used/"
    log_success "Directorio: $RUNDIR"
}

#==========================================
# SETUP INICIAL
#==========================================
setup_initial_files() {
    log_step "Copiando archivos iniciales"
    cd "$RUNDIR/00_setup" || exit 1

    cp "$INITIAL_DIR/$BASE_PROT/$PROT/"*.pdb . 2>/dev/null || true
    cp "$INITIAL_DIR/$BASE_PROT/$PROT/proteina.gro" . 2>/dev/null || true
    cp "$INITIAL_DIR/$BASE_PROT/$PROT/topol.top" . 2>/dev/null || true
    cp "$INITIAL_DIR/$BASE_PROT/$PROT/posre.itp" . 2>/dev/null || true

    if [ "$HAS_LIGAND" = true ]; then
        for f in ligando.itp ligando.gro ligando.pdb ligando.prm; do
            [ -f "$INITIAL_DIR/$BASE_LIG/$LIG/$f" ] && cp "$INITIAL_DIR/$BASE_LIG/$LIG/$f" .
        done
    fi

    FF_DIR="${SELECTED_FF_DIR:-}"
    FF_IS_LOCAL="${SELECTED_FF_LOCAL:-false}"
    if [ -n "$FF_DIR" ] && [ "$FF_IS_LOCAL" = true ] && [ -d "$INITIAL_DIR/$FF_DIR" ]; then
        cp -r "$INITIAL_DIR/$FF_DIR" .
        log_success "$FF_DIR copiado"
    fi

    adapt_mdp_for_forcefield "$RUNDIR/mdp_used"
    log_success "Archivos iniciales listos"
}

#==========================================
# LIGAND RESTRAINTS
#==========================================
generate_ligand_restraints() {
    [ "$HAS_LIGAND" != true ] && return
    log_step "Generando restraints del ligando"
    cd "$RUNDIR/00_setup" || exit 1

    local last_idx
    last_idx=$(echo q | "$GMX" make_ndx -f ligando.gro 2>&1 | grep -oP '^\s*\K\d+(?=\s)' | tail -1)
    local new_grp=$((last_idx + 1))

    run_gmx make_ndx -f ligando.gro -o lig_noh.ndx &> "$RUNDIR/logs/make_ndx_lig.log" <<EOF
r LIG & !a H*
name ${new_grp} LIG-H
q
EOF

    echo "LIG-H" | run_gmx genrestr -f ligando.gro -n lig_noh.ndx -o posre_ligando.itp \
        -fc 1000 1000 1000 &> "$RUNDIR/logs/genrestr.log"
    log_success "posre_ligando.itp generado"
}

#==========================================
# TOPOLOGÍA
#==========================================
update_topology() {
    [ "$HAS_LIGAND" != true ] && return
    log_step "Actualizando topología para ligando"
    cd "$RUNDIR/00_setup" || exit 1

    # Restraints en ligando.itp
    if ! grep -q "POSRES_LIG" ligando.itp; then
        cat <<'EOF' >> ligando.itp

; Ligand position restraints
#ifdef POSRES_LIG
#include "posre_ligando.itp"
#endif
EOF
        log_success "Restraints añadidos a ligando.itp"
    fi

    cp topol.top topol.top.backup

    # Incluir ligando.prm si existe
    if [ -f "ligando.prm" ]; then
        local ff_line=$(grep -n '#include.*forcefield\.itp"' topol.top | cut -d: -f1)
        if [ -n "$ff_line" ]; then
            sed -i "${ff_line}a\\
; Include ligand parameters\\
#include \"ligando.prm\"" topol.top
        fi
    fi

    # Incluir ligando.itp
    if ! grep -q 'ligando.itp' topol.top; then
        if [ -f "ligando.prm" ]; then
            local insert_line=$(grep -n 'ligando\.prm' topol.top | cut -d: -f1)
        else
            local insert_line=$(grep -n '#include.*forcefield\.itp"' topol.top | cut -d: -f1)
        fi
        if [ -n "$insert_line" ]; then
            sed -i "${insert_line}a\\
; Include ligand topology\\
#include \"ligando.itp\"" topol.top
        fi
    fi

    # Añadir molécula LIG
    if ! grep -q '^LIG' topol.top; then
        if grep -q '^\[ molecules \]' topol.top; then
            local prot_name=$(awk '/^\[ molecules \]/{flag=1; next} flag && NF>0 && !/^;/{print $1; exit}' topol.top)
            sed -i "/^\[ molecules \]/,/^${prot_name}/s/^\(${prot_name}.*\)/\1\nLIG                 1/" topol.top
        fi
    fi
    log_success "Topología actualizada"
}

#==========================================
# CONSTRUCCIÓN DEL SISTEMA
#==========================================
build_system() {
    local STAGE="build_system"
    is_stage_done "$STAGE" && { log_info "Saltando: $STAGE"; return; }
    log_step "Construyendo sistema"
    cd "$RUNDIR/00_setup" || exit 1

    if [ "$HAS_LIGAND" = true ]; then
        run_gmx editconf -f proteina.gro -o prot.gro &> "$RUNDIR/logs/editconf_prot.log"
        run_gmx insert-molecules -f prot.gro -ci ligando.gro -o complex.gro -nmol 1 \
            &> "$RUNDIR/logs/insert_molecules.log"
        log_success "Complejo proteína-ligando ensamblado"
    else
        local pdb_file=$(ls *.pdb 2>/dev/null | head -1)
        if [ -f "proteina.gro" ]; then
            cp proteina.gro complex.gro
        elif [ -n "$pdb_file" ]; then
            cp "$pdb_file" complex.gro
        fi
        log_success "Estructura copiada"
    fi

    # Caja + solvatar
    run_gmx editconf -f complex.gro -o boxed.gro -d "$BOX_DIST" -bt "$BOX_TYPE" \
        &> "$RUNDIR/logs/editconf_box.log"
    run_gmx solvate -cp boxed.gro -cs "$WATER_FILE" -o solv.gro -p topol.top \
        &> "$RUNDIR/logs/solvate.log"
    log_success "Sistema solvatado ($BOX_TYPE, d=$BOX_DIST nm)"

    # Iones
    run_gmx grompp -f "$RUNDIR/mdp_used/ions.mdp" -c solv.gro -p topol.top -o ions.tpr \
        -maxwarn 2 &> "$RUNDIR/logs/grompp_ions.log"
    if (( $(echo "$ION_CONC > 0" | bc -l) )); then
        echo "SOL" | run_gmx genion -s ions.tpr -o system.gro -p topol.top \
            -pname NA -nname CL -neutral -conc "$ION_CONC" &> "$RUNDIR/logs/genion.log"
    else
        echo "SOL" | run_gmx genion -s ions.tpr -o system.gro -p topol.top -neutral \
            &> "$RUNDIR/logs/genion.log"
    fi
    log_success "Iones añadidos ($ION_CONC M)"
    mark_stage_done "$STAGE"
}

#==========================================
# GRUPOS DE ÍNDICE
#==========================================
create_index_groups() {
    local STAGE="index_groups"
    is_stage_done "$STAGE" && { log_info "Saltando: $STAGE"; return; }
    log_step "Creando grupos para termostatos"
    cd "$RUNDIR/00_setup" || exit 1

    local last_idx
    last_idx=$(echo q | "$GMX" make_ndx -f system.gro 2>&1 | grep -oP '^\s*\K\d+(?=\s)' | tail -1)
    local g1=$((last_idx + 1)) g2=$((last_idx + 2))

    if [ "$HAS_LIGAND" = true ]; then
        run_gmx make_ndx -f system.gro -o index.ndx &> "$RUNDIR/logs/make_ndx.log" <<EOF
1 | r LIG
r SOL | r NA | r CL
name ${g1} Protein_Ligand
name ${g2} Solvent
q
EOF
    else
        run_gmx make_ndx -f system.gro -o index.ndx &> "$RUNDIR/logs/make_ndx.log" <<EOF
1
r SOL | r NA | r CL
name ${g1} Protein_Ligand
name ${g2} Solvent
q
EOF
    fi
    log_success "index.ndx creado"
    mark_stage_done "$STAGE"
}

#==========================================
# HELPER: Copiar archivos comunes
#==========================================
copy_common_files() {
    local t="$1"
    cp -f "$RUNDIR/00_setup/topol.top" "$t/"
    cp -f "$RUNDIR/00_setup/index.ndx" "$t/" 2>/dev/null || true
    for itp in "$RUNDIR/00_setup"/*.itp; do [ -f "$itp" ] && cp -f "$itp" "$t/"; done
    if [ "$FF_IS_LOCAL" = true ]; then
        for ff in "$RUNDIR/00_setup"/*.ff; do [ -d "$ff" ] && cp -r "$ff" "$t/"; done
    fi
    [ -f "$RUNDIR/00_setup/ligando.prm" ] && cp -f "$RUNDIR/00_setup/ligando.prm" "$t/"
}

#==========================================
# MINIMIZACIÓN
#==========================================
run_minimization() {
    local STAGE="minimization"
    is_stage_done "$STAGE" && { log_info "Saltando: $STAGE"; return; }
    log_step "Minimización de energía"
    cd "$RUNDIR/01_minimization" || exit 1; copy_common_files .
    cp "$RUNDIR/00_setup/system.gro" .

    run_gmx grompp -f "$RUNDIR/mdp_used/em.mdp" -c system.gro -p topol.top -o em.tpr \
        -maxwarn 2 &> "$RUNDIR/logs/grompp_em.log"
    $MDRUN -deffnm em -v &> "$RUNDIR/logs/mdrun_em.log"
    [ ! -f em.gro ] && { log_error "Minimización falló"; exit 1; }
    log_success "Minimización completada"
    mark_stage_done "$STAGE"
}

#==========================================
# NVT EQUILIBRACIÓN
#==========================================
run_nvt_equilibration() {
    local STAGE="nvt"
    is_stage_done "$STAGE" && { log_info "Saltando: $STAGE"; return; }
    log_step "Equilibración NVT (100 ps)"
    cd "$RUNDIR/02_nvt" || exit 1; copy_common_files .
    cp "$RUNDIR/01_minimization/em.gro" .

    local nvt_mdp="nvt_temp.mdp"
    cp "$RUNDIR/mdp_used/nvt.mdp" "$nvt_mdp"
    sed -i 's/^tc-grps.*/tc-grps                  = Protein_Ligand Solvent/' "$nvt_mdp"
    sed -i "s/^ref_t.*/ref_t                    = ${METAD_TEMP}     ${METAD_TEMP}/" "$nvt_mdp"
    sed -i "s/^gen_temp.*/gen_temp    = ${METAD_TEMP}/" "$nvt_mdp"

    run_gmx grompp -f "$nvt_mdp" -c em.gro -r em.gro -p topol.top \
        -n index.ndx -o nvt.tpr -maxwarn 2 &> "$RUNDIR/logs/grompp_nvt.log"
    $MDRUN -deffnm nvt -v &> "$RUNDIR/logs/mdrun_nvt.log"
    [ ! -f nvt.gro ] && { log_error "NVT falló"; exit 1; }
    log_success "NVT completada"
    mark_stage_done "$STAGE"
}

#==========================================
# NPT EQUILIBRACIÓN
#==========================================
run_npt_equilibration() {
    local STAGE="npt"
    is_stage_done "$STAGE" && { log_info "Saltando: $STAGE"; return; }
    log_step "Equilibración NPT (100 ps)"
    cd "$RUNDIR/03_npt" || exit 1; copy_common_files .
    cp "$RUNDIR/02_nvt/nvt.gro" .
    cp "$RUNDIR/02_nvt/nvt.cpt" .

    local npt_mdp="npt_temp.mdp"
    cp "$RUNDIR/mdp_used/npt.mdp" "$npt_mdp"
    sed -i 's/^tc-grps.*/tc-grps                  = Protein_Ligand Solvent/' "$npt_mdp"
    sed -i "s/^ref_t.*/ref_t                    = ${METAD_TEMP}     ${METAD_TEMP}/" "$npt_mdp"

    run_gmx grompp -f "$npt_mdp" -c nvt.gro -t nvt.cpt -r nvt.gro -p topol.top \
        -n index.ndx -o npt.tpr -maxwarn 2 &> "$RUNDIR/logs/grompp_npt.log"
    $MDRUN -deffnm npt -v &> "$RUNDIR/logs/mdrun_npt.log"
    [ ! -f npt.gro ] && { log_error "NPT falló"; exit 1; }
    log_success "NPT completada"
    mark_stage_done "$STAGE"
}

#==========================================
# GENERAR PLUMED INPUT
#==========================================
generate_plumed_input() {
    local STAGE="plumed_input"
    is_stage_done "$STAGE" && { log_info "Saltando: $STAGE"; return; }
    log_step "Generando plumed.dat"

    local plumed_args=()
    for cv_type in "${CV_TYPES[@]}"; do
        plumed_args+=("--cv-type" "$cv_type")
    done
    for cv_atom in "${CV_ATOMS[@]}"; do
        plumed_args+=("--cv-atoms" "$cv_atom")
    done
    plumed_args+=("--sigma" "${SIGMAS[@]}")
    plumed_args+=("--height" "$METAD_HEIGHT")
    plumed_args+=("--pace" "$METAD_PACE")
    plumed_args+=("--biasfactor" "$METAD_BIASFACTOR")
    plumed_args+=("--temp" "$METAD_TEMP")
    plumed_args+=("--dt" "${DT:-0.002}")
    plumed_args+=("--output" "$RUNDIR/04_metadynamics/plumed.dat")

    for wall in "${WALL_SPECS[@]+"${WALL_SPECS[@]}"}"; do
        plumed_args+=("--wall" "$wall")
    done

    # Grid (mejora O(N) -> O(1) en lookup de gaussianas)
    if [ -n "${GRID_MIN:-}" ] && [ -n "${GRID_MAX:-}" ]; then
        plumed_args+=("--grid-min" "$GRID_MIN")
        plumed_args+=("--grid-max" "$GRID_MAX")
        [ -n "${GRID_BIN:-}" ] && plumed_args+=("--grid-bin" "$GRID_BIN")
        log_info "Grid habilitado: [$GRID_MIN] → [$GRID_MAX]"
    fi

    python3 "$SCRIPT_DIR/generate_plumed.py" "${plumed_args[@]}"

    [ ! -f "$RUNDIR/04_metadynamics/plumed.dat" ] && { log_error "plumed.dat no generado"; exit 1; }
    log_success "plumed.dat generado"
    mark_stage_done "$STAGE"
}

#==========================================
# METADINÁMICA (PRODUCCIÓN)
#==========================================
run_metadynamics() {
    local STAGE="metadynamics"
    is_stage_done "$STAGE" && { log_info "Saltando: $STAGE"; return; }
    log_step "Ejecutando Well-Tempered Metadynamics ($METAD_NS ns)"
    cd "$RUNDIR/04_metadynamics" || exit 1; copy_common_files .
    cp "$RUNDIR/03_npt/npt.gro" .
    cp "$RUNDIR/03_npt/npt.cpt" .

    local md_mdp="md_metad_temp.mdp"
    cp "$RUNDIR/mdp_used/md_metad.mdp" "$md_mdp"
    sed -i "s/^nsteps.*/nsteps                   = ${METAD_NSTEPS}      ; ${METAD_NS} ns/" "$md_mdp"
    sed -i "s/^dt.*/dt                       = ${DT:-0.002}/" "$md_mdp"
    sed -i 's/^tc-grps.*/tc-grps                  = Protein_Ligand Solvent/' "$md_mdp"
    sed -i "s/^ref_t.*/ref_t                    = ${METAD_TEMP}     ${METAD_TEMP}/" "$md_mdp"

    run_gmx grompp -f "$md_mdp" -c npt.gro -t npt.cpt -p topol.top \
        -n index.ndx -o metad.tpr -maxwarn 2 &> "$RUNDIR/logs/grompp_metad.log"

    log_success "Sistema preparado para WTMetaD"
    echo -e "\n${YELLOW}Ejecutando metadinámica (esto puede tardar)...${NC}\n"

    # Si hay HILLS previo (reanudando), activar RESTART en plumed.dat
    if [ -f metad.cpt ] && [ -f HILLS ] && [ -s HILLS ]; then
        log_info "Checkpoint + HILLS detectados — activando RESTART en PLUMED"
        if ! grep -q 'RESTART' plumed.dat; then
            sed -i '/^METAD/a   RESTART=YES' plumed.dat
        fi
        log_info "Continuando simulación desde checkpoint..."
        $MDRUN -deffnm metad -plumed plumed.dat -cpi metad.cpt \
            &> "$RUNDIR/logs/mdrun_metad.log"
    else
        $MDRUN -deffnm metad -plumed plumed.dat \
            &> "$RUNDIR/logs/mdrun_metad.log"
    fi

    [ ! -f HILLS ] && { log_error "HILLS no generado — ¿PLUMED funciona?"; exit 1; }
    log_success "Metadinámica completada"
    log_info "HILLS: $(wc -l < HILLS) líneas"
    log_info "COLVAR: $(wc -l < COLVAR) líneas"
    mark_stage_done "$STAGE"
}

#==========================================
# ANÁLISIS
#==========================================
run_analysis() {
    local STAGE="analysis"
    is_stage_done "$STAGE" && { log_info "Saltando: $STAGE"; return; }
    log_step "Análisis de convergencia y FES"

    local analysis_dir="$RUNDIR/05_analysis"
    local metad_dir="$RUNDIR/04_metadynamics"

    python3 "$SCRIPT_DIR/analyze_convergence.py" \
        --hills "$metad_dir/HILLS" \
        --colvar "$metad_dir/COLVAR" \
        --temp "$METAD_TEMP" \
        --biasfactor "$METAD_BIASFACTOR" \
        --output-dir "$analysis_dir" \
        --n-blocks 10

    log_success "Análisis completado"
    mark_stage_done "$STAGE"
}

#==========================================
# CONTROL DE CALIDAD (QC)
#==========================================
run_qc() {
    local STAGE="qc"
    is_stage_done "$STAGE" && { log_info "Saltando: $STAGE"; return; }
    log_step "Quality Control: energía y RMSD del backbone"

    local qc_dir="$RUNDIR/06_qc"
    local metad_dir="$RUNDIR/04_metadynamics"
    mkdir -p "$qc_dir"

    # Energía: Temperatura, Presión, Potencial
    if [ -f "$metad_dir/metad.edr" ]; then
        log_info "Extrayendo energía..."
        printf 'Temperature\nPressure\nPotential\n0\n' \
            | run_gmx energy -f "$metad_dir/metad.edr" \
                             -o "$qc_dir/energy.xvg" \
                             &> "$RUNDIR/logs/qc_energy.log" || \
            log_warning "gmx energy falló — revisa logs/qc_energy.log"
        log_success "Energía guardada: 06_qc/energy.xvg"
    else
        log_warning "metad.edr no encontrado, saltando QC de energía"
    fi

    # RMSD del backbone proteico
    if [ -f "$metad_dir/metad.xtc" ] && [ -f "$metad_dir/metad.tpr" ]; then
        log_info "Calculando RMSD del backbone..."
        printf 'Backbone\nBackbone\n' \
            | run_gmx rms \
                -s "$metad_dir/metad.tpr" \
                -f "$metad_dir/metad.xtc" \
                -n "$RUNDIR/04_metadynamics/index.ndx" \
                -o "$qc_dir/rmsd_backbone.xvg" \
                -tu ns \
                &> "$RUNDIR/logs/qc_rmsd.log" || \
            log_warning "gmx rms falló — revisa logs/qc_rmsd.log"
        log_success "RMSD guardado: 06_qc/rmsd_backbone.xvg"
    else
        log_warning "Trayectoria no encontrada, saltando QC de RMSD"
    fi

    mark_stage_done "$STAGE"
}

#==========================================
# VISUALIZACIÓN
#==========================================
run_plots() {
    local STAGE="plots"
    is_stage_done "$STAGE" && { log_info "Saltando: $STAGE"; return; }
    log_step "Generando gráficos"

    local plot_dir="$RUNDIR/07_plots"
    local metad_dir="$RUNDIR/04_metadynamics"
    local analysis_dir="$RUNDIR/05_analysis"

    local plot_args=(
        "--colvar" "$metad_dir/COLVAR"
        "--hills" "$metad_dir/HILLS"
        "--output-dir" "$plot_dir"
    )

    # FES si existe
    local fes_file="$analysis_dir/metad_fes.dat"
    [ -f "$fes_file" ] && plot_args+=("--fes" "$fes_file")

    # ΔG si existe
    local dg_file="$analysis_dir/deltaG_vs_time.dat"
    [ -f "$dg_file" ] && plot_args+=("--deltaG" "$dg_file")

    python3 "$SCRIPT_DIR/plot_metad.py" "${plot_args[@]}"

    log_success "Gráficos generados en $plot_dir/"
    mark_stage_done "$STAGE"
}

#==========================================
# RESUMEN FINAL
#==========================================
generate_summary() {
    log_step "Resumen de la simulación"
    echo -e "${GREEN}═══════════════════════════════════════════${NC}"
    echo -e "${GREEN}  SIMULACIÓN COMPLETADA${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════${NC}"
    echo ""
    echo -e "  ${CYAN}Directorio:${NC}  $RUNDIR"
    echo -e "  ${CYAN}Proteína:${NC}    $PROT"
    [ "$HAS_LIGAND" = true ] && echo -e "  ${CYAN}Ligando:${NC}     $LIG"
    echo -e "  ${CYAN}Force field:${NC} ${FF_DIR:-N/A}"
    echo -e "  ${CYAN}Tiempo:${NC}      $METAD_NS ns  |  dt=${DT:-0.002} ps"
    echo -e "  ${CYAN}CVs:${NC}         ${#CV_TYPES[@]} (${CV_TYPES[*]})"
    echo -e "  ${CYAN}Biasfactor:${NC}  $METAD_BIASFACTOR"
    [ -n "${GRID_MIN:-}" ] && echo -e "  ${CYAN}Grid:${NC}        [$GRID_MIN] → [$GRID_MAX]"
    echo ""
    echo -e "  ${BLUE}Archivos clave:${NC}"
    echo "    HILLS:     04_metadynamics/HILLS"
    echo "    COLVAR:    04_metadynamics/COLVAR"
    echo "    FES:       05_analysis/metad_fes.dat"
    echo "    Reporte:   05_analysis/convergence_report.txt"
    echo "    QC:        06_qc/  (energy.xvg, rmsd_backbone.xvg)"
    echo "    Gráficos:  07_plots/"
    echo ""
}

#==========================================
# CLEANUP
#==========================================
cleanup_run() {
    local run_dir="$1"
    if [ ! -d "$run_dir" ]; then
        log_error "Directorio no encontrado: $run_dir"; exit 1
    fi
    log_step "Limpiando archivos temporales en $run_dir"
    find "$run_dir" -name "*.trr" -delete 2>/dev/null && log_success "Eliminados .trr"
    find "$run_dir" -name "*_temp.mdp" -delete 2>/dev/null && log_success "Eliminados MDPs temporales"
    find "$run_dir" -name "#*#" -delete 2>/dev/null && log_success "Eliminados backups"
    log_success "Limpieza completada"
}

#==========================================
# RESUME
#==========================================
resume_run() {
    local run_dir="$1"
    if [ ! -d "$run_dir" ]; then
        log_error "Directorio no encontrado: $run_dir"; exit 1
    fi
    RUNDIR="$run_dir"
    log_info "Reanudando desde: $RUNDIR"
    # Cargar configuración guardada si existe
    if [ -f "$RUNDIR/config.txt" ]; then
        source "$RUNDIR/config.txt"
    fi
}

#==========================================
# MAIN
#==========================================
main() {
    # Parse CLI arguments
    case "${1:-}" in
        --help|-h)    show_help ;;
        --cleanup)    cleanup_run "${2:-}"; exit 0 ;;
        --resume)     resume_run "${2:-}" ;;
        *)
            validate_dependencies
            get_user_input
            validate_mdp_files
            setup_directory_structure
            setup_initial_files
            generate_ligand_restraints
            update_topology
            save_config   # Persiste configuración para --resume
            ;;
    esac

    # Pipeline principal
    build_system
    create_index_groups
    run_minimization
    run_nvt_equilibration
    run_npt_equilibration
    generate_plumed_input
    run_metadynamics
    run_qc            # QC: energy + RMSD backbone
    run_analysis
    run_plots
    generate_summary

    # Reset trap
    trap - EXIT
}

main "$@"
