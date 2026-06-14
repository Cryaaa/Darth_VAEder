# Estructura de datos - Multinucleation Big

## Nomenclatura unificada

Todos los archivos siguen el patrón: `{objetivo}-{condición}-Foto{N}{sufijo}.{ext}`

- **Separador**: `-`
- **Objetivo**: `10X` o `20X`
- **Condiciones**: `CTRL`, `MATURE`, `CMs25d`
- **Foto**: `Foto{N}` (N entero sin zero-padding)
- **Tipos de archivo por foto**:
  - Foto cruda: `{objetivo}-{condición}-Foto{N}.tif` (o `.czi` en N2)
  - Segmentación nuclear: `{objetivo}-{condición}-Foto{N}_NucleiSeg.tif`
  - Segmentación celular: `{objetivo}-{condición}-Foto{N}_cp_masks.png`

## Réplicas biológicas (5 carpetas = 5 réplicas independientes)

| Carpeta | Objetivo | Condiciones | Raw ext |
|---------|----------|-------------|---------|
| ID18    | 10X      | CTRL, MATURE | .tif   |
| ID19    | 20X      | CTRL, MATURE | .tif   |
| ID23    | 10X      | CTRL, MATURE, CMs25d | .tif |
| N2      | 10X      | CTRL, MATURE, CMs25d | .czi |
| N3      | 10X      | CTRL, MATURE, CMs25d | .tif |

## Inventario por carpeta y condición

### ID18 (10X)
- CTRL: Foto1-8 → raw=8, NucleiSeg=8, cp_masks=8 ✓ completo
- MATURE: Foto1-8 → raw=8, NucleiSeg=8, cp_masks=8 ✓ completo

### ID19 (20X)
- CTRL: Foto1-12,14 → raw=13, NucleiSeg=13, cp_masks=13 ✓ completo (nota: no hay Foto13)
- MATURE: Foto1-19 → raw=19, NucleiSeg=19, cp_masks=18 ⚠ falta cp_masks Foto2

### ID23 (10X)
- CTRL: Foto1-2 → raw=2, NucleiSeg=2, cp_masks=2 ✓ completo
- MATURE: Foto1-2 → raw=2, NucleiSeg=2, cp_masks=2 ✓ completo
- CMs25d: Foto1-5 → raw=5, NucleiSeg=0, cp_masks=5 ⚠ sin segmentación nuclear

### N2 (10X)
- CTRL: Foto1-5 → raw=5(.czi), NucleiSeg=5, cp_masks=5 ✓ completo
- MATURE: Foto1-5 → raw=5(.czi), NucleiSeg=5, cp_masks=5 ✓ completo
- CMs25d: Foto1-6 → raw=5(.czi), NucleiSeg=5, cp_masks=5 ⚠ Foto4 sin .czi; Foto6 solo tiene .czi (sin segmentaciones)

### N3 (10X)
- CTRL: Foto1-6 → raw=6, NucleiSeg=6, cp_masks=6 ✓ completo
- MATURE: Foto1-6 → raw=6, NucleiSeg=6, cp_masks=6 ✓ completo
- CMs25d: Foto1-5 → raw=5, NucleiSeg=5, cp_masks=5 ✓ completo

## Faltantes conocidos

1. `ID19/20X-MATURE-Foto2_cp_masks.png` — falta segmentación celular
2. `ID23/10X-CMs25d-Foto{1-5}_NucleiSeg.tif` — falta toda la segmentación nuclear de CMs25d
3. `N2/10X-CMs25d-Foto4.czi` — falta la imagen cruda
4. `N2/10X-CMs25d-Foto6` — solo tiene .czi, sin segmentaciones

## Notas para el pipeline

- Para cuantificación se necesitan las 3 capas: raw + NucleiSeg + cp_masks. Las fotos incompletas deben excluirse o completarse antes.
- ID18 e ID19 no tienen condición CMs25d.
- ID19 es la única réplica con objetivo 20X (las demás son 10X).
- N2 tiene raws en .czi en vez de .tif — el pipeline debe soportar ambos formatos o convertir previamente.
