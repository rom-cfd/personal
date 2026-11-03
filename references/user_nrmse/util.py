import re
import numpy as np
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from skimage.metrics import structural_similarity as ssim


PALETTE = [
    'k',        # black
    '#1f77b4',  # blue
    '#d62728',  # red
    '#2ca02c',  # green
    '#9467bd',  # purple
    '#ff7f0e',  # orange
    '#17becf',  # cyan
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # gray
    '#bcbd22',  # olive
    '#006D77',  # deep-teal
]

__all__ = [
    'PALETTE',
    'find_anchor_upwards',
    'extract_grid_from_path',
    'extract_ann_type_from_path',
    'short_path',
    'load_tec_ascii_fromstring',
    'figsize_from_px',
    'make_fig_with_inner_size',
    'plot_1D',
    'compare_species_1d_by_lists',
    '_grid_key',
    '_subset_method_from_group_paths',
    '_ylabel_for_method',
    'chem_to_tex_label',
    'extract_ann_model_type_from_path',
    'extract_model_type_from_path',
    'species_color',
    'place_info_text_auto',
]


def find_anchor_upwards(start: Path, anchor_name: str) -> Path:
    p = start.resolve()
    for q in [p] + list(p.parents):
        if (q / anchor_name).is_dir():
            return q / anchor_name
    raise RuntimeError(f"Could not find '{anchor_name}' folder in parent paths. Start: {p}")


def extract_grid_from_path(path: Path) -> str:
    """Return grid info like '640' from path name such as 'dns_640'."""
    m = re.search(r'(\d+)', path.name)
    if not m:
        raise ValueError(f"Cannot find grid info in: {path}")
    return m.group(1)


def extract_ann_type_from_path(path: Path) -> str:
    """
    Return high-level ANN type (e.g., 'max_value', 'max_value_manual', 'max_value_width').
    Raise ValueError if not an ANN folder (defensive).
    """
    name = path.name.lower()
    if name.startswith("dns"):
        raise ValueError(f"ANN path expected, got: {path.name}")
    else:
        return path.parent.parent.name


def short_path(filepath: Path, tail=3) -> str:
        path=Path(filepath)
        parts = path.parts
        return "/".join(parts[-tail:])


def _parse_header(lines, filepath, data_dimension):
    """
    Parse VARIABLES block, detect packing (BLOCK/POINT), read I/J/K,
    and locate the first data line.
    """
    # VARIABLES block (collect multiline and normalize smart quotes)
    
    low_line = lambda s: s.strip().lower()
    
    i = 0
    while i < len(lines) and not low_line(lines[i]).startswith("variables"):
        i += 1
    if i >= len(lines):
        raise ValueError("VARIABLES not found")
    
    j = i + 1
    while j < len(lines):
        lj = low_line(lines[j])
        
        if lj.startswith("zone"):
            break

        if lj.startswith(("title", "variables")):
            raise ValueError(f"Unexpected header between VARIABLES and ZONE: {lines[j].strip()}")
            
        j += 1

    if j >= len(lines):
        raise ValueError("ZONE not found after VARIABLES")
        
    var_block = " ".join(lines[i:j])   # var_block is from the "variable =" line to the line before "zone"

    nl=j-i-1

    def _normalize_variables_block(var_block: str, *, sep_mode="space") -> list[str]:
        """
        Normalize quotes in VARIABLES block and robustly extract variable names.
        Handles smart quotes, mixed closing quotes, and adjacent quotes like "x""y".
        Returns a clean list of variable names.
        """
        trans = str.maketrans({
            "“": '"', "”": '"', "‟": '"', "＂": '"',   # smart/double variants -> "
            "‘": "'", "’": "'", "‛": "'", "＇": "'",   # smart/single variants -> '
        })
        s = var_block.translate(trans)

        comma_count = s.count(",")
        sep_mode = "comma" if comma_count >= 1 else "space"
    
        # "x""y" -> "x", "y"
        if sep_mode == "comma":
            s = re.sub(r'""', r'", "', s)  # Insert comma
        elif sep_mode == "space":
            s = re.sub(r'""', r'" "', s)   # Insert space
        else:
            raise ValueError("sep_mode must be 'comma' or 'space'")
    
        # Mixed closing: "Y CO' -> "Y CO"
        s = re.sub(r'"([^"]+)\'+(?=\s|")', r'"\1"', s)
    
        # Extract by double quotes (after normalization)
        vars_list = [n.strip() for n in re.findall(r'"([^"]+)"', s)]
    
        # As a safety net, strip stray single-quotes at both ends (keep inner apostrophes)
        vars_list = [re.sub(r"^'+|'+$", "", v) for v in vars_list]
    
        return vars_list

    
    vars_list = _normalize_variables_block(var_block)


    if len(vars_list) == 0:
        raise ValueError("NO variables found between VARIABLES and ZONE")
    elif len(vars_list) < nl:
        raise ValueError("The number of variables is less than expected")

    print(f"{short_path(filepath)} has {len(vars_list)} variables")
        

    # ZONE line
    z=j
    header = "\n".join(lines[:z+1])
    # packing
    low_header = header.lower()
    if "datapacking=block" in low_header or "f=block" in low_header:
        packing = "BLOCK"
    elif "datapacking=point" in low_header or "f=point" in low_header:
        packing = "POINT"
    else:
        raise ValueError(f"{short_path(filepath)} : DATAPACKING type not found in header")

    # I, J, K
    dim = data_dimension
    if dim not in {"1D", "2D", "3D"}:
        raise ValueError(f"{short_path(filepath)}: data dimension must be '1D'|'2D'|'3D', got {data_dimension!r}")
        
    def _find_int(tag):
        m = re.search(rf"\b{tag}\s*=\s*(\d+)", header, flags=re.I)
        return int(m.group(1)) if m else None
    I, J, K = _find_int("I"), _find_int("J"), _find_int("K")

    if dim == "1D":
        if I is None:
            raise ValueError(f"{short_path(filepath)}: Missing grid info - 'I' is required in ZONE header for 1D data")
    elif dim == "2D":
        if I is None or J is None:
            raise ValueError(f"{short_path(filepath)}: Missing grid info - 'I' and 'J' are required in ZONE header for 2D data")
    else:
        if I is None or J is None or K is None:
            raise ValueError(f"{short_path(filepath)}: Missing grid info - 'I' and 'J' and 'K' are required in ZONE header for 3D data")
                

    # start of data: first line after ZONE that likely contains numeric text
    if z < len(lines):
        start = z+1
    else:
        raise ValueError(f"{short_path(filepath)}: Numerical data not found after ZONE header")
    digit_like = re.compile(r"^[\s0-9eEdD\+\-\,\.\t nNaAiIfF]+$")
    while start < len(lines) and not digit_like.match(lines[start]):
        start += 1

    if start >= len(lines):
        raise ValueError(f"{short_path(filepath)}: No numeric-looking line found after ZONE header")

    print(f"{short_path(filepath)}: ZONE line is '{j}', and numerical data start line is '{start}'")

    return vars_list, packing, (I, J, K), start


def _grab_numbers_via_fromstring(lines, start_line):
    """
    Fast and robust numeric extraction using numpy.fromstring (C parser).
    Minimal normalization:
      - Unicode minus (U+2212) -> '-'
      - Fortran D/d exponent -> 'e'
      - Commas -> spaces (defensive for CSV/thousands separators)
    If expected_count is given, truncate to that many tokens.
    """
    data_text = "\n".join(lines[start_line:])
    norm = (data_text
            .replace("\u2212", "-")        # Unicode minus
            .replace("D", "e").replace("d", "e")  # Fortran exponent
            .replace(",", " "))            # commas -> spaces
    nums = np.fromstring(norm, sep=" ")
    return nums


def detect_coord_indices_from_header(vars_list, data_dimension: str, file_path: Path):
    """
    Based on the header's data_dimension ('1D'|'2D'|'3D'),
    assuming the 'first n' variables in the variable list are coordinates,
    check whether their names are within {x,y,z} and are distinct.
    Returns: (coord_indices_tuple, coord_labels_tuple)
    Example: ['y','z'] in 2D -> ((0,1), ('y','z'))
    """
    dim = str(data_dimension).strip().upper()
    need_n = {'1D': 1, '2D': 2, '3D': 3}.get(dim)
    file_path=Path(file_path)
    if need_n is None:
        raise ValueError(f"data_dimension must be '1D'|'2D'|'3D', got {data_dimension!r}")
    if len(vars_list) < need_n:
        raise ValueError(f"{short_path(file_path)}: not enough variables for {dim}: got {len(vars_list)}, need {need_n}")

    # Check only the first n variables
    names = [vars_list[i].strip().lower() for i in range(need_n)]
    allowed = {"x","y","z"}

    # All must belong to the allowed set and there must be no duplicates
    if not all(n in allowed for n in names):
        bad = [vars_list[i] for i,n in enumerate(names) if n not in allowed]
        raise ValueError(f"{short_path(file_path)}: coordinate names must be among x/y/z (got {bad})")

    if len(set(names)) != len(names):
        raise ValueError(f"{short_path(file_path)}: duplicate coordinate names in first {need_n}: {names}")

    # The index is fixed at 0..need_n-1, and the label reflects the order given by the file
    coord_indices = tuple(range(need_n))
    coord_labels  = tuple(names)
    return coord_indices, coord_labels


def _actual_dim_from_header(I,J,K):
    for name, val in (('I', I), ('J', J), ('K', K)):
        if val is not None and val < 1:
            raise ValueError(f"{name} must be >= 1 (got {val})")
    d = 1
    if J is not None and J > 1:
        d = 2
    if K is not None and K > 1:
        d = 3
    return f"{d}D"


def load_tec_ascii_fromstring(filepath: str,
                              data_dimension: str,               # '1D' | '2D' | '3D'
                             ):
    """
    Load Tecplot ASCII and reshape into (ny, nx) grids keyed by variable names.
    Numeric extraction uses numpy.fromstring (fast, edge-case friendly).
    Assumptions:
      - 2D ordered zone; first two variables are X and Y
      - DATAPACKING may be BLOCK or POINT (unspecified treated as POINT)
      - Species mass fractions may be prefixed with "Y "
    """
    # 1) Read file & parse header
    try:
        text = Path(filepath).read_text(encoding="utf-8", errors="strict")
    except UnicodeDecodeError as e:
        raise ValueError(f"{short_path(filepath)}: encoding error - {e}") from None

    lines = text.splitlines()
    dim = str(data_dimension).strip().upper()
        
    vars_list, packing, (I, J, K), start = _parse_header(lines, filepath, dim)
    M = len(vars_list)

    actual_dim = _actual_dim_from_header(I, J, K)
    order = {'1D': 1, '2D': 2, '3D': 3}
    if order[dim] != order[actual_dim]:
        raise ValueError(f"{short_path(filepath)}: Please check the data dimension values.\n"
        "The dimension values in the data header and the data dimension values you entered do not match!\n"
        f"got: {dim}, header: {actual_dim}")

    # 2) Expected count from header (if present) -> extra-safety trimming
    if dim == "1D":
        expected_count = I * M
        print(f"{short_path(filepath)}: file has '{I}' data in 1D")
    elif dim == "2D":
        expected_count = I * J * M
        print(f"{short_path(filepath)}: file has '{I}' x '{J}' data in 2D")
    elif dim == "3D":
        expected_count = I * J * K * M
        print(f"{short_path(filepath)}: file has '{I}' x '{J}' x '{K}' data in 3D")
    else:
        raise ValueError(f"data dimension must be '1D' | '2D' | '3D'")

    # 3) Extract numbers (fast C parser)
    nums = _grab_numbers_via_fromstring(lines, start)

    # 4) number of data checks
    L = len(nums)
    if L != expected_count:
        raise ValueError(f"{short_path(filepath)}: numeric count mismatch; header expects {expected_count}, got {L}")

    N = expected_count // M


    # 5) Arrange data according to the shape described in the header by packing
    if packing == "BLOCK":
        try:
            blocks = np.vstack([nums[m*N:(m+1)*N] for m in range(M)])
            arr = blocks.T
        except Exception as e:
            raise ValueError(f"{short_path(filepath)}: Failed BLOCK reshaping ({e})")
        r = min(3,M)
        if not np.array_equal(arr[:,:r].ravel(order="F"), nums[:r*N]):
            raise AssertionError(f"{short_path(filepath)}: BLOCK reshape mismatch for first {r} variables")
            
    elif packing == "POINT":
        try:
            arr = nums.reshape(N, M)
        except Exception as e:
            raise ValueError(f"{short_path(filepath)}: Failed POINT reshaping ({e})")
        r = min(3,N)
        if not np.array_equal(arr[:r, :].ravel(order="C"), nums[:r*M]):
            raise AssertionError(f"{short_path(filepath)}: POINT reshape order mismatch for first {r} points")
        

    # 6) Grid reconstruction (no sorting):
    #    - Detect coordinate columns (x/y/z) from header (must be the first N columns)
    #    - Reshape to (J, I) or (K, J, I) using header I/J/K (row-major, I fastest)
    #    - Verify rectilinearity (X row-constant, Y col-constant; 3D: axis-specific invariance)
    #    - Enforce strict monotonicity along the varying axis of each coordinate
    (coord_idx, coord_labels) = detect_coord_indices_from_header(vars_list, dim, filepath)

    def _to_grid(data, data_dimension, I, J, K):
        if data_dimension == "1D":
            return data
        elif data_dimension == "2D":
            return data.reshape(J, I, order='C')    # I fastest
        else:
            return data.reshape(K, J, I, order='C')
    
    if dim == '1D':
        (ix,), (ax,) = coord_idx, coord_labels  # ax is 'x' or 'y'/'z'
        if ix != 0:
            raise ValueError(f"Coordinate columns must be the first ones in the file: expected {ax} to occupy the first column")
            
        x = _to_grid(arr[:, ix], dim, I, J, K)
        # Monotonic increase (strict)
        if not (np.diff(x) > 0).all():
            raise ValueError(f"{short_path(filepath)}: 1D {ax} not strictly increasing")
        coords = {
            ax: x,
            'info': [ax]
        }  # Save the coordinate name written in the file as a key
    
    elif dim == '2D':
        (ix, iy), (ax, ay) = coord_idx, coord_labels  # ex: ('y','z')
        if ix != 0:
            raise ValueError(f"Coordinate columns must be the first ones in the file: expected {ax} to occupy the first column")
        if iy != 1:
            raise ValueError(f"Coordinate columns must be the first ones in the file: expected {ay} to occupy the second column")
            
        x = arr[:, ix]; y = arr[:, iy]
        X = _to_grid(x, dim, I, J, K)
        Y = _to_grid(y, dim, I, J, K)
    
        # Orthogonality check
        if not (X == X[0:1, :]).all():
            raise ValueError(f"{short_path(filepath)}: {ax.upper()} grid not rectilinear across rows")
        if not (Y == Y[:, 0:1]).all():
            raise ValueError(f"{short_path(filepath)}: {ay.upper()} grid not rectilinear across columns")
        if not (np.diff(X[0, :]) > 0).all():
            raise ValueError(f"{short_path(filepath)}: {ax} not strictly increasing along I")
        if not (np.diff(Y[:, 0]) > 0).all():
            raise ValueError(f"{short_path(filepath)}: {ay} not strictly increasing along J")
    
        coords = {
            ax: X, ay: Y,
            'info': [ax, ay]
        }   # Save the coordinate name written in the file as a key and Save axis name sequence chart
    
    else:  # '3D'
        (ix, iy, iz), (ax, ay, az) = coord_idx, coord_labels
        if ix != 0:
            raise ValueError(f"Coordinate columns must be the first ones in the file: expected {ax} to occupy the first column")
        if iy != 1:
            raise ValueError(f"Coordinate columns must be the first ones in the file: expected {ay} to occupy the second column")
        if iz != 2:
            raise ValueError(f"Coordinate columns must be the first ones in the file: expected {az} to occupy the third column")
            
        x = arr[:, ix]; y = arr[:, iy]; z = arr[:, iz]
        X = _to_grid(x, dim, I, J, K)
        Y = _to_grid(y, dim, I, J, K)
        Z = _to_grid(z, dim, I, J, K)
        # If necessary, perform only a few orthogonal and monotonic checks with k-slice samples (saving computation) -> to be improved later
        # X change in i-direction only
        if not (np.diff(X[0,0,:]) > 0).all(): raise ValueError("X not strictly increasing along I")
        if not (X == X[0:1,0:1,:]).all(): raise ValueError("X varies along J/K (shouldn’t)")
        
        # Y change in j-direction only
        if not (np.diff(Y[0,:,0]) > 0).all(): raise ValueError("Y not strictly increasing along J")
        if not (Y == Y[0:1,:,0:1]).all(): raise ValueError("Y varies along I/K (shouldn’t)")
        
        # Z change in k-direction only
        if not (np.diff(Z[:,0,0]) > 0).all(): raise ValueError("Z not strictly increasing along K")
        if not ((Z == Z[:,0:1,0:1])).all(): raise ValueError("Z varies along I/J (shouldn’t)")

        coords = {
            ax: X, ay: Y, az: Z,
            'info': [ax, ay, az]
        }

    # 7) Variable classification (light heuristics)
    fields, species = {}, {}
    for k, raw in enumerate(vars_list):
        if k < order[dim]:
            continue
        name = raw.strip()
        low = name.lower()
        Z = _to_grid(arr[:, k], dim, I, J, K)

        if low.startswith("y "):                     # e.g., "Y H2" -> species["H2"]
            species[name[2:].strip()] = Z
        elif low in ("t (k)", "temperature", "t"):
            fields["T"] = Z
        elif low.startswith("u"):
            fields["u"] = Z
        elif low.startswith("v") and "vort" not in low:
            fields["v"] = Z
        elif low.startswith("w"):
            fields["w"] = Z
        elif "vort" in low:
            fields["vort"] = Z
        else:
            fields[name] = Z

    meta = {
        "vars": vars_list,
        "packing": packing,
        "dimension": dim,
        "N": N, "numbers_total": L,
        "I": I, "J": J, "K": K,
        "data info": short_path(filepath),
    }
    return {"coords": coords, "fields": fields, "species": species, "meta": meta}


def figsize_from_px(w_px: int, h_px: int, dpi: int = 300):
    return (w_px / dpi, h_px / dpi)


def make_fig_with_inner_size(fig_w, fig_h, dpi,
                             left_in=1.6, right_in=0.6,
                             top_in=0.6, bottom_in=1.2,
                             symmetric_lr=False):
    """
    fig_w, fig_h : inner axes area's width/height (inches)
    left_in, right_in, top_in, bottom_in : empty space size (inches)
    if symmetric_lr=True, right_in = left_in
    """
    if symmetric_lr:
        right_in = left_in

    total_w_in = left_in + fig_w + right_in
    total_h_in = bottom_in + fig_h + top_in

    fig = plt.figure(figsize=(total_w_in, total_h_in), dpi=dpi)

    ax_left = left_in / total_w_in
    ax_bottom = bottom_in / total_h_in
    ax_width = fig_w / total_w_in
    ax_height = fig_h / total_h_in

    ax = fig.add_axes([ax_left, ax_bottom, ax_width, ax_height])
    return fig, ax


def chem_to_tex(name: str) -> str:
    # Convert consecutive digits to _{...}: H2O -> H_{2}O, NNH remains unchanged
    tex = re.sub(r'(\d+)', r'_{\1}', name)
    # Force upright roman via \mathrm{} (for consistent Times in math mode)
    return rf'\mathrm{{{tex}}}'


def plot_1D(grid, data, label, fig_px, fig_dpi):
    W_px, H_px = int(round(fig_px[0])), int(round(fig_px[1]))
    fig_w, fig_h = figsize_from_px(W_px, H_px, fig_dpi)
    fig, ax = make_fig_with_inner_size(
            fig_w, fig_h, fig_dpi,
            left_in=2.1, right_in=0.3, top_in=0.3, bottom_in=1.0, symmetric_lr=False
        )
    x=grid
    y=data
    label=chem_to_tex(label)
    ax.set_xlabel(r'$\boldsymbol{y}$ [\si{cm}]', fontdict={'family': 'Times New Roman', 'size': 36})
    ax.set_ylabel(ylabel = rf'$\boldsymbol{{Y}}_{{{label}}}$', fontdict={'family': 'Times New Roman', 'size': 36}, labelpad=10)
    ax.tick_params(axis='both', which='major', direction='in', length=10, width=2.0, labelsize=26)
    ax.grid(False)
    ax.plot(x, y, linestyle=(0, ()), linewidth=5, color="k", label=label)
    plt.show()


def _auto_atol_from_steps(xf, xc, atol_scale=1e-6):
    def _min_step_1d(a):
        d = np.diff(a)
        d = d[np.isfinite(d) & (np.abs(d) > 0)]
        if d.size == 0:
            raise ValueError("No positive finite step found in axis (degenerate or constant grid).")
        return float(np.min(np.abs(d)))

    step_candidates = [
        _min_step_1d(xf),
        _min_step_1d(xc),
    ]
    base_step = min(step_candidates)
    return base_step * float(atol_scale)


def _can_stride_downsample(
    xf, xc,
    *,
    rtol=0.0,
    atol=None,
    atol_scale=1e-6,
    debug=False
):

    xf = np.asarray(xf, np.float64)
    xc = np.asarray(xc, np.float64)
    nxf = xf.size
    nxc = xc.size

    # integer ratio check
    if (nxf % nxc) != 0:
        if debug:
            print(f"[can_stride] not integer ratio: (nxf)=({nxf}) vs (nxc)=({nxc})")
        return False, None, None

    fx = nxf // nxc

    # absolute tolerance auto estimate
    if atol is None:
        atol = _auto_atol_from_steps(xf, xc, atol_scale=atol_scale)

    # try all offsets
    found = False
    ox = None
    for ox_try in range(fx):
        if np.allclose(xf[ox_try::fx], xc, rtol=rtol, atol=atol):
            found = True
            ox = ox_try
            break

    if debug:
        if found:
            print(f"[can_stride] success: fx={fx}, ox={ox}, rtol={rtol}, atol={atol}")
        else:
            # diagnostics: show the best (smallest) max-abs-diff candidate
            def _maxdiff(a, b):
                return float(np.max(np.abs(a - b)))
            best_xdiff = None; best_ox = None
            for ox_try in range(fx):
                if xf[ox_try::fx].size == xc.size:
                    d = _maxdiff(xf[ox_try::fx], xc)
                    if best_xdiff is None or d < best_xdiff:
                        best_xdiff, best_ox = d, ox_try
            print(f"[can_stride] fail: fx={fx}, rtol={rtol}, atol={atol}, "
                  f"best_ox={best_ox}, best_x_maxΔ={best_xdiff}")

    return (True, fx, ox) if found else (False, fx, None)


def _interp_to_grid(Z_src, X_src, X_tgt):
    """
    Strict interpolation:
      - Source/Target axes must be strictly increasing (no sorting/reindexing).
      - If target is outside source domain, raise early ValueError (no extrapolation).
    """
    # Common helper: Strict monotonicity verification (tells you where it breaks)
    def _assert_strict_increasing(axis_1d, name="axis"):
        d = np.diff(axis_1d)
        if not np.all(d > 0):
            bad = int(np.where(~(d > 0))[0][0])
            raise ValueError(
                f"[interp] {name} not strictly increasing at idx={bad}: "
                f"{name}[{bad}]={axis_1d[bad]}, {name}[{bad+1}]={axis_1d[bad+1]}"
            )

    # source axis
    xf = np.asarray(X_src, np.float64)
    Z = np.asarray(Z_src, np.float64)
    if Z.shape != xf.shape:
        raise ValueError(f"[interp] Z.shape {Z.shape} != grid shape {xf.shape}")

    # Source axis: Strict monotonic increasing verification
    _assert_strict_increasing(xf, "X_src")

    # Target axis
    xt = np.asarray(X_tgt, np.float64)

    # Target axis: Strict monotonic increasing verification
    _assert_strict_increasing(xt, "X_tgt")

    # Range check: Since monotonic increase is guaranteed, only the two end values are compared (same as minimum/maximum values)
    x_min, x_max = float(xf[0]), float(xf[-1])

    xt_min, xt_max = float(xt[0]), float(xt[-1])

    if not (x_min <= xt_min and xt_max <= x_max):
        raise ValueError(
            f"[interp] target grid outside source domain: "
            f"X_tgt∈[{xt_min},{xt_max}] vs X_src∈[{x_min},{x_max}]"
        )

    # Interpolation (no sorting/reindexing, no extrapolation)
    rgi = RegularGridInterpolator((xf,), Z, method="linear", bounds_error=True)
    Zi = rgi(xt).reshape(X_tgt.shape)
    return Zi


def _align_1d(
    A0, X0, A1, X1,
    *,
    align_mode="coarse",
    stride_rtol=0.0,
    stride_atol=None,
    stride_atol_scale=1e-6,
    debug_stride=False
):
    """
    Align two 1D fields to a common grid.

    Policy:
      - If shapes & coords match (allclose): return as-is.
      - Else choose the target grid:
          * 'coarse' -> grid with fewer points (min nx)
          * 'fine'   -> grid with more points  (max nx)
      - If target is 'coarse' and the other grid is an exact integer multiple,
        try stride-downsample with offsets (no interpolation) under rtol/atol.
      - Otherwise use linear interpolation to the chosen target grid.

    Returns: A0_aligned, A1_aligned, X_tgt, info_dict
    """
    X0 = np.asarray(X0, np.float64);  A0 = np.asarray(A0, np.float64)
    X1 = np.asarray(X1, np.float64);  A1 = np.asarray(A1, np.float64)

    # Same grid? (shape + coordinate arrays within tolerance)
    if X0.shape == X1.shape:
        if np.allclose(X0, X1, rtol=0, atol=1e-12):
            return A0.copy(), A1.copy(), X0.copy(), {"mode":"as_is"}
        else:
            raise ValueError(f"The number of grids is the same, but the actual grid values are different")

    nx0 = X0.size
    nx1 = X1.size
    if align_mode == "coarse":
        take_0 = nx0 <= nx1
    elif align_mode == "fine":
        take_0 = nx0 >= nx1
    else:
        raise ValueError("align_mode must be 'coarse' or 'fine'")

    if take_0:
        X_tgt = X0
        # If A0's grid > A1's grid, the remainder is not 0, so ok automatically becomes False
        ok, fx, ox = _can_stride_downsample(
            X1, X0, rtol=stride_rtol, atol=stride_atol,
            atol_scale=stride_atol_scale, debug=debug_stride
        )
        if ok and align_mode == "coarse":
            A1a = A1[ox::fx]
            return A0.copy(), A1a, X_tgt.copy(), {
                "mode":"stride", "fx":fx, "ox":ox, "target":"0"
            }
        else:
            A1a = _interp_to_grid(A1, X1, X_tgt)
            return A0.copy(), A1a, X_tgt.copy(), {"mode":"interp", "target":"0"}
    else:
        X_tgt = X1
        ok, fx, ox = _can_stride_downsample(
            X0, X1, rtol=stride_rtol, atol=stride_atol,
            atol_scale=stride_atol_scale, debug=debug_stride
        )
        if ok and align_mode == "coarse":
            A0a = A0[ox::fx]
            return A0a, A1.copy(), X_tgt.copy(), {
                "mode":"stride", "fx":fx, "ox":ox, "target":"1"
            }
        else:
            A0a = _interp_to_grid(A0, X0, X_tgt)
            return A0a, A1.copy(), X_tgt.copy(), {"mode":"interp", "target":"1"}


def _ssim_1d(A, B, *, gaussian_weights=True, win_size="basic"):
    """Return 1 - SSIM for 1D arrays (same shape)."""
    A = np.asarray(A, np.float64); B = np.asarray(B, np.float64)
    dmin = float(min(A.min(), B.min())); dmax = float(max(A.max(), B.max()))
    data_range = dmax - dmin
    if data_range < 0:
        raise ValueError("data range is strange!!")
    elif data_range == 0:
        return 0.0

    if win_size == "basic":
        if gaussian_weights == "basic":
            val = ssim(A, B, data_range=data_range, use_sample_covariance=False, channel_axis=None)
        else:
            val = ssim(A, B, data_range=data_range, gaussian_weights=gaussian_weights, use_sample_covariance=False, channel_axis=None)
    else:
        if gaussian_weights == "basic":
            val = ssim(A, B, data_range=data_range, win_size=win_size, use_sample_covariance=False, channel_axis=None)
        else:
            val = ssim(A, B, data_range=data_range, gaussian_weights=gaussian_weights, win_size=win_size, use_sample_covariance=False, channel_axis=None)
    return float(1.0 - val)


def _nrmse_1d(A, B, *, norm="rms"):
    """
    Normalized RMSE:
      - 'rms'     : RMSE(A,B) / RMS(A)        = RMSE / sqrt(mean(A**2))
      - 'min-max' : RMSE(A,B) / (max(A,B)-min(A,B))
      - 'std'     : RMSE(A,B) / std(A)
      - 'mean'    : RMSE(A,B) / mean(|A|)
    """
    # The data that serves as the basis for comparison is A
    A = np.asarray(A, np.float64); B = np.asarray(B, np.float64)
    dmin = float(min(A.min(), B.min())); dmax = float(max(A.max(), B.max()))
    data_range = dmax - dmin
    if data_range < 0:
        raise ValueError("data range is strange!!")
    elif data_range == 0:
        return 0.0
        
    mse = np.mean((A - B)**2)
    rmse = float(np.sqrt(mse))
    if norm == "rms":
        denom = float(np.sqrt(np.mean(A**2)))
    elif norm == "min-max":
        dmin = float(min(A.min(), B.min())); dmax = float(max(A.max(), B.max()))
        denom = dmax - dmin
    elif norm == "std":
        denom = float(np.std(A))
    elif norm == "mean":
        denom = float(np.mean(np.abs(A)))
    else:
        raise ValueError("norm must be 'rms', 'min-max', 'std', or 'mean'")

    if denom == 0:
        raise ValueError("ref data value is 0")
    return float(rmse / denom)


def compare_1d_field(
    A0, grid0, A1, grid1,
    grid_info,
    *,
    metric="SSIM",            # 'SSIM' | 'NRMSE'
    align_mode="coarse",      # 'coarse' (fewer points) | 'fine' (denser)
    gaussian_weights="basic",
    win_size="basic",
    nrmse_norm="rms",        # Select the type of denominator to be used in the nrmse calculation   'rms' | 'min-max' | 'std' | 'mean'
    nrmse_ref="A",           # Select data A or B to use as a reference when calculating nrmse
    # stride params:
    stride_rtol=0.0,
    stride_atol=None,
    stride_atol_scale=1e-6,
    debug_stride=False
):
    """
    The two 1D fields (A0, A1) are aligned to their respective coordinates (grid0, grid1), and the similarity is calculated using a metric selected from SSIM, NRMSE.
    The coordinate alignment automatically selects stride (integer ratio) or interpolation (interp).
    The coordinate names are specified as grid_info = (ax0, ax1).
    
    Note:
    - A0, A1 and X must all be one-dimensional.
    - An error occurs if there are NaN/Inf values.
    - The axis names (ax0) follow the strings defined in grid_info.
    """
    ax0, ax1 = grid_info  # ex: ('y','y')
    if not (isinstance(grid0, dict) and ax0 in grid0):
        raise ValueError(f"[compare_1d_field] grid0 must have keys {{{ax0!r}}}")
    if not (isinstance(grid1, dict) and ax1 in grid1):
        raise ValueError(f"[compare_1d_field] grid1 must have keys {{{ax1!r}}}")

    A0 = np.asarray(A0, np.float64)
    A1 = np.asarray(A1, np.float64)

    if ax0 != ax1:
        raise ValueError(f"The axes must match: A axis={ax0}, B axis={ax1}")

    X0 = np.asarray(grid0[ax0], np.float64)
    X1 = np.asarray(grid1[ax1], np.float64)

    if not (A0.ndim == 1 and A1.ndim == 1):
        raise ValueError(f"Input data type must be 1D arrays")

    if not (X0.shape == A0.shape):
        raise ValueError(f"[compare_1d_field] grid0 shapes X0={X0.shape}, must match A0={A0.shape}")
    if not (X1.shape == A1.shape):
        raise ValueError(f"[compare_1d_field] grid1 shapes X0={X1.shape}, must match A1={A1.shape}")

    if not (np.isfinite(A0).all() and np.isfinite(A1).all()):
        raise ValueError("[compare_1d_field] A0/A1 contain NaN/Inf")
    if not (np.isfinite(X0).all() and np.isfinite(X1).all()):
        raise ValueError("[compare_1d_field] grids contain NaN/Inf")

    
    A0a, A1a, Xt, info = _align_1d(
        A0, X0, A1, X1,
        align_mode=align_mode,
        stride_rtol=stride_rtol, stride_atol=stride_atol,
        stride_atol_scale=stride_atol_scale,
        debug_stride=debug_stride
    )
    if metric.upper() == "SSIM":
        score = _ssim_1d(A0a, A1a, gaussian_weights=gaussian_weights, win_size=win_size)
    elif metric.upper() == "NRMSE":
        if nrmse_ref.upper() == "A":
            score = _nrmse_1d(A0a, A1a, norm=nrmse_norm)
        elif nrmse_ref.upper() == "B":
            score = _nrmse_1d(A1a, A0a, norm=nrmse_norm)
        else:
            raise ValueError(f"Incorrect nrmse_ref selection")
    else:
        raise ValueError("metric must be 'SSIM', 'NRMSE'")
    return {
        "score": np.float64(score),
        "metric": metric.upper(),
        "aligned_shape": A0a.shape,
        "align_info": info,          # {'mode': 'as_is'|'stride'|'interp', ...}
        ax0: Xt,
    }


def compare_species_1d_by_lists(
    dA, dB,
    species_list,        # full species list
    subset_list,         # subset inside species_list (others become non-subset)
    exclude_list=None,        # Chemical species to be excluded from calculations
    *,
    metric="SSIM",       # 'SSIM' | 'NRMSE'
    align_mode="coarse", # 'coarse' -> stride if integer factor; else interpolate
    gaussian_weights="basic",
    win_size="basic",
    nrmse_norm="rms",    # Select the type of denominator to be used in the nrmse calculation   'rms' | 'min-max' | 'std' | 'mean'
    nrmse_ref="A",       # Select data A or B to use as a reference when calculating nrmse
    # stride params:
    stride_rtol=0.0,
    stride_atol=None,
    stride_atol_scale=1e-6,
    debug_stride=False,
    on_missing="error",   # 'error' | 'nan' for species in species_list missing in A or B
):
    """
    Compares the 1D fields for each chemical species in two datasets (dA and dB) and
    computes the metric score for each species and the subset/non-subset average.
    
    Species included in the exclude_list are excluded from the calculations,
    and the on_missing option specifies how missing species are handled ('error' or 'nan').
    
    Note:
    - dA and dB must both have the {"coords", "species"} key.
    - The axis information in grid_info['info'] must match for comparison.
    - The alignment information (align_info) of the first species represents the synchronization state of the entire grid, as all species share the same spatial coordinates.
    """
        
    species_list = list(species_list)
    subset_list  = list(subset_list)
    
    # Guard: species_list must be unique (prevent double-evaluation & double-count)
    if len(set(species_list)) != len(species_list):
        raise ValueError("[species-check] species_list has duplicates")
    
    grid_info_A_x = dA["coords"]['info'][0]
    grid_info_B_x = dB["coords"]['info'][0]
    grid_info = (grid_info_A_x, grid_info_B_x)
    
    if grid_info[0] != grid_info[1]:
        raise ValueError(f"The axes must match: dA axis={grid_info[0]}, dB axis={grid_info[1]}")
    
    gridA = {grid_info[0]: dA["coords"][grid_info[0]]}
    gridB = {grid_info[1]: dB["coords"][grid_info[1]]}
    
    if exclude_list is None:
        if debug_stride:
            print("Calculate all chemical species")
    else:
        if not all(isinstance(sp, str) for sp in exclude_list):
            raise ValueError(f"exclude_list must be a list of str")
        else:
            exclude_set=set(exclude_list)
            invalid_exclude = [s for s in exclude_list if s not in species_list]
            if invalid_exclude:
                raise ValueError(f"exclude_list must be in species_list")
            species_list = [s for s in species_list if s not in exclude_set]
            subset_list  = [s for s in subset_list  if s not in exclude_set]
            
    
    subset_set = set(subset_list)
    subset_present   = [nm for nm in subset_list if nm in species_list]
    ignored_subset   = [nm for nm in subset_list if nm not in species_list]
    nonsubset_names  = [nm for nm in species_list if nm not in subset_present]
    
    if len(subset_set) != len(subset_list) and len(subset_list) != len(subset_present):
        raise ValueError(
            f"check the subset list={subset_list}!! There are duplicate values in subset_list and\n"
            f"There are currently species not included in species_list={ignored_subset}"
                        )
    elif len(subset_set) != len(subset_list):
        raise ValueError(f"check the subset list={subset_list}!! There are duplicate values in subset_list")
    elif len(subset_list) != len(subset_present):
        raise ValueError(
            f"check the subset list={subset_list}!!\n" 
            f"There are currently species not included in species_list={ignored_subset}"
        )
    else:
        # subset count is number of (species + T)
        print(f"[subset-check] valid subset count: {len(subset_present)+1} | [calc_species-check] calc species count: {len(species_list)}")
        if debug_stride:
            print(f"[subset-check] subset species: {', '.join(subset_present)}\n"
                  f"[calc_species-check] calc species: {', '.join(species_list)}")
    
    names, vals = [], []
    align_info_first = None
    x_tgt = None
    aligned_shape = None
    missing_list = []
    first = True
    
    for nm in species_list:
        hasA = nm in dA["species"] and dA["species"][nm] is not None
        hasB = nm in dB["species"] and dB["species"][nm] is not None
        if not (hasA and hasB):
            missing_list.append(nm)
            if on_missing == "error":
                cases = []
                if not hasA: cases.append("A")
                if not hasB: cases.append("B")
                raise KeyError(f"Species '{nm}' missing in case {', '.join(cases)}")
            elif on_missing == "nan":
                names.append(nm); vals.append(float('nan'))
                continue
            else:
                raise ValueError("on_missing must be 'error', or 'nan'")
    
        A = dA["species"][nm]
        B = dB["species"][nm]
        out = compare_1d_field(
            A, gridA, B, gridB,
            grid_info,
            metric=metric,
            align_mode=align_mode,
            gaussian_weights=gaussian_weights,
            win_size=win_size,
            nrmse_norm=nrmse_norm,
            nrmse_ref=nrmse_ref,
            stride_rtol=stride_rtol,
            stride_atol=stride_atol,
            stride_atol_scale=stride_atol_scale,
            debug_stride=debug_stride
        )
        names.append(nm)
        vals.append(float(out["score"]))
    
        if first:
            align_info_first = out["align_info"]
            x_tgt = out[grid_info[0]]
            aligned_shape = out["aligned_shape"]
            first = False
        else:
            Xt = out[grid_info[0]]
            if (Xt.shape != x_tgt.shape) or not np.array_equal(Xt, x_tgt):
                raise ValueError("Per-species aligned grids differ; check inputs.")
    
    
    # Guard clause: if no species were evaluated (e.g. all excluded or skipped)
    # stop early to prevent meaningless downstream averages or NaN results
    # This can occur when:
    #   - all species are removed by exclude_list
    #   - all species are missing in dA/dB and on_missing='skip'
    #   - or species_list itself is empty
    if not names:
        raise ValueError("No species evaluated. Check 'species_list' and 'on_missing' policy.")
    
    vals = np.asarray(vals, np.float64)
    
    # ----- buckets -----
    name_to_idx = {nm: i for i, nm in enumerate(names)}
    
    def _bucket(list_names, vals):
        if not list_names:
            return {
                "species": [],
                "values": {},
                "min_species": None,
                "min":np.float64('nan'),
                "avg": np.float64('nan'),
                "max":np.float64('nan'),
                "max_species": None,
            }
        idx = [name_to_idx[nm] for nm in list_names]   # must exist; fail-fast if not
        arr = vals[idx]
        
        use_nan=False
        if on_missing == 'nan':
            if not np.any(np.isfinite(arr)):
                return{
                    "species": list_names,
                    "values": {sn: v for sn, v in zip(list_names, arr)},
                    "min":np.float64('nan'),
                    "min_species": None,
                    "avg": np.float64('nan'),
                    "max":np.float64('nan'),
                    "max_species": None,
                }
            stat_min = np.float64(np.nanmin(arr))
            stat_avg = np.float64(np.nanmean(arr))
            stat_max = np.float64(np.nanmax(arr))
            use_nan = True
            
        return {
            "species": list_names,
            "values":  {sn: v for sn, v in zip(list_names, arr)},
            "min":     np.float64(np.min(arr)) if not use_nan else stat_min,
            "min_species": list_names[np.argmin(arr)] if not use_nan else list_names[np.nanargmin(arr)],
            "avg":     np.float64(np.mean(arr)) if not use_nan else stat_avg,
            "max":     np.float64(np.max(arr)) if not use_nan else stat_max,
            "max_species": list_names[np.argmax(arr)] if not use_nan else list_names[np.nanargmax(arr)],
        }
        
    
    fvals={sn: v for sn, v in zip(names, vals)}
    
    if on_missing == 'nan':
        if not np.any(np.isfinite(vals)):
            species_all_v = {
                "species": names,
                "values": {sn: v for sn, v in zip(names, vals)},
                "min":np.float64('nan'),
                "min_species": None,
                "avg": np.float64('nan'),
                "max":np.float64('nan'),
                "max_species": None,
            }
        else:
            stat_min = np.float64(np.nanmin(vals))
            stat_avg = np.float64(np.nanmean(vals))
            stat_max = np.float64(np.nanmax(vals))
            species_all_v = {
                "species": names,
                "values": fvals,
                "min": stat_min,
                "min_species": names[np.nanargmin(vals)],
                "avg": stat_avg,
                "max": stat_max,
                "max_species": names[np.nanargmax(vals)],
            }
    else:
        species_all_v = {
            "species": names,
            "values": fvals,
            "min": np.float64(np.min(vals)),
            "min_species": names[np.argmin(vals)],
            "avg": np.float64(np.mean(vals)),
            "max": np.float64(np.max(vals)),
            "max_species": names[np.argmax(vals)],
        }
            
    out = {
        "metric": metric.upper(),
        "align_mode": align_mode,
        "aligned_shape": aligned_shape,
        "align_info": align_info_first,
        f"{grid_info[0]}_target": x_tgt,
    
        "species_all": species_all_v,
        "subset":      _bucket(subset_present, vals),
        "nonsubset":   _bucket(nonsubset_names, vals),
    
        "missing_list":   missing_list,     # data don't have specific species
    }
    return out



def _grid_key(g):
    return int(g)


def _subset_method_from_group_paths(group_paths):
    """Extract the method token from the first element of short_path (e.g., 'max_value_manual/...')."""
    if not group_paths:
        raise ValueError("group path is missing!!")
    dns_check = short_path(group_paths[0], tail=2)
    dns_tag = str(dns_check).split('/')[0]
    if dns_tag.startswith("dns"):
        return "dns_laminar"
    else:
        s = short_path(group_paths[0])
        s = str(s).split('/')[0]
        return s


def _ylabel_for_method(mname: str):
    return 'NRMSE' if mname == "NRMSE" else rf'1-{mname.replace("_","-")}'


def chem_to_tex_label(name: str) -> str:
    # Convert consecutive digits to _{...}: H2O -> H_{2}O, NNH remains unchanged
    tex = re.sub(r'(\d+)', r'_{\1}', name)
    # Force upright roman via \mathrm{} (for consistent Times in math mode)
    return rf'$\mathrm{{{tex}}}$'


def extract_ann_model_type_from_path(path: Path) -> str:
    """
    Return 'best', 'medi', worst', or 'dns' from the last folder name.
    ex) ann_640_best -> best
        ann_640_dns -> dns
    """
    name = path.name.lower()
    last = name.split("_")[-1]
    return last


def extract_model_type_from_path(path: Path) -> str:
    """Return 'dns' or 'ann' inferred from the last folder name."""
    name = path.name.lower()
    if name.startswith("dns"):
        return "dns"
    elif name.startswith("ann"):
        return "ann"
    else:
        raise ValueError(f"Cannot infer model type (dns/ann) from folder name: {path.name}")


def species_color(idx: int):
    return mpl.colors.to_rgb(PALETTE[idx % len(PALETTE)])


def place_info_text_auto(ax, text, *, pad=0.02, _store_attr='_auto_info_text', **text_kwargs):
    """Place text at the best of 8 candidate axes-fraction positions.

    Candidates (ax-fraction): top-right, top-center, top-left, middle-left,
    bottom-left, bottom-center, bottom-right, middle-right. For each, an
    invisible probe text is rendered to measure its bbox, then scored by:
      score = (data points inside the bbox) + 1000 * (legend overlap area).
    The min-score candidate wins. A previous placement on the same ax made
    by this function is removed first, so it is safe to call once per scale
    (e.g., before linear save and again before log save).
    """
    fig = ax.figure

    prev = getattr(ax, _store_attr, None)
    if prev is not None:
        try:
            prev.remove()
        except Exception:
            pass
        try:
            delattr(ax, _store_attr)
        except AttributeError:
            pass

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    candidates = [
        (1 - pad, 1 - pad,  'right',  'top'),
        (0.5,     1 - pad,  'center', 'top'),
        (pad,     1 - pad,  'left',   'top'),
        (pad,     0.5,      'left',   'center'),
        (pad,     pad,      'left',   'bottom'),
        (0.5,     pad,      'center', 'bottom'),
        (1 - pad, pad,      'right',  'bottom'),
        (1 - pad, 0.5,      'right',  'center'),
    ]

    data_pts_list = []
    for line in ax.get_lines():
        xy = line.get_xydata()
        if xy.size == 0:
            continue
        disp = ax.transData.transform(xy)
        axf = ax.transAxes.inverted().transform(disp)
        data_pts_list.append(axf)
    all_pts = np.vstack(data_pts_list) if data_pts_list else np.zeros((0, 2))

    # collect ALL legends on the axes (ax.legend() + any added via add_artist), not just ax.get_legend()
    legend_bboxes = []
    _inv_leg = ax.transAxes.inverted()
    for _art in ax.get_children():
        if isinstance(_art, mpl.legend.Legend):
            try:
                lbb = _art.get_window_extent(renderer=renderer)
                (lx0, ly0), (lx1, ly1) = _inv_leg.transform([(lbb.x0, lbb.y0), (lbb.x1, lbb.y1)])
                legend_bboxes.append((min(lx0, lx1), min(ly0, ly1), max(lx0, lx1), max(ly0, ly1)))
            except Exception:
                pass

    invisible = [ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va,
                         alpha=0, **text_kwargs)
                 for (x, y, ha, va) in candidates]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    inv = ax.transAxes.inverted()
    best_idx = 0
    best_score = float('inf')
    for i, t in enumerate(invisible):
        bb = t.get_window_extent(renderer=renderer)
        (tx0, ty0), (tx1, ty1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        tx0, tx1 = min(tx0, tx1), max(tx0, tx1)
        ty0, ty1 = min(ty0, ty1), max(ty0, ty1)

        if all_pts.size > 0:
            in_box = ((all_pts[:, 0] >= tx0) & (all_pts[:, 0] <= tx1) &
                      (all_pts[:, 1] >= ty0) & (all_pts[:, 1] <= ty1))
            data_count = int(in_box.sum())
        else:
            data_count = 0

        legend_overlap = 0.0
        for (lx0, ly0, lx1, ly1) in legend_bboxes:
            ox0 = max(tx0, lx0)
            oy0 = max(ty0, ly0)
            ox1 = min(tx1, lx1)
            oy1 = min(ty1, ly1)
            if ox1 > ox0 and oy1 > oy0:
                legend_overlap += (ox1 - ox0) * (oy1 - oy0)

        score = data_count + legend_overlap * 1000.0
        if score < best_score:
            best_score = score
            best_idx = i

    for t in invisible:
        t.remove()

    x, y, ha, va = candidates[best_idx]
    text_obj = ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, **text_kwargs)
    setattr(ax, _store_attr, text_obj)
    return text_obj
