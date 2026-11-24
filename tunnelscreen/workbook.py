"""Excel workbook of the trough and of one Stage 2 calculation, written with live formulas.

The workbook is an independent re-statement of the method in spreadsheet formulas: nothing in it is
a pasted number from Python apart from the inputs. tests/test_workbook.py recalculates it and
compares every result with the Python engine.
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

N_GRID = 41  # points per span on the Stage 2 sheet; the test calls the Python engine with the same number
TROUGH_Y = [float(v) for v in range(-60, 61, 2)]
SPAN_BLOCK_ROW = 30  # first row of the helper grid on the Stage2 sheet

NOTICE = ("Student-level screening study with textbook empirical methods. Not a design, not checked by a "
          "qualified geotechnical engineer, not for use on any project. The tunnel, ground and building "
          "below are SYNTHETIC.")

_HEAD = Font(bold=True)
_INPUT = PatternFill("solid", fgColor="FFF2CC")


def _put(ws, cell, value, bold=False, fill=None):
    ws[cell] = value
    if bold:
        ws[cell].font = _HEAD
    if fill is not None:
        ws[cell].fill = fill


def build(path: Path, *, D: float, z0: float, K: float, VL: float, y0_first: float, y0_second: float,
          second_factor: float, facade_start: float, facade_end: float, H: float, e_over_g: float,
          nu: float, building_label: str) -> Path:
    wb = Workbook()

    # ------------------------------------------------------------------ Inputs
    ws = wb.active
    ws.title = "Inputs"
    ws.column_dimensions["A"].width = 46
    ws.column_dimensions["B"].width = 14
    ws.column_dimensions["C"].width = 70
    _put(ws, "A1", "Greenfield trough and Stage 2 check: inputs", bold=True)
    ws["A2"] = NOTICE
    ws["A2"].alignment = Alignment(wrap_text=True)
    ws.merge_cells("A2:C2")
    ws.row_dimensions[2].height = 45
    rows = [
        (4, "Excavated diameter D (m)", D, "ILLUSTRATIVE"),
        (5, "Depth to tunnel axis z0 (m)", z0, "SYNTHETIC alignment"),
        (6, "Trough width parameter K", K, "Design value, source S1 Table 14.1"),
        (7, "Volume loss V (%)", VL, "Design value, source S1 Table 14.1"),
        (8, "Centre line of first drive y (m)", y0_first, "SYNTHETIC"),
        (9, "Centre line of second drive y (m)", y0_second, "SYNTHETIC"),
        (10, "Volume loss factor on second drive", second_factor, "ASSUMPTION"),
    ]
    for r, label, value, note in rows:
        ws[f"A{r}"] = label
        _put(ws, f"B{r}", value, fill=_INPUT)
        ws[f"C{r}"] = note
    _put(ws, "A12", "Derived", bold=True)
    derived = [
        (13, "i = K z0 (m)", "=B6*B5", "S3 Eq. 2"),
        (14, "Vs first drive = V pi D^2 / 4 (m3 per m)", "=B7/100*PI()*B4^2/4", "S3 Eq. 4"),
        (15, "Smax first drive (m)", "=B14/(SQRT(2*PI())*B13)", "S3 Eq. 3"),
        (16, "Smax first drive (mm)", "=B15*1000", ""),
        (17, "Smax by the LTA formula 0.0031 V D^2 / i (mm)", "=0.0031*B7*B4^2/B13*1000", "S1 Cl. 14.2.1.2 (coefficient rounded)"),
        (18, "Smax second drive (m)", "=B15*B10", ""),
    ]
    for r, label, formula, note in derived:
        ws[f"A{r}"] = label
        ws[f"B{r}"] = formula
        ws[f"C{r}"] = note

    # ------------------------------------------------------------------ Trough
    wt = wb.create_sheet("Trough")
    heads = ["y (m)", "S first (mm)", "S second (mm)", "S total (mm)", "slope total (1 in ...)",
             "slope total (-)", "curvature total (1/m)", "u total (mm, + = towards +y)", "eps_h total (%, + = tension)"]
    for c, h in enumerate(heads, start=1):
        cell = wt.cell(row=1, column=c, value=h)
        cell.font = _HEAD
        wt.column_dimensions[get_column_letter(c)].width = 22
    # In Excel a leading minus binds tighter than ^, so -x^2 is +x^2: every square is bracketed first.
    for k, y in enumerate(TROUGH_Y):
        r = k + 2
        d1, d2 = f"(A{r}-Inputs!$B$8)", f"(A{r}-Inputs!$B$9)"
        s1 = f"Inputs!$B$15*EXP(-({d1}^2)/(2*Inputs!$B$13^2))"
        s2 = f"Inputs!$B$18*EXP(-({d2}^2)/(2*Inputs!$B$13^2))"
        wt[f"A{r}"] = y
        wt[f"B{r}"] = f"=1000*{s1}"
        wt[f"C{r}"] = f"=1000*{s2}"
        wt[f"D{r}"] = f"=B{r}+C{r}"
        wt[f"F{r}"] = f"=-{d1}/Inputs!$B$13^2*{s1}-{d2}/Inputs!$B$13^2*{s2}"
        wt[f"E{r}"] = f"=IF(ABS(F{r})>0,1/ABS(F{r}),0)"
        wt[f"G{r}"] = f"={s1}/Inputs!$B$13^2*({d1}^2/Inputs!$B$13^2-1)+{s2}/Inputs!$B$13^2*({d2}^2/Inputs!$B$13^2-1)"
        wt[f"H{r}"] = f"=1000*(-{d1}/Inputs!$B$5*{s1}-{d2}/Inputs!$B$5*{s2})"
        wt[f"I{r}"] = f"=100*({s1}/Inputs!$B$5*({d1}^2/Inputs!$B$13^2-1)+{s2}/Inputs!$B$5*({d2}^2/Inputs!$B$13^2-1))"
    last = len(TROUGH_Y) + 1
    wt["K1"] = "Max S total (mm)"
    wt["L1"] = f"=MAX(D2:D{last})"
    wt["K2"] = "Trough volume by trapezium rule (m3 per m)"
    wt["L2"] = "=" + "+".join(f"(D{r}+D{r + 1})/2*(A{r + 1}-A{r})/1000" for r in range(2, last))
    wt["K3"] = "Vs first + second (m3 per m)"
    wt["L3"] = "=Inputs!B14*(1+Inputs!B10)"
    wt.column_dimensions["K"].width = 42

    # ------------------------------------------------------------------ Stage 2
    s2ws = wb.create_sheet("Stage2")
    s2ws.column_dimensions["A"].width = 52
    for col in "BCD":
        s2ws.column_dimensions[col].width = 18
    _put(s2ws, "A1", f"Stage 2 (limiting tensile strain) for {building_label}: facade across the first drive only", bold=True)
    s2ws["A2"] = "Method: source S3 (Mair, Taylor and Burland 1996), Eq. 7 to 10; categories: source S1 Table 14.2."
    inp = [(4, "Facade start y (m)", facade_start), (5, "Facade end y (m)", facade_end), (6, "Height H (m)", H),
           (7, "E/G", e_over_g), (8, "Poisson's ratio in Eq. 10", nu)]
    for r, label, value in inp:
        s2ws[f"A{r}"] = label
        _put(s2ws, f"B{r}", value, fill=_INPUT)
    s2ws["A9"] = "Centre line c of first drive (m)"
    s2ws["B9"] = "=Inputs!B8"
    s2ws["A10"] = "i (m)"
    s2ws["B10"] = "=Inputs!B13"
    for c, h in zip("BCD", ("left hogging", "sagging", "right hogging")):
        _put(s2ws, f"{c}12", h, bold=True)
    span_bounds = {
        "B": ("=MAX($B$4,$B$9-2.5*$B$10)", "=MIN($B$5,$B$9-$B$10)"),
        "C": ("=MAX($B$4,$B$9-$B$10)", "=MIN($B$5,$B$9+$B$10)"),
        "D": ("=MAX($B$4,$B$9+$B$10)", "=MIN($B$5,$B$9+2.5*$B$10)"),
    }
    labels = {13: "Span start a (m)", 14: "Span end b (m)", 15: "Span length L (m); 0 if the span is absent",
              16: "Relative deflection Delta (m)", 17: "Deflection ratio Delta/L", 18: "u at a (m)", 19: "u at b (m)",
              20: "Average horizontal strain eps_h", 21: "I per unit thickness (m3)", 22: "t (m)",
              23: "Eq. 7 factor: L/(12t) + 3I(E/G)/(2tLH)", 24: "Eq. 8 factor: 1 + HL^2/(18I(E/G))",
              25: "eps_b", 26: "eps_d", 27: "eps_bt = eps_h + eps_b (Eq. 9)", 28: "eps_dt (Eq. 10, general Poisson's ratio)"}
    for r, text in labels.items():
        s2ws[f"A{r}"] = text
    first, lastg = SPAN_BLOCK_ROW + 1, SPAN_BLOCK_ROW + N_GRID
    block_cols = {"B": ("F", "G", "H", "I"), "C": ("K", "L", "M", "N"), "D": ("P", "Q", "R", "S")}
    smax, cen, ii, z0c = "Inputs!$B$15", "$B$9", "$B$10", "Inputs!$B$5"

    def s_of(x):
        return f"{smax}*EXP(-(({x}-{cen})^2)/(2*{ii}^2))"

    def u_of(x):
        return f"(-({x}-{cen})/{z0c}*{s_of(x)})"

    for c, (a_f, b_f) in span_bounds.items():
        cx, cs, cc, cd = block_cols[c]
        s2ws[f"{c}13"] = a_f
        s2ws[f"{c}14"] = b_f
        s2ws[f"{c}15"] = f"=IF({c}14-{c}13>0.000001,{c}14-{c}13,0)"
        s2ws[f"{c}16"] = f"=IF({c}15>0,MAX({cd}{first}:{cd}{lastg}),0)"
        s2ws[f"{c}17"] = f"=IF({c}15>0,{c}16/{c}15,0)"
        s2ws[f"{c}18"] = f"={u_of(f'{c}13')}"
        s2ws[f"{c}19"] = f"={u_of(f'{c}14')}"
        s2ws[f"{c}20"] = f"=IF({c}15>0,({c}19-{c}18)/{c}15,0)"
        if c == "C":  # sagging: neutral axis at mid-height
            s2ws[f"{c}21"] = "=$B$6^3/12"
            s2ws[f"{c}22"] = "=$B$6/2"
        else:  # hogging: neutral axis at the bottom edge
            s2ws[f"{c}21"] = "=$B$6^3/3"
            s2ws[f"{c}22"] = "=$B$6"
        s2ws[f"{c}23"] = f"=IF({c}15>0,{c}15/(12*{c}22)+3*{c}21*$B$7/(2*{c}22*{c}15*$B$6),1)"
        s2ws[f"{c}24"] = f"=IF({c}15>0,1+$B$6*{c}15^2/(18*{c}21*$B$7),1)"
        s2ws[f"{c}25"] = f"={c}17/{c}23"
        s2ws[f"{c}26"] = f"={c}17/{c}24"
        s2ws[f"{c}27"] = f"={c}20+{c}25"
        s2ws[f"{c}28"] = f"={c}20*(1-$B$8)/2+SQRT(({c}20*(1+$B$8)/2)^2+{c}26^2)"
        for h, col in zip(("x (m)", "S (m)", "chord (m)", "|S - chord| (m)"), (cx, cs, cc, cd)):
            _put(s2ws, f"{col}{SPAN_BLOCK_ROW}", f"{s2ws[f'{c}12'].value}: {h}", bold=True)
        for k in range(N_GRID):
            r = first + k
            s2ws[f"{cx}{r}"] = f"=${c}$13+(${c}$14-${c}$13)*{k}/{N_GRID - 1}"
            s2ws[f"{cs}{r}"] = f"={s_of(f'{cx}{r}')}"
            s2ws[f"{cc}{r}"] = (f"=IF(${c}$15>0,${cs}${first}+(${cs}${lastg}-${cs}${first})*({cx}{r}-${c}$13)/${c}$15,{cs}{r})")
            s2ws[f"{cd}{r}"] = f"=ABS({cs}{r}-{cc}{r})"
    _put(s2ws, "F4", "Result", bold=True)
    s2ws["F5"] = "Maximum tensile strain (strain)"
    s2ws["H5"] = ("=MAX(0,IF(B15>0,MAX(B27,B28),0),IF(C15>0,MAX(C27,C28),0),IF(D15>0,MAX(D27,D28),0))")
    s2ws["F6"] = "Maximum tensile strain (%)"
    s2ws["H6"] = "=H5*100"
    s2ws["F7"] = "Damage category (S1 Table 14.2)"
    s2ws["H7"] = "=IF(H6>=0.3,4,IF(H6>=0.15,3,IF(H6>=0.075,2,IF(H6>=0.05,1,0))))"
    s2ws["F8"] = "Category name"
    s2ws["H8"] = ('=IF(H7=0,"Negligible",IF(H7=1,"Very Slight",IF(H7=2,"Slight",IF(H7=3,"Moderate","Severe to Very Severe"))))')
    s2ws.column_dimensions["F"].width = 34

    # ------------------------------------------------------------------ About
    wa = wb.create_sheet("About")
    wa.column_dimensions["A"].width = 120
    lines = [
        "Read this first", NOTICE,
        "Personal portfolio project, built with AI assistance. Independent: not affiliated with or endorsed by the Land Transport Authority (LTA).",
        "Yellow cells are inputs. Every other cell is a formula.",
        "Sign conventions: settlement positive downwards; u positive in the +y direction; strain positive in tension.",
        "Sources: S1 = LTA Civil Design Criteria E/GD/09/106/A3 (Dec 2025); S3 = Mair, Taylor and Burland (1996). Full list in DATA_SOURCES.md.",
        "The Stage2 sheet treats the first drive alone. The Python package also checks both drives together and the facade parallel to the tunnel.",
        "Author: Wenjie Luo (GitHub LUOaini1213).",
    ]
    for r, text in enumerate(lines, start=1):
        wa[f"A{r}"] = text
        wa[f"A{r}"].alignment = Alignment(wrap_text=True)
    wa["A1"].font = _HEAD

    wb.properties.creator = "Wenjie Luo (LUOaini1213)"
    wb.properties.title = "Greenfield trough and Stage 2 check (SYNTHETIC example)"
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    _fix_timestamps(path)
    return path


STAMP = "2026-11-24T00:00:00Z"


def _fix_timestamps(path: Path) -> None:
    """Rewrite the saved file with fixed dates, so that the same inputs give the same file."""
    with zipfile.ZipFile(path) as z:
        items = [(info.filename, z.read(info.filename)) for info in z.infolist()]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in items:
            if name == "docProps/core.xml":
                data = re.sub(rb"(<dcterms:(created|modified)[^>]*>)[^<]*(</dcterms:)", rb"\g<1>" + STAMP.encode() + rb"\g<3>", data)
            info = zipfile.ZipInfo(name, date_time=(2026, 11, 24, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, data)
    path.write_bytes(buf.getvalue())
