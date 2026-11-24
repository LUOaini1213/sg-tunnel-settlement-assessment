"""Write the four-page technical note, docs/technical_note.pdf, from results/ (reportlab).

Usage: python scripts/build_note.py
The numbers come from the same context as the README, so the two cannot disagree.
"""
from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import build_docs  # noqa: E402

OUT = ROOT / "docs" / "technical_note.pdf"
FIG = ROOT / "docs" / "figures"
NOTICE = ("Read this first. Personal portfolio project, built with AI assistance; independent, not affiliated with or endorsed "
          "by the Land Transport Authority (LTA); no non-public information. Student-level screening study with textbook "
          "empirical methods: not a design, not checked by a qualified geotechnical engineer, not for use on any project. "
          "The alignment, buildings and readings are SYNTHETIC. Author: Wenjie Luo (GitHub LUOaini1213).")

ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["BodyText"], fontName="Helvetica", fontSize=8.6, leading=11.2, alignment=TA_JUSTIFY, spaceAfter=3.5)
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=13, leading=16, spaceAfter=4)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=10, leading=12.5, spaceBefore=4, spaceAfter=2.5)
BOX = ParagraphStyle("box", parent=BODY, fontSize=8, leading=10, borderColor=colors.HexColor("#c62828"), borderWidth=0.8,
                     borderPadding=4, backColor=colors.HexColor("#fff5f5"), spaceAfter=7)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=7.1, leading=8.6, alignment=0, spaceAfter=0)
EQ = ParagraphStyle("eq", parent=BODY, fontName="Courier", fontSize=8.2, leading=11, alignment=1, spaceBefore=1, spaceAfter=4)
CAP = ParagraphStyle("cap", parent=BODY, fontSize=7.4, leading=9, textColor=colors.HexColor("#444444"))


def P(text: str, style=BODY):
    return Paragraph(text, style)


def tbl(headers, rows, widths):
    data = [[Paragraph(f"<b>{h}</b>", CELL) for h in headers]] + [[Paragraph(str(c), CELL) for c in r] for r in rows]
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#999999")),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
                           ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 2.5),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 2.5), ("TOPPADDING", (0, 0), (-1, -1), 1.5),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]))
    return t


def img(name: str, width_mm: float):
    from PIL import Image as PILImage

    with PILImage.open(FIG / name) as im:
        w, h = im.size
    return Image(str(FIG / name), width=width_mm * mm, height=width_mm * mm * h / w)


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawString(18 * mm, 10 * mm, "sg-tunnel-settlement-assessment: student screening study on SYNTHETIC inputs. Not a design. Not for use on any project.")
    canvas.drawRightString(192 * mm, 10 * mm, f"Page {doc.page} of 4")
    canvas.restoreState()


def build(out: Path = OUT) -> int:
    """Write the note to ``out`` and return the number of pages."""
    c = build_docs.context()
    ver = build_docs.load("verification.json")
    scr = build_docs.load("screening.json")["design"]
    unc = build_docs.load("uncertainty.json")
    back = build_docs.load("backanalysis.json")
    ext = build_docs.load("existing_tunnel.json")["design"]
    story = []

    # ------------------------------------------------------------------ page 1: method
    story += [P("Greenfield settlement and building damage screening for a twin bored tunnel drive", H1),
              P("Technical note 1 of 1. Method, verification, results, limits.", CAP), P(NOTICE, BOX)]
    story += [P("1. Purpose and scope", H2),
              P("This note works through the first-pass assessment made before a bored tunnel is driven under buildings: predict the "
                "free-field (greenfield) ground movement, screen each building for damage, set monitoring review levels, and "
                "back-analyse the readings. The method is the one named in LTA's Civil Design Criteria (CDC), Revision A3 of "
                "December 2025, Chapters 13 and 14. It is applied to an invented 900 m twin-tunnel drive with thirteen invented "
                "buildings. The purpose is to show the method implemented correctly and checked against its sources. Source IDs "
                "(S1, S3, ...) refer to DATA_SOURCES.md in the repository; S1 is the CDC."),
              P("2. Ground movement", H2),
              P("The transverse settlement trough of one tunnel is the Gaussian curve of Peck (1969), with the trough width of "
                "O'Reilly and New (1982). With y the offset from the centre line, i the distance to the point of inflexion, z0 the depth "
                "to the tunnel axis, K the trough width parameter, D the excavated diameter and V the volume loss in per cent of the "
                "face area (S1 Cl. 14.2.1.2; S3 Eq. 1 to 5):"),
              P("S = Smax exp(-y^2 / (2 i^2)) &nbsp;&nbsp;&nbsp;&nbsp; i = K z0<br/>"
                "Vs = sqrt(2 pi) i Smax = (V/100) pi D^2 / 4 &nbsp;&nbsp;&nbsp;&nbsp; Smax = 0.0031 V D^2 / i", EQ),
              P("Horizontal movement is taken as directed at the tunnel axis (S1 Cl. 14.2.1.8), which gives the horizontal strain eh. "
                "It is compressive between the points of inflexion, zero at them and tensile outside:"),
              P("u = (y / z0) S towards the centre line &nbsp;&nbsp;&nbsp; eh = (S / z0) (y^2 / i^2 - 1), tension positive", EQ),
              P("Ahead of the face the trough develops as a cumulative normal curve (Attewell and Woodman 1982, as reported in S3 and S6). "
                "The troughs of the two drives are added (S3 section 2.4)."),
              P("3. Building screening", H2),
              P("<b>Stage 1.</b> A building is of negligible risk if the settlement under it is below 10 mm and the slope flatter than 1:500 "
                "(S1 Section 14.3.6). <b>Stage 2.</b> The building is treated as a beam with no stiffness that follows the greenfield profile "
                "(Mair, Taylor and Burland 1996, S3). Each facade is cut at the points of inflexion and at 2.5 i. For each span of "
                "length L and height H, the deflection ratio Delta/L (largest departure of the profile from the chord, divided by L) gives "
                "the bending strain eb and the diagonal strain ed (S3 Eq. 7 and 8):"),
              P("Delta/L = eb [ L/(12 t) + 3 I (E/G) / (2 t L H) ] &nbsp;&nbsp;&nbsp;&nbsp; Delta/L = ed [ 1 + H L^2 / (18 I (E/G)) ]", EQ),
              P("In sagging the neutral axis is at mid-height (I = H^3/12, t = H/2); in hogging it is at the bottom edge (I = H^3/3, t = H). "
                "E/G is 2.6 for masonry and 12.5 for framed buildings. The average horizontal strain eh of the span (difference of the "
                "end movements divided by L) is combined with each (S3 Eq. 9 and 10, Poisson's ratio 0.3):"),
              P("ebt = eh + eb &nbsp;&nbsp;&nbsp;&nbsp; edt = 0.35 eh + sqrt( (0.65 eh)^2 + ed^2 )", EQ),
              P("The larger value is compared with the limiting tensile strains of S1 Table 14.2: negligible below 0.05%, very slight to "
                "0.075%, slight to 0.15%, moderate to 0.3%, severe above. Sensitivity indices for structural form, heritage features and "
                "existing condition are added to the category (S1 Section 14.3.10). Under Revision A3, negligible and very slight are low "
                "risk; slight and above go to Stage 3 detailed evaluation, which is outside this study, as are piled buildings."),
              P("4. Inputs", H2),
              P(f"Design basis: the typical K and V of S1 Table 14.1 ({c['design_values']}). Observed volume loss by ground type: "
                "mean and 95% characteristic value from an open-access study of recent Singapore drives (S5, Ma et al. 2023). "
                "Tunnels: 6.6 m excavated diameter, 15 m apart, axis 20 to 26 m deep. All geometry and buildings are SYNTHETIC; every "
                "assumed factor is listed in the provenance table of the README."),
              PageBreak()]

    # ------------------------------------------------------------------ page 2: verification
    story += [P("5. Verification", H2),
              P(f"{c['n_checks']} published values or identities are reproduced by the code; each is a test. Tolerances are those the "
                "published figure allows (a coefficient printed to two figures, a value read from a chart)."),
              tbl(["ID", "Quantity", "Source", "Published", "This code", "Tolerance"],
                  [[v["id"], v["what"], v["source"], build_docs.published(v["published"]), build_docs.published(v["computed"]), v["tolerance"]]
                   for v in ver["checks"]], [8, 62, 30, 22, 26, 26]),
              Spacer(1, 4),
              P("Further checks made as tests: the two beam factors are re-derived from a centrally loaded Timoshenko beam; the diagonal "
                "strain formula is compared with the largest eigenvalue of the strain tensor; slope, curvature and horizontal strain are "
                "compared with numerical derivatives; the numerical volume of the trough equals the volume loss; a hogging span is worked "
                "by hand. The Excel workbook is recalculated from its formulas and agrees with the Python code to nine figures. "
                f"Hand-written mutation testing (errors planted one at a time) killed {c['mut_first_killed']} of {c['mut_first_total']} "
                f"on the first run; the final review added {c['mut_added_total']}, of which {c['mut_added_killed']} were killed at once, "
                f"and after tests were written for the survivors {c['mut_final_killed']} of {c['mut_final_total']} are killed."),
              P("Four things the checks turned up. (a) One secondary source prints the diagonal-strain formula with (1 - nu)/2 inside the "
                "square root; the primary paper, a second source and Mohr's circle give (1 + nu)/2, which is used. (b) From the beam "
                "equations alone, hogging is worse than sagging for the same deflection ratio only for spans of about three times the "
                "height or more; for a span equal to the height the sagging beam has the larger bending strain. (c) The first draft of "
                "the workbook returned an absurd settlement because in Excel -x^2 means (-x)^2; the recalculation test found it. "
                "(d) Eq. 9 and 10 as written let a compressive average strain in a sagging span offset the bending strain, and the "
                f"screening follows them. Setting that strain to zero instead, a cautious reading, {c['variant_text']}"),
              img("fig7_chart_check.png", 78),
              P("Figure 1. The published damage chart for L/H = 1 in hogging (S3 Fig. 4, after Burland 1995), redrawn from the implemented "
                "equations. Circles are the intercepts read from the printed chart.", CAP),
              PageBreak()]

    # ------------------------------------------------------------------ page 3: results
    rows = [[r["rank"], r["id"], r["zone"], r["foundation"], f"{r['max_settlement_mm']:.1f}", f"1:{r['max_slope_1_in']:.0f}",
             "-" if r["eps_max_pct"] is None else f"{r['eps_max_pct']:.3f}",
             "-" if r["stage2_category"] is None else r["stage2_category"], r["sensitivity_index_sum"],
             "-" if r["final_category"] is None else r["final_category"], r["action"].split(" (")[0].replace("; outside this study", "")]
            for r in scr]
    mcb = unc["base"]["buildings"]
    story += [P("6. Results", H2),
              P(f"<b>Design-basis screening.</b> {c['n_stage1']} of {c['n_buildings']} buildings are negligible at Stage 1 and {c['n_low']} are "
                f"low risk at Stage 2. {c['n_more']} need work outside this study: {c['n_stage3_shallow']} on shallow foundations go to "
                f"Stage 3, {c['n_mixed']} on mixed foundations goes to Stage 3 and {c['n_piled']} on piles need a pile assessment. The rank is "
                "the suggested order of priority for instrumentation."),
              tbl(["Rank", "Bldg", "Zone", "Foundation", "Settlement (mm)", "Slope", "Tensile strain (%)", "Stage 2 cat.", "Index sum", "Final cat.", "Next step"],
                  rows, [9, 10, 10, 18, 17, 14, 17, 13, 12, 12, 42]),
              Spacer(1, 3),
              P(f"<b>Design basis against experience.</b> The observed mean volume loss is far below the design value "
                f"({c['vl_mean_kf']}% against {c['vl_design_kf']}% in Kallang Formation). Sampling volume loss from a lognormal matched to "
                f"the published mean and 95% value, and K across a stated range ({c['n_mc']} samples), the probability that a shallow-founded "
                f"building reaches Stage 2 category 1 or higher is at most {c['max_p_cat1']}% (building {c['max_p_cat1_id']}), and the "
                f"probability that volume loss exceeds the design value is {c['p_exceed_design_min']}% to {c['p_exceed_design_max']}%. "
                f"In the sensitivity study {c['tornado_order']}. Building B06 is slight or worse in every sample "
                "because of its structural-form index alone."),
              P(f"<b>Existing tunnel (SYNTHETIC).</b> Free-field settlement at the invert level of an existing tunnel {ext['clear_distance_m']:g} m "
                f"above the new crowns is {c['et_design']} mm on the design basis, above the {c['et_allow']} mm allowable total movement of "
                f"LTA's Code of Practice for Railway Protection (S2 Table 2(b)); the steepest gradient along it, 1:{c['et_slope_design']}, is "
                f"steeper than the 1:{c['et_diff']} differential limit. At the characteristic volume loss the figures are {c['et_char']} mm "
                f"and 1:{c['et_slope_char']}, within both. The structure is not modelled."),
              P(f"<b>Monitoring.</b> Following S1 Chapter 13: {c['n_d']} Type D points, {c['n_e']} Type E arrays, {c['n_f']} Type F array, "
                f"{c['n_bldg_points']} building points. {c['n_rt']} buildings in use have part of a tunnel under them ({c['rt_edge_ids']} only "
                "its edge) and are monitored in real time (S1 Section 13.4.9). Work suspension level = design-basis prediction "
                "(S1 Cl. 13.7.4); alert level = "
                f"{c['alert_factor']} of it (ASSUMPTION). <b>Back-analysis.</b> From {c['ba_n']} SYNTHETIC readings with {c['ba_sigma']} mm noise the "
                f"least-squares fit gives V = {c['ba_vl']}% ({c['ba_vl_lo']} to {c['ba_vl_hi']}) and K = {c['ba_k']} ({c['ba_k_lo']} to "
                f"{c['ba_k_hi']}) for a true {c['ba_true_vl']}% and {c['ba_true_k']}. No marker reaches the work suspension level, but the "
                f"fitted trough is steeper (1:{c['ba_slope_fit']}) than the design trough (1:{c['ba_slope_design']}); the projection of final "
                f"settlement reaches the alert level {c['ba_lead']} tunnel diameters of advance before the reading does."),
              img("fig6_backanalysis.png", 150),
              P("Figure 2. Left: one array of SYNTHETIC readings with the fitted, design-basis and expected troughs. Right: the centre-line "
                "marker against the review levels as the face advances.", CAP),
              PageBreak()]

    # ------------------------------------------------------------------ page 4: limits
    worst = sorted(((k, v["p_stage2_category_at_least"]["1"]) for k, v in mcb.items() if "p_stage2_category_at_least" in v),
                   key=lambda kv: -kv[1]["p"])[:4]
    story += [P("7. What the greenfield assumption does", H2),
              P("The method assumes the building has no stiffness and follows the free-field ground. For most buildings that over-predicts: "
                "S1 calls Stage 2 a conservative assumption because the stiffness of buildings reduces both the deflection ratio and the "
                "horizontal strain, and S3 calls it usually very conservative. Buildings on strip footings or rafts are likely to pick up "
                "little horizontal strain. A Stage 2 category is therefore an upper estimate of possible damage, not a prediction."),
              P("It can under-predict in particular cases, which the CDC routes to other checks. Piles with toes directly above a tunnel were "
                "found to settle more than the ground surface (S1 Section 14.3.7). In sands the towards-axis rule can under-estimate "
                "horizontal movement near the edge of the trough (S4). Superposition may be unconservative when the clear separation of the "
                f"tunnels is under one diameter (here {c['clear_sep']} m against {c['diameter']} m). Consolidation settlement, which S1 requires "
                "to be added, is left out. Volume loss is higher at stoppages. The probabilities in section 6 are probabilities of exceeding "
                "the screening criterion under the stated distributions; ignoring stiffness makes them too high, and leaving out "
                "consolidation, stoppages and interaction makes them too low. Neither effect is quantified here."),
              tbl(["Building", "P(Stage 2 category 1 or higher), %", "95% interval, %"],
                  [[k, f"{100 * p['p']:.2f}", f"{100 * p['lo95']:.2f} to {100 * p['hi95']:.2f}"] for k, p in worst], [25, 60, 45]),
              Spacer(1, 3),
              img("fig4_uncertainty.png", 112),
              P("Figure 3. Maximum tensile strain of each shallow-founded building: design-basis value against the Monte Carlo range. "
                "Dashed lines are the category boundaries of S1 Table 14.2.", CAP),
              P("8. Limits", H2),
              P("1. Student-level screening with textbook empirical methods. Not a design, not checked by a qualified geotechnical engineer, "
                "not for use on any project. No finite-element or commercial ground-movement software was used.<br/>"
                "2. No ground investigation data, soil model, groundwater or consolidation. Ground parameters are published typical "
                "values or stated assumptions.<br/>"
                "3. The alignment, buildings, existing tunnel and readings are SYNTHETIC and must not be attached to any real line, "
                "station or building.<br/>"
                "4. Volume-loss statistics come from one open-access paper for earth pressure balance drives. The lognormal shape and the "
                "ranges of K are the author's assumptions; the paper notes that the data do not always follow a normal distribution.<br/>"
                "5. The longitudinal curve puts half the final settlement above the face, which the literature gives for open-face "
                "tunnelling; closed-face machines develop less at the face, so the timing of the review-level example is indicative only.<br/>"
                "6. The subsurface relation was published for tunnels in clay and is applied to a stylised Old Alluvium zone with a scaling "
                "that is an assumption.<br/>"
                "7. The alert factor, heave levels, noise level and the thresholds of the departure test are assumptions; LTA defines the "
                "review levels but gives no ratio between them.<br/>"
                "8. Stage 3 evaluation, pile assessment, utilities, protective works and deep excavations are outside the study.<br/>"
                f"9. Of {c['n_facts']} sourced statements, {c['n_verified']} were matched against the text of a document opened on "
                f"{c['access_date']}; {c['n_unverified']} primary papers could not be opened and are relied on only as reported by a "
                "source that was."),
              P("9. Sources used most", H2),
              P("S1: LTA, Civil Design Criteria for Civil and Structural Works, E/GD/09/106/A3, Dec 2025. S2: LTA, Code of Practice for "
                "Railway Protection, 2024 Edition. S3: Mair, Taylor and Burland (1996), Prediction of ground movements and assessment of "
                "risk of building damage due to bored tunnelling. S4: Mair and Taylor (1997), Bored tunnelling in the urban environment. "
                "S5: Ma, Su, Kumarasamy and Goh (2023), Tunnel volume loss study on various geological conditions in Singapore. "
                "S9: Peck (1969), Deep excavations and tunneling in soft ground. Full list with URLs, access dates and terms: DATA_SOURCES.md.",
                CAP)]

    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=14 * mm, bottomMargin=16 * mm,
                            title="Greenfield settlement and building damage screening for a twin bored tunnel drive (SYNTHETIC study)",
                            author="Wenjie Luo (LUOaini1213)", subject="Student-level screening study; not a design", invariant=1)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return doc.page


def page_texts(path: Path) -> list[str]:
    """Text of each page, for the drift test."""
    from pypdf import PdfReader

    return [page.extract_text() for page in PdfReader(str(path)).pages]


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    pages = build(target)
    print(f"wrote {target.name} ({pages} pages)")
    sys.exit(0 if pages == 4 else 1)
