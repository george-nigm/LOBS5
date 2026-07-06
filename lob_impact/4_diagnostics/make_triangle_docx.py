#!/usr/bin/env python3
"""
Build the explanatory Word report (Control Triangle) from the figures + numbers.json
produced by control_triangle_report.py. Dependency-free OOXML writer (no python-docx
in the lobs5 env).

  python 4_diagnostics/make_triangle_docx.py --fig_dir <triangle_out> \
      --out <triangle_out>/Control_Triangle_Report.docx
"""
import os, json, struct, zipfile, argparse
from xml.sax.saxutils import escape

EMU_IN = 914400
PAGE_W_IN = 6.7  # usable width with 0.9in margins on Letter


def png_size(path):
    with open(path, 'rb') as f:
        head = f.read(33)
    w, h = struct.unpack('>II', head[16:24])
    return w, h


class Doc:
    def __init__(self):
        self.body = []
        self.images = []  # (rid, filename, path)

    # ---- building blocks -------------------------------------------------
    def p(self, runs, style=None, space_after=120, align=None):
        """runs: list of (text, bold, italic, color, size_halfpt) or plain str."""
        if isinstance(runs, str):
            runs = [(runs, False, False, None, None)]
        ppr = ['<w:pPr>']
        if style:
            ppr.append(f'<w:pStyle w:val="{style}"/>')
        if align:
            ppr.append(f'<w:jc w:val="{align}"/>')
        ppr.append(f'<w:spacing w:after="{space_after}"/>')
        ppr.append('</w:pPr>')
        xml = ['<w:p>', ''.join(ppr)]
        for t, b, i, col, sz in runs:
            rpr = ['<w:rPr>']
            if b:
                rpr.append('<w:b/>')
            if i:
                rpr.append('<w:i/>')
            if col:
                rpr.append(f'<w:color w:val="{col}"/>')
            if sz:
                rpr.append(f'<w:sz w:val="{sz}"/><w:szCs w:val="{sz}"/>')
            rpr.append('</w:rPr>')
            xml.append(f'<w:r>{"".join(rpr)}<w:t xml:space="preserve">{escape(t)}</w:t></w:r>')
        xml.append('</w:p>')
        self.body.append(''.join(xml))

    def h(self, text, lvl=1):
        self.p(text, style=f'Heading{lvl}', space_after=80)

    def bullet(self, runs):
        if isinstance(runs, str):
            runs = [(runs, False, False, None, None)]
        xml = ['<w:p><w:pPr><w:pStyle w:val="ListParagraph"/>'
               '<w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr>'
               '<w:spacing w:after="60"/></w:pPr>']
        for t, b, i, col, sz in runs:
            rpr = '<w:rPr>' + ('<w:b/>' if b else '') + ('<w:i/>' if i else '') + \
                  (f'<w:color w:val="{col}"/>' if col else '') + '</w:rPr>'
            xml.append(f'<w:r>{rpr}<w:t xml:space="preserve">{escape(t)}</w:t></w:r>')
        xml.append('</w:p>')
        self.body.append(''.join(xml))

    def image(self, path, caption=None, width_in=PAGE_W_IN):
        rid = f'rIdImg{len(self.images) + 1}'
        fname = f'image{len(self.images) + 1}.png'
        self.images.append((rid, fname, path))
        w, h = png_size(path)
        cx = int(min(width_in, PAGE_W_IN) * EMU_IN)
        cy = int(cx * h / w)
        n = len(self.images)
        self.body.append(
            f'<w:p><w:pPr><w:jc w:val="center"/><w:spacing w:after="60"/></w:pPr>'
            f'<w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">'
            f'<wp:extent cx="{cx}" cy="{cy}"/><wp:docPr id="{n}" name="fig{n}"/>'
            f'<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
            f'<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            f'<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            f'<pic:nvPicPr><pic:cNvPr id="{n}" name="fig{n}"/><pic:cNvPicPr/></pic:nvPicPr>'
            f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
            f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
            f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
            f'</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>')
        if caption:
            self.p([(caption, False, True, '52514E', 18)], space_after=200, align='center')

    def table(self, rows, header=True, widths=None):
        ncol = len(rows[0])
        widths = widths or [int(9360 / ncol)] * ncol
        xml = ['<w:tbl><w:tblPr><w:tblW w:w="9360" w:type="dxa"/>'
               '<w:tblBorders>'
               '<w:top w:val="single" w:sz="4" w:color="C3C2B7"/>'
               '<w:bottom w:val="single" w:sz="4" w:color="C3C2B7"/>'
               '<w:insideH w:val="single" w:sz="4" w:color="E1E0D9"/>'
               '</w:tblBorders>'
               '<w:tblCellMar><w:left w:w="80" w:type="dxa"/><w:right w:w="80" w:type="dxa"/>'
               '</w:tblCellMar></w:tblPr><w:tblGrid>']
        for wd in widths:
            xml.append(f'<w:gridCol w:w="{wd}"/>')
        xml.append('</w:tblGrid>')
        for ri, row in enumerate(rows):
            xml.append('<w:tr>')
            for ci, cell in enumerate(row):
                bold = header and ri == 0
                shade = '<w:shd w:val="clear" w:fill="F0EFEC"/>' if bold else ''
                rpr = '<w:rPr>' + ('<w:b/>' if bold else '') + \
                      '<w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr>'
                xml.append(
                    f'<w:tc><w:tcPr><w:tcW w:w="{widths[ci]}" w:type="dxa"/>{shade}</w:tcPr>'
                    f'<w:p><w:pPr><w:spacing w:after="20"/></w:pPr>'
                    f'<w:r>{rpr}<w:t xml:space="preserve">{escape(str(cell))}</w:t></w:r></w:p></w:tc>')
            xml.append('</w:tr>')
        xml.append('</w:tbl>')
        self.body.append(''.join(xml))
        self.p('', space_after=120)

    # ---- packaging --------------------------------------------------------
    def save(self, out_path):
        doc_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document '
            'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
            'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing">'
            '<w:body>' + ''.join(self.body) +
            '<w:sectPr><w:pgSz w:w="12240" w:h="15840"/>'
            '<w:pgMar w:top="1150" w:right="1300" w:bottom="1150" w:left="1300" '
            'w:header="720" w:footer="720"/></w:sectPr></w:body></w:document>')

        rels = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                '<Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/'
                'officeDocument/2006/relationships/styles" Target="styles.xml"/>'
                '<Relationship Id="rIdNum" Type="http://schemas.openxmlformats.org/'
                'officeDocument/2006/relationships/numbering" Target="numbering.xml"/>']
        for rid, fname, _ in self.images:
            rels.append(f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/'
                        f'officeDocument/2006/relationships/image" Target="media/{fname}"/>')
        rels.append('</Relationships>')

        content_types = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Default Extension="png" ContentType="image/png"/>'
            '<Override PartName="/word/document.xml" ContentType="application/vnd.'
            'openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            '<Override PartName="/word/styles.xml" ContentType="application/vnd.'
            'openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
            '<Override PartName="/word/numbering.xml" ContentType="application/vnd.'
            'openxmlformats-officedocument.wordprocessingml.numbering+xml"/></Types>')

        root_rels = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/'
            '2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>')

        def hstyle(sid, size, color, before, after, bold=True):
            return (f'<w:style w:type="paragraph" w:styleId="{sid}"><w:name w:val="{sid}"/>'
                    f'<w:basedOn w:val="Normal"/>'
                    f'<w:pPr><w:keepNext/><w:spacing w:before="{before}" w:after="{after}"/></w:pPr>'
                    f'<w:rPr>{"<w:b/>" if bold else ""}<w:color w:val="{color}"/>'
                    f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/></w:rPr></w:style>')

        styles = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            '<w:docDefaults><w:rPrDefault><w:rPr>'
            '<w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:cs="Calibri"/>'
            '<w:sz w:val="21"/><w:szCs w:val="21"/><w:color w:val="0B0B0B"/>'
            '</w:rPr></w:rPrDefault>'
            '<w:pPrDefault><w:pPr><w:spacing w:line="276" w:lineRule="auto"/></w:pPr></w:pPrDefault>'
            '</w:docDefaults>'
            '<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>'
            + hstyle('Title', 40, '0B0B0B', 0, 60)
            + hstyle('Heading1', 28, '0B0B0B', 320, 100)
            + hstyle('Heading2', 23, '2A78D6', 240, 80)
            + '<w:style w:type="paragraph" w:styleId="ListParagraph">'
              '<w:name w:val="List Paragraph"/><w:basedOn w:val="Normal"/>'
              '<w:pPr><w:ind w:left="360"/></w:pPr></w:style>'
            '</w:styles>')

        numbering = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:numbering xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            '<w:abstractNum w:abstractNumId="0"><w:lvl w:ilvl="0"><w:start w:val="1"/>'
            '<w:numFmt w:val="bullet"/><w:lvlText w:val="•"/>'
            '<w:pPr><w:ind w:left="360" w:hanging="200"/></w:pPr></w:lvl></w:abstractNum>'
            '<w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num></w:numbering>')

        with zipfile.ZipFile(out_path, 'w', zipfile.ZIP_DEFLATED) as z:
            z.writestr('[Content_Types].xml', content_types)
            z.writestr('_rels/.rels', root_rels)
            z.writestr('word/document.xml', doc_xml)
            z.writestr('word/_rels/document.xml.rels', ''.join(rels))
            z.writestr('word/styles.xml', styles)
            z.writestr('word/numbering.xml', numbering)
            for _, fname, path in self.images:
                z.write(path, f'word/media/{fname}')


# ---------------------------------------------------------------------------
MODEL_META = {  # grid label -> (display label, conditioning length)
    'Mamba3': ('Mamba3-78M', 500),
    'Mamba3_4k': ('Mamba3-4k (2k-finetuned checkpoint)', 4000),
    'S5_4k': ('S5-4k', 4000),
}


def build_report(fig_dir, out_path, model='Mamba3'):
    N = json.load(open(os.path.join(fig_dir, 'numbers.json')))
    emp = N['empirical']
    label, n_cond = MODEL_META.get(model, (model, 500))

    def f(key, field, d=1):
        return f"{N[key][field]:+.{d}f}"

    vb, vs = N['visible-buy'], N['visible-sell']
    ib, isl = N['invisible-buy'], N['invisible-sell']
    no = N['noins']
    r_sat = emp.get('R_131', emp.get('R_100'))
    resp = vb.get('resp_mean') or []
    r_model_100 = next((resp[m] for m in range(min(100, len(resp) - 1), 0, -1)
                        if resp[m] == resp[m]), float('nan'))
    overshoot = r_model_100 / r_sat if r_sat else float('nan')
    # buy/sell average in trade direction cancels any COMMON generator drift
    # (drift D enters +D on buy, -D on sell), isolating the directional response.
    vis_dir = (vb['final_mean'] + vs['final_mean']) / 2
    inv_dir = (ib['final_mean'] + isl['final_mean']) / 2
    pct_inv = 100 * inv_dir / vis_dir if vis_dir else float('nan')
    hot = 'sell' if vs['final_mean'] > vb['final_mean'] else 'buy'
    asym = abs(vs['final_mean'] - vb['final_mean']) / max(abs(vb['final_mean']),
                                                          abs(vs['final_mean']), 1e-9)
    mech_pct = 100 * vb['mech_mean'] / vb['final_mean'] if vb['final_mean'] else float('nan')
    drift_per_win = vb['drift_mean'] / 99  # 100 insertions -> 99 between-windows
    d = Doc()

    d.p([('Where Does the Simulated Market Impact Come From?', True, False, None, 40)],
        style='Title', space_after=40)
    d.p([(f'A control-triangle diagnostic — EA · {label} · Shape I (100 child market orders) '
          '· grid_v2 · 2026-07-06', False, True, '52514E', 20)], space_after=240)

    d.h('1. Summary', 1)
    d.p([('The pipeline is mechanically correct; the large impact is a property of the model, '
          'and we can now say precisely which property. ', True, False, None, None),
         ('We generated the same metaorder experiment three times, changing exactly one ingredient '
          'at a time: whether the child market orders are applied to the simulated order book, and '
          'whether the model sees them in its context. With both switches on, the mid-price moves ',
          False, False, None, None),
         (f'{vb["final_mean"]:+.0f} ticks (buy) / {vs["final_mean"]:+.0f} ticks (sell)',
          True, False, '2A78D6', None),
         ('. Turn off only the model’s visibility of the metaorder — the book still gets '
          'eaten by every child order — and the impact collapses to ', False, False, None, None),
         (f'{ib["final_mean"]:+.0f} / {isl["final_mean"]:+.0f} ticks', True, False, '199E70', None),
         ('. Averaging buy and sell in the trade direction (which cancels any common generator '
          'drift) leaves an invisible-regime directional response of ', False, False, None, None),
         (f'{inv_dir:+.1f} ticks — {pct_inv:.0f}% of the visible {vis_dir:+.0f}', True, False,
          '199E70', None),
         (' — on the scale of the no-insertion drift baseline of ', False, False, None, None),
         (f'{no["final_mean"]:+.0f} ticks', True, False, 'C98500', None),
         (f'. Essentially all ({max(0.0, 100 - max(pct_inv, 0)):.0f}%+) of the simulated impact '
          'is therefore the model’s learned '
          'directional reaction to seeing the child orders in the order flow — not book mechanics, '
          'not an unconditional drift of the generator, and not degradation over the long rollout.',
          False, False, None, None)])
    d.p([('The reaction itself is miscalibrated in one specific way: after each ~16-share child '
          'order the model keeps pushing the price with little or no saturation, while real '
          f'EA data shows the response saturating at {r_sat:+.2f} ticks within ~60 messages. '
          f'Per event the model overshoots by ×{overshoot:.1f}; accumulated over 100 children '
          'this compounds to far above a square-root-law expectation. The model has '
          'learned order-flow momentum but not market resilience — which also makes the measured '
          'build-up exponent δ≈1 a mathematical necessity rather than an empirical discovery.',
          False, False, None, None)])

    d.h('2. The three regimes', 1)
    d.p(f'All three runs share everything: the same {label} checkpoint, the same trading days and '
        f'{n_cond}-message historical conditioning, the same insertion machinery, and rollouts of '
        '~12.7–13k generated messages. Only two switches differ. In the visible regime each child '
        'market order is executed against the simulated book AND encoded into the token stream the '
        'model conditions on. In the invisible regime the child orders still hit the book (the '
        'liquidity really disappears, the mid really jumps), but the model’s context is rolled '
        'forward as if they never happened. In the no-insertion regime nothing is injected at all — '
        'the model simply generates 13,000 messages, which measures its unconditional drift and its '
        f'stability at ~{13000 // n_cond}× the conditioning window.')
    d.image(os.path.join(fig_dir, 'fig1_regimes.png'),
            f'Figure 1. The control triangle: one experiment, two switches. Differencing the three '
            f'outcomes attributes the impact to mechanics, model reaction, or drift. Each box '
            f'states the number of runs averaged.')
    d.table([
        ['Regime', 'Child MOs hit the book', 'Model sees them', 'Final impact (buy / sell, ticks)'],
        ['Visible', 'yes', 'yes', f'{vb["final_mean"]:+.1f} / {vs["final_mean"]:+.1f}  '
         f'(n={vb["n"]}/{vs["n"]})'],
        ['Invisible', 'yes', 'no', f'{ib["final_mean"]:+.1f} / {isl["final_mean"]:+.1f}  '
         f'(n={ib["n"]}/{isl["n"]})'],
        ['No insertions', 'no', 'no', f'{no["final_mean"]:+.1f} ± {no["final_se"]:.1f} (n={no["n"]})'],
    ])
    d.p([('The differencing logic: ', True, False, None, None),
         ('Visible − Invisible isolates the model’s behavioural response to the metaorder; '
          'Invisible − No-insertion isolates the persistence of pure book mechanics; No-insertion '
          'vs zero isolates generator bias. Impact values are reported in the trade direction '
          '(a sell metaorder that lowers the price counts as positive impact).',
          False, False, None, None)])

    d.h('3. What actually happens along the rollout', 1)
    asym_note = (f' — the {hot} side runs noticeably hotter, a buy/sell asymmetry worth '
                 f'reporting on its own' if asym > 0.2 else
                 ' — buy and sell are roughly symmetric')
    d.p(f'Figure 2 overlays the mean mid-price trajectory of all five runs on a common axis '
        f'(mean ± 2 s.e. bands). Both visible curves build up steadily in the direction of the '
        f'observed flow, and only they do: buy moves the mid {vb["final_mean"]:+.0f} ticks '
        f'and sell {vs["final_mean"]:+.0f} ticks in the trade direction{asym_note}. The controls '
        f'stay near zero for the entire rollout: their buy/sell average in trade-direction units '
        f'(which cancels common generator drift) is {inv_dir:+.1f} ticks '
        f'(invisible: {ib["final_mean"]:+.1f} buy / {isl["final_mean"]:+.1f} sell), i.e. mostly a '
        f'common small bias shared with the no-insertion run, not a directional response. '
        f'The transition from history to generation is seamless '
        f'in all regimes (boundary jump ≤0.12 ticks; zero crossed or invalid book states across '
        f'the whole grid), so the divergence between the curves cannot be a bookkeeping artifact.')
    d.image(os.path.join(fig_dir, 'fig2_trajectories.png'),
            f'Figure 2. Cumulative mid move in the trade direction (mean ± 2 s.e.; n={vb["n"]} runs '
            f'per curve). Visible '
            f'buy/sell build to {vb["final_mean"]:+.0f}/{vs["final_mean"]:+.0f} ticks; invisible '
            'and no-insertion runs are flat.')

    d.h('4. Decomposing the visible impact: mechanics vs model reaction', 1)
    d.p([('Within the visible regime we can split every trajectory exactly: the ', False, False, None, None),
         ('mechanical', True, False, '4A3AA7', None),
         (' part is the mid jump at each insertion row (the child order eating the touch level), and the ',
          False, False, None, None),
         ('model-generated', True, False, 'EB6834', None),
         (' part is everything the model writes between insertions. Mechanics contribute '
          f'{vb["mech_mean"]:+.1f} ticks (buy) — about {mech_pct:.0f}% of the total. The other '
          f'{vb["drift_mean"]:+.1f} ticks are messages the model chose to generate after watching '
          'the child orders: quotes walking away, same-side follow-on flow, cancellations on the '
          f'attacked side. Between insertions the mid keeps moving with the trade '
          f'({drift_per_win:+.2f} ticks per window on the buy side) instead '
          'of reverting — the signature of momentum without resilience.', False, False, None, None)])
    d.image(os.path.join(fig_dir, 'fig3_decomposition.png'),
            f'Figure 3. Mechanical vs model-generated contribution per regime (trade direction; bars '
            f'are means over n={vb["n"]} runs each). '
            'In the invisible regime the model-generated part matches the no-insertion baseline.')
    d.p('A note on the invisible bars: the legacy invisible code path logs the book state before '
        'the insertion is applied, so its mechanical component is partially hidden in the saved '
        'files; what matters here is the model-generated component, which is baseline-flat.')

    d.h('5. Per-event calibration against real data', 1)
    d.p(f'The cleanest calibration test is the event response R(m): the average mid move m messages '
        f'after an execution, measured identically in the model rollouts and in the real EA message '
        f'stream ({emp["n_events"]} real executions from the conditioning data). Real EA responses '
        f'rise to about {emp.get("R_30", 0):+.2f} ticks after 30 messages and saturate at '
        f'{r_sat:+.2f} ticks — transient impact decays into a small permanent component. The '
        f'visible-regime model response climbs to ~{r_model_100:.1f} ticks at m=100, '
        f'×{overshoot:.1f} above the real-data curve — and the real-data '
        f'number is itself an upper '
        f'bound on causal impact, since organic order flow is autocorrelated. The invisible-regime '
        f'response is flat, confirming that the excess is driven entirely by what the model sees.')
    d.image(os.path.join(fig_dir, 'fig4_event_response.png'),
            f'Figure 4. Event response R(m) in ticks: model (visible / invisible; '
            f'~{vb.get("resp_n_events", 0)} child events pooled from n={vb["n"]} runs per curve) '
            f'vs real EA data ({emp["n_events"]} executions). '
            'The model overshoots and never saturates; real impact saturates within ~60 messages.')
    d.p('Two unit caveats worth carrying into any write-up: the children actually execute ~16 '
        'shares, not the configured 75 (the insertion clips at the top-of-queue order), and the '
        '10% participation target is defined on trade count — in volume terms participation is '
        '~4–5%, which makes a square-root-law benchmark even smaller.')

    hs = N.get('hist_spread', float('nan'))
    d.h('6. Validity gate V0: book health (a sanity check, not a criterion)', 1)
    d.p([('Book health is deliberately a gate, not a headline criterion: ', True, False, None, None),
         ('it certifies that long-horizon generation is trustworthy enough for the impact '
          'measurements to mean anything, and nothing more. A model does not "win" by keeping a '
          'tidy book — it merely qualifies for the real tests. The reference point is the real '
          f'data itself: the historical EA streams have a mean spread of ~{hs:.1f} ticks '
          '(dashed green line in Figure 5).', False, False, None, None)])
    d.p(f'Against that anchor the story is clean: the no-insertion '
        f'({no["spread_first"]:.1f}→{no["spread_last"]:.1f} ticks) and invisible '
        f'({ib["spread_first"]:.1f}→{ib["spread_last"]:.1f}) runs hold essentially the historical '
        f'level for all 13k messages — generating at ~{13000 // n_cond}× the conditioning window '
        f'does not by itself '
        f'break the book. The visible run degrades from ~{vb["spread_first"]:.1f} to '
        f'~{vb["spread_last"]:.1f} ticks, i.e. the degradation is metaorder-induced, part of the '
        f'same overreaction as the price drift. Two housekeeping notes: L10 depth roughly doubles '
        f'over long rollouts (the book slowly "fills up"), so depth-sensitive metrics need care; '
        f'and a V0 failure does not invalidate a model’s impact numbers by itself — it flags that '
        f'they must be read jointly with the book distortion.')
    d.image(os.path.join(fig_dir, 'fig5_spread.png'),
            f'Figure 5. Mean bid–ask spread along the rollout vs the historical level (dashed '
            'green). Controls hold the real-data level; only the visible-metaorder regime '
            'degrades.')

    d.h('7. The framework: properties, definitions, worked example', 1)
    d.p([('Each property below is a concrete statistic with a pass criterion, so the framework is '
          'falsifiable and model-agnostic. To keep it abstract we walk one model through all of '
          'them — the Hawkes baseline (numbers from the full-grid audit, 30 samples/side). ',
          False, False, None, None),
         ('The ordering is load-bearing: V0 gates, P1–P2 are controls, P3 is the property being '
          'evaluated, P4–P5 only mean something once P3 holds.', True, False, None, None)])
    d.table([
        ['Property', 'Statistic (how measured)', 'Pass criterion', 'Hawkes (worked example)'],
        ['V0 Validity (gate)',
         'Crossed/sentinel book rows; history→generation boundary jump; spread & depth of the '
         'NO-insertion run vs the historical level over the full horizon',
         'Zero unlawful rows; jump ≈ 0; no trend absent intervention',
         'Books lawful, but spread creeps 5.3→8.2 ticks even though the generator never sees the '
         'metaorder — an intrinsic ratchet. V0 FLAG: fix before quoting impact.'],
        ['P1 No unconditional drift',
         'No-insertion run: mean signed final mid move ± s.e. over samples',
         '|mean| ≲ 2 s.e. and ≪ the impact scale',
         'Passes by construction: stationary intensities cannot drift.'],
        ['P2 Blind ⇒ mechanics only',
         'Invisible run: total impact minus the mechanical jumps at insertions, buy vs sell',
         'Non-mechanical part ≈ P1 baseline and does NOT flip sign with trade direction',
         'Passes vacuously: intensities never condition on the flow, so visible ≡ invisible.'],
        ['P3 Directional response',
         'Δ = visible − invisible final impact in the trade direction; antisymmetry buy vs sell',
         'Δ > 0 and buy/sell antisymmetric (this is the property under evaluation)',
         'FAILS: mech ~2 ticks after 100 children, drift direction-independent (±2) → Δ ≈ 0. '
         'No learned reaction, no permanent impact.'],
        ['P4 Per-event calibration',
         'R_model(m) per child vs R_real(m) on real data at matched child size (Figure 4)',
         'Ratio ≈ 1 at the saturation horizon',
         'Under-responds: mechanical kick, then flat ≈ 0 vs real +0.56 ticks.'],
        ['P5 Resilience (conditional on P3)',
         '(a) R(m) saturates: R(131)−R(60) ≈ 0 within 2 s.e.; (b) inter-insertion reversion ≤ 0; '
         '(c) Shape II relaxation ratio I_end/I_peak < 1 after execution stops',
         'All three; real data: (a) holds, impact decays to a partial permanent level',
         'Passes trivially (reversion +0.01 ≈ 0, R(m) flat) — but vacuously, since there is no '
         'response to relax. P5 without P3 is meaningless.'],
    ], widths=[1500, 2900, 2200, 2760])
    d.p('Hawkes thus lands in one failure mode: trivially resilient but impact-blind (P3 fail, '
        'P5 vacuous). The neural models are the exact mirror image: strong P3 (Sections 3–4), '
        'overshooting P4 by ×2–3 (Section 5), failing P5 on all three statistics — no R(m) '
        'saturation, positive inter-insertion drift (+1.1 ticks/window), Shape II relaxation '
        'ratio ≈ 0.9–1.0 where real markets decay to a partial permanent level. The framework’s '
        'value is that it separates these two failure modes cleanly: a usable simulator must pass '
        'P3 AND P4–P5 simultaneously, and currently no model in our zoo does.')

    d.h('8. What β/δ measures — and what it does not', 1)
    d.p([('β/δ is a build-up exponent measured DURING execution, not a statement about the final '
          'price or about temporary impact decaying into permanent. ', True, False, None, None),
         ('Concretely: at each insertion k we take I(k) = signed mid move from the start of the '
          'metaorder to insertion k, and Q(k) = cumulative executed volume (= k × child size '
          'within a day); δ is the slope of log I on log Q pooled over k, samples and days. The '
          'final mid enters only as the last point of the build-up curve; nothing after the last '
          'child is used.', False, False, None, None)])
    d.p('The temporary-vs-permanent question is a separate measurement — Shape II: 10 insertions '
        'followed by 100 cooling windows, where execution stops and we watch I(t) relax. Its '
        'headline statistic is the relaxation ratio I_end/I_peak (P5c above): real markets decay '
        'to a partial permanent level (propagator-literature ballpark ~2/3 of peak); our neural '
        'models sit at ≈0.9–1.0, i.e. nothing decays. The two measurements are linked by an '
        'identity: if the per-child response is constant in k (P4, flat marginal response) and '
        'nothing relaxes between children (P5b), then I(k) ∝ k and the fitted δ is exactly 1. '
        'So δ is DETERMINED by P4+P5 — a resilience diagnostic, not the impact-vs-size law '
        'I(Q)~Q^δ, which would require varying total metaorder size independently. This also '
        'explains why all well-behaved generative models cluster near the same δ.')

    d.h('9. Using this in the paper', 1)
    d.p([('This diagnostic is arguably more publishable than the raw impact curves. ', True, False, None, None),
         ('It gives the paper a falsifiable protocol: any generative LOB simulator claiming '
          'realistic impact should pass the control triangle (P1–P3) and the calibration/'
          'resilience tests (P4–P5) as defined in Section 7. Suggested placement: a "Where does '
          'the impact come from?" subsection in Results using Figures 1–4 (Figure 5 and the V0 '
          'gate go to the appendix), with the Section 7 table as the evaluation rubric applied '
          'to every model. All figures are 200-dpi PNGs with a colorblind-safe palette; regime '
          'colors are consistent across figures (blue = visible, teal = invisible, yellow = no '
          'insertions, green = real data).', False, False, None, None)])

    d.h('10. Caveats', 1)
    d.bullet('The invisible regime uses the legacy code branch, which conditions the model on the '
             'pre-insertion book while decoding prices against the post-insertion mid (a "decode '
             'ratchet"). In practice it produced no directional push, but the branch should be '
             'rewritten symmetrically to the visible one before publishing invisible-regime numbers.')
    d.bullet('Insertions clip at the top-of-queue order: configured 75 shares, executed ~16 on '
             'average, and the touch level is fully wiped in only 3–15% of insertions. Mechanical '
             'impact is genuinely small partly for this reason.')
    d.bullet('The "Mamba3-4k" checkpoint is not actually 4k-trained (it re-saves 2k-finetuned '
             'weights); its results should be labelled accordingly. The S5-4k checkpoint is '
             'genuinely 4k-trained but has no recorded validation.')
    d.bullet('The real-data R(m) is an upper bound on causal impact (organic flow autocorrelation '
             'is included); the model exceeds even this bound.')
    d.bullet(f'Sample sizes: visible/invisible n={vb["n"]}/{ib["n"]} per side here; '
             f'no-insertion n={no["n"]}. Error bands in Figures 2 and 4 are ±2 s.e.')

    d.h('11. Reproduction', 1)
    d.p([('Data: ', True, False, None, None),
         (f'/lus/lfs1aip2/projects/u6gb/lob_impact_grid_v2/EA-{model}-beta (visible) and '
          f'/lus/lfs1aip2/projects/u6gb/lob_impact_controls_v2/{{invisible,noins}}/EA-{model}-beta '
          '(controls, generated with the METAORDER_VISIBLE=0 and N_INS_OVERRIDE=1 '
          'MB_OVERRIDE=13000 knobs of lob_impact/3_scenarios/run_experiments.sh). ',
          False, False, None, None),
         ('Analysis: ', True, False, None, None),
         ('lob_impact/4_diagnostics/{control_triangle_report.py, verify_integrity.py, '
          'empirical_response.py}; figures and numbers.json regenerate with a single sbatch of '
          'run_triangle_report.sh.', False, False, None, None)])

    d.save(out_path)
    print(f'DOCX_DONE -> {out_path}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--fig_dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', default='Mamba3', help='grid model label (see MODEL_META)')
    a = ap.parse_args()
    build_report(a.fig_dir, a.out, a.model)
