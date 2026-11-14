"""Combine saved Cartpole, Rocket and Reacher trajectory figures as vector art.

Uses the system Python with pypdf and Pillow, plus the pdftocairo executable.
Optional axis changes redraw saved Cartpole records using Matplotlib and NumPy.
No controller or simulator is run.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import platform
import subprocess


import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.plots.common import PAPER_RESULTS, REPOSITORY, paper_font, replay_input_path
FONT_FAMILY = paper_font()

from PIL import Image, ImageChops
from pypdf import PdfReader, PdfWriter, Transformation
from pypdf.generic import NameObject, RectangleObject
import pypdf


TASKS = ('cartpole', 'rocket', 'reacher')
DEFAULT_ROOT = (Path(__file__).resolve().parents[2] / "results/paper")
METHOD_COLORS = {'BD-χ²': '#2ca02c', 'Select-DPC': '#ff7f0e'}
LAYOUT_DESCRIPTIONS = {
    'original': 'Outer whitespace is cropped and columns are vertically centered at their '
                'original physical scale. Aspect ratios are preserved.',
    'equal-height': 'Outer whitespace is cropped and each column is scaled uniformly to '
                    'the tallest source column. All columns share the same top and bottom '
                    'edges; aspect ratios are preserved.',
    'equal-size': 'Outer whitespace is cropped and all columns are stretched to the same '
                  'height and width. The height is that of the tallest source column and '
                  'the width is the mean original column width. Horizontal and vertical '
                  'scale factors differ: drawing geometry and text proportions change.',
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def content_bounds(page, png_path, padding=3.):
    """Locate outer whitespace in the matching PNG, expressed in PDF points."""
    with Image.open(png_path) as image:
        rgb = image.convert('RGB')
        bounds = ImageChops.difference(rgb, Image.new('RGB', rgb.size, 'white')).getbbox()
        if bounds is None:
            raise ValueError(f'Empty source figure: {png_path}')
        width, height = image.size
    page_width, page_height = float(page.mediabox.width), float(page.mediabox.height)
    left, top, right, bottom = bounds
    return [max(0, left * page_width / width - padding),
            max(0, page_height - bottom * page_height / height - padding),
            min(page_width, right * page_width / width + padding),
            min(page_height, page_height - top * page_height / height + padding)]


def remove_method_titles(page):
    """Remove only exact method-name text blocks from an in-memory PDF page."""
    contents = page.get_contents()
    operations = []
    text_block = None
    removed = []
    for operands, operator in contents.operations:
        if operator == b'BT':
            text_block = []
        if text_block is None:
            operations.append((operands, operator))
            continue
        text_block.append((operands, operator))
        if operator != b'ET':
            continue
        fragments = []
        for arguments, instruction in text_block:
            if instruction == b'TJ':
                fragments.extend(value for value in arguments[0] if isinstance(value, str))
            elif instruction == b'Tj':
                fragments.append(str(arguments[0]))
        text = ''.join(fragments)
        if text in METHOD_COLORS:
            removed.append(text)
        else:
            operations.extend(text_block)
        text_block = None
    if text_block is not None or sorted(removed) != sorted(METHOD_COLORS):
        raise ValueError(f'Expected one standalone title per method; found {removed}')
    contents.operations = operations
    page[NameObject('/Contents')] = contents
    return removed


def shared_legend(width, height=26.):
    """Build an embedded-font vector legend matching the source plots."""
    import matplotlib
    from matplotlib import font_manager
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    font = Path(font_manager.findfont(font_manager.FontProperties(family=FONT_FAMILY),
                                     fallback_to_default=False))
    with matplotlib.rc_context({'pdf.fonttype': 42}):
        figure = Figure(figsize=(width / 72, height / 72))
        handles = [Line2D([], [], color=color, linewidth=2, linestyle='--', label=method)
                   for method, color in METHOD_COLORS.items()]
        figure.legend(handles=handles, loc='center', ncol=2, frameon=False,
                      prop=font_manager.FontProperties(fname=font, size=12),
                      handlelength=2.5, columnspacing=2.5)
        buffer = BytesIO()
        figure.savefig(buffer, format='pdf', transparent=True)
    page = PdfReader(buffer).pages[0]
    return page, {'font': str(font), 'font_sha256': sha256(font),
                  'colors': METHOD_COLORS, 'matplotlib': matplotlib.__version__}


def redraw_cartpole(root, source_folder, output, x_limits):
    """Draw the saved Cartpole trajectory with the specified viewing limits."""
    from scripts.plots import cartpole
    import numpy as np
    from matplotlib import font_manager, pyplot as plt

    source_manifest = json.loads((source_folder / 'manifest.json').read_text())
    seed = source_manifest['selected_seed']
    pair = {method: cartpole.load_trial(root / 'cartpole' / method / f'seed_{seed}', method)
            for method in cartpole.METHODS}
    geometry = cartpole.pole_geometry(pair)
    drawn_x = []
    for trial in pair.values():
        angles = trial.states[:, 1]
        tip_x = (trial.states[:, 0] + geometry[-1, 0] * np.cos(angles)
                 + geometry[-1, 2] * np.sin(angles))
        drawn_x.extend(tip_x)
        for index in cartpole.trajectory_pose_indices(trial):
            state = trial.states[index]
            drawn_x.extend(cartpole.pose_coordinates(state, geometry)[0])
            drawn_x.extend([state[0] - .12, state[0] + .12])
    drawn_bounds = [float(min(drawn_x)), float(max(drawn_x))]
    if x_limits[0] >= drawn_bounds[0] or x_limits[1] <= drawn_bounds[1]:
        raise ValueError(f'Cartpole x limits must contain the full drawing: {drawn_bounds}')
    font = Path(font_manager.findfont(FONT_FAMILY, fallback_to_default=False))
    cartpole.pin(font)
    output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({'font.family': FONT_FAMILY, 'font.size': 10,
                         'axes.labelsize': 11, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'svg.fonttype': 'none'}):
        figure, axes = plt.subplots(2, 1, figsize=(4.4, 5.2), sharex=True, sharey=True)
        figure.subplots_adjust(left=.16, right=.98, bottom=.11, top=.93, hspace=.28)
        cartpole.draw_trajectory_axes(axes, pair, geometry)
        for axis in axes:
            axis.set_xlim(*x_limits)
        cartpole.save_figure(figure, output, 'trajectory_comparison')
    manifest = {'selected_seed': seed, 'xlim_m': list(x_limits), 'ylim_m': [-.75, .75],
                'aspect': 'equal', 'drawn_x_bounds_m': drawn_bounds,
                'source_manifest_sha256': sha256(source_folder / 'manifest.json'),
                'drawing_script': str(Path(cartpole.__file__).resolve()),
                'drawing_script_sha256': sha256(Path(cartpole.__file__)),
                'input_sha256': dict(cartpole.INPUT_HASHES),
                'simulation_steps': 0, 'controller_calls': 0,
                'output_sha256': {path.name: sha256(path) for path in sorted(output.iterdir())
                                  if path.name.startswith('trajectory_comparison.')}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--layout', choices=LAYOUT_DESCRIPTIONS, default='equal-height')
    parser.add_argument('--shared-legend', action=argparse.BooleanOptionalAction, default=True,
                        help='Replace repeated method titles with a single legend.')
    parser.add_argument('--cartpole-xlim', type=float, nargs=2, metavar=('MIN', 'MAX'), default=(-0.7, 0.8),
                        help='Redraw Cartpole with narrower x limits, preserving equal aspect.')
    parser.add_argument('--gap', type=float, default=14., help='Gap between columns in PDF points.')
    arguments = parser.parse_args()
    if arguments.gap < 0:
        raise ValueError('Column gap must be nonnegative')
    if arguments.cartpole_xlim and not arguments.cartpole_xlim[0] < arguments.cartpole_xlim[1]:
        raise ValueError('Cartpole x limits must be increasing')
    comparisons = arguments.root / 'figures'
    output = comparisons / 'three_tasks'
    output = arguments.output or output
    output.mkdir(parents=True, exist_ok=True)
    sources, pages = [], []
    for task in TASKS:
        folder = comparisons / task
        if task == 'cartpole' and arguments.cartpole_xlim:
            folder = redraw_cartpole(arguments.root, folder, output / 'cartpole_view',
                                     arguments.cartpole_xlim)
        manifest = json.loads((folder / 'manifest.json').read_text())
        paths = [folder / f'trajectory_comparison.{extension}' for extension in ('pdf', 'png')]
        for path in paths:
            if sha256(path) != manifest['output_sha256'][path.name]:
                raise ValueError(f'Source does not match its saved manifest: {path}')
        reader = PdfReader(paths[0])
        if len(reader.pages) != 1:
            raise ValueError(f'Expected a single-page figure: {paths[0]}')
        page = reader.pages[0]
        if page.rotation or float(page.mediabox.left) or float(page.mediabox.bottom):
            raise ValueError('Source figures require unrotated, zero-origin PDF pages')
        bounds = content_bounds(page, paths[1])
        page.cropbox = RectangleObject(bounds)
        removed = remove_method_titles(page) if arguments.shared_legend else []
        pages.append(page)
        sources.append({'task': task, 'selected_seed': manifest.get('selected_seed'),
                        'pdf': str(paths[0].resolve()), 'png': str(paths[1].resolve()),
                        'input_sha256': {path.name: sha256(path) for path in paths},
                        'source_manifest_sha256': sha256(folder / 'manifest.json'),
                        'crop_points': bounds, 'removed_method_titles': removed})
    margin = 4.
    original_widths = [float(page.cropbox.width) for page in pages]
    original_heights = [float(page.cropbox.height) for page in pages]
    common_height = max(original_heights)
    if arguments.layout == 'original':
        widths, heights = original_widths, original_heights
    elif arguments.layout == 'equal-height':
        widths = [width * common_height / height
                  for width, height in zip(original_widths, original_heights)]
        heights = [common_height] * len(pages)
    else:
        widths = [sum(original_widths) / len(pages)] * len(pages)
        heights = [common_height] * len(pages)
    height = common_height + 2 * margin
    width = sum(widths) + arguments.gap * (len(pages)-1) + 2 * margin
    writer = PdfWriter()
    canvas = writer.add_blank_page(width=width, height=height)
    x = margin
    for page, source, panel_width, panel_height in zip(pages, sources, widths, heights):
        y = (height - panel_height) / 2
        scale_x = panel_width / float(page.cropbox.width)
        scale_y = panel_height / float(page.cropbox.height)
        transform = (Transformation()
                     .translate(-float(page.cropbox.left), -float(page.cropbox.bottom))
                     .scale(scale_x, scale_y).translate(x, y))
        canvas.merge_transformed_page(page, transform)
        source['placement_points'] = [x, y, panel_width, panel_height]
        source['scale_xy'] = [scale_x, scale_y]
        x += panel_width + arguments.gap
    legend = None
    if arguments.shared_legend:
        legend_page, legend = shared_legend(width)
        legend_height = float(legend_page.mediabox.height)
        # Reuse the space freed by the top-row titles; keep column geometry fixed.
        legend_y = height - legend_height
        canvas.merge_translated_page(legend_page, 0, legend_y)
        legend['placement_points'] = [0, legend_y, width, legend_height]
    writer.add_metadata({'/Title': 'Cartpole, Rocket, Reacher trajectory comparisons',
                         '/Subject': LAYOUT_DESCRIPTIONS[arguments.layout]})
    pdf = output / 'trajectory_comparison.pdf'
    with pdf.open('wb') as stream:
        writer.write(stream)
    subprocess.run(['pdftocairo', '-png', '-singlefile', '-r', '240', str(pdf),
                    str(output / 'trajectory_comparison')], check=True)
    subprocess.run(['pdftocairo', '-svg', str(pdf), str(output / 'trajectory_comparison.svg')], check=True)
    layout_argument = '' if arguments.layout == 'original' else f' --layout {arguments.layout}'
    if arguments.shared_legend:
        layout_argument += ' --shared-legend'
    axis_description = ''
    if arguments.cartpole_xlim:
        left, right = arguments.cartpole_xlim
        layout_argument += f' --cartpole-xlim {left:g} {right:g}'
        axis_description = (f'Cartpole uses x limits [{left:g}, {right:g}] m and '
                            'y limits [-0.75, 0.75] m, with equal physical scaling on both axes. '
                            'The full tip paths and every displayed cart-and-pole pose are retained. ')
    label_description = (
        'One shared legend uses green for BD-χ² and orange for Select-DPC. '
        if arguments.shared_legend else 'Each panel is labeled with its method. ')
    (output / 'README.md').write_text(
        '# Three-task trajectory comparison\n\n'
        'Left to right: Cartpole (seed 10006), Rocket (seed 3005), Reacher (seed 4000). '
        'Each column retains BD-χ² above Select-DPC. '
        f'{LAYOUT_DESCRIPTIONS[arguments.layout]} '
        f'{label_description}{axis_description}\n\n'
        '[PDF](trajectory_comparison.pdf) · [SVG](trajectory_comparison.svg) · [PNG](trajectory_comparison.png)\n\n'
        'Regenerate from the repository root using the system Python (pypdf and Pillow installed):\n\n'
        '```bash\nPYTHONDONTWRITEBYTECODE=1 python '
        f'scripts/plots/combine_trajectories.py{layout_argument}\n```\n\n'
        'The script also requires `pdftocairo`; the shared legend requires Matplotlib. '
        'Figures use Times New Roman when available, otherwise DejaVu Serif.\n')
    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'order': list(TASKS),
              'sources': sources, 'layout': arguments.layout,
              'layout_description': LAYOUT_DESCRIPTIONS[arguments.layout],
              'shared_legend': legend,
              'cartpole_xlim_m': arguments.cartpole_xlim,
              'size_points': [width, height],
              'script_sha256': sha256(Path(__file__)), 'python': platform.python_version(),
              'pypdf': pypdf.__version__, 'controller_calls': 0, 'simulation_steps': 0,
              'outputs': {path.name: sha256(path) for path in sorted(output.iterdir())
                          if path.name in ('trajectory_comparison.pdf', 'trajectory_comparison.png',
                                           'trajectory_comparison.svg', 'README.md')}}
    (output / 'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    print(pdf)


if __name__ == '__main__':
    main()
