from random import randint

from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Display.SimpleGui import init_display


class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

    @staticmethod
    def random_color():
        return Quantity_Color(randint(0, 128) / 255.0,
                              randint(0, 128) / 255.0,
                              randint(0, 128) / 255.0,
                              Quantity_TOC_RGB)


def viz_shapes(shapes, transparency=0, backend_str=None):
    display, start_display, add_menu, add_function_to_menu = init_display(backend_str=backend_str)
    if len(shapes) == 1:
        display.DisplayShape(shapes[0], update=True, transparency=transparency)
    else:
        for shape in shapes:
            display.DisplayShape(shape, update=True, color=Colors.random_color(), transparency=transparency)
    display.FitAll()
    start_display()


def export_edges(l_v, v_file):
    with open(v_file, "w") as f:
        line_str = ""
        num_points = 0
        for edge in l_v:
            for v in edge:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for i in range(0, edge.shape[0] - 1):
                line_str += f"l {i + num_points + 1} {i + num_points + 2}\n"
            num_points += edge.shape[0]
        f.write(line_str)
