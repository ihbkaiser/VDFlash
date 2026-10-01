import sys

import evaluate
from evaluate.utils import launch_gradio_widget


sys.path = [p for p in sys.path if p != "/path/to/local-resource"]
module = evaluate.load("mauve")
sys.path = ["/path/to/local-resource"] + sys.path

launch_gradio_widget(module)
