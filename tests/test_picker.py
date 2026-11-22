import pytest
from PIL import Image

from artgen.config import ConfigError
from artgen.mask import overlay_box
from artgen.picker import edit_box, pick, show_styles
from artgen.pnginfo import read_params

from test_forge import ENV, FakePipe, forge, pipe, repo, write  # noqa: F401  (fixtures)


class _Widget:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class Button(_Widget):
    def on_click(self, f):
        self._handler = f

    def click(self):
        self._handler(self)


class IntSlider(_Widget):
    def observe(self, f, names=None):
        self.__dict__["_observer"] = f

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name == "value" and "_observer" in self.__dict__:
            self._observer({"name": "value", "new": value})


class Image_(_Widget):
    pass


class Label(_Widget):
    pass


class HTML(_Widget):
    pass


class VBox(_Widget):
    pass


class HBox(_Widget):
    pass


class FakeWidgets:
    def __init__(self):
        self._made = []

    def created(self, name):
        return [w for n, w in self._made if n == name]

    def _factory(self, name, cls):
        def make(*a, **kw):
            if a:
                kw["children"] = a[0]
            w = cls(**kw)
            self._made.append((name, w))
            return w
        return make

    def __getattr__(self, name):
        classes = {"Button": Button, "IntSlider": IntSlider, "Image": Image_, "Label": Label,
                   "HTML": HTML, "VBox": VBox, "HBox": HBox}
        if name not in classes:
            raise AttributeError(name)
        return self._factory(name, classes[name])


@pytest.fixture
def fake_widgets():
    return FakeWidgets()


def noop(*a, **k):
    return None


def test_overlay_box_still_importable():
    im = Image.new("RGB", (10, 10), (255, 255, 255))
    assert overlay_box(im, (1, 1, 5, 5), width=1).size == (10, 10)


def test_pick_adopt_button_adopts(forge, fake_widgets):
    paths = forge.generate("c1", "face", seeds=[1, 2])
    pick(forge, paths, widgets=fake_widgets, display=noop)
    buttons = fake_widgets.created("Button")
    assert [b.description for b in buttons] == ["採用", "左右反転して採用"] * 2
    buttons[1].click()
    assert read_params(forge.selected.face_path("c1"))["flipped"] is True
    assert "採用: " in fake_widgets.created("Label")[0].value


def test_pick_defaults_to_latest_run(forge, fake_widgets):
    forge.generate("c1", "face", seeds=[1])
    pick(forge, widgets=fake_widgets, display=noop)
    assert len(fake_widgets.created("Button")) == 2


def test_pick_has_no_flip_for_expressions(forge, fake_widgets):
    forge.adopt(forge.generate("c1", "face", seeds=[1])[0])
    paths = forge.inpaint("c1", "hurt", seeds=[3])
    pick(forge, paths, widgets=fake_widgets, display=noop)
    assert [b.description for b in fake_widgets.created("Button")] == ["採用"]


def test_pick_shows_error_text_instead_of_raising(forge, fake_widgets):
    p = forge.generate("c1", "face", seeds=[1])[0]
    pick(forge, [p], widgets=fake_widgets, display=noop)
    p.unlink()
    fake_widgets.created("Button")[0].click()
    assert "採用できない: " in fake_widgets.created("Label")[-1].value


def test_edit_box_saves_selection(forge, fake_widgets):
    forge.adopt(forge.generate("c1", "face", seeds=[1])[0])
    edit_box(forge, "c1", widgets=fake_widgets, display=noop)
    x0, y0, x1, y1 = fake_widgets.created("IntSlider")
    assert (x0.value, y0.value, x1.value, y1.value) == (100, 100, 400, 300)
    x1.value = 600
    fake_widgets.created("Button")[0].click()
    assert forge.selected.read_box("c1") == (100, 100, 600, 300)
    assert fake_widgets.created("Button")[0].description == "この範囲で決める"


def test_edit_box_rejects_bad_box(forge, fake_widgets):
    forge.adopt(forge.generate("c1", "face", seeds=[1])[0])
    edit_box(forge, "c1", widgets=fake_widgets, display=noop)
    x0, y0, x1, y1 = fake_widgets.created("IntSlider")
    x1.value = x0.value + 16  # feather 24 の 2 倍より狭い
    fake_widgets.created("Button")[0].click()
    assert forge.selected.read_box("c1") is None
    assert "決められない: " in fake_widgets.created("Label")[-1].value


def test_edit_box_without_face_says_what_to_do(forge, fake_widgets):
    with pytest.raises(ConfigError, match="顔を採用"):
        edit_box(forge, "c1", widgets=fake_widgets, display=noop)
    assert fake_widgets._made == []


def test_show_styles_one_composite_and_row_labels(forge, fake_widgets):
    import io

    paths = forge.compare_styles("c1", "face", styles=["s", "t"], seeds=[1, 2])
    show_styles(paths, display=noop, widgets=fake_widgets, thumb=32)
    images = fake_widgets.created("Image")
    assert len(images) == 1
    with Image.open(io.BytesIO(images[0].value)) as im:
        assert im.size == (64, 64)  # 2 列 x 2 行
    labels = [h.value for h in fake_widgets.created("HTML")]
    assert any("<b>s</b>" in v for v in labels) and any("<b>t</b>" in v for v in labels)
