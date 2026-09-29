"""Small template helpers: percentages, severity/status colours, list lookups."""

from django import template

register = template.Library()

STATUS_COLOURS = {
    "Classified": "success", "Uncertain": "warning", "Alert Generated": "danger", "Manual Review": "info",
    "Reviewed": "primary", "Closed": "secondary", "Uploaded": "light",
}
CONSISTENCY_COLOURS = {"Acceptable Match": "success", "Weak Match": "warning", "Model Disagreement": "danger", "Uncertain Result": "secondary"}
ALERT_COLOURS = {"active": "danger", "escalated": "dark", "acknowledged": "warning", "dismissed": "secondary", "resolved": "success"}
QUALITY_COLOURS = {"Good": "success", "Acceptable": "primary", "Poor": "warning", "Unusable": "danger"}


@register.filter
def pct(value, digits=0):
    """0.9234 -> '92%' (or '92.3%' with digits=1); '—' for None; tiny values show as '<0.1%' instead of 0."""
    if value is None or value == "":
        return "—"
    digits = int(digits)
    percent = float(value) * 100
    smallest = 10 ** -digits
    if 0 < percent < smallest:
        return f"<{smallest:.{digits}f}%"
    return f"{percent:.{digits}f}%"


@register.filter
def width(value):
    """A probability as a CSS width percentage for bars."""
    try:
        return f"{max(0.0, min(1.0, float(value))) * 100:.1f}"
    except (TypeError, ValueError):
        return "0"


@register.filter
def sev(value):
    return f"sev sev-{str(value).lower()}"


@register.filter
def status_colour(value):
    return STATUS_COLOURS.get(value, "secondary")


@register.filter
def consistency_colour(value):
    return CONSISTENCY_COLOURS.get(value, "secondary")


@register.filter
def alert_colour(value):
    return ALERT_COLOURS.get(value, "secondary")


@register.filter
def quality_colour(value):
    return QUALITY_COLOURS.get(value, "secondary")


@register.filter
def top_class(scores):
    """The best class of a {class: probability} dict."""
    if not scores:
        return ""
    return max(scores, key=scores.get)


@register.filter
def get(mapping, key):
    return (mapping or {}).get(key)


@register.filter
def filesize(value):
    size = float(value or 0)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024


@register.simple_tag(takes_context=True)
def query(context, **changes):
    """The current page's query string with some parameters changed ("" removes one): keeps filters on links."""
    params = context["request"].GET.copy()
    for key, value in changes.items():
        if value in (None, ""):
            params.pop(key, None)
        else:
            params[key] = value
    return params.urlencode()


@register.filter
def index(sequence, position):
    """sequence[position] (for tables that show several parallel lists)."""
    try:
        return sequence[int(position)]
    except (IndexError, TypeError, ValueError):
        return ""


@register.filter
def chart_height(rows):
    """Height in px for a horizontal bar chart with this many rows (about 30 px a bar, at least 160)."""
    return max(160, int(rows) * 30 + 40)
