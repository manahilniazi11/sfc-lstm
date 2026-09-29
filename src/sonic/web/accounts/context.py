"""Template context: `can.<capability>` tells templates which menu entries the current user may see."""

from . import roles as role_rules


def roles(request):
    user = request.user
    if not user.is_authenticated:
        return {"can": {}}
    return {"can": {cap: user.can(cap) for cap in role_rules.CAPABILITIES}}
