"""Administrator settings: decision thresholds, alert rules and data retention (FR xxxv, xxxvi, liii, lxxx)."""

from __future__ import annotations

import json

from django.contrib import messages
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_POST

from sonic.detection import rules as R

from ..accounts.audit import Action, record as audit
from ..accounts.decorators import capability_required
from . import anomalies, retention
from .forms import AlertRuleForm, DecisionSettingsForm
from .models import AdminNotice, AlertRule, DecisionSettings, current_ruleset


def changes(form) -> dict:
    """{field: [old, new]} for the fields the form changed (kept in the audit trail)."""
    return {name: [form.initial.get(name), form.cleaned_data.get(name)] for name in form.changed_data}


@capability_required("administer")
def settings_page(request):
    config = DecisionSettings.load()
    form = DecisionSettingsForm(request.POST or None, instance=config)
    if request.method == "POST" and form.is_valid():
        changed = changes(form)
        saved = form.save(commit=False)
        saved.updated_by = request.user
        saved.save()
        if changed:
            audit(Action.SETTINGS, request, target=saved, changed=json.loads(json.dumps(changed, default=str)))
        messages.success(request, "Settings saved." if changed else "Nothing changed.")
        return redirect("settings:decision")
    return render(request, "settings/decision.html", {"form": form, "config": config, "plan": retention.plan()})


@require_POST
@capability_required("administer")
def run_retention(request):
    result = retention.apply(user=request.user, request=request)
    messages.success(request, f"Retention applied: {result.audio_files} audio file(s) and {result.event_records} "
                              f"event record(s) deleted; {result.kept_open} old record(s) kept because they still "
                              f"need review or have an open alert.")
    return redirect("settings:decision")


@capability_required("administer")
def rule_list(request):
    current_ruleset()  # seeds the rules from alert_rules.json the first time
    return render(request, "settings/rules.html", {"rules": AlertRule.objects.all()})


@capability_required("administer")
def rule_edit(request, category: str):
    rule = get_object_or_404(AlertRule, category=category)
    form = AlertRuleForm(request.POST or None, instance=rule)
    if request.method == "POST" and form.is_valid():
        changed = changes(form)
        form.save()
        if changed:
            audit(Action.SETTINGS, request, target=rule, rule=rule.category,
                  changed=json.loads(json.dumps(changed, default=str)))
        messages.success(request, f"Rule for {rule.category} saved." if changed else "Nothing changed.")
        return redirect("settings:rules")
    return render(request, "settings/rule_edit.html", {"form": form, "rule": rule})


@capability_required("administer")
def rules_export(request):
    """The current rules in the alert_rules.json format, to keep in the repository or move to another install."""
    data = R.dump_rules(current_ruleset())
    audit(Action.EXPORT, request, what="alert rules", rules=len(data["rules"]))
    response = HttpResponse(json.dumps(data, indent=2), content_type="application/json")
    response["Content-Disposition"] = f'attachment; filename="alert_rules_{timezone.localtime():%Y%m%d-%H%M}.json"'
    return response


@capability_required("administer")
def notices(request):
    anomalies.check()
    return render(request, "settings/notices.html", {
        "open": AdminNotice.objects.filter(acknowledged_at__isnull=True),
        "done": AdminNotice.objects.filter(acknowledged_at__isnull=False).select_related("acknowledged_by")[:50],
        "limits": anomalies,
    })


@require_POST
@capability_required("administer")
def acknowledge_notice(request, pk: int | None = None):
    pending = AdminNotice.objects.filter(acknowledged_at__isnull=True)
    if pk is not None:
        pending = pending.filter(pk=pk)
    count = pending.update(acknowledged_at=timezone.now(), acknowledged_by=request.user)
    audit(Action.SETTINGS, request, acknowledged_notices=count)
    messages.success(request, f"{count} notice{'s' if count != 1 else ''} acknowledged.")
    return redirect("settings:notices")
