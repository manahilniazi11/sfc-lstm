"""Readable record IDs such as AUD-000042: the prefix names the record type, the number is the primary key."""

from __future__ import annotations

from django.db import models, transaction


class CodedModel(models.Model):
    """Abstract base: fills `code` with PREFIX-000001 on the first save."""

    CODE_PREFIX = "REC"
    code = models.CharField(max_length=16, unique=True, editable=False, blank=True)

    class Meta:
        abstract = True

    def save(self, *args, **kwargs):
        if self.code:
            return super().save(*args, **kwargs)
        with transaction.atomic():
            super().save(*args, **kwargs)
            self.code = f"{self.CODE_PREFIX}-{self.pk:06d}"
            type(self).objects.filter(pk=self.pk).update(code=self.code)

    def __str__(self) -> str:
        return self.code
