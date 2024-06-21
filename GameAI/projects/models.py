from django.db import models


class Paragraphs(models.Model):
    detail = models.CharField(max_length=2000)

    def __str__(self):
        return self.detail

class Answer(models.Model):
    answer = models.FloatField()
    details = models.ForeignKey(Paragraphs, on_delete=models.CASCADE)