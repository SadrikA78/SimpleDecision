# Generated by Django 2.1 on 2020-11-16 21:15

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('polls', '0008_auto_20201117_0008'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='criteria',
            name='criteria',
        ),
        migrations.AddField(
            model_name='criteria',
            name='criteria',
            field=models.ManyToManyField(blank=True, related_name='criteria', to='polls.Conkurs', verbose_name='Конкурс'),
        ),
    ]
