# Generated by Django 2.1 on 2020-11-16 20:33

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('polls', '0005_docexample_conkurs'),
    ]

    operations = [
        migrations.AddField(
            model_name='conkurs',
            name='criteria',
            field=models.CharField(default='', max_length=2000, verbose_name='Критерии проверки'),
        ),
    ]
