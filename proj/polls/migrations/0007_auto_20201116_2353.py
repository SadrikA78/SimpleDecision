# Generated by Django 2.1 on 2020-11-16 20:53

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('polls', '0006_conkurs_criteria'),
    ]

    operations = [
        migrations.AlterField(
            model_name='conkurs',
            name='criteria',
            field=models.CharField(default=['Выявление печатей', 'Сверка реквизитов', 'Печать без реквизитов', 'Разные печати', 'Выявленеи подписей', 'Разные подписи', 'ОС-1', 'ОС-6', 'Пустые страницы'], max_length=2000, verbose_name='Критерии проверки'),
        ),
    ]
