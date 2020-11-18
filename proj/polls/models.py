#-*- coding: utf-8 -*-
from django.core.files.storage import FileSystemStorage
from django.conf import settings

from django.db import models
from django.contrib.auth.models import AbstractUser
import datetime
from django.db import models
from django.utils import timezone
from django.contrib.auth.models import AbstractUser

# Create your models here.
private_storage = FileSystemStorage(location=settings.PRIVATE_STORAGE_ROOT)
media_storage = FileSystemStorage(location=settings.MEDIA_ROOT)
USER_STATUSES = (
        ('A', 'Администратор'),
        ('S', 'Специалист'),
        ('O', 'Организация'),
    )
TYPE_PROOF = (
        ('STAMP1', 'Выявление печатей'),
        ('REC', 'Сверка реквизитов'),
        ('STAMP2', 'Печать без реквизитов'),
        ('STAMP3', 'Разные печати'),
        ('SIGN1', 'Выявленеи подписей'),
        ('SIGN2', 'Разные подписи'),
        ('1', 'ОС-1'),
        ('6', 'ОС-6'),
        ('NULL', 'Пустые страницы'),
        
    )
list_PROOF = {'item':['Выявление печатей','Сверка реквизитов', 'Печать без реквизитов','Разные печати','Выявленеи подписей','Разные подписи', 'ОС-1','ОС-6','Пустые страницы']}

class CreateUser(models.Model):
    ROLES = (
        ('A', 'Администратор'),
        ('S', 'Специалист'),
        ('O', 'Организация'),

    )

    name= models.CharField(max_length=100, null=True)
    last_name= models.CharField(max_length=100, null=True)
    roles = models.CharField(max_length=50, choices = ROLES, null=True)
    date_joined = models.DateField(auto_now_add=True)

    def __str__(self):
        return self.name

    class Meta:
        verbose_name = 'Роль'
        verbose_name_plural = 'Роли'

#Класс конкурса
class Conkurs(models.Model):
    name = models.CharField(max_length = 128, verbose_name='Название конкурса')
    logo = models.FileField('Изображение', storage=private_storage, default='settings.MEDIA_ROOT/anonymous.png')
    definition = models.CharField(max_length = 1000, verbose_name='Описание')
    #criteria = models.CharField(default=list_PROOF, max_length = 2000, verbose_name='Критерии проверки')
    def __str__(self):
        return self.name

    class Meta:
        verbose_name = 'Конкурс'
        verbose_name_plural = 'Конкурсы'

#Класс документа-образца
class DocExample(models.Model):
    name = models.CharField(max_length = 1000, verbose_name='Название документа')
    doc = models.FileField('Документ-образец', storage=private_storage, blank=True)
    mask = models.FileField('Документ-маска', storage=private_storage, blank=True)
    conkurs = models.ForeignKey(Conkurs, on_delete=models.CASCADE, null=True, verbose_name='Конкурс', related_name='conkurs')

    def __str__(self):
        return self.name

    class Meta:
        verbose_name = 'Образец'
        verbose_name_plural = 'Образцы'

class Criteria(models.Model):
    name = models.CharField(max_length = 128, verbose_name='Критерий')
    criteria = models.ManyToManyField(Conkurs, blank=True, verbose_name='Конкурс', related_name='criteria')
    def __str__(self):
        return self.name

    class Meta:
        verbose_name = 'Критерий'
        verbose_name_plural = 'Критерии'
#Организации
class Org(models.Model):
    name = models.CharField(max_length = 500, verbose_name='Название')
    INN = models.CharField(max_length = 12, verbose_name='ИНН')
    conkurs = models.ManyToManyField(Conkurs, blank=True)
    def __str__(self):
        return self.name

    class Meta:
        verbose_name = 'Организация'
        verbose_name_plural = 'Организации'
# Заявка
class Zayvka(models.Model):
    conkurs = models.ForeignKey(Conkurs, on_delete=models.CASCADE, null=True, verbose_name='Конкурс', related_name='zay')    
    org = models.ForeignKey(Org, on_delete=models.CASCADE, null=True, verbose_name='Организация', related_name='org')


    class Meta:
        verbose_name = 'Заявка'
        verbose_name_plural = 'Завки'
# Документы к заявке
class DocZayvka(models.Model):
    name = models.CharField(max_length = 1000, verbose_name='Название документа')
    doc = models.FileField('Документ', storage=private_storage, blank=True)
    mask = models.FileField('Маска', storage=private_storage, blank=True)
    zay = models.ForeignKey(Zayvka, on_delete=models.CASCADE, null=True, verbose_name='Заявка', related_name='zay')

    def __str__(self):
        return self.name

    class Meta:
        verbose_name = 'Документ в заявке'
        verbose_name_plural = 'Документы в заявке'