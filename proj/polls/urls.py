from django.urls import path
from polls.views import *
from . import views
app_name = 'polls'
urlpatterns = [
    path('', views.index, name='index'),
    path('conkurs/<int:conkurs_id>/', views.conkurs, name='conkurs'),
    path('doc/<int:doc_id>/', views.doc, name='doc'),
    path('doc_z/<int:doc_id>/', views.doc_z, name='doc_z'),
    path('list_zay/', views.list_zay, name='list_zay'),######################
    path('results/<int:results_id>/', views.results, name='results'),
    # #ДОБАВЛЕНИЕ НОВЫХ ОБЪЕКТОВ
    # #Конкурс
    path('new_conkurs/', views.new_conkurs, name='new_conkurs'),
    # # Организация
    path('new_org/', views.new_org, name='new_org'),
    # #Заявка
    path('new_zay/', views.new_zay, name='new_zay'),
    
    
    # #РЕДАКТИРОВАНИЕ ОБЪЕКТОВ
    path('edit_conkurs/<int:conkurs_id>/', views.edit_conkurs, name='edit_conkurs'),
]