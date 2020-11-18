# -*- coding: utf-8 -*-
from django.contrib.admin.models import LogEntry
from django.template import loader
from django.http import HttpResponse, HttpResponseRedirect
from .models import *
from django.http import Http404
from django.shortcuts import get_object_or_404, render
from django.urls import reverse
from django.views import generic
from django.utils import timezone
from django.contrib.auth.decorators import login_required
import plotly.offline as opy
import plotly.graph_objs as go
import datetime
from plotly.offline import plot
from plotly.graph_objs import *
import plotly.express as px
import numpy as np
import pandas as pd
import sys
import random
from .forms import *
from django.core.files.storage import FileSystemStorage
from django.http import QueryDict
from django.utils.datastructures import MultiValueDict

from absl import logging
import os
from pdf2image import convert_from_path
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from scipy.spatial import cKDTree
from skimage.feature import plot_matches
from skimage.measure import ransac
from skimage.transform import AffineTransform
from six import BytesIO
import tensorflow as tf
import tensorflow_hub as hub
from six.moves.urllib.request import urlopen
import cv2
import matplotlib.pyplot as plt
from skimage import measure, morphology
from skimage.color import label2rgb
from skimage.measure import regionprops
import filetype
from wand.image import Image
#from PIL import Image, ImageFont
import cv2 as cv
import matplotlib.pyplot as plt
from skimage import measure, morphology
from skimage.color import label2rgb
from skimage.measure import regionprops
import matplotlib.patches as mpatches
private_storage = FileSystemStorage(location=settings.PRIVATE_STORAGE_ROOT)
media_storage = FileSystemStorage(location=settings.MEDIA_ROOT)
delf = hub.load('model').signatures['default']
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images
def download_and_resize(name,new_width=256, new_height=256):
  image = Image.open(name)
  image = ImageOps.fit(image, (new_width, new_height), Image.ANTIALIAS)
  image = image.convert('RGB')
  return image
def resize_image(name, destfile, new_width=256, new_height=256):
    pil_image = Image.open(name)
    pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.ANTIALIAS)
    pil_image_rgb = pil_image.convert('RGB')
    pil_image_rgb.save(destfile, format='JPEG', quality=90)
def run_delf(image):
  np_image = np.array(image)
  float_image = tf.image.convert_image_dtype(np_image, tf.float32)

  return delf(
      image=float_image,
      score_threshold=tf.constant(100.0),
      image_scales=tf.constant([0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0]),
      max_feature_num=tf.constant(1000))
def filter_value( someList, value ):
    for x, y in someList:
        if x == value :
            yield x,y
def match_images(image1, image2):
  result1 = run_delf(image1)
  result2 = run_delf(image2)
  distance_threshold = 0.8

  # Read features.
  num_features_1 = result1['locations'].shape[0]
  print("Документ 1: %d признаков" % num_features_1)
  
  num_features_2 = result2['locations'].shape[0]
  print("Документ 2: %d признаков" % num_features_2)

  # Find nearest-neighbor matches using a KD tree.
  d1_tree = cKDTree(result1['descriptors'])
  _, indices = d1_tree.query(
      result2['descriptors'],
      distance_upper_bound=distance_threshold)

  # Select feature locations for putative matches.
  locations_2_to_use = np.array([
      result2['locations'][i,]
      for i in range(num_features_2)
      if indices[i] != num_features_1
  ])
  locations_1_to_use = np.array([
      result1['locations'][indices[i],]
      for i in range(num_features_2)
      if indices[i] != num_features_1
  ])

  # Perform geometric verification using RANSAC.
  _, inliers = ransac(
      (locations_1_to_use, locations_2_to_use),
      AffineTransform,
      min_samples=3,
      residual_threshold=20,
      max_trials=1000)

  print('Найдено %d схожих дескрипторов' % sum(inliers))
  return {"inliers": sum(inliers)}
class CVOutput:
    def __init__(self,image,error):
        self.image = image
        self.error = error
        
class SignatureItem:
    def __init__(self,minr,minc,maxr,maxc):
        self.minr = minr
        self.minc = minc
        self.maxr = maxr
        self.maxc = maxc
MIN_CONNECTED_THRESHOLD = 680
MAX_CONNECTED_THRESHOLD = 8000
DebugMode = True
def signature(INPUTIMAGE):
    image = cv2.imread(INPUTIMAGE,cv2.IMREAD_GRAYSCALE)


    # We NEED thresholding to remove vertical and horizontal lines
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Remove horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
    remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(image, [c], -1, (255,255,255), 5)
         
    # Remove vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,40))
    remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(image, [c], -1, (255,255,255), 5)


    ObjOut = CVOutput(image,None)
    img = ObjOut.image

    # Uncomment the following line to create an image without lines       
    #cv2.imwrite("./inputs/out5.png",ObjOut.image)

    #img = image
    # connected component analysis by scikit-learn framework
    blobs = img > img.mean()
    blobs_labels = measure.label(blobs, background=1)
    image_label_overlay = label2rgb(blobs_labels, image=img)

    fig, ax = plt.subplots(figsize=(10, 6))

    the_biggest_component = 0

    AllSignatures = []

    for region in regionprops(blobs_labels):

        if (region.area >= MIN_CONNECTED_THRESHOLD and region.area < MAX_CONNECTED_THRESHOLD):
            if (region.area > the_biggest_component):
                the_biggest_component = region.area
                
            minr, minc, maxr, maxc = region.bbox
            signature = SignatureItem(minr, minc, maxr, maxc)
            #print("debug:"+str(minr))
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            AllSignatures.append(signature)
        
    print("Найдено рукописных подписей:" + str(len(AllSignatures)))
    return len(AllSignatures)

#Главная страница
@login_required
def index(request):
    conkurs = Conkurs.objects.all()
    docs = DocExample.objects.all()
    orgs = Org.objects.all()
    zayas = Zayvka.objects.all()
    content = {'conkurs':conkurs, 'docs':docs, 'orgs':orgs, 'zayas':zayas}
    return render(request, 'polls/index.html', content)
    
    
#Новый конкурс
@login_required
def new_conkurs(request):
    conkurs = Conkurs.objects.all()
    if request.POST:
            
        new_conkurs = Conkurs(
                name=request.POST['name'], definition=request.POST['definition'], logo=request.FILES['logo']
                )
        
        new_conkurs.save()
        for i  in request.FILES.getlist('files[]'):
            new_doc = DocExample (
                        name=str(i), doc=i, conkurs = new_conkurs
                        )
            new_doc.save()
        return HttpResponseRedirect('/')
    content = {'conkurs':conkurs}
    return render(request, 'polls/create/new_conkurs.html', content)

#Просмотр конкурса
@login_required
def conkurs(request, conkurs_id):
    conkurs = Conkurs.objects.all()
    real_conkurs = Conkurs.objects.get(id=conkurs_id)
    docs = DocExample.objects.filter(conkurs=real_conkurs)
    print(request.user)
    content = {'conkurs':conkurs, 'real_conkurs':real_conkurs, 'docs':docs, 'TYPE_PROOF':TYPE_PROOF, 'user':request.user}
    return render(request, 'polls/conkurs.html', content)

#Настройка проверки конкурса
@login_required
def edit_conkurs(request, conkurs_id):
    conkurs = Conkurs.objects.all()
    real_conkurs = Conkurs.objects.get(id=conkurs_id)
    docs = DocExample.objects.filter(conkurs=real_conkurs)
    if request.POST:
        
        new_conkurs = Conkurs(
                    name=real_conkurs.name, definition=request.POST['definition'], logo=real_conkurs.logo
                    )
        new_conkurs.save()
        for i  in request.FILES.getlist('files[]'):
            new_doc = DocExample (
                        name=str(i), doc=i, conkurs = new_conkurs
                        )
            new_doc.save()
        for i  in request.POST.getlist('docs'):
            new_doc = DocExample (
                        name=str(i), doc=i, conkurs = new_conkurs
                        )
            new_doc.save()
        # for i in request.POST.getlist('criteria'):
            # new_criteria = Criteria(name=str(i), criteria=new_conkurs)
        docs.delete()
        real_conkurs.delete()
        return HttpResponseRedirect('/conkurs/'+str(new_conkurs.id))
    content = {'conkurs':conkurs, 'real_conkurs':real_conkurs, 'docs':docs, 'TYPE_PROOF':TYPE_PROOF, 'user':request.user}
    return render(request, 'polls/edit_conkurs.html', content)    
#Просмотр документа
@login_required
def doc(request, doc_id):
    conkurs = Conkurs.objects.all()
    doc = DocExample.objects.get(id=doc_id)
    content = {'conkurs':conkurs, 'doc':doc}
    return render(request, 'polls/doc.html', content)
@login_required
def doc_z(request, doc_id):
    conkurs = Conkurs.objects.all()
    doc = DocZayvka.objects.get(id=doc_id)
    content = {'conkurs':conkurs, 'doc':doc}
    return render(request, 'polls/doc.html', content)    
#Список заявок
@login_required
def list_zay(request):
    conkurs = Conkurs.objects.all()
    content = {'conkurs':conkurs}
    return render(request, 'polls/list_zay.html', content)
    
# Новая организация
@login_required
def new_org(request):
    conkurs = Conkurs.objects.all()
    if request.POST:
        new_org = Org(name=request.POST['name'], INN = request.POST['INN'])
        new_org.save()
        return HttpResponseRedirect('/')
    content = {'conkurs':conkurs}
    return render(request, 'polls/create/new_org.html', content)

# Новая заявка
@login_required
def new_zay(request):
    conkurs = Conkurs.objects.all()
    orgs = Org.objects.all()
    if request.POST:
        new_zay = Zayvka(conkurs=Conkurs.objects.get(id=request.POST['conkurs']), org=Org.objects.get(id=request.POST['orgs']))
        new_zay.save()
        for i  in request.FILES.getlist('files[]'):
            new_doc = DocZayvka (
                        name=str(i), doc=i, zay = new_zay
                        )
            new_doc.save()
        return HttpResponseRedirect('/results/'+str(new_zay.id))
    content = {'conkurs':conkurs, 'orgs': orgs}
    return render(request, 'polls/create/new_zay.html', content)
#Результат
@login_required
def results(request, results_id):
    conkurs = Conkurs.objects.all()
    zay = Zayvka.objects.get(id=results_id)
    docs_z = DocZayvka.objects.filter(zay=zay)
    docs_example = DocExample.objects.filter(conkurs=zay.conkurs)
    list_examples_files = []
    result = {}
    list_doc_z = []
    for i in docs_example:
        kind = filetype.guess(i.doc)
        if kind.extension=="pdf":
            inputpath = "./polls/static/files/" + str(i)
            outputpath = (str(settings.PRIVATE_STORAGE_ROOT)+'/examples/'+str(str(i).split('.pdf')[0]))
            list_examples_files.append(outputpath)
            try:
                os.mkdir(outputpath)
                with(Image(filename=inputpath,resolution=80)) as source:
                    images=source.sequence
                    pages=len(images)
                    for j in range(pages):
                        Image(images[j]).save(filename=outputpath+'/'+str(j)+'.png')
            except FileExistsError:
                pass
            
    for i in docs_z:
        kind = filetype.guess(i.doc)
        if kind.extension=="pdf":
            inputpath = "./polls/static/files/" + str(i)
            outputpath = (str(settings.PRIVATE_STORAGE_ROOT)+'/'+str(str(i).split('.pdf')[0]))
            try:
                os.mkdir(outputpath)
                with(Image(filename=inputpath,resolution=80)) as source:
                    images=source.sequence
                    pages=len(images)
                    for j in range(pages):
                        Image(images[j]).save(filename=outputpath+'/'+str(j)+'.png')
                        
            except FileExistsError:
                pass
            list_rasbor=[]
            for png in os.listdir(outputpath):
                
                sign = random.randint(1,27)
                rasbor = {'page':png, 'sign':sign}
                list_rasbor.append(rasbor)
            doc_to_test = load_images_from_folder(outputpath)
            doc_to_example = load_images_from_folder(list_examples_files[0])
            list_page =[]
            for index, page in enumerate(doc_to_test):
                for index2, example in enumerate(doc_to_example):
                    try:
                        info_page = {'page': index, 'example':index2, 'error': None, 'results':match_images(page, example)}
                        
                    except ValueError:
                        info_page = {'page': index, 'example':index2, 'error': 'pull', 'results':None}
                    list_page.append(info_page)
            list_doc_z.append({'name':i.name, 'type': kind.extension, 'test': list_examples_files[0], 'example': list_examples_files[0], 'sign': list_rasbor, 'page':list_page})
        elif kind.extension=="jpg" or kind.extension=="png":
            list_doc_z.append({'name':i.name, 'type': kind.extension})
        #result1 = run_delf(i.doc)
    result['res'] = list_doc_z
    content = {'conkurs':conkurs, 'zay':zay, 'docs_z':docs_z, 'result':result, 'docs_example':docs_example}
    return render(request, 'polls/results.html', content)